/**
 * Transcribes stored segments after the fact, without ever blocking capture.
 *
 * Recording and recognising have opposite constraints. Capture must never stall — a
 * dropped buffer is audio that no longer exists. Recognition is expensive and bursty,
 * and on a phone it competes with everything else for CPU and battery. Coupling them
 * means the recogniser's worst moment becomes the recorder's worst moment, so here they
 * share nothing but a directory.
 *
 * Results are keyed by the segment's fingerprint rather than its position in a list, so
 * the queue is idempotent: running it twice transcribes nothing twice, an interrupted
 * run resumes exactly where it stopped, and a segment whose bytes somehow changed is
 * redone rather than trusted. That property is what lets it be killed at any moment,
 * which on a phone it will be.
 *
 * The fingerprint describes the audio and nothing else, which for a while meant a
 * transcript was kept forever regardless of what produced it. When a decode-path bug was
 * found, every segment it had already mangled stayed mangled: the audio was still on
 * disk, the fix was in the build, and the queue skipped all of it because a result
 * existed. So each result also records which recogniser produced it, and results from a
 * different one are treated as absent — the audio is re-read and re-transcribed. Cheap
 * relative to being permanently stuck with output from a version known to be wrong.
 *
 * Only one segment is in flight at a time. Depth is not the bottleneck — the model is —
 * and a bounded queue keeps a backlog from turning into memory pressure on the device
 * that is still recording.
 */

import { fileIo } from '@kit.CoreFileKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { SegmentRecord } from './SegmentStore';

const DOMAIN = 0x0001;
const TAG = 'TranscriptionQueue';
const TRANSCRIPT_FILE: string = 'transcripts.jsonl';
/** Beyond this a segment is assumed lost rather than slow; see markFailed(). */
const ATTEMPT_LIMIT: number = 3;

export interface TranscriptRecord {
  segmentId: string;
  fingerprint: string;
  text: string;
  startedAt: number;
  durationMs: number;
  transcribedAt: number;
  attempts: number;
  /**
   * Which recogniser produced this text. Results from any other one are re-done rather
   * than shown, so improving recognition also repairs everything already on disk.
   */
  engine: string;
}

/** Hands a segment's audio to whoever owns the recogniser; resolves with the text. */
export type TranscribeFn = (record: SegmentRecord, bytes: Uint8Array) => Promise<string>;

export class TranscriptionQueue {
  private transcriptPath: string = '';
  private done: Map<string, TranscriptRecord> = new Map<string, TranscriptRecord>();
  private attempts: Map<string, number> = new Map<string, number>();
  private running: boolean = false;
  private engine: string = '';
  /** How many stored results the current recogniser disowned on load. */
  private staleOnLoad: number = 0;

  /**
   * @param engine identifies the recogniser whose output may be trusted. Anything on
   *   disk from a different one is discarded and re-transcribed.
   */
  init(filesDir: string, engine: string): void {
    this.transcriptPath = `${filesDir}/${TRANSCRIPT_FILE}`;
    this.engine = engine;
    this.done = new Map<string, TranscriptRecord>();
    const existing: TranscriptRecord[] = this.load();
    for (let i: number = 0; i < existing.length; i++) {
      this.done.set(existing[i].fingerprint, existing[i]);
    }
    hilog.info(DOMAIN, TAG,
      `restored ${this.done.size} transcripts, ${this.staleOnLoad} to redo (engine ${engine})`);
  }

  /** How many stored results were rejected as coming from an older recogniser. */
  staleCount(): number {
    return this.staleOnLoad;
  }

  /** Segment ids with no transcript yet — what retention must not delete. */
  pendingIds(segments: SegmentRecord[]): Set<string> {
    const pending: Set<string> = new Set<string>();
    for (let i: number = 0; i < segments.length; i++) {
      if (!this.done.has(segments[i].fingerprint)) {
        pending.add(segments[i].id);
      }
    }
    return pending;
  }

  /**
   * Clears the failure counters so exhausted segments are eligible again.
   *
   * Called when something that could have caused the failures has changed — the
   * recogniser finishing its load, for instance. Attempt limits exist to stop a
   * genuinely broken segment spinning forever, not to permanently condemn audio that
   * happened to arrive while the engine was still starting.
   */
  resetAttempts(): void {
    this.attempts.clear();
  }

  /** The transcript for one segment, or null while it is still queued. */
  transcriptFor(fingerprint: string): TranscriptRecord | null {
    return this.done.get(fingerprint) ?? null;
  }

  /** Everything transcribed so far, oldest first. */
  transcripts(): TranscriptRecord[] {
    const all: TranscriptRecord[] = [];
    this.done.forEach((value: TranscriptRecord) => {
      all.push(value);
    });
    all.sort((a: TranscriptRecord, b: TranscriptRecord): number =>
      a.startedAt - b.startedAt);
    return all;
  }

  /**
   * Works through everything untranscribed, oldest first.
   *
   * Re-entrant calls return immediately rather than queueing: this is driven by capture
   * events and by the UI, and both can fire while a run is already under way.
   */
  async drain(segments: SegmentRecord[], transcribe: TranscribeFn): Promise<number> {
    if (this.running) {
      return 0;
    }
    this.running = true;
    let completed: number = 0;
    try {
      const ordered: SegmentRecord[] = segments.slice().sort(
        (a: SegmentRecord, b: SegmentRecord): number => a.startedAt - b.startedAt);
      for (let i: number = 0; i < ordered.length; i++) {
        const record: SegmentRecord = ordered[i];
        if (this.done.has(record.fingerprint)) {
          continue;
        }
        const tried: number = this.attempts.get(record.fingerprint) ?? 0;
        if (tried >= ATTEMPT_LIMIT) {
          continue;
        }
        const bytes: Uint8Array | null = this.readSegment(record);
        if (bytes === null) {
          // The file is gone but the index still lists it. Recording it as empty stops
          // the queue retrying a segment that can never succeed.
          this.markDone(record, '', tried + 1);
          continue;
        }
        try {
          const text: string = await transcribe(record, bytes);
          this.markDone(record, text.trim(), tried + 1);
          completed++;
        } catch (error) {
          this.attempts.set(record.fingerprint, tried + 1);
          hilog.warn(DOMAIN, TAG,
            `transcribe ${record.id} failed (attempt ${tried + 1}): ${error}`);
        }
      }
    } finally {
      this.running = false;
    }
    return completed;
  }

  private markDone(record: SegmentRecord, text: string, attempts: number): void {
    const entry: TranscriptRecord = {
      segmentId: record.id,
      fingerprint: record.fingerprint,
      text: text,
      startedAt: record.startedAt,
      durationMs: record.durationMs,
      transcribedAt: Date.now(),
      attempts: attempts,
      engine: this.engine,
    };
    this.done.set(record.fingerprint, entry);
    this.append(entry);
  }

  private readSegment(record: SegmentRecord): Uint8Array | null {
    try {
      if (!fileIo.accessSync(record.path)) {
        return null;
      }
      const size: number = fileIo.statSync(record.path).size;
      if (size <= 0) {
        return null;
      }
      const buffer: ArrayBuffer = new ArrayBuffer(size);
      const file: fileIo.File = fileIo.openSync(record.path, fileIo.OpenMode.READ_ONLY);
      const read: number = fileIo.readSync(file.fd, buffer);
      fileIo.closeSync(file);
      return new Uint8Array(buffer, 0, read);
    } catch (error) {
      hilog.warn(DOMAIN, TAG, `read ${record.id} failed: ${error}`);
      return null;
    }
  }

  private load(): TranscriptRecord[] {
    const records: TranscriptRecord[] = [];
    this.staleOnLoad = 0;
    try {
      if (!fileIo.accessSync(this.transcriptPath)) {
        return records;
      }
      const raw: string = fileIo.readTextSync(this.transcriptPath);
      const lines: string[] = raw.split('\n');
      for (let i: number = 0; i < lines.length; i++) {
        const line: string = lines[i].trim();
        if (line.length === 0) {
          continue;
        }
        try {
          const parsed: TranscriptRecord = JSON.parse(line) as TranscriptRecord;
          if (!parsed.fingerprint) {
            continue;
          }
          // A line written before results carried an engine id has none, so it compares
          // unequal and is redone — which is the intent: those are exactly the ones
          // produced before this was being tracked at all.
          if (parsed.engine !== this.engine) {
            this.staleOnLoad++;
            continue;
          }
          records.push(parsed);
        } catch (_error) {
          hilog.warn(DOMAIN, TAG, `dropping unparsable transcript line ${i}`);
        }
      }
    } catch (error) {
      hilog.warn(DOMAIN, TAG, `read transcripts failed: ${error}`);
    }
    return records;
  }

  private append(entry: TranscriptRecord): void {
    try {
      const file: fileIo.File = fileIo.openSync(
        this.transcriptPath,
        fileIo.OpenMode.CREATE | fileIo.OpenMode.APPEND | fileIo.OpenMode.WRITE_ONLY);
      fileIo.writeSync(file.fd, `${JSON.stringify(entry)}\n`);
      fileIo.fsyncSync(file.fd);
      fileIo.closeSync(file);
    } catch (error) {
      hilog.error(DOMAIN, TAG, `append transcript failed: ${error}`);
    }
  }
}
