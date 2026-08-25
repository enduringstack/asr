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
import { SpeakerCandidate, SpeakerTurn } from '../../common/SpeakerTypes';
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
  /**
   * Wall-clock milliseconds the recogniser held this segment.
   *
   * Recorded because the one question the screen keeps raising — why segments sit unread
   * while recording continues — cannot be answered from a phone by looking at it. Whether
   * a 30 s segment takes 8 seconds or 40 decides whether the queue can ever keep pace
   * with capture, and that number only exists on the device. It travels out with the
   * export so the next one settles the question instead of reopening it.
   */
  decodeMs: number;
  meetingId?: string;
  speakerTurns?: SpeakerTurn[];
  /** Local-only enrollment candidates. RecordingExporter intentionally excludes these. */
  speakerCandidates?: SpeakerCandidate[];
}

export interface TranscriptionResult {
  text: string;
  meetingId: string;
  speakerTurns: SpeakerTurn[];
  speakerCandidates: SpeakerCandidate[];
}

/** Hands a segment's audio to whoever owns the recogniser; resolves with structured output. */
export type TranscribeFn = (record: SegmentRecord,
  bytes: Uint8Array) => Promise<TranscriptionResult>;

export class TranscriptionQueue {
  private transcriptPath: string = '';
  private done: Map<string, TranscriptRecord> = new Map<string, TranscriptRecord>();
  private attempts: Map<string, number> = new Map<string, number>();
  private running: boolean = false;
  private engine: string = '';
  /** How many stored results the current recogniser disowned on load. */
  private staleOnLoad: number = 0;
  /** Fingerprint of the segment inside the recogniser right now; empty when idle. */
  private inFlight: string = '';

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

  /** True while this exact segment is inside the recogniser. */
  isTranscribing(fingerprint: string): boolean {
    return this.inFlight.length > 0 && this.inFlight === fingerprint;
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
          this.markDone(record, {
            text: '', meetingId: '', speakerTurns: [], speakerCandidates: [],
          }, tried + 1, 0);
          continue;
        }
        // Published so the screen can say "transcribing" rather than "queued". They look
        // the same from outside and mean opposite things: one is the queue working, the
        // other is the queue waiting, and only the first is a reason to keep watching.
        this.inFlight = record.fingerprint;
        const began: number = Date.now();
        try {
          const result: TranscriptionResult = await transcribe(record, bytes);
          result.text = result.text.trim();
          this.markDone(record, result, tried + 1, Date.now() - began);
          completed++;
        } catch (error) {
          this.attempts.set(record.fingerprint, tried + 1);
          hilog.warn(DOMAIN, TAG,
            `transcribe ${record.id} failed (attempt ${tried + 1}): ${error}`);
        } finally {
          this.inFlight = '';
        }
      }
    } finally {
      this.running = false;
    }
    return completed;
  }

  private markDone(record: SegmentRecord, result: TranscriptionResult, attempts: number,
    decodeMs: number): void {
    const entry: TranscriptRecord = {
      segmentId: record.id,
      fingerprint: record.fingerprint,
      text: result.text,
      startedAt: record.startedAt,
      durationMs: record.durationMs,
      transcribedAt: Date.now(),
      attempts: attempts,
      engine: this.engine,
      decodeMs: decodeMs,
      meetingId: result.meetingId,
      speakerTurns: result.speakerTurns,
      speakerCandidates: result.speakerCandidates,
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
      // Query embeddings are biometric material. They may live in memory long enough
      // for the user to confirm "记住此人", but only SpeakerProfileStore may persist
      // embeddings and it does so through HUKS. The resumable transcript log keeps the
      // editable turns while deliberately dropping enrollment candidates.
      const persisted: TranscriptRecord = {
        segmentId: entry.segmentId,
        fingerprint: entry.fingerprint,
        text: entry.text,
        startedAt: entry.startedAt,
        durationMs: entry.durationMs,
        transcribedAt: entry.transcribedAt,
        attempts: entry.attempts,
        engine: entry.engine,
        decodeMs: entry.decodeMs,
        meetingId: entry.meetingId,
        speakerTurns: this.persistableTurns(entry.speakerTurns),
        speakerCandidates: [],
      };
      const file: fileIo.File = fileIo.openSync(
        this.transcriptPath,
        fileIo.OpenMode.CREATE | fileIo.OpenMode.APPEND | fileIo.OpenMode.WRITE_ONLY);
      fileIo.writeSync(file.fd, `${JSON.stringify(persisted)}\n`);
      fileIo.fsyncSync(file.fd);
      fileIo.closeSync(file);
    } catch (error) {
      hilog.error(DOMAIN, TAG, `append transcript failed: ${error}`);
    }
  }

  /** Match scores and enrollment candidates are not needed after the live session. */
  private persistableTurns(turns?: SpeakerTurn[]): SpeakerTurn[] {
    const persisted: SpeakerTurn[] = [];
    if (!turns) {
      return persisted;
    }
    for (let i: number = 0; i < turns.length; i++) {
      const turn: SpeakerTurn = turns[i];
      persisted.push({
        turnId: turn.turnId,
        meetingId: turn.meetingId,
        clusterId: turn.clusterId,
        startMs: turn.startMs,
        endMs: turn.endMs,
        text: turn.text,
        identity: turn.identity,
        personId: turn.personId,
        displayName: turn.displayName,
      });
    }
    return persisted;
  }
}
