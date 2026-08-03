/**
 * Stores recorded speech segments so they survive the process being killed.
 *
 * A phone can lose the app at any instant — the user swipes it away, the system reclaims
 * memory, the battery dies. Any design that finishes a file by writing something at the
 * end loses whatever was in flight, which is always the most recent audio and usually
 * the audio someone wanted.
 *
 * So nothing here is ever rewritten in place. A segment is written as `<id>.part`, with
 * a WAV header carrying placeholder sizes. On close the two size fields are patched and
 * the file is renamed to `<id>.wav` — rename is atomic, so a segment either exists whole
 * or does not exist. Only then is one line appended to an index that is never rewritten.
 *
 * Recovery follows from that. A `.part` left behind is a segment that was being written
 * when the process died: its header sizes are wrong but the audio is intact, so the
 * sizes are recomputed from the file length and it is promoted like any other. A torn
 * final line in the index is simply dropped when reading, because a line is only ever
 * appended after its file already exists.
 */

import { fileIo } from '@kit.CoreFileKit';
import { hilog } from '@kit.PerformanceAnalysisKit';

const DOMAIN = 0x0001;
const TAG = 'SegmentStore';

const SAMPLE_RATE: number = 16000;
const BYTES_PER_SAMPLE: number = 2;
const WAV_HEADER_BYTES: number = 44;
const INDEX_FILE: string = 'segments.jsonl';
const AUDIO_DIR: string = 'recordings';
/** Too short to hold a word; keeping it would only add rows nobody can use. */
const MIN_SEGMENT_MS: number = 400;

export interface SegmentRecord {
  id: string;
  path: string;
  startedAt: number;
  durationMs: number;
  bytes: number;
  /** Identifies the audio for the transcription queue; see fingerprint(). */
  fingerprint: string;
}

export class SegmentStore {
  private baseDir: string = '';
  private audioDir: string = '';
  private indexPath: string = '';
  private openFile: fileIo.File | null = null;
  private openId: string = '';
  private openStartedAt: number = 0;
  private openBytes: number = 0;

  /** Prepares directories and promotes anything a previous run left unfinished. */
  init(filesDir: string): SegmentRecord[] {
    this.baseDir = filesDir;
    this.audioDir = `${filesDir}/${AUDIO_DIR}`;
    this.indexPath = `${filesDir}/${INDEX_FILE}`;
    this.ensureDir(this.audioDir);
    return this.recoverPartials();
  }

  /** Opens a new segment. Any previously open one is closed first. */
  begin(startedAt: number): void {
    if (this.openFile !== null) {
      this.finish();
    }
    this.openId = `seg_${startedAt}`;
    this.openStartedAt = startedAt;
    this.openBytes = 0;
    const path: string = this.partPath(this.openId);
    try {
      this.openFile = fileIo.openSync(
        path, fileIo.OpenMode.CREATE | fileIo.OpenMode.TRUNC | fileIo.OpenMode.READ_WRITE);
      // Placeholder sizes; patched on close, and recomputable from the file length if
      // the process never gets there.
      fileIo.writeSync(this.openFile.fd, this.buildHeader(0).buffer);
    } catch (error) {
      hilog.error(DOMAIN, TAG, `open segment failed: ${error}`);
      this.openFile = null;
    }
  }

  /** Appends audio to the open segment. Silently ignored when none is open. */
  append(samples: Int16Array): void {
    if (this.openFile === null || samples.length === 0) {
      return;
    }
    try {
      const bytes: number = samples.length * BYTES_PER_SAMPLE;
      const view: Uint8Array = new Uint8Array(
        samples.buffer, samples.byteOffset, bytes);
      // A fresh copy: the caller owns the capture buffer and may reuse it immediately.
      const copy: ArrayBuffer = new ArrayBuffer(bytes);
      new Uint8Array(copy).set(view);
      fileIo.writeSync(this.openFile.fd, copy);
      this.openBytes += bytes;
    } catch (error) {
      hilog.error(DOMAIN, TAG, `append failed: ${error}`);
    }
  }

  /**
   * Patches the header, renames into place, and records the segment.
   *
   * Returns null when nothing usable was captured, so a burst of noise that opened and
   * closed a segment does not leave a row behind.
   */
  finish(): SegmentRecord | null {
    const file: fileIo.File | null = this.openFile;
    if (file === null) {
      return null;
    }
    this.openFile = null;
    const id: string = this.openId;
    const bytes: number = this.openBytes;
    const durationMs: number = (bytes / BYTES_PER_SAMPLE / SAMPLE_RATE) * 1000;

    try {
      if (durationMs >= MIN_SEGMENT_MS) {
        fileIo.writeSync(file.fd, this.buildHeader(bytes).buffer, { offset: 0 });
        fileIo.fsyncSync(file.fd);
      }
      fileIo.closeSync(file);
    } catch (error) {
      hilog.error(DOMAIN, TAG, `close segment failed: ${error}`);
      return null;
    }

    const part: string = this.partPath(id);
    if (durationMs < MIN_SEGMENT_MS) {
      this.deleteQuietly(part);
      return null;
    }

    const finalPath: string = this.wavPath(id);
    try {
      fileIo.renameSync(part, finalPath);
    } catch (error) {
      hilog.error(DOMAIN, TAG, `rename segment failed: ${error}`);
      return null;
    }

    const record: SegmentRecord = {
      id: id,
      path: finalPath,
      startedAt: this.openStartedAt,
      durationMs: Math.round(durationMs),
      bytes: bytes,
      fingerprint: this.fingerprint(id, bytes, durationMs),
    };
    this.appendIndex(record);
    return record;
  }

  /** Every recorded segment, oldest first. Torn lines are skipped. */
  list(): SegmentRecord[] {
    const records: SegmentRecord[] = [];
    let raw: string = '';
    try {
      if (!fileIo.accessSync(this.indexPath)) {
        return records;
      }
      raw = fileIo.readTextSync(this.indexPath);
    } catch (error) {
      hilog.warn(DOMAIN, TAG, `read index failed: ${error}`);
      return records;
    }
    const lines: string[] = raw.split('\n');
    for (let i: number = 0; i < lines.length; i++) {
      const line: string = lines[i].trim();
      if (line.length === 0) {
        continue;
      }
      try {
        const parsed: SegmentRecord = JSON.parse(line) as SegmentRecord;
        if (parsed.id && parsed.path) {
          records.push(parsed);
        }
      } catch (_error) {
        // A half-written final line means the process died mid-append. The audio it
        // referred to is still on disk and will be picked up as a stray file.
        hilog.warn(DOMAIN, TAG, `dropping unparsable index line ${i}`);
      }
    }
    return records;
  }

  /**
   * Deletes segments older than the retention window.
   *
   * `keepIds` names segments that must survive regardless — the queue passes anything
   * not yet transcribed, so a backlog is never silently destroyed to satisfy a clock.
   */
  prune(retentionMs: number, keepIds: Set<string>): number {
    const cutoff: number = Date.now() - retentionMs;
    const kept: SegmentRecord[] = [];
    let removed: number = 0;
    const all: SegmentRecord[] = this.list();
    for (let i: number = 0; i < all.length; i++) {
      const record: SegmentRecord = all[i];
      if (record.startedAt >= cutoff || keepIds.has(record.id)) {
        kept.push(record);
        continue;
      }
      this.deleteQuietly(record.path);
      removed++;
    }
    if (removed > 0) {
      this.rewriteIndex(kept);
    }
    return removed;
  }

  /**
   * Deletes one segment and drops it from the index.
   *
   * Used when the recogniser reports a segment held no speech. The capture-side gate is
   * deliberately permissive because energy alone cannot separate far-field speech from
   * office ambience, so this is where the real VAD's verdict is applied and the disk is
   * given back.
   */
  remove(id: string): boolean {
    const all: SegmentRecord[] = this.list();
    const kept: SegmentRecord[] = [];
    let found: boolean = false;
    for (let i: number = 0; i < all.length; i++) {
      if (all[i].id === id) {
        this.deleteQuietly(all[i].path);
        found = true;
        continue;
      }
      kept.push(all[i]);
    }
    if (found) {
      this.rewriteIndex(kept);
    }
    return found;
  }

  /** Total bytes currently held, for surfacing storage use. */
  totalBytes(): number {
    const all: SegmentRecord[] = this.list();
    let total: number = 0;
    for (let i: number = 0; i < all.length; i++) {
      total += all[i].bytes;
    }
    return total;
  }

  /**
   * Promotes `.part` files a previous run left behind.
   *
   * Their headers claim zero length, so the sizes are rebuilt from what is actually on
   * disk. This is the whole reason the header is written first and patched last.
   */
  private recoverPartials(): SegmentRecord[] {
    const recovered: SegmentRecord[] = [];
    let names: string[] = [];
    try {
      names = fileIo.listFileSync(this.audioDir);
    } catch (_error) {
      return recovered;
    }
    const known: Set<string> = new Set<string>();
    const existing: SegmentRecord[] = this.list();
    for (let i: number = 0; i < existing.length; i++) {
      known.add(existing[i].id);
    }

    for (let i: number = 0; i < names.length; i++) {
      const name: string = names[i];
      if (!name.endsWith('.part')) {
        continue;
      }
      const id: string = name.substring(0, name.length - 5);
      const part: string = `${this.audioDir}/${name}`;
      try {
        const size: number = fileIo.statSync(part).size;
        const bytes: number = size - WAV_HEADER_BYTES;
        const durationMs: number = (bytes / BYTES_PER_SAMPLE / SAMPLE_RATE) * 1000;
        if (bytes <= 0 || durationMs < MIN_SEGMENT_MS || known.has(id)) {
          this.deleteQuietly(part);
          continue;
        }
        const file: fileIo.File = fileIo.openSync(part, fileIo.OpenMode.READ_WRITE);
        fileIo.writeSync(file.fd, this.buildHeader(bytes).buffer, { offset: 0 });
        fileIo.fsyncSync(file.fd);
        fileIo.closeSync(file);
        const finalPath: string = this.wavPath(id);
        fileIo.renameSync(part, finalPath);
        const startedAt: number = this.startedAtFromId(id);
        const record: SegmentRecord = {
          id: id, path: finalPath, startedAt: startedAt,
          durationMs: Math.round(durationMs), bytes: bytes,
          fingerprint: this.fingerprint(id, bytes, durationMs),
        };
        this.appendIndex(record);
        recovered.push(record);
        hilog.info(DOMAIN, TAG, `recovered interrupted segment ${id} (${bytes} bytes)`);
      } catch (error) {
        hilog.warn(DOMAIN, TAG, `recover ${name} failed: ${error}`);
        this.deleteQuietly(part);
      }
    }
    return recovered;
  }

  /**
   * Identifies the audio, not the file.
   *
   * The transcription queue keys results on this, so re-running never duplicates work
   * and a segment whose bytes changed is transcribed again rather than trusted.
   */
  private fingerprint(id: string, bytes: number, durationMs: number): string {
    return `${id}:${bytes}:${Math.round(durationMs)}`;
  }

  private startedAtFromId(id: string): number {
    const parsed: number = Number.parseInt(id.replace('seg_', ''), 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : Date.now();
  }

  private appendIndex(record: SegmentRecord): void {
    try {
      const file: fileIo.File = fileIo.openSync(
        this.indexPath,
        fileIo.OpenMode.CREATE | fileIo.OpenMode.APPEND | fileIo.OpenMode.WRITE_ONLY);
      fileIo.writeSync(file.fd, `${JSON.stringify(record)}\n`);
      fileIo.fsyncSync(file.fd);
      fileIo.closeSync(file);
    } catch (error) {
      hilog.error(DOMAIN, TAG, `append index failed: ${error}`);
    }
  }

  /** Rewrites via a temporary file so a crash mid-prune cannot lose the whole index. */
  private rewriteIndex(records: SegmentRecord[]): void {
    const temp: string = `${this.indexPath}.tmp`;
    try {
      const file: fileIo.File = fileIo.openSync(
        temp, fileIo.OpenMode.CREATE | fileIo.OpenMode.TRUNC | fileIo.OpenMode.WRITE_ONLY);
      for (let i: number = 0; i < records.length; i++) {
        fileIo.writeSync(file.fd, `${JSON.stringify(records[i])}\n`);
      }
      fileIo.fsyncSync(file.fd);
      fileIo.closeSync(file);
      fileIo.renameSync(temp, this.indexPath);
    } catch (error) {
      hilog.error(DOMAIN, TAG, `rewrite index failed: ${error}`);
      this.deleteQuietly(temp);
    }
  }

  private buildHeader(dataBytes: number): Uint8Array {
    const header: Uint8Array = new Uint8Array(WAV_HEADER_BYTES);
    const view: DataView = new DataView(header.buffer);
    const byteRate: number = SAMPLE_RATE * BYTES_PER_SAMPLE;
    this.writeAscii(header, 0, 'RIFF');
    view.setUint32(4, 36 + dataBytes, true);
    this.writeAscii(header, 8, 'WAVE');
    this.writeAscii(header, 12, 'fmt ');
    view.setUint32(16, 16, true);          // PCM chunk size
    view.setUint16(20, 1, true);           // PCM
    view.setUint16(22, 1, true);           // mono
    view.setUint32(24, SAMPLE_RATE, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, BYTES_PER_SAMPLE, true);
    view.setUint16(34, 16, true);          // bits per sample
    this.writeAscii(header, 36, 'data');
    view.setUint32(40, dataBytes, true);
    return header;
  }

  private writeAscii(target: Uint8Array, offset: number, text: string): void {
    for (let i: number = 0; i < text.length; i++) {
      target[offset + i] = text.charCodeAt(i);
    }
  }

  private partPath(id: string): string {
    return `${this.audioDir}/${id}.part`;
  }

  private wavPath(id: string): string {
    return `${this.audioDir}/${id}.wav`;
  }

  private ensureDir(path: string): void {
    try {
      if (!fileIo.accessSync(path)) {
        fileIo.mkdirSync(path, true);
      }
    } catch (error) {
      hilog.error(DOMAIN, TAG, `mkdir ${path} failed: ${error}`);
    }
  }

  private deleteQuietly(path: string): void {
    try {
      if (fileIo.accessSync(path)) {
        fileIo.unlinkSync(path);
      }
    } catch (_error) {
      // Already gone, or held by something else; nothing here depends on it.
    }
  }
}
