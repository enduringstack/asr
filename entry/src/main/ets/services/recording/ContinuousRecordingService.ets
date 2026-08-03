/**
 * Records continuously, stores only speech, and transcribes later.
 *
 * The pieces are deliberately separate — a segmenter that decides boundaries, a store
 * that survives being killed, a queue that catches up on its own schedule — because they
 * fail differently and should not be able to take each other down. This is the seam that
 * joins them, and its one rule is that nothing on the capture path may wait for anything
 * on the recognition path.
 *
 * Capture therefore only ever writes bytes to an open file. Transcription is kicked off
 * after a segment closes, and if it is already busy the new work is simply picked up on
 * its next pass; there is no queue to overflow and no callback to block.
 */

import { hilog } from '@kit.PerformanceAnalysisKit';
import { AudioChunk, AudioRecordingService } from '../audio/AudioRecordingService';
import { SegmentEvent, SpeechSegmenter } from './SpeechSegmenter';
import { SegmentRecord, SegmentStore } from './SegmentStore';
import { TranscribeFn, TranscriptRecord, TranscriptionQueue } from './TranscriptionQueue';

const DOMAIN = 0x0001;
const TAG = 'ContinuousRecording';

/** Audio older than this is discarded, unless it is still waiting to be transcribed. */
const DEFAULT_RETENTION_MS: number = 24 * 60 * 60 * 1000;

export interface RecordingStatus {
  recording: boolean;
  segments: number;
  transcribed: number;
  pending: number;
  storedBytes: number;
}

/** One stored segment paired with its transcript, if it has one yet. */
export interface TimelineRow {
  segment: SegmentRecord;
  transcript: TranscriptRecord | null;
}

export type StatusCallback = (status: RecordingStatus) => void;

export class ContinuousRecordingService {
  private static instance: ContinuousRecordingService | null = null;
  private audio: AudioRecordingService = AudioRecordingService.getInstance();
  private segmenter: SpeechSegmenter = new SpeechSegmenter();
  private store: SegmentStore = new SegmentStore();
  private queue: TranscriptionQueue = new TranscriptionQueue();
  private segments: SegmentRecord[] = [];
  private transcribe: TranscribeFn | null = null;
  private onStatus: StatusCallback | null = null;
  private recording: boolean = false;
  private retentionMs: number = DEFAULT_RETENTION_MS;

  static getInstance(): ContinuousRecordingService {
    if (ContinuousRecordingService.instance === null) {
      ContinuousRecordingService.instance = new ContinuousRecordingService();
    }
    return ContinuousRecordingService.instance;
  }

  /**
   * Prepares storage and adopts anything the previous run left behind.
   *
   * Recovery happens before the first byte of new audio, so an interrupted segment is
   * already a normal segment by the time capture resumes.
   */
  init(filesDir: string, transcribe: TranscribeFn): void {
    this.transcribe = transcribe;
    const recovered: SegmentRecord[] = this.store.init(filesDir);
    this.queue.init(filesDir);
    this.segments = this.store.list();
    if (recovered.length > 0) {
      hilog.info(DOMAIN, TAG, `adopted ${recovered.length} interrupted segment(s)`);
    }
    this.pruneExpired();
    this.notify();
    // Anything left untranscribed from last time is caught up now rather than waiting
    // for the user to speak again.
    this.kickTranscription();
  }

  setStatusCallback(callback: StatusCallback | null): void {
    this.onStatus = callback;
  }

  setRetentionMs(retentionMs: number): void {
    this.retentionMs = Math.max(60 * 1000, retentionMs);
  }

  async start(): Promise<void> {
    if (this.recording) {
      return;
    }
    this.segmenter.reset();
    this.audio.setOnChunkCallback((chunk: AudioChunk) => {
      this.onChunk(chunk);
    });
    await this.audio.start();
    this.recording = true;
    this.notify();
  }

  async stop(): Promise<void> {
    if (!this.recording) {
      return;
    }
    this.recording = false;
    await this.audio.stop();
    // Close whatever was mid-utterance when the user stopped; that last segment is the
    // one they are most likely to be waiting for.
    this.applyEvents(this.segmenter.flush());
    this.notify();
    this.kickTranscription();
  }

  transcripts(): TranscriptRecord[] {
    return this.queue.transcripts();
  }

  /**
   * Every stored segment with its transcript, or null while one is still queued.
   *
   * The list is built from segments rather than from transcripts so that audio which
   * has been captured but not yet recognised is visible too. Showing only finished rows
   * would make the gap between recording and transcribing invisible, which is exactly
   * the thing someone watching this screen is trying to judge — and it would make a
   * stalled queue look like silence.
   *
   * Newest first: the reason to open this screen is almost always what just happened.
   */
  timeline(): TimelineRow[] {
    const rows: TimelineRow[] = [];
    for (let i: number = 0; i < this.segments.length; i++) {
      const segment: SegmentRecord = this.segments[i];
      rows.push({
        segment: segment,
        transcript: this.queue.transcriptFor(segment.fingerprint),
      });
    }
    rows.sort((a: TimelineRow, b: TimelineRow): number =>
      b.segment.startedAt - a.segment.startedAt);
    return rows;
  }

  status(): RecordingStatus {
    const pending: Set<string> = this.queue.pendingIds(this.segments);
    return {
      recording: this.recording,
      segments: this.segments.length,
      transcribed: this.segments.length - pending.size,
      pending: pending.size,
      storedBytes: this.store.totalBytes(),
    };
  }

  private onChunk(chunk: AudioChunk): void {
    if (!this.recording) {
      return;
    }
    try {
      this.applyEvents(this.segmenter.push(new Int16Array(chunk.pcmData)));
    } catch (error) {
      // A failure here must not stop capture; the next buffer is independent.
      hilog.error(DOMAIN, TAG, `chunk handling failed: ${error}`);
    }
  }

  private applyEvents(events: SegmentEvent[]): void {
    for (let i: number = 0; i < events.length; i++) {
      const event: SegmentEvent = events[i];
      if (event.kind === 'open') {
        this.store.begin(Date.now() - Math.round(event.offsetMs));
      } else if (event.kind === 'append') {
        this.store.append(event.samples);
      } else {
        const record: SegmentRecord | null = this.store.finish();
        if (record !== null) {
          this.segments.push(record);
          this.notify();
          this.kickTranscription();
        }
      }
    }
  }

  /**
   * Starts a catch-up pass without waiting for it.
   *
   * Awaiting here would put recognition on the capture path, which is the one thing this
   * design exists to avoid. The queue ignores overlapping calls, so firing on every
   * closed segment is safe.
   */
  private kickTranscription(): void {
    const transcribe: TranscribeFn | null = this.transcribe;
    if (transcribe === null) {
      return;
    }
    this.queue.drain(this.segments, transcribe)
      .then((completed: number): void => {
        if (completed > 0) {
          this.pruneExpired();
          this.notify();
        }
      })
      .catch((error: Error): void => {
        hilog.warn(DOMAIN, TAG, `transcription pass failed: ${error.message}`);
      });
  }

  /** Drops expired audio, never touching segments that still owe a transcript. */
  private pruneExpired(): void {
    const keep: Set<string> = this.queue.pendingIds(this.segments);
    const removed: number = this.store.prune(this.retentionMs, keep);
    if (removed > 0) {
      this.segments = this.store.list();
      hilog.info(DOMAIN, TAG, `pruned ${removed} expired segment(s)`);
    }
  }

  private notify(): void {
    const callback: StatusCallback | null = this.onStatus;
    if (callback !== null) {
      callback(this.status());
    }
  }
}
