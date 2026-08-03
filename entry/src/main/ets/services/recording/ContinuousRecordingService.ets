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
import { SegmentEvent, SegmenterLevels, SpeechSegmenter } from './SpeechSegmenter';
import { SegmentRecord, SegmentStore } from './SegmentStore';
import { TranscribeFn, TranscriptRecord, TranscriptionQueue } from './TranscriptionQueue';

const DOMAIN = 0x0001;
const TAG = 'ContinuousRecording';

/** Audio older than this is discarded, unless it is still waiting to be transcribed. */
const DEFAULT_RETENTION_MS: number = 24 * 60 * 60 * 1000;
/** Backstop for a backlog no event will move on its own. */
const RETRY_INTERVAL_MS: number = 5000;

export interface RecordingStatus {
  recording: boolean;
  segments: number;
  transcribed: number;
  pending: number;
  storedBytes: number;
  /** Most recent frame level, and the bar it had to clear to start a segment. */
  inputLevel: number;
  inputThreshold: number;
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
  private lastLevelNotifyAt: number = 0;
  private engineReady: boolean = false;
  private retryTimer: number = -1;

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
   *
   * @param engine names the recogniser behind `transcribe`. Stored results from a
   *   different one are discarded on load and the audio is transcribed again, so a fix
   *   to recognition also repairs recordings it already got wrong.
   */
  init(filesDir: string, transcribe: TranscribeFn, engine: string): void {
    this.transcribe = transcribe;
    const recovered: SegmentRecord[] = this.store.init(filesDir);
    this.queue.init(filesDir, engine);
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

  /**
   * How many stored transcripts the current recogniser disowned at startup.
   *
   * Worth surfacing: after an upgrade the pending count jumps for reasons the screen
   * cannot otherwise explain, and "the recogniser changed, this is a re-run" is a very
   * different thing to read than an unexplained backlog.
   */
  redoCount(): number {
    return this.queue.staleCount();
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
      const row: TimelineRow = {
        segment: segment,
        transcript: this.queue.transcriptFor(segment.fingerprint),
      };
      rows.push(row);
    }
    rows.sort((a: TimelineRow, b: TimelineRow): number =>
      b.segment.startedAt - a.segment.startedAt);
    return rows;
  }

  status(): RecordingStatus {
    const pending: Set<string> = this.queue.pendingIds(this.segments);
    const levels: SegmenterLevels = this.segmenter.levelSnapshot();
    const snapshot: RecordingStatus = {
      recording: this.recording,
      segments: this.segments.length,
      transcribed: this.segments.length - pending.size,
      pending: pending.size,
      storedBytes: this.store.totalBytes(),
      inputLevel: levels.level,
      inputThreshold: levels.threshold,
    };
    return snapshot;
  }

  private onChunk(chunk: AudioChunk): void {
    if (!this.recording) {
      return;
    }
    try {
      this.applyEvents(this.segmenter.push(new Int16Array(chunk.pcmData)));
      // Push the level up a few times a second even when no segment opens. Without it
      // the screen is identical whether audio is arriving and being rejected or not
      // arriving at all — which is exactly the ambiguity that made the first broken
      // threshold so hard to see.
      const now: number = Date.now();
      if (now - this.lastLevelNotifyAt >= 300) {
        this.lastLevelNotifyAt = now;
        this.notify();
      }
    } catch (error) {
      // A failure here must not stop capture; the next buffer is independent.
      hilog.error(DOMAIN, TAG, `chunk handling failed: ${error}`);
    }
  }

  private applyEvents(events: SegmentEvent[]): void {
    for (let i: number = 0; i < events.length; i++) {
      const event: SegmentEvent = events[i];
      if (event.kind === 'open') {
        // backdateMs is the pre-roll only, so this is genuinely when the speech started.
        this.store.begin(Date.now() - Math.round(event.backdateMs));
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
  /**
   * Tells the service the recogniser can accept work.
   *
   * Segments recorded while the models were still loading used to be handed over anyway,
   * fail for that reason alone, and burn their retry budget doing it - three of them
   * were enough to exhaust it, after which nothing ever tried again and they sat at
   * "pending" forever. Readiness is a precondition for attempting, not a kind of
   * failure, so the queue is not touched until it holds.
   */
  setEngineReady(ready: boolean): void {
    if (this.engineReady === ready) {
      return;
    }
    this.engineReady = ready;
    if (ready) {
      // Anything that failed before the engine existed deserves a clean slate.
      this.queue.resetAttempts();
      this.kickTranscription();
      return;
    }
    // The recogniser is going away; a pending retry would only hand work to a worker
    // that no longer exists.
    if (this.retryTimer !== -1) {
      clearTimeout(this.retryTimer);
      this.retryTimer = -1;
    }
  }

  private kickTranscription(): void {
    const transcribe: TranscribeFn | null = this.transcribe;
    if (transcribe === null || !this.engineReady) {
      return;
    }
    this.queue.drain(this.segments, transcribe)
      .then((completed: number): void => {
        if (completed > 0) {
          this.discardSilentSegments();
          this.pruneExpired();
          this.notify();
        }
        this.scheduleRetryIfPending();
      })
      .catch((error: Error): void => {
        hilog.warn(DOMAIN, TAG, `transcription pass failed: ${error.message}`);
        this.scheduleRetryIfPending();
      });
  }

  /**
   * Re-drives the queue while anything is still outstanding.
   *
   * Every other trigger is an event - a segment closing, the engine becoming ready - and
   * events stop arriving. Without a timer, a backlog left after the last segment closes
   * has nothing to move it, which is how three recordings ended up sitting at "pending"
   * with no way to progress short of leaving the page and coming back.
   */
  private scheduleRetryIfPending(): void {
    if (this.retryTimer !== -1) {
      return;
    }
    if (this.queue.pendingIds(this.segments).size === 0) {
      return;
    }
    this.retryTimer = setTimeout((): void => {
      this.retryTimer = -1;
      this.kickTranscription();
    }, RETRY_INTERVAL_MS);
  }

  /**
   * Deletes segments the recogniser found no speech in.
   *
   * This is the second half of a deliberately permissive capture gate. Energy alone
   * cannot separate far-field speech from office ambience — measured on real meeting
   * audio the two overlap almost entirely — so the gate errs towards recording and the
   * Silero VAD inside the recogniser makes the real decision here.
   *
   * Only an empty result from a *successful* pass counts. A failure throws and is
   * retried by the queue, so a segment is never deleted because recognition had a bad
   * moment.
   */
  private discardSilentSegments(): void {
    const keep: SegmentRecord[] = [];
    let dropped: number = 0;
    for (let i: number = 0; i < this.segments.length; i++) {
      const segment: SegmentRecord = this.segments[i];
      const transcript = this.queue.transcriptFor(segment.fingerprint);
      if (transcript !== null && transcript.text.length === 0) {
        if (this.store.remove(segment.id)) {
          dropped++;
          continue;
        }
      }
      keep.push(segment);
    }
    if (dropped > 0) {
      this.segments = keep;
      hilog.info(DOMAIN, TAG, `discarded ${dropped} segment(s) with no speech`);
    }
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
