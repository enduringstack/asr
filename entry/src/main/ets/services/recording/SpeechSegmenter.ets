/**
 * Decides where a continuously recorded stream should be cut into stored segments.
 *
 * Storing everything is the obvious approach and the wrong one: 16 kHz mono S16LE runs
 * 115 MB per hour, and in an office most of that is nobody talking. Cutting on speech
 * instead of on a timer costs far less disk and, more importantly, means every stored
 * segment starts and ends in silence — the recogniser never has to begin mid-word, and
 * later work can treat one segment as one unit without re-deciding where it belongs.
 *
 * Two details matter more than the detector itself. A segment opens with pre-roll
 * already buffered, because by the time energy crosses a threshold the first phoneme is
 * gone. And it closes only after silence persists, because a natural pause between
 * clauses is shorter than the gap between utterances and cutting on the first quiet
 * frame would shred sentences.
 *
 * Detection is deliberately energy-based rather than the Silero model that ships in the
 * worker. This runs on the capture callback, where the budget is microseconds, and its
 * job is only to find boundaries — the worker still applies real VAD before recognising.
 * A boundary that is slightly generous costs a little disk; a boundary that is late
 * costs a word.
 */

const FRAME_SAMPLES: number = 320;            // 20 ms at 16 kHz
const PREROLL_FRAMES: number = 25;            // 500 ms kept before speech is confirmed
const OPEN_AFTER_SPEECH_FRAMES: number = 5;   // 100 ms of speech opens a segment
const CLOSE_AFTER_SILENCE_FRAMES: number = 40; // 800 ms of silence closes it
const MAX_SEGMENT_SAMPLES: number = 16000 * 30; // hard cap so one monologue is still usable
/** Within this much of the cap, take the first silence rather than waiting for the cap. */
const SOFT_CAP_SLACK: number = 16000 * 5;
/** Noise floor tracker: rises quickly to a louder room, falls slowly back. */
const FLOOR_ATTACK: number = 0.02;
const FLOOR_RELEASE: number = 0.002;

/*
 * Thresholds, measured rather than guessed.
 *
 * On real far-field meeting audio the median speech frame sits at 144 on the Int16
 * mean-absolute scale while real office ambience sits at 113 — they overlap almost
 * completely. An energy detector cannot separate them, and an earlier build that tried
 * (requiring 180) rejected 59% of genuine speech frames and recorded nothing at all on
 * device, where UNPROCESSED capture leaves levels around 20-47 with no gain control.
 *
 * So this gate is deliberately permissive: it decides where audio *might* be worth
 * keeping, and the Silero VAD inside the recogniser decides whether it actually was.
 * Segments whose transcript comes back empty are deleted afterwards. Recording some
 * noise costs disk that retention reclaims; missing speech costs audio that never
 * existed.
 *
 * Opening needs a clearer margin than staying open, so one hesitation mid-sentence does
 * not end the segment.
 */
const OPEN_OVER_FLOOR: number = 1.6;
const SUSTAIN_OVER_FLOOR: number = 1.15;
/** Only guards against digital silence driving the threshold to zero. */
const MIN_FLOOR: number = 8.0;

export type SegmentEventKind = 'open' | 'append' | 'close';

/** What the detector last saw, and what it needed to see. */
export interface SegmenterLevels {
  level: number;
  threshold: number;
  open: boolean;
}

export interface SegmentEvent {
  kind: SegmentEventKind;
  /** Samples to write. Empty on 'close'. */
  samples: Int16Array;
  /** Milliseconds since capture started, at the first sample of this event. */
  offsetMs: number;
}

export class SpeechSegmenter {
  private floor: number = MIN_FLOOR;
  private speechRun: number = 0;
  private silenceRun: number = 0;
  private open: boolean = false;
  private openSamples: number = 0;
  private totalSamples: number = 0;
  private lastLevel: number = 0;
  /** Ring of recent frames, so a segment can start before speech was confirmed. */
  private preroll: Int16Array[] = [];

  reset(): void {
    this.floor = MIN_FLOOR;
    this.speechRun = 0;
    this.silenceRun = 0;
    this.open = false;
    this.openSamples = 0;
    this.totalSamples = 0;
    this.lastLevel = 0;
    this.preroll = [];
  }

  /** True while a segment is open, so the caller knows a final close is still owed. */
  isOpen(): boolean {
    return this.open;
  }

  /**
   * The most recent frame level and the bar it had to clear.
   *
   * Surfaced because the first version of this detector shipped with a threshold that
   * silently rejected every frame, and nothing on screen could distinguish "no audio is
   * arriving" from "audio is arriving and being discarded". One number beside the other
   * answers that at a glance.
   */
  levelSnapshot(): SegmenterLevels {
    const ratio: number = this.open ? SUSTAIN_OVER_FLOOR : OPEN_OVER_FLOOR;
    const snapshot: SegmenterLevels = {
      level: Math.round(this.lastLevel),
      threshold: Math.round(Math.max(this.floor * ratio, MIN_FLOOR)),
      open: this.open,
    };
    return snapshot;
  }

  /**
   * Closes any open segment, for use when recording stops.
   *
   * Without this the last utterance would sit in the writer forever, which is exactly
   * the audio a user is most likely to go looking for.
   */
  flush(): SegmentEvent[] {
    if (!this.open) {
      return [];
    }
    this.open = false;
    this.openSamples = 0;
    return [{ kind: 'close', samples: new Int16Array(0), offsetMs: this.offsetMs() }];
  }

  /** Feeds one capture buffer and returns what the writer should do with it. */
  push(samples: Int16Array): SegmentEvent[] {
    const events: SegmentEvent[] = [];
    let index: number = 0;
    while (index + FRAME_SAMPLES <= samples.length) {
      const frame: Int16Array = samples.subarray(index, index + FRAME_SAMPLES);
      this.consumeFrame(frame, events);
      index += FRAME_SAMPLES;
    }
    // Whatever is left is shorter than a frame; carry it as audio without re-deciding.
    if (index < samples.length && this.open) {
      const tail: Int16Array = samples.slice(index);
      this.openSamples += tail.length;
      this.totalSamples += tail.length;
      events.push({ kind: 'append', samples: tail, offsetMs: this.offsetMs() });
    } else if (index < samples.length) {
      this.totalSamples += samples.length - index;
    }
    return events;
  }

  private consumeFrame(frame: Int16Array, events: SegmentEvent[]): void {
    const level: number = this.frameLevel(frame);
    this.lastLevel = level;
    // Hysteresis: a segment is harder to start than to continue, so a pause between
    // words does not end it and a single loud clatter does not begin one.
    const ratio: number = this.open ? SUSTAIN_OVER_FLOOR : OPEN_OVER_FLOOR;
    const isSpeech: boolean = level > Math.max(this.floor * ratio, MIN_FLOOR);
    // Track the floor only on quiet frames; adapting during speech would chase the
    // talker and make the detector progressively deafer as they keep talking.
    if (!isSpeech) {
      const rate: number = level > this.floor ? FLOOR_ATTACK : FLOOR_RELEASE;
      this.floor = this.floor * (1 - rate) + level * rate;
      if (this.floor < MIN_FLOOR) {
        this.floor = MIN_FLOOR;
      }
    }

    if (isSpeech) {
      this.speechRun++;
      this.silenceRun = 0;
    } else {
      this.silenceRun++;
      this.speechRun = 0;
    }

    if (!this.open) {
      this.rememberPreroll(frame);
      if (this.speechRun >= OPEN_AFTER_SPEECH_FRAMES) {
        this.openSegment(events);
      }
      this.totalSamples += frame.length;
      return;
    }

    this.openSamples += frame.length;
    this.totalSamples += frame.length;
    events.push({ kind: 'append', samples: frame, offsetMs: this.offsetMs() });

    // The cap has to be absolute. An earlier version only applied it once a silent frame
    // appeared, which in a room whose noise never drops below the gate meant it never
    // applied at all - segments ran to 67 seconds. Preferring a clean boundary is right,
    // but not at the cost of having no boundary.
    const tooLong: boolean = this.openSamples >= MAX_SEGMENT_SAMPLES;
    const nearCap: boolean = this.openSamples >= MAX_SEGMENT_SAMPLES - SOFT_CAP_SLACK;
    if (this.silenceRun >= CLOSE_AFTER_SILENCE_FRAMES || tooLong ||
      (nearCap && this.silenceRun > 0)) {
      this.open = false;
      this.openSamples = 0;
      events.push({ kind: 'close', samples: new Int16Array(0), offsetMs: this.offsetMs() });
    }
  }

  private openSegment(events: SegmentEvent[]): void {
    this.open = true;
    this.openSamples = 0;
    let prerollSamples: number = 0;
    for (let i: number = 0; i < this.preroll.length; i++) {
      prerollSamples += this.preroll[i].length;
    }
    const opened: SegmentEvent = {
      kind: 'open',
      samples: new Int16Array(0),
      // The segment begins where the pre-roll begins, not where speech was detected.
      offsetMs: Math.max(0, this.offsetMs() - (prerollSamples / 16)),
    };
    events.push(opened);
    for (let i: number = 0; i < this.preroll.length; i++) {
      this.openSamples += this.preroll[i].length;
      events.push({ kind: 'append', samples: this.preroll[i], offsetMs: 0 });
    }
    this.preroll = [];
  }

  private rememberPreroll(frame: Int16Array): void {
    this.preroll.push(frame.slice());
    while (this.preroll.length > PREROLL_FRAMES) {
      this.preroll.shift();
    }
  }

  /** Mean absolute amplitude: cheaper than RMS and just as usable for a threshold. */
  private frameLevel(frame: Int16Array): number {
    let sum: number = 0;
    for (let i: number = 0; i < frame.length; i++) {
      sum += frame[i] < 0 ? -frame[i] : frame[i];
    }
    return sum / Math.max(frame.length, 1);
  }

  private offsetMs(): number {
    return this.totalSamples / 16;
  }
}
