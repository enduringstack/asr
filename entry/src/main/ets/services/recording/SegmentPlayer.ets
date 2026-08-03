/**
 * Plays back a stored segment so a transcript can be checked against what was said.
 *
 * A transcript on its own cannot be trusted or corrected — the only way to know whether
 * "改到明天了" was really said is to hear it. That makes playback part of the feature
 * rather than a convenience, especially while recognition quality is still being tuned.
 *
 * One segment plays at a time. Starting a second stops the first rather than layering
 * them, because two overlapping recordings of a room are indistinguishable to a
 * listener, and mixing them is never what someone tapping a second row intended.
 *
 * Files are handed to the player by descriptor rather than by path. The segments live in
 * the app sandbox where a URL would have to be granted separately, and an fd the app
 * already opened needs no such grant.
 */

import { fileIo } from '@kit.CoreFileKit';
import { media } from '@kit.MediaKit';
import { hilog } from '@kit.PerformanceAnalysisKit';

const DOMAIN = 0x0001;
const TAG = 'SegmentPlayer';

export interface PlaybackState {
  /** Segment id currently sounding, or empty when nothing is. */
  segmentId: string;
  positionMs: number;
  durationMs: number;
}

export type PlaybackCallback = (state: PlaybackState) => void;

export class SegmentPlayer {
  private player: media.AVPlayer | null = null;
  private file: fileIo.File | null = null;
  private currentId: string = '';
  private onChange: PlaybackCallback | null = null;
  /** Set while tearing down, so lifecycle callbacks do not re-enter the teardown. */
  private releasing: boolean = false;

  setCallback(callback: PlaybackCallback | null): void {
    this.onChange = callback;
  }

  playing(): string {
    return this.currentId;
  }

  /** Starts `path`, or stops if that segment is already the one playing. */
  async toggle(segmentId: string, path: string): Promise<void> {
    if (this.currentId === segmentId) {
      await this.stop();
      return;
    }
    await this.stop();
    await this.start(segmentId, path);
  }

  async stop(): Promise<void> {
    const player: media.AVPlayer | null = this.player;
    this.player = null;
    this.currentId = '';
    this.releasing = true;
    if (player !== null) {
      try {
        await player.release();
      } catch (error) {
        hilog.warn(DOMAIN, TAG, `release failed: ${error}`);
      }
    }
    this.closeFile();
    this.releasing = false;
    this.notify(0, 0);
  }

  private async start(segmentId: string, path: string): Promise<void> {
    try {
      if (!fileIo.accessSync(path)) {
        hilog.warn(DOMAIN, TAG, `segment missing: ${path}`);
        return;
      }
      const size: number = fileIo.statSync(path).size;
      this.file = fileIo.openSync(path, fileIo.OpenMode.READ_ONLY);
      // Held as a non-nullable const because the callbacks below close over it. A
      // nullable local cannot be narrowed inside a closure, and re-reading this.player
      // there would be wrong anyway: these handlers belong to *this* player and must
      // keep working on it even after the field has moved on to another segment.
      const active: media.AVPlayer = await media.createAVPlayer();
      this.player = active;
      this.currentId = segmentId;

      active.on('stateChange', (state: string): void => {
        if (this.releasing || this.player !== active) {
          return;
        }
        if (state === 'initialized') {
          active.prepare();
        } else if (state === 'prepared') {
          active.play();
          this.notify(0, active.duration);
        } else if (state === 'completed') {
          // Reaching the end is a normal stop, not an error; clear state so the row
          // stops showing itself as playing.
          this.stop();
        }
      });
      active.on('timeUpdate', (time: number): void => {
        if (this.player === active) {
          this.notify(time, active.duration);
        }
      });
      active.on('error', (error: Error): void => {
        hilog.warn(DOMAIN, TAG, `playback error: ${error.message}`);
        this.stop();
      });

      active.fdSrc = { fd: this.file.fd, offset: 0, length: size };
    } catch (error) {
      hilog.error(DOMAIN, TAG, `start failed: ${error}`);
      this.player = null;
      this.currentId = '';
      this.closeFile();
      this.notify(0, 0);
    }
  }

  private closeFile(): void {
    const file: fileIo.File | null = this.file;
    this.file = null;
    if (file === null) {
      return;
    }
    try {
      fileIo.closeSync(file);
    } catch (error) {
      hilog.warn(DOMAIN, TAG, `close fd failed: ${error}`);
    }
  }

  private notify(positionMs: number, durationMs: number): void {
    const callback: PlaybackCallback | null = this.onChange;
    if (callback !== null) {
      callback({
        segmentId: this.currentId,
        positionMs: positionMs,
        durationMs: durationMs,
      });
    }
  }
}
