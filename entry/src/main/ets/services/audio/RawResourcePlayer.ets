import { resourceManager } from '@kit.LocalizationKit';
import { media } from '@kit.MediaKit';
import { hilog } from '@kit.PerformanceAnalysisKit';

const DOMAIN: number = 0x0001;
const TAG: string = 'RawResourcePlayer';

export interface RawPlaybackState {
  audioId: string;
  positionMs: number;
  durationMs: number;
  errorMessage: string;
}

export type RawPlaybackCallback = (state: RawPlaybackState) => void;

/** Plays one bundled rawfile at a time without copying large audio into memory. */
export class RawResourcePlayer {
  private player: media.AVPlayer | null = null;
  private manager: resourceManager.ResourceManager | null = null;
  private rawPath: string = '';
  private currentId: string = '';
  private callback: RawPlaybackCallback | null = null;
  private releasing: boolean = false;

  setCallback(callback: RawPlaybackCallback | null): void {
    this.callback = callback;
  }

  playing(): string {
    return this.currentId;
  }

  async toggle(audioId: string, rawPath: string,
    manager: resourceManager.ResourceManager): Promise<void> {
    if (this.currentId === audioId) {
      await this.stop();
      return;
    }
    await this.stop();
    await this.start(audioId, rawPath, manager);
  }

  async stop(): Promise<void> {
    const active: media.AVPlayer | null = this.player;
    const manager: resourceManager.ResourceManager | null = this.manager;
    const rawPath: string = this.rawPath;
    this.player = null;
    this.manager = null;
    this.rawPath = '';
    this.currentId = '';
    this.releasing = true;
    if (active !== null) {
      try {
        await active.release();
      } catch (error) {
        hilog.warn(DOMAIN, TAG, `release failed: ${error}`);
      }
    }
    if (manager !== null && rawPath.length > 0) {
      try {
        await manager.closeRawFd(rawPath);
      } catch (error) {
        hilog.warn(DOMAIN, TAG, `close raw fd failed: ${error}`);
      }
    }
    this.releasing = false;
    this.notify('', 0, 0, '');
  }

  private async start(audioId: string, rawPath: string,
    manager: resourceManager.ResourceManager): Promise<void> {
    try {
      const descriptor: resourceManager.RawFileDescriptor = await manager.getRawFd(rawPath);
      this.manager = manager;
      this.rawPath = rawPath;
      const active: media.AVPlayer = await media.createAVPlayer();
      this.player = active;
      this.currentId = audioId;
      this.notify(audioId, 0, 0, '');

      active.on('stateChange', (state: string): void => {
        if (this.releasing || this.player !== active) {
          return;
        }
        if (state === 'initialized') {
          active.prepare();
        } else if (state === 'prepared') {
          active.play();
          this.notify(audioId, 0, active.duration, '');
        } else if (state === 'completed') {
          this.stop().catch((error: Error): void => {
            hilog.warn(DOMAIN, TAG, `completed cleanup failed: ${error.message}`);
          });
        }
      });
      active.on('timeUpdate', (positionMs: number): void => {
        if (this.player === active) {
          this.notify(audioId, positionMs, active.duration, '');
        }
      });
      active.on('error', (error: Error): void => {
        if (this.player !== active) {
          return;
        }
        hilog.warn(DOMAIN, TAG, `playback error: ${error.message}`);
        this.notify(audioId, 0, 0, error.message);
        this.stop().catch((cleanupError: Error): void => {
          hilog.warn(DOMAIN, TAG, `error cleanup failed: ${cleanupError.message}`);
        });
      });

      active.fdSrc = {
        fd: descriptor.fd,
        offset: descriptor.offset,
        length: descriptor.length,
      };
    } catch (error) {
      const message: string = `${error}`;
      hilog.error(DOMAIN, TAG, `start failed: ${message}`);
      this.notify(audioId, 0, 0, message);
      await this.stop();
      throw new Error(message);
    }
  }

  private notify(audioId: string, positionMs: number, durationMs: number,
    errorMessage: string): void {
    const callback: RawPlaybackCallback | null = this.callback;
    if (callback !== null) {
      callback({
        audioId: audioId,
        positionMs: positionMs,
        durationMs: durationMs,
        errorMessage: errorMessage,
      });
    }
  }
}
