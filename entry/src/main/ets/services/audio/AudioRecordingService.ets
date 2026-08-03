import { audio } from '@kit.AudioKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { StereoBeamformer } from './StereoBeamformer';

const DOMAIN = 0x0001;
const TAG = 'AudioRecordingService';

// ASR parameters
const SAMPLE_RATE = 16000;       // 16kHz for wenet
const SAMPLE_FORMAT = audio.AudioSampleFormat.SAMPLE_FORMAT_S16LE;
const BYTES_PER_SAMPLE = 2;

/**
 * Source types to try, best first.
 *
 * The previous choice, SOURCE_TYPE_VOICE_RECOGNITION, asks the system for a stream tuned
 * to the person holding the phone: gain control, noise suppression and beamforming aimed
 * at keeping their voice and discarding the room. This app needs the room. Measured on
 * real far-field meeting audio, that style of single-channel enhancement cost 4.6 points
 * of character error rate on exactly the ambient speech we are trying to keep.
 *
 * UNPROCESSED asks for the microphones before any of it. Devices are allowed to decline —
 * the docs describe it as a request, not a guarantee — so MIC is the fallback and the
 * old behaviour remains the last resort rather than a silent failure.
 */
const SOURCE_TYPE_PREFERENCE: audio.SourceType[] = [
  audio.SourceType.SOURCE_TYPE_UNPROCESSED,
  audio.SourceType.SOURCE_TYPE_MIC,
  audio.SourceType.SOURCE_TYPE_VOICE_RECOGNITION,
];

/**
 * Channel counts to try, best first.
 *
 * Two microphones let the capture side steer at whoever is talking before the recogniser
 * ever sees the audio: measured 19.3% -> 17.5% overall and 26.3% -> 24.1% on overlapped
 * speech, at a real-time factor of 0.005. Downstream still receives mono, because
 * StereoBeamformer collapses the pair here.
 */
const CHANNEL_PREFERENCE: audio.AudioChannel[] = [
  audio.AudioChannel.CHANNEL_2,
  audio.AudioChannel.CHANNEL_1,
];

export interface AudioChunk {
  pcmData: ArrayBuffer;
  timestamp: number;
  durationMs: number;
}

export type AudioChunkCallback = (chunk: AudioChunk) => void;
export type StateCallback = (state: 'idle' | 'recording' | 'error', error?: string) => void;

export class AudioRecordingService {
  private audioCapturer: audio.AudioCapturer | null = null;
  private isRecording: boolean = false;
  private isStarting: boolean = false;
  private isStopping: boolean = false;
  private onChunk: AudioChunkCallback | null = null;
  private onStateChange: StateCallback | null = null;
  private readDataCallback: ((buffer: ArrayBuffer) => void) | null = null;
  /** What the device actually granted; 1 whenever it declined the stereo request. */
  private captureChannels: number = 1;
  private beamformer: StereoBeamformer = new StereoBeamformer();
  private static instance: AudioRecordingService | null = null;

  static getInstance(): AudioRecordingService {
    if (!AudioRecordingService.instance) {
      AudioRecordingService.instance = new AudioRecordingService();
    }
    return AudioRecordingService.instance;
  }

  setOnChunkCallback(callback: AudioChunkCallback): void {
    this.onChunk = callback;
  }

  setStateCallback(callback: StateCallback): void {
    this.onStateChange = callback;
  }

  clearCallbacks(): void {
    this.onChunk = null;
    this.onStateChange = null;
  }

  async start(): Promise<void> {
    if (this.isRecording || this.isStarting) {
      hilog.warn(DOMAIN, TAG, 'Already recording');
      return;
    }

    this.isStarting = true;
    let capturer: audio.AudioCapturer | null = null;
    let readCallback: ((buffer: ArrayBuffer) => void) | null = null;
    try {
      capturer = await this.createPreferredCapturer();
      this.beamformer.reset();
      readCallback = (buffer: ArrayBuffer) => {
        this.handleAudioData(buffer);
      };
      capturer.on('readData', readCallback);

      hilog.info(DOMAIN, TAG, 'AudioCapturer created, starting recording...');
      await capturer.start();

      this.audioCapturer = capturer;
      this.readDataCallback = readCallback;
      this.isRecording = true;
      this.onStateChange?.('recording');

    } catch (error) {
      const errMsg = error instanceof Error ? error.message : String(error);
      hilog.error(DOMAIN, TAG, `Failed to start recording: ${errMsg}`);
      if (capturer && readCallback) {
        try {
          capturer.off('readData', readCallback);
        } catch (offError) {
          hilog.warn(DOMAIN, TAG, `Error unregistering failed capturer callback: ${offError}`);
        }
      }
      if (capturer) {
        try {
          await capturer.release();
        } catch (releaseError) {
          hilog.warn(DOMAIN, TAG, `Error releasing failed capturer: ${releaseError}`);
        }
      }
      this.audioCapturer = null;
      this.readDataCallback = null;
      this.isRecording = false;
      this.onStateChange?.('error', errMsg);
      throw new Error(errMsg);
    } finally {
      this.isStarting = false;
    }
  }

  /**
   * Tries each source type and channel count in preference order.
   *
   * Both requests can be declined per device, and a decline surfaces as a failure to
   * create the capturer rather than as a flag to read afterwards. Walking the list keeps
   * a device that refuses raw stereo working on exactly the configuration it used
   * before, and the log line records what it actually settled on — without it, a silent
   * downgrade would look identical to the feature working.
   */
  private async createPreferredCapturer(): Promise<audio.AudioCapturer> {
    let lastError: string = '';
    for (const source of SOURCE_TYPE_PREFERENCE) {
      for (const channels of CHANNEL_PREFERENCE) {
        const options: audio.AudioCapturerOptions = {
          streamInfo: {
            samplingRate: audio.AudioSamplingRate.SAMPLE_RATE_16000,
            channels: channels,
            sampleFormat: SAMPLE_FORMAT,
            encodingType: audio.AudioEncodingType.ENCODING_TYPE_RAW
          },
          capturerInfo: { source: source, capturerFlags: 0 }
        };
        try {
          const capturer = await audio.createAudioCapturer(options);
          this.captureChannels = channels === audio.AudioChannel.CHANNEL_2 ? 2 : 1;
          hilog.info(DOMAIN, TAG,
            `AudioCapturer created source=${source} channels=${this.captureChannels}`);
          return capturer;
        } catch (error) {
          lastError = error instanceof Error ? error.message : String(error);
          hilog.debug(DOMAIN, TAG,
            `source=${source} channels=${channels} unavailable: ${lastError}`);
        }
      }
    }
    throw new Error(`no usable capture configuration: ${lastError}`);
  }

  private handleAudioData(buffer: ArrayBuffer): void {
    if (!this.isRecording || buffer.byteLength === 0) {
      return;
    }

    try {
      const channels = this.captureChannels;
      // Frames, not samples: a stereo buffer holds two samples per instant in time, so
      // dividing by the byte size alone would report double the real duration.
      const frames = buffer.byteLength / BYTES_PER_SAMPLE / channels;
      const durationMs = (frames / SAMPLE_RATE) * 1000;
      const pcmCopy = channels > 1
        ? this.beamformToMono(buffer, channels)
        : this.copyPcmBuffer(buffer);

      this.onChunk?.({
        pcmData: pcmCopy,
        timestamp: Date.now(),
        durationMs: durationMs
      });
    } catch (error) {
      if (this.isRecording) {
        hilog.error(DOMAIN, TAG, `Audio data callback error: ${error}`);
      }
    }
  }

  /**
   * Collapses a multi-channel buffer to the mono stream the rest of the pipeline reads.
   *
   * Everything downstream — VAD, the recogniser, the WAV writer — is built around one
   * 16 kHz channel, so the extra microphone is spent here on steering rather than being
   * passed along as a second channel nothing knows how to use.
   */
  private beamformToMono(buffer: ArrayBuffer, channels: number): ArrayBuffer {
    const interleaved = new Int16Array(buffer);
    const mono = this.beamformer.process(interleaved, channels);
    const out = new ArrayBuffer(mono.length * BYTES_PER_SAMPLE);
    new Int16Array(out).set(mono);
    return out;
  }

  private copyPcmBuffer(buffer: ArrayBuffer): ArrayBuffer {
    const copy = new ArrayBuffer(buffer.byteLength);
    new Uint8Array(copy).set(new Uint8Array(buffer));
    return copy;
  }

  async stop(): Promise<void> {
    if (this.isStopping) {
      return;
    }
    this.isStopping = true;
    if (this.audioCapturer) {
      const capturer = this.audioCapturer;
      const readCallback = this.readDataCallback;
      this.audioCapturer = null;
      try {
        await capturer.stop();
      } catch (error) {
        hilog.warn(DOMAIN, TAG, `Error stopping capturer stream: ${error}`);
      }
      if (readCallback) {
        try {
          capturer.off('readData', readCallback);
        } catch (error) {
          hilog.warn(DOMAIN, TAG, `Error unregistering capturer callback: ${error}`);
        }
      }
      try {
        await capturer.release();
      } catch (error) {
        hilog.warn(DOMAIN, TAG, `Error releasing capturer: ${error}`);
      }
      this.readDataCallback = null;
    }

    this.isRecording = false;
    this.isStopping = false;
    this.onStateChange?.('idle');
    hilog.info(DOMAIN, TAG, 'Recording stopped');
  }

  getIsRecording(): boolean {
    return this.isRecording;
  }
}
