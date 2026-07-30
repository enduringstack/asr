import { audio } from '@kit.AudioKit';
import { hilog } from '@kit.PerformanceAnalysisKit';

const DOMAIN = 0x0001;
const TAG = 'AudioRecordingService';

// ASR parameters
const SAMPLE_RATE = 16000;       // 16kHz for wenet
const CHANNEL_COUNT = 1;         // mono
const SAMPLE_FORMAT = audio.AudioSampleFormat.SAMPLE_FORMAT_S16LE;
const BYTES_PER_SAMPLE = 2;

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
      // Create AudioCapturer
      const audioStreamInfo: audio.AudioStreamInfo = {
        samplingRate: audio.AudioSamplingRate.SAMPLE_RATE_16000,
        channels: audio.AudioChannel.CHANNEL_1,
        sampleFormat: SAMPLE_FORMAT,
        encodingType: audio.AudioEncodingType.ENCODING_TYPE_RAW
      };

      const audioCapturerInfo: audio.AudioCapturerInfo = {
        source: audio.SourceType.SOURCE_TYPE_VOICE_RECOGNITION,
        capturerFlags: 0
      };

      const audioCapturerOptions: audio.AudioCapturerOptions = {
        streamInfo: audioStreamInfo,
        capturerInfo: audioCapturerInfo
      };

      capturer = await audio.createAudioCapturer(audioCapturerOptions);
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

  private handleAudioData(buffer: ArrayBuffer): void {
    if (!this.isRecording || buffer.byteLength === 0) {
      return;
    }

    try {
      const durationMs = (buffer.byteLength / BYTES_PER_SAMPLE / SAMPLE_RATE) * 1000;
      const pcmCopy = this.copyPcmBuffer(buffer);

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
