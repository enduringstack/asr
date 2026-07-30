export interface AudioDecodeResult {
  samples: ArrayBuffer;
  sampleRate: number;
  originalSampleRate: number;
  channels: number;
  durationSeconds: number;
}

export interface AudioProbeResult {
  durationSeconds: number;
  sampleRate: number;
  channels: number;
}

export type Mp3DecodeResult = AudioDecodeResult;
export type M4aDecodeResult = AudioDecodeResult;

export const add: (a: number, b: number) => number;
export const decodeMp3ToMono16k: (data: ArrayBuffer | Uint8Array) => Mp3DecodeResult;
export const decodeM4aToMono16k: (
  data: ArrayBuffer | Uint8Array,
  startSeconds?: number,
  endSeconds?: number
) => M4aDecodeResult;
export const probeM4aInfo: (data: ArrayBuffer | Uint8Array) => AudioProbeResult;
