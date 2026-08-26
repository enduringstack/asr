import { ErrorEvent, MessageEvents, ThreadWorkerGlobalScope, util, worker } from '@kit.ArkTS';
import { Context } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { decodeM4aToMono16k, decodeMp3ToMono16k, probeM4aInfo, AudioDecodeResult, AudioProbeResult } from 'libentry.so';
import {
  FastClusteringConfig,
  OfflineSpeakerDiarization,
  OfflineSpeakerDiarizationConfig,
  OfflineSpeakerDiarizationSegment,
  OfflineSpeakerSegmentationModelConfig,
  OfflineSpeakerSegmentationPyannoteModelConfig,
  OfflinePunctuation,
  OfflinePunctuationConfig,
  OfflinePunctuationModelConfig,
  OfflineModelConfig,
  OfflineRecognizer,
  OfflineRecognizerConfig,
  OfflineRecognizerResult,
  OfflineStream,
  OnlineModelConfig,
  OnlineRecognizer,
  OnlineRecognizerConfig,
  OnlineRecognizerResult,
  OnlineStream,
  readWaveFromBinary,
  Samples,
  SpeakerEmbeddingExtractor,
  SpeakerEmbeddingExtractorConfig,
  SileroVadConfig,
  TenVadConfig,
  Vad,
  VadConfig
} from 'sherpa_onnx';
import {
  buildFixedLengthSegments,
  computeRms,
  coverWholeSignal,
  DecodeBlockOptions,
  DecodePlan,
  FixedSegmentOptions,
  isCorroborated,
  isDecodeRepetitionCollapse,
  planDecodeBlocks,
  SampleRange,
  speechFlagsToRegions,
  SpeechRegionOptions
} from '../common/AudioSegmentation';
import { resampleAntiAliased } from '../common/Resampler';
import {
  EnrollmentQuality,
  SPEAKER_ACTIVE_ENROLL_MAX_MS,
  SPEAKER_ACTIVE_ENROLL_MIN_MS,
  SPEAKER_CLUSTER_THRESHOLD,
  SPEAKER_EMBEDDING_DIM,
  SPEAKER_EMBEDDING_THREADS,
  SPEAKER_IDENTIFY_MIN_SEGMENTS,
  SPEAKER_IDENTIFY_MIN_SPEECH_MS,
  SPEAKER_IDENTITY_THRESHOLD,
  SPEAKER_MAX_PROFILES,
  SPEAKER_MODEL_ID,
  SPEAKER_MODEL_SHA256,
  SpeakerCandidate,
  SpeakerSearchProfile,
  SpeakerTurn,
  VoiceprintExtractionResult,
} from '../common/SpeakerTypes';
import { TranscriptBuilder } from '../common/TranscriptBuilder';
import { averageEmbeddings, cosineSimilarity, normalizeEmbedding } from '../services/speaker/SpeakerMath';

const workerPort: ThreadWorkerGlobalScope = worker.workerPort;
const DOMAIN: number = 0x0001;
const TAG: string = 'StreamingSherpaWorker';

const SAMPLE_RATE = 16000;
const MODEL_DIR = 'sherpa-onnx-streaming-paraformer-bilingual-zh-en';
const FINAL_ASR_MODEL_DIR = 'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09';
const OFFLINE_PARAFORMER_MODEL_DIR = 'sherpa-onnx-paraformer-zh-2024-03-09';
const PUNCTUATION_MODEL_DIR = 'sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8';
const SPEAKER_SEGMENTATION_MODEL = 'sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx';
const SPEAKER_EMBEDDING_MODEL = 'sherpa-onnx-campplus-sv-zh-en-16k/model.onnx';
const PRE_ROLL_SAMPLES = 9600;
const TRAILING_SILENCE_SAMPLES = 24000;
const FINAL_TAIL_PADDING_SAMPLES = 16000;
const DIRECT_TAIL_PADDING_SAMPLES = 16000;
const MIC_FALLBACK_MIN_SECONDS = 0.8;
const MIC_FALLBACK_LONG_SESSION_SECONDS = 12;
const MIC_FALLBACK_MIN_RMS = 0.003;
const MIC_FALLBACK_MIN_CHARS_PER_SECOND = 0.8;
const DIARIZATION_SEGMENT_PADDING_SECONDS = 0.20;
const DIARIZATION_MERGE_GAP_SECONDS = 1.20;
const DIARIZATION_RAW_MIN_SEGMENT_SECONDS = 0.35;
const DIARIZATION_SHORT_ISLAND_SECONDS = 2.20;
const DIARIZATION_OVERLAP_ABSORB_RATIO = 0.45;
const DIARIZATION_NEIGHBOR_GAP_SECONDS = 1.00;
const DIARIZATION_MIN_SPEAKER_TOTAL_SECONDS = 18.00;
const DIARIZATION_MIN_RELIABLE_TURN_SECONDS = 8.00;
const DIARIZATION_UNSTABLE_SPEAKER_MERGE_GAP_SECONDS = 8.00;
const DIARIZATION_MAX_DISPLAY_SEGMENTS = 36;
const DIARIZATION_MAX_RESCORE_SEGMENTS = 18;
const DIARIZATION_MAX_RESCORE_SECONDS = 190;
const DIARIZATION_AUTO_RESCORE_MAX_SEGMENTS = 18;
const DIARIZATION_AUTO_RESCORE_MAX_SECONDS = 190;
const DIARIZATION_CLUSTER_THRESHOLD = SPEAKER_CLUSTER_THRESHOLD;
const VOICEPRINT_MATCH_THRESHOLD = SPEAKER_IDENTITY_THRESHOLD;
const VOICEPRINT_MIN_IDENTIFY_SAMPLES = 16000;
const VOICEPRINT_MAX_IDENTIFY_SAMPLES = 320000;
const LONG_AUDIO_SEGMENT_SECONDS = 10;
const LONG_AUDIO_OVERLAP_SECONDS = 0.25;
const LONG_AUDIO_MIN_SEGMENTED_SECONDS = 20;
const LONG_AUDIO_BOUNDARY_SEARCH_SECONDS = 0.6;
const LONG_AUDIO_BOUNDARY_WINDOW_SECONDS = 0.16;
const LONG_AUDIO_MIN_SEGMENT_SECONDS = 6;
// VAD-driven decode blocks for the offline path. Blocks may be much longer than the old
// fixed 10 s cut because they always start and end in silence: more context for the
// acoustic model and for punctuation, and no audio overlap to undo afterwards.
const VAD_BLOCK_MAX_SECONDS = 24;
const VAD_BLOCK_MAX_GAP_SECONDS = 1.5;
const VAD_BLOCK_PAD_SECONDS = 0.20;
const VAD_MIN_SPEECH_SECONDS = 0.20;
const VAD_BRIDGE_SILENCE_SECONDS = 0.60;
// Offline decoding is batch work off the real-time path, so it may use more cores than
// the streaming pass, which has to stay cheap while the microphone is live.
const STREAMING_ASR_NUM_THREADS = 2;
const OFFLINE_ASR_NUM_THREADS = 4;
const PUNCTUATION_NUM_THREADS = 2;
const WARM_UP_SAMPLES = 8000;
const TEXT_OVERLAP_MAX_CHARS = 32;
const TEXT_OVERLAP_SEARCH_CHARS = 80;
const TEXT_TAIL_REPLACE_MIN_OVERLAP_CHARS = 8;
const TEXT_TAIL_REPLACE_MAX_CHARS = 48;
const MIXED_ASCII_NOISE_MAX_CHARS = 3;
const FILE_ASR_MODE_STREAMING = 'streaming';
const FILE_ASR_MODE_SENSEVOICE = 'sensevoice';
const FILE_ASR_MODE_OFFLINE_PARAFORMER = 'offline_paraformer_large';

interface WorkerRequest {
  msgType: string;
  context?: Context;
  pcm?: ArrayBuffer;
  data?: Uint8Array;
  audioFormat?: string;
  fileName?: string;
  fileAsrMode?: string;
  enableSpeakerDiarization?: boolean;
  enableSegmentRescore?: boolean;
  compareEnhanced?: boolean;
  speakerName?: string;
  voiceprints?: SpeakerSearchProfile[];
  meetingId?: string;
  /** Correlates a deferred segment transcription with its reply. */
  requestId?: string;
}

interface AsrTelemetry {
  chunkDecodeMs?: number;
  partialIntervalMs?: number;
  finalSegmentMs?: number;
  finalRedecodeMs?: number;
  finalRedecodeRtf?: number;
  audioDurationSec?: number;
}

interface ReplacementRule {
  from: string;
  to: string;
}

interface WorkerResponse {
  msgType: string;
  message?: string;
  text?: string;
  punctuationReady?: boolean;
  segmentIndex?: number;
  segmentTotal?: number;
  segmentStart?: number;
  segmentEnd?: number;
  audioDuration?: number;
  speakerName?: string;
  speakerNames?: string[];
  embedding?: number[];
  quality?: EnrollmentQuality;
  meetingId?: string;
  speakerTurns?: SpeakerTurn[];
  speakerCandidates?: SpeakerCandidate[];
  voiceprintReady?: boolean;
  telemetry?: AsrTelemetry;
  /** Echoed back from the request so the caller can match the reply. */
  requestId?: string;
}

interface NormalizedSpeakerSegment {
  start: number;
  end: number;
  speaker: number;
}

interface SpeakerOverlapRange {
  start: number;
  end: number;
}

interface AudioSamples {
  samples: Float32Array;
  sampleRate: number;
}

let recognizer: OnlineRecognizer | undefined;
let finalRecognizer: OfflineRecognizer | undefined;
let offlineParaformerRecognizer: OfflineRecognizer | undefined;
let micStream: OnlineStream | undefined;
let vad: Vad | undefined;
// A second VAD instance dedicated to offline segmentation, so planning a file decode can
// never disturb the streaming VAD state of a live microphone session.
let fileVad: Vad | undefined;
let punctuation: OfflinePunctuation | undefined;
let speakerDiarization: OfflineSpeakerDiarization | undefined;
let speakerEmbeddingExtractor: SpeakerEmbeddingExtractor | undefined;
let restoredVoiceprints: SpeakerSearchProfile[] = [];
let appContext: Context | undefined;
let latestText: string = '';
let lastPartialEmitMs: number = 0;
const REPLACEMENTS_FILE = 'replacements.json';
let replacementRules: ReplacementRule[] = [];
const HOTWORDS_FILE = 'hotwords.txt';
const HOTWORDS_SCORE = 1.5;
let hotwordsText: string = '';
let hotwordsBufByteLength: number = 0;
let inSpeech: boolean = false;
let silenceSamples: number = 0;
let preRollChunks: Float32Array[] = [];
let preRollLength: number = 0;
let micChunks: Float32Array[] = [];
let micLength: number = 0;
let sessionChunks: Float32Array[] = [];
let sessionLength: number = 0;
let voiceprintChunks: Float32Array[] = [];
let voiceprintLength: number = 0;
let finishingSegment: boolean = false;
let collectedFinalSegments: string[] | undefined;

function postResponse(response: WorkerResponse): void {
  if (collectedFinalSegments) {
    if (response.msgType === 'sherpa-asr-final' && response.text && response.text.trim() !== '') {
      collectedFinalSegments.push(response.text.trim());
    }
    if (response.msgType === 'sherpa-asr-partial' || response.msgType === 'sherpa-asr-final') {
      return;
    }
  }
  workerPort.postMessage(response);
}

function postError(message: string, msgType: string = 'sherpa-asr-error'): void {
  const response: WorkerResponse = { msgType, message };
  postResponse(response);
}

// Throttle high-frequency chunk telemetry so the message bus is not flooded.
const TELEMETRY_CHUNK_MIN_INTERVAL_MS = 500;
let lastChunkTelemetryMs: number = 0;

function postTelemetry(telemetry: AsrTelemetry): void {
  const response: WorkerResponse = { msgType: 'sherpa-asr-telemetry', telemetry };
  postResponse(response);
}

function initRecognizer(context: Context): OnlineRecognizer {
  const modelConfig = new OnlineModelConfig();
  modelConfig.paraformer.encoder = `${MODEL_DIR}/encoder.int8.onnx`;
  modelConfig.paraformer.decoder = `${MODEL_DIR}/decoder.int8.onnx`;
  modelConfig.tokens = `${MODEL_DIR}/tokens.txt`;
  modelConfig.modelType = 'paraformer';
  modelConfig.numThreads = STREAMING_ASR_NUM_THREADS;
  modelConfig.debug = false;
  modelConfig.provider = 'cpu';

  const config = new OnlineRecognizerConfig();
  config.modelConfig = modelConfig;
  config.enableEndpoint = false;
  config.rule1MinTrailingSilence = 2.0;
  config.rule2MinTrailingSilence = 1.2;
  config.rule3MinUtteranceLength = 20;
  // Built-in static hotword list (rawfile hotwords.txt). NOTE: per sherpa-onnx, hotword
  // biasing only takes effect for transducer / LLM-ASR models; the streaming Paraformer
  // (CIF) is expected to treat this as a no-op. Wired here so the empirical benchmark
  // (run_asr_benchmark.py --hotwords-file) can confirm that expectation on-device.
  if (hotwordsText.length > 0) {
    config.hotwordsBuf = hotwordsText;
    config.hotwordsBufSize = hotwordsBufByteLength;
    config.hotwordsScore = HOTWORDS_SCORE;
  }

  return new OnlineRecognizer(config, context.resourceManager);
}

function initFinalRecognizer(context: Context): OfflineRecognizer {
  const modelConfig = new OfflineModelConfig();
  modelConfig.senseVoice.model = `${FINAL_ASR_MODEL_DIR}/model.int8.onnx`;
  modelConfig.senseVoice.language = 'zh';
  modelConfig.senseVoice.useItn = true;
  modelConfig.tokens = `${FINAL_ASR_MODEL_DIR}/tokens.txt`;
  modelConfig.numThreads = OFFLINE_ASR_NUM_THREADS;
  modelConfig.debug = false;
  modelConfig.provider = 'cpu';

  const config = new OfflineRecognizerConfig();
  config.featConfig.sampleRate = SAMPLE_RATE;
  config.featConfig.featureDim = 80;
  config.modelConfig = modelConfig;
  config.decodingMethod = 'greedy_search';

  return new OfflineRecognizer(config, context.resourceManager);
}

function ensureFinalRecognizer(): OfflineRecognizer {
  if (finalRecognizer) {
    return finalRecognizer;
  }
  if (!appContext) {
    throw new Error('application context is not initialized');
  }
  finalRecognizer = initFinalRecognizer(appContext);
  return finalRecognizer;
}

function initOfflineParaformerRecognizer(context: Context): OfflineRecognizer {
  const modelConfig = new OfflineModelConfig();
  modelConfig.paraformer.model = `${OFFLINE_PARAFORMER_MODEL_DIR}/model.int8.onnx`;
  modelConfig.tokens = `${OFFLINE_PARAFORMER_MODEL_DIR}/tokens.txt`;
  modelConfig.numThreads = OFFLINE_ASR_NUM_THREADS;
  modelConfig.debug = false;
  modelConfig.provider = 'cpu';

  const config = new OfflineRecognizerConfig();
  config.featConfig.sampleRate = SAMPLE_RATE;
  config.featConfig.featureDim = 80;
  config.modelConfig = modelConfig;
  config.decodingMethod = 'greedy_search';
  // hotwordsFfile expects a filesystem path; we do not write one in-app (Option B).
  // The empirical no-op confirmation for offline paraformer hotwords is done via the
  // Python benchmark (run_asr_benchmark.py --hotwords-file --decoder offline-paraformer).
  config.hotwordsFfile = '';

  return new OfflineRecognizer(config, context.resourceManager);
}

function ensureOfflineParaformerRecognizer(): OfflineRecognizer {
  if (offlineParaformerRecognizer) {
    return offlineParaformerRecognizer;
  }
  if (!appContext) {
    throw new Error('application context is not initialized');
  }
  offlineParaformerRecognizer = initOfflineParaformerRecognizer(appContext);
  return offlineParaformerRecognizer;
}

function initSpeakerDiarization(context: Context): OfflineSpeakerDiarization {
  const pyannoteConfig = new OfflineSpeakerSegmentationPyannoteModelConfig();
  pyannoteConfig.model = SPEAKER_SEGMENTATION_MODEL;

  const segmentationConfig = new OfflineSpeakerSegmentationModelConfig();
  segmentationConfig.pyannote = pyannoteConfig;
  segmentationConfig.numThreads = 2;
  segmentationConfig.debug = false;
  segmentationConfig.provider = 'cpu';

  const embeddingConfig = new SpeakerEmbeddingExtractorConfig();
  embeddingConfig.model = SPEAKER_EMBEDDING_MODEL;
  embeddingConfig.numThreads = SPEAKER_EMBEDDING_THREADS;
  embeddingConfig.debug = false;
  embeddingConfig.provider = 'cpu';

  const clusteringConfig = new FastClusteringConfig();
  clusteringConfig.numClusters = -1;
  clusteringConfig.threshold = DIARIZATION_CLUSTER_THRESHOLD;

  const config = new OfflineSpeakerDiarizationConfig();
  config.segmentation = segmentationConfig;
  config.embedding = embeddingConfig;
  config.clustering = clusteringConfig;
  config.minDurationOn = 0.25;
  config.minDurationOff = 0.50;

  return new OfflineSpeakerDiarization(config, context.resourceManager);
}

function ensureSpeakerDiarization(): OfflineSpeakerDiarization {
  if (speakerDiarization) {
    return speakerDiarization;
  }
  if (!appContext) {
    throw new Error('application context is not initialized');
  }
  speakerDiarization = initSpeakerDiarization(appContext);
  return speakerDiarization;
}

function initSpeakerEmbeddingExtractor(context: Context): SpeakerEmbeddingExtractor {
  const config = new SpeakerEmbeddingExtractorConfig();
  config.model = SPEAKER_EMBEDDING_MODEL;
  config.numThreads = SPEAKER_EMBEDDING_THREADS;
  config.debug = false;
  config.provider = 'cpu';
  return new SpeakerEmbeddingExtractor(config, context.resourceManager);
}

function ensureSpeakerEmbeddingExtractor(): SpeakerEmbeddingExtractor {
  if (speakerEmbeddingExtractor) {
    return speakerEmbeddingExtractor;
  }
  if (!appContext) {
    throw new Error('application context is not initialized');
  }
  speakerEmbeddingExtractor = initSpeakerEmbeddingExtractor(appContext);
  return speakerEmbeddingExtractor;
}

function initVad(context: Context): Vad {
  const config = new VadConfig(
    new SileroVadConfig(
      'silero_vad.onnx',
      0.35,
      0.10,
      0.80,
      512,
      20
    ),
    new TenVadConfig(
      '',
      0.35,
      0.10,
      0.80,
      256,
      20
    ),
    SAMPLE_RATE,
    false,
    1,
    'cpu'
  );

  return new Vad(config, 60, context.resourceManager);
}

function initPunctuation(context: Context): OfflinePunctuation {
  const modelConfig = new OfflinePunctuationModelConfig();
  modelConfig.ctTransformer = `${PUNCTUATION_MODEL_DIR}/model.int8.onnx`;
  modelConfig.numThreads = PUNCTUATION_NUM_THREADS;
  modelConfig.provider = 'cpu';

  const config = new OfflinePunctuationConfig();
  config.model = modelConfig;
  return new OfflinePunctuation(config, context.resourceManager);
}

function loadReplacementDictionary(context: Context): void {
  try {
    const bytes = context.resourceManager.getRawFileContentSync(REPLACEMENTS_FILE);
    const text = new util.TextDecoder('utf-8').decode(bytes).trim();
    if (text.length === 0) {
      replacementRules = [];
      return;
    }
    const parsed = JSON.parse(text) as ReplacementRule[];
    if (!Array.isArray(parsed)) {
      replacementRules = [];
      return;
    }
    // Longest-match-first so a longer target wins over a prefixing shorter one.
    const rules: ReplacementRule[] = [];
    for (let i = 0; i < parsed.length; i++) {
      const rule = parsed[i];
      if (rule && typeof rule.from === 'string' && typeof rule.to === 'string' && rule.from.length > 0) {
        rules.push({ from: rule.from, to: rule.to });
      }
    }
    rules.sort((a: ReplacementRule, b: ReplacementRule) => b.from.length - a.from.length);
    replacementRules = rules;
  } catch (_error) {
    // Missing rawfile or invalid JSON is non-fatal: dict stays empty (no-op).
    replacementRules = [];
  }
}

// UTF-8 byte length without depending on a TextEncoder global. sherpa's hotwordsBufSize
// expects the byte count, not the UTF-16 code-unit count, so CJK phrases must be measured
// this way or the native side reads a truncated buffer.
function utf8ByteLength(s: string): number {
  let n = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c < 0x80) {
      n += 1;
    } else if (c < 0x800) {
      n += 2;
    } else if (c >= 0xD800 && c <= 0xDBFF) {
      // High surrogate of a supplementary-plane pair: one UTF-16 code unit maps to 4 UTF-8 bytes.
      n += 4;
      i += 1;
    } else {
      n += 3;
    }
  }
  return n;
}

function loadHotwordsBuf(context: Context): void {
  try {
    const bytes = context.resourceManager.getRawFileContentSync(HOTWORDS_FILE);
    hotwordsText = new util.TextDecoder('utf-8').decode(bytes).trim();
    hotwordsBufByteLength = utf8ByteLength(hotwordsText);
  } catch (_error) {
    // Missing rawfile is non-fatal: hotwords stay empty (no-op). This is also the
    // "no-hotwords" arm of the empirical no-op benchmark.
    hotwordsText = '';
    hotwordsBufByteLength = 0;
  }
}

function applyReplacementDictionary(text: string): string {
  if (replacementRules.length === 0 || text.length === 0) {
    return text;
  }
  let ans = text;
  for (let i = 0; i < replacementRules.length; i++) {
    const rule = replacementRules[i];
    if (ans.indexOf(rule.from) < 0) {
      continue;
    }
    // split/join replaces all occurrences (ArkTS String.replace with a string only replaces the first).
    ans = ans.split(rule.from).join(rule.to);
  }
  return ans;
}

// Conservative ITN-style normalization. ONLY resolves full-width digit/punctuation
// inconsistency (０１２ → 012, ％ → %) so the streaming-Paraformer and SenseVoice final
// paths render numbers in the same width style. Bridging ASCII digits ("123") to CJK
// numerals ("一百二十三") or vice-versa requires a real ITN model and is deferred — a
// partial hand-rolled conversion would only create new inconsistency, so it is not done.
function normalizeNumbers(text: string): string {
  if (text.length === 0) {
    return text;
  }
  let ans = '';
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    // Full-width digits ０..９ (U+FF10..U+FF19) -> ASCII 0..9.
    if (code >= 0xFF10 && code <= 0xFF19) {
      ans += String.fromCharCode(code - 0xFF10 + 0x0030);
      continue;
    }
    // Full-width percent ％ (U+FF05) -> %.
    if (code === 0xFF05) {
      ans += '%';
      continue;
    }
    // Full-width comma between digits ，(U+FF0C) as a thousands separator -> ','.
    if (code === 0xFF0C && ans.length > 0 && i + 1 < text.length) {
      const prev = ans.charCodeAt(ans.length - 1);
      const next = text.charCodeAt(i + 1);
      if (prev >= 0x0030 && prev <= 0x0039 && next >= 0x0030 && next <= 0x0039) {
        ans += ',';
        continue;
      }
    }
    ans += text.charAt(i);
  }
  return ans;
}

function addPunctuation(text: string): string {
  const cleanText = cleanDecodedText(text);
  if (cleanText === '') {
    return cleanText;
  }
  let result = cleanText;
  if (punctuation) {
    try {
      result = cleanDecodedText(punctuation.addPunct(cleanText));
    } catch (_error) {
      result = cleanText;
    }
  }
  // Apply the user/domain dictionary on the final (punctuated) text, normalize number
  // widths for cross-model consistency, then a final punctuation-collapse in case a
  // replacement re-introduced duplicated punctuation.
  return collapseRepeatedPunctuation(normalizeNumbers(applyReplacementDictionary(result)));
}

function resetPreRoll(): void {
  preRollChunks = [];
  preRollLength = 0;
}

function resetMicChunks(): void {
  micChunks = [];
  micLength = 0;
}

function resetSessionChunks(): void {
  sessionChunks = [];
  sessionLength = 0;
}

function resetVoiceprintChunks(): void {
  voiceprintChunks = [];
  voiceprintLength = 0;
}

function appendMicChunks(samples: Float32Array): void {
  micChunks.push(samples);
  micLength += samples.length;
}

function appendSessionChunks(samples: Float32Array): void {
  sessionChunks.push(samples);
  sessionLength += samples.length;
}

function appendVoiceprintChunks(samples: Float32Array): void {
  voiceprintChunks.push(samples);
  voiceprintLength += samples.length;
}

function appendPreRoll(samples: Float32Array): void {
  preRollChunks.push(samples);
  preRollLength += samples.length;

  while (preRollLength > PRE_ROLL_SAMPLES && preRollChunks.length > 0) {
    const overflow = preRollLength - PRE_ROLL_SAMPLES;
    const first = preRollChunks[0];
    if (first.length <= overflow) {
      preRollChunks.shift();
      preRollLength -= first.length;
    } else {
      preRollChunks[0] = first.subarray(overflow);
      preRollLength -= overflow;
      break;
    }
  }
}

function concatChunks(chunks: Float32Array[]): Float32Array {
  let total = 0;
  for (let i = 0; i < chunks.length; ++i) {
    total += chunks[i].length;
  }

  const ans = new Float32Array(total);
  let offset = 0;
  for (let i = 0; i < chunks.length; ++i) {
    ans.set(chunks[i], offset);
    offset += chunks[i].length;
  }
  return ans;
}

function isLongAudio(samples: Float32Array, sampleRate: number): boolean {
  return samples.length >= Math.floor(sampleRate * LONG_AUDIO_MIN_SEGMENTED_SECONDS);
}

function ensureFileVad(): Vad | undefined {
  if (fileVad) {
    return fileVad;
  }
  if (!appContext) {
    return undefined;
  }
  try {
    fileVad = initVad(appContext);
  } catch (_error) {
    // Without a VAD the offline path falls back to fixed-length segmentation.
    fileVad = undefined;
  }
  return fileVad;
}

/**
 * Run the offline VAD over the whole signal and return one speech/non-speech decision
 * per VAD window.
 */
function detectSpeechFlags(samples: Float32Array): boolean[] {
  const detector = ensureFileVad();
  if (!detector) {
    return [];
  }

  detector.reset();
  const windowSize = detector.config.sileroVad.windowSize;
  const flags: boolean[] = [];
  for (let i = 0; i < samples.length; i += windowSize) {
    detector.acceptWaveform(samples.subarray(i, Math.min(i + windowSize, samples.length)));
    flags.push(detector.isDetected());
    while (!detector.isEmpty()) {
      detector.pop();
    }
  }
  detector.flush();
  while (!detector.isEmpty()) {
    detector.pop();
  }
  detector.reset();
  return flags;
}

function buildFallbackPlan(samples: Float32Array, sampleRate: number): DecodePlan {
  const options: FixedSegmentOptions = {
    segmentSamples: Math.max(1, Math.floor(sampleRate * LONG_AUDIO_SEGMENT_SECONDS)),
    searchRadiusSamples: Math.max(1, Math.floor(sampleRate * LONG_AUDIO_BOUNDARY_SEARCH_SECONDS)),
    boundaryWindowSamples: Math.max(1, Math.floor(sampleRate * LONG_AUDIO_BOUNDARY_WINDOW_SECONDS)),
    minSegmentSamples: Math.max(1, Math.floor(sampleRate * LONG_AUDIO_MIN_SEGMENT_SECONDS))
  };
  return {
    segments: buildFixedLengthSegments(samples, options),
    padSamples: Math.max(0, Math.floor(sampleRate * LONG_AUDIO_OVERLAP_SECONDS)),
    mergeTextOverlap: true
  };
}

/**
 * Plan how a long recording is cut into decode calls.
 *
 * Prefers VAD boundaries (blocks begin and end in silence, silence is skipped, no audio
 * overlap needed) and falls back to fixed-length quiet-window cuts when the VAD is
 * unavailable or found no speech at all.
 */
function buildDecodePlan(samples: Float32Array, sampleRate: number): DecodePlan {
  // Short audio is decoded in one call either way, so there is nothing to gain from VAD
  // planning and no reason to risk the VAD trimming a soft onset.
  if (!isLongAudio(samples, sampleRate)) {
    return buildFallbackPlan(samples, sampleRate);
  }

  // The VAD model is built for 16 kHz; every caller normalises first, but do not silently
  // feed it a different rate if that ever changes.
  const detector = sampleRate === SAMPLE_RATE ? ensureFileVad() : undefined;
  if (!detector) {
    return buildFallbackPlan(samples, sampleRate);
  }

  let flags: boolean[] = [];
  try {
    flags = detectSpeechFlags(samples);
  } catch (_error) {
    return buildFallbackPlan(samples, sampleRate);
  }

  const regionOptions: SpeechRegionOptions = {
    minSpeechSamples: Math.floor(sampleRate * VAD_MIN_SPEECH_SECONDS),
    minSilenceSamples: Math.floor(sampleRate * VAD_BRIDGE_SILENCE_SECONDS),
    padSamples: Math.floor(sampleRate * VAD_BLOCK_PAD_SECONDS)
  };
  const regions = speechFlagsToRegions(
    flags,
    detector.config.sileroVad.windowSize,
    samples.length,
    regionOptions
  );

  const blockOptions: DecodeBlockOptions = {
    maxBlockSamples: Math.max(1, Math.floor(sampleRate * VAD_BLOCK_MAX_SECONDS)),
    maxGapSamples: Math.max(0, Math.floor(sampleRate * VAD_BLOCK_MAX_GAP_SECONDS))
  };
  if (regions.length === 0) {
    // Nothing detected anywhere. The VAD has no boundaries to offer, so there is nothing
    // to build on and the fixed-length planner takes over — and it is measurably the
    // better choice here: scored against the article being read, its shorter blocks
    // recovered slightly more from these segments than two long ones did.
    return buildFallbackPlan(samples, sampleRate);
  }

  // The VAD picks the cuts; it does not get to decide what goes undecoded. Handing it
  // that decision cost two thirds of the words in the quietest segments of a real
  // recording, because it hears far-field speech as silence. See coverWholeSignal.
  const blocks = planDecodeBlocks(coverWholeSignal(regions, samples.length), blockOptions);
  if (blocks.length === 0) {
    return buildFallbackPlan(samples, sampleRate);
  }

  return {
    segments: blocks,
    padSamples: 0,
    mergeTextOverlap: false
  };
}

function isAsciiWordChar(ch: string): boolean {
  if (ch.length === 0) {
    return false;
  }
  const code = ch.charCodeAt(0);
  return (code >= 48 && code <= 57) ||
    (code >= 65 && code <= 90) ||
    (code >= 97 && code <= 122);
}

function isAsciiLetter(ch: string): boolean {
  if (ch.length === 0) {
    return false;
  }
  const code = ch.charCodeAt(0);
  return (code >= 65 && code <= 90) || (code >= 97 && code <= 122);
}

function isCjkChar(ch: string): boolean {
  if (ch.length === 0) {
    return false;
  }
  const code = ch.charCodeAt(0);
  return (code >= 0x3400 && code <= 0x4DBF) || (code >= 0x4E00 && code <= 0x9FFF);
}

function isChineseNumeralChar(ch: string): boolean {
  return '零一二三四五六七八九十百千万两俩0123456789'.indexOf(ch) >= 0;
}

function isChinesePunctuationChar(ch: string): boolean {
  return '，。！？；：、（）《》“”‘’'.indexOf(ch) >= 0;
}

function shouldDropAsciiNoise(token: string, previous: string, next: string): boolean {
  if (token.length === 0 || token.length > MIXED_ASCII_NOISE_MAX_CHARS) {
    return false;
  }

  const lower = token.toLowerCase();
  if (lower === 's' || lower === 'unk') {
    return true;
  }
  if (!isCjkChar(previous) && !isChinesePunctuationChar(previous) && !isChineseNumeralChar(previous)) {
    return false;
  }
  if (!isCjkChar(next) && !isChinesePunctuationChar(next) && !isChineseNumeralChar(next)) {
    return false;
  }

  return true;
}

function cleanMixedAsciiNoise(text: string): string {
  let ans = '';
  let i = 0;
  while (i < text.length) {
    const ch = text.charAt(i);
    if (!isAsciiLetter(ch)) {
      ans += ch;
      i += 1;
      continue;
    }

    const start = i;
    while (i < text.length && isAsciiLetter(text.charAt(i))) {
      i += 1;
    }
    const token = text.slice(start, i);
    const previous = start > 0 ? text.charAt(start - 1) : '';
    const next = i < text.length ? text.charAt(i) : '';

    if (token.toLowerCase() === 'bug' && isChineseNumeralChar(previous) && isChineseNumeralChar(next)) {
      ans += '个';
      continue;
    }
    if (shouldDropAsciiNoise(token, previous, next)) {
      continue;
    }
    ans += token;
  }
  return ans;
}

function collapseRepeatedPunctuation(text: string): string {
  let ans = '';
  for (let i = 0; i < text.length; ++i) {
    const ch = text.charAt(i);
    if (isChinesePunctuationChar(ch) && ans.length > 0 && isChinesePunctuationChar(ans.charAt(ans.length - 1))) {
      ans = `${ans.slice(0, ans.length - 1)}${ch}`;
      continue;
    }
    ans += ch;
  }
  return ans;
}

function cleanDecodedText(text: string): string {
  let ans = text.trim();
  ans = ans.replace(/<\s*\/\s*s\s*>/gi, '');
  ans = ans.replace(/<\s*s\s*>/gi, '');
  ans = ans.replace(/<\s*unk\s*>/gi, '');
  ans = cleanMixedAsciiNoise(ans);
  ans = collapseRepeatedPunctuation(ans);
  return ans.trim();
}

function appendTextWithOverlap(base: string, next: string): string {
  const left = cleanDecodedText(base);
  const right = cleanDecodedText(next);
  if (left.length === 0) {
    return right;
  }
  if (right.length === 0) {
    return left;
  }

  const maxOverlap = Math.min(TEXT_OVERLAP_MAX_CHARS, left.length, right.length);
  for (let overlap = maxOverlap; overlap >= 2; --overlap) {
    if (left.slice(left.length - overlap) === right.slice(0, overlap)) {
      return `${left}${right.slice(overlap)}`;
    }
  }

  const searchStart = Math.max(0, left.length - TEXT_OVERLAP_SEARCH_CHARS);
  const searchArea = left.slice(searchStart);
  for (let overlap = maxOverlap; overlap >= TEXT_TAIL_REPLACE_MIN_OVERLAP_CHARS; --overlap) {
    const prefix = right.slice(0, overlap);
    const prefixAt = searchArea.lastIndexOf(prefix);
    if (prefixAt >= 0) {
      const tailStart = searchStart + prefixAt;
      if (left.length - tailStart <= TEXT_TAIL_REPLACE_MAX_CHARS) {
        return `${left.slice(0, tailStart)}${right}`;
      }
      return `${left}${right.slice(overlap)}`;
    }
  }

  const last = left.charAt(left.length - 1);
  const first = right.charAt(0);
  return isAsciiWordChar(last) && isAsciiWordChar(first) ? `${left} ${right}` : `${left}${right}`;
}

function embeddingToNumbers(embedding: Float32Array): number[] {
  const ans: number[] = [];
  for (let i = 0; i < embedding.length; ++i) {
    ans.push(embedding[i]);
  }
  return ans;
}

function cloneEmbedding(embedding: Float32Array): Float32Array {
  const ans = new Float32Array(embedding.length);
  ans.set(embedding);
  return ans;
}

function cleanSpeakerName(name?: string): string {
  const cleanName = name ? name.trim() : '';
  if (cleanName.length === 0) {
    return '我';
  }
  return cleanName.length > 24 ? cleanName.slice(0, 24) : cleanName;
}

function getRegisteredSpeakerNames(): string[] {
  const names: string[] = [];
  for (let i: number = 0; i < restoredVoiceprints.length; i++) {
    names.push(restoredVoiceprints[i].displayName);
  }
  return names;
}

function computeSpeakerEmbedding(samples: Float32Array, sampleRate: number): Float32Array {
  if (sampleRate !== SAMPLE_RATE) {
    throw new Error(`sample rate ${sampleRate} is not ${SAMPLE_RATE}`);
  }
  if (samples.length < VOICEPRINT_MIN_IDENTIFY_SAMPLES) {
    throw new Error('voiceprint audio is too short');
  }

  const extractor = ensureSpeakerEmbeddingExtractor();
  const stream = extractor.createStream();
  const wave: Samples = { samples, sampleRate };
  stream.acceptWaveform(wave);
  stream.inputFinished();
  if (!extractor.isReady(stream)) {
    throw new Error('voiceprint model is not ready for this audio');
  }
  return cloneEmbedding(extractor.compute(stream));
}

function restoreVoiceprints(profiles: SpeakerSearchProfile[] = []): void {
  const restored: SpeakerSearchProfile[] = [];
  for (let i: number = 0; i < profiles.length && restored.length < SPEAKER_MAX_PROFILES; i++) {
    const profile: SpeakerSearchProfile = profiles[i];
    if (!profile || profile.modelId !== SPEAKER_MODEL_ID ||
      profile.modelSha256 !== SPEAKER_MODEL_SHA256 ||
      profile.embeddingDim !== SPEAKER_EMBEDDING_DIM ||
      profile.centroid.length !== SPEAKER_EMBEDDING_DIM) {
      continue;
    }
    restored.push({
      personId: profile.personId,
      displayName: cleanSpeakerName(profile.displayName),
      modelId: profile.modelId,
      modelSha256: profile.modelSha256,
      embeddingDim: profile.embeddingDim,
      centroid: normalizeEmbedding(profile.centroid),
    });
  }
  restoredVoiceprints = restored;
  if (restored.length > 0) {
    ensureSpeakerEmbeddingExtractor();
  }
}

function analyzeVoiceprintQuality(samples: Float32Array,
  activeEnrollment: boolean): EnrollmentQuality {
  let speechMs: number = Math.round(samples.length * 1000 / SAMPLE_RATE);
  let voiceActivityRatio: number = samples.length > 0 ? 1 : 0;
  try {
    const flags: boolean[] = detectSpeechFlags(samples);
    if (flags.length > 0) {
      let speechFrames: number = 0;
      for (let i: number = 0; i < flags.length; i++) {
        if (flags[i]) {
          speechFrames += 1;
        }
      }
      voiceActivityRatio = speechFrames / flags.length;
      const detector: Vad | undefined = ensureFileVad();
      if (detector) {
        speechMs = Math.min(speechMs, Math.round(speechFrames *
          detector.config.sileroVad.windowSize * 1000 / SAMPLE_RATE));
      }
    }
  } catch (_error) {
    // The energy/clipping checks remain available if the VAD cannot be used.
  }
  const rms: number = computeRms(samples);
  let clipped: number = 0;
  for (let i: number = 0; i < samples.length; i++) {
    if (Math.abs(samples[i]) >= 0.985) {
      clipped += 1;
    }
  }
  const clippingRatio: number = samples.length > 0 ? clipped / samples.length : 0;
  let reason: string = '';
  if (activeEnrollment && speechMs < SPEAKER_ACTIVE_ENROLL_MIN_MS) {
    reason = '请自然说满 4 秒';
  } else if (activeEnrollment && speechMs > SPEAKER_ACTIVE_ENROLL_MAX_MS) {
    reason = '单次录音请控制在 4–6 秒';
  } else if (samples.length < VOICEPRINT_MIN_IDENTIFY_SAMPLES || speechMs < 1000) {
    reason = '有效语音太短';
  } else if (voiceActivityRatio < 0.35) {
    reason = '语音占比太低或背景噪声过多，请换安静环境重试';
  } else if (rms < 0.005) {
    reason = '声音太小或静音过多，请靠近手机重试';
  } else if (clippingRatio > 0.015) {
    reason = '声音出现削波，请降低音量或远离麦克风重试';
  }
  const durationScore: number = Math.min(1, speechMs /
    (activeEnrollment ? SPEAKER_ACTIVE_ENROLL_MIN_MS : SPEAKER_IDENTIFY_MIN_SPEECH_MS));
  const loudnessScore: number = Math.min(1, rms / 0.04);
  const clippingScore: number = Math.max(0, 1 - clippingRatio / 0.015);
  const activityScore: number = Math.min(1, voiceActivityRatio / 0.75);
  return {
    accepted: reason.length === 0,
    reason: reason,
    speechMs: speechMs,
    rms: rms,
    clippingRatio: clippingRatio,
    voiceActivityRatio: voiceActivityRatio,
    qualityScore: Math.max(0, Math.min(1,
      durationScore * 0.35 + loudnessScore * 0.25 + clippingScore * 0.20 +
      activityScore * 0.20)),
  };
}

function trimVoiceprintSilence(samples: Float32Array): Float32Array {
  const frameSamples: number = 320;
  const threshold: number = 0.004;
  let first: number = -1;
  let last: number = -1;
  for (let start: number = 0; start < samples.length; start += frameSamples) {
    const end: number = Math.min(samples.length, start + frameSamples);
    let sum: number = 0;
    for (let i: number = start; i < end; i++) {
      sum += samples[i] * samples[i];
    }
    const rms: number = end > start ? Math.sqrt(sum / (end - start)) : 0;
    if (rms >= threshold) {
      if (first < 0) {
        first = start;
      }
      last = end;
    }
  }
  if (first < 0 || last <= first) {
    return new Float32Array(0);
  }
  const pad: number = 1600;
  return samples.subarray(Math.max(0, first - pad), Math.min(samples.length, last + pad));
}

function extractVoiceprint(samples: Float32Array,
  activeEnrollment: boolean): VoiceprintExtractionResult {
  const quality: EnrollmentQuality = analyzeVoiceprintQuality(samples, activeEnrollment);
  if (!quality.accepted) {
    return { embedding: [], quality: quality };
  }
  const embedding: Float32Array = computeSpeakerEmbedding(samples, SAMPLE_RATE);
  if (embedding.length !== SPEAKER_EMBEDDING_DIM) {
    throw new Error(`声纹模型输出维度异常: ${embedding.length}`);
  }
  return { embedding: embeddingToNumbers(embedding), quality: quality };
}

function collectSpeakerSamples(
  samples: Float32Array,
  segments: NormalizedSpeakerSegment[],
  speaker: number
): Float32Array {
  const chunks: Float32Array[] = [];
  let total = 0;
  for (let i = 0; i < segments.length && total < VOICEPRINT_MAX_IDENTIFY_SAMPLES; ++i) {
    const segment = segments[i];
    if (segment.speaker !== speaker) {
      continue;
    }
    const startSample = Math.max(0, Math.floor(segment.start * SAMPLE_RATE));
    const endSample = Math.min(
      samples.length,
      Math.ceil(segment.end * SAMPLE_RATE)
    );
    if (endSample <= startSample) {
      continue;
    }
    let chunk = samples.subarray(startSample, endSample);
    const remaining = VOICEPRINT_MAX_IDENTIFY_SAMPLES - total;
    if (chunk.length > remaining) {
      chunk = chunk.subarray(0, remaining);
    }
    chunks.push(chunk);
    total += chunk.length;
  }
  return concatChunks(chunks);
}

interface SpeakerMatch {
  personId: string;
  displayName: string;
  score: number;
}

interface SpeakerCandidateBuild {
  bySpeaker: Map<number, SpeakerCandidate>;
  candidates: SpeakerCandidate[];
}

function findBestSpeakerMatch(query: number[]): SpeakerMatch | undefined {
  let best: SpeakerSearchProfile | undefined = undefined;
  let bestScore: number = -1;
  for (let i: number = 0; i < restoredVoiceprints.length; i++) {
    const score: number = cosineSimilarity(query, restoredVoiceprints[i].centroid);
    if (score > bestScore) {
      best = restoredVoiceprints[i];
      bestScore = score;
    }
  }
  if (!best || bestScore < VOICEPRINT_MATCH_THRESHOLD) {
    return undefined;
  }
  return { personId: best.personId, displayName: best.displayName, score: bestScore };
}

function intersectsSpeakerOverlap(segment: NormalizedSpeakerSegment,
  overlaps: SpeakerOverlapRange[]): boolean {
  const paddedStart: number = segment.start - DIARIZATION_SEGMENT_PADDING_SECONDS;
  const paddedEnd: number = segment.end + DIARIZATION_SEGMENT_PADDING_SECONDS;
  for (let i: number = 0; i < overlaps.length; i++) {
    if (paddedStart < overlaps[i].end && paddedEnd > overlaps[i].start) {
      return true;
    }
  }
  return false;
}

function buildSpeakerCandidates(samples: Float32Array, segments: NormalizedSpeakerSegment[],
  meetingId: string, overlaps: SpeakerOverlapRange[]): SpeakerCandidateBuild {
  const bySpeaker: Map<number, SpeakerCandidate> = new Map<number, SpeakerCandidate>();
  const candidates: SpeakerCandidate[] = [];
  const seen: Set<number> = new Set<number>();
  for (let i: number = 0; i < segments.length; i++) {
    const speaker: number = segments[i].speaker;
    if (seen.has(speaker)) {
      continue;
    }
    seen.add(speaker);
    let segmentCount: number = 0;
    let speechMs: number = 0;
    const turnEmbeddings: number[][] = [];
    const safeSegments: NormalizedSpeakerSegment[] = [];
    for (let j: number = 0; j < segments.length; j++) {
      const segment: NormalizedSpeakerSegment = segments[j];
      if (segment.speaker !== speaker || intersectsSpeakerOverlap(segment, overlaps)) {
        continue;
      }
      safeSegments.push(segment);
      segmentCount += 1;
      speechMs += Math.round(speakerSegmentDuration(segment) * 1000);
      const startSample: number = Math.max(0, Math.floor(segment.start * SAMPLE_RATE));
      const maxEndSample: number = startSample + 8 * SAMPLE_RATE;
      const endSample: number = Math.min(samples.length, maxEndSample,
        Math.ceil(segment.end * SAMPLE_RATE));
      if (endSample - startSample < VOICEPRINT_MIN_IDENTIFY_SAMPLES) {
        continue;
      }
      try {
        const embedding: Float32Array = computeSpeakerEmbedding(
          samples.subarray(startSample, endSample), SAMPLE_RATE);
        turnEmbeddings.push(embeddingToNumbers(embedding));
      } catch (_error) {
      }
    }

    const combinedSamples: Float32Array = collectSpeakerSamples(samples, safeSegments, speaker);
    const quality: EnrollmentQuality = analyzeVoiceprintQuality(combinedSamples, false);
    let query: number[] = averageEmbeddings(turnEmbeddings);
    if (query.length === 0 && combinedSamples.length >= VOICEPRINT_MIN_IDENTIFY_SAMPLES) {
      try {
        query = normalizeEmbedding(embeddingToNumbers(
          computeSpeakerEmbedding(combinedSamples, SAMPLE_RATE)));
      } catch (_error) {
        query = [];
      }
    }

    const qualified: boolean = segmentCount >= SPEAKER_IDENTIFY_MIN_SEGMENTS ||
      speechMs >= SPEAKER_IDENTIFY_MIN_SPEECH_MS;
    const match: SpeakerMatch | undefined = qualified && quality.accepted &&
      query.length === SPEAKER_EMBEDDING_DIM ? findBestSpeakerMatch(query) : undefined;
    const candidate: SpeakerCandidate = {
      meetingId: meetingId,
      clusterId: `S${speaker + 1}`,
      segmentCount: segmentCount,
      speechMs: speechMs,
      qualityScore: quality.qualityScore,
      queryEmbedding: quality.accepted && query.length === SPEAKER_EMBEDDING_DIM ? query : undefined,
      personId: match ? match.personId : undefined,
      displayName: match ? match.displayName : undefined,
      matchScore: match ? match.score : undefined,
    };
    bySpeaker.set(speaker, candidate);
    candidates.push(candidate);
  }
  return { bySpeaker: bySpeaker, candidates: candidates };
}

function formatTime(seconds: number): string {
  const totalSeconds = Math.max(0, seconds);
  const minutes = Math.floor(totalSeconds / 60);
  const sec = totalSeconds - minutes * 60;
  const secInt = Math.floor(sec);
  const tenths = Math.floor((sec - secInt) * 10);
  const secText = secInt < 10 ? `0${secInt}` : `${secInt}`;
  return `${minutes}:${secText}.${tenths}`;
}

function speakerSegmentDuration(segment: NormalizedSpeakerSegment): number {
  return Math.max(0, segment.end - segment.start);
}

function cloneSpeakerSegment(segment: NormalizedSpeakerSegment): NormalizedSpeakerSegment {
  return {
    start: segment.start,
    end: segment.end,
    speaker: segment.speaker
  };
}

function pushSpeakerSegment(segments: NormalizedSpeakerSegment[], segment: NormalizedSpeakerSegment): void {
  if (speakerSegmentDuration(segment) < DIARIZATION_RAW_MIN_SEGMENT_SECONDS) {
    return;
  }

  const previous = segments.length > 0 ? segments[segments.length - 1] : undefined;
  if (previous && previous.speaker === segment.speaker &&
    segment.start - previous.end <= DIARIZATION_MERGE_GAP_SECONDS) {
    previous.end = Math.max(previous.end, segment.end);
    return;
  }

  segments.push(segment);
}

function resolveSpeakerOverlaps(segments: NormalizedSpeakerSegment[]): NormalizedSpeakerSegment[] {
  const resolved: NormalizedSpeakerSegment[] = [];
  for (let i = 0; i < segments.length; ++i) {
    const segment = cloneSpeakerSegment(segments[i]);
    if (speakerSegmentDuration(segment) < DIARIZATION_RAW_MIN_SEGMENT_SECONDS) {
      continue;
    }

    const previous = resolved.length > 0 ? resolved[resolved.length - 1] : undefined;
    if (previous && segment.start < previous.end) {
      if (previous.speaker === segment.speaker) {
        previous.end = Math.max(previous.end, segment.end);
        continue;
      }

      const overlap = previous.end - segment.start;
      const segmentDuration = speakerSegmentDuration(segment);
      const previousDuration = speakerSegmentDuration(previous);
      if (segmentDuration <= DIARIZATION_SHORT_ISLAND_SECONDS ||
        overlap / segmentDuration >= DIARIZATION_OVERLAP_ABSORB_RATIO) {
        previous.end = Math.max(previous.end, segment.end);
        continue;
      }

      if (previousDuration <= DIARIZATION_SHORT_ISLAND_SECONDS &&
        overlap / previousDuration >= DIARIZATION_OVERLAP_ABSORB_RATIO) {
        previous.speaker = segment.speaker;
        previous.end = Math.max(previous.end, segment.end);
        continue;
      }

      segment.start = previous.end;
      if (speakerSegmentDuration(segment) < DIARIZATION_RAW_MIN_SEGMENT_SECONDS) {
        continue;
      }
    }

    pushSpeakerSegment(resolved, segment);
  }

  return resolved;
}

/** Cross-speaker overlaps are excluded from every enrollment candidate. */
function findSpeakerOverlapRanges(
  segments: OfflineSpeakerDiarizationSegment[]): SpeakerOverlapRange[] {
  const valid: OfflineSpeakerDiarizationSegment[] = segments
    .filter((segment: OfflineSpeakerDiarizationSegment): boolean => segment.end > segment.start)
    .sort((left: OfflineSpeakerDiarizationSegment,
      right: OfflineSpeakerDiarizationSegment): number => left.start - right.start);
  const raw: SpeakerOverlapRange[] = [];
  for (let i: number = 0; i < valid.length; i++) {
    for (let j: number = i + 1; j < valid.length && valid[j].start < valid[i].end; j++) {
      if (valid[i].speaker === valid[j].speaker) {
        continue;
      }
      const range: SpeakerOverlapRange = {
        start: Math.max(valid[i].start, valid[j].start),
        end: Math.min(valid[i].end, valid[j].end),
      };
      if (range.end > range.start) {
        raw.push(range);
      }
    }
  }
  raw.sort((left: SpeakerOverlapRange, right: SpeakerOverlapRange): number =>
    left.start - right.start);
  const merged: SpeakerOverlapRange[] = [];
  for (let i: number = 0; i < raw.length; i++) {
    const previous: SpeakerOverlapRange | undefined = merged.length > 0 ?
      merged[merged.length - 1] : undefined;
    if (previous && raw[i].start <= previous.end) {
      previous.end = Math.max(previous.end, raw[i].end);
    } else {
      merged.push({ start: raw[i].start, end: raw[i].end });
    }
  }
  return merged;
}

function smoothShortSpeakerIslands(segments: NormalizedSpeakerSegment[]): NormalizedSpeakerSegment[] {
  const smoothed: NormalizedSpeakerSegment[] = [];
  for (let i = 0; i < segments.length; ++i) {
    const segment = cloneSpeakerSegment(segments[i]);
    const duration = speakerSegmentDuration(segment);
    const previous = smoothed.length > 0 ? smoothed[smoothed.length - 1] : undefined;
    const next = i + 1 < segments.length ? segments[i + 1] : undefined;

    if (duration <= DIARIZATION_SHORT_ISLAND_SECONDS) {
      const previousGap = previous ? segment.start - previous.end : Number.MAX_VALUE;
      const nextGap = next ? next.start - segment.end : Number.MAX_VALUE;
      if (previous && next && previous.speaker === next.speaker &&
        previousGap <= DIARIZATION_NEIGHBOR_GAP_SECONDS &&
        nextGap <= DIARIZATION_NEIGHBOR_GAP_SECONDS) {
        previous.end = Math.max(previous.end, next.end);
        i += 1;
        continue;
      }

      if (previous && previousGap <= DIARIZATION_NEIGHBOR_GAP_SECONDS) {
        segment.speaker = previous.speaker;
      } else if (next && nextGap <= DIARIZATION_NEIGHBOR_GAP_SECONDS) {
        segment.speaker = next.speaker;
      }
    }

    pushSpeakerSegment(smoothed, segment);
  }

  return smoothed;
}

function findNextStableSegment(
  segments: NormalizedSpeakerSegment[],
  startIndex: number,
  unstableSpeakers: Set<number>
): NormalizedSpeakerSegment | undefined {
  for (let i = startIndex; i < segments.length; ++i) {
    if (!unstableSpeakers.has(segments[i].speaker)) {
      return segments[i];
    }
  }
  return undefined;
}

function mergeUnstableSpeakers(segments: NormalizedSpeakerSegment[]): NormalizedSpeakerSegment[] {
  if (countSpeakerIds(segments) <= 2) {
    return segments;
  }

  const totalDurationBySpeaker: Map<number, number> = new Map();
  const maxTurnBySpeaker: Map<number, number> = new Map();
  for (let i = 0; i < segments.length; ++i) {
    const speaker = segments[i].speaker;
    const duration = speakerSegmentDuration(segments[i]);
    totalDurationBySpeaker.set(speaker, (totalDurationBySpeaker.get(speaker) ?? 0) + duration);
    maxTurnBySpeaker.set(speaker, Math.max(maxTurnBySpeaker.get(speaker) ?? 0, duration));
  }

  const unstableSpeakers: Set<number> = new Set();
  totalDurationBySpeaker.forEach((totalDuration: number, speaker: number) => {
    const maxTurn = maxTurnBySpeaker.get(speaker) ?? 0;
    if (totalDuration < DIARIZATION_MIN_SPEAKER_TOTAL_SECONDS &&
      maxTurn < DIARIZATION_MIN_RELIABLE_TURN_SECONDS) {
      unstableSpeakers.add(speaker);
    }
  });

  if (unstableSpeakers.size === 0 || unstableSpeakers.size >= totalDurationBySpeaker.size) {
    return segments;
  }

  const merged: NormalizedSpeakerSegment[] = [];
  for (let i = 0; i < segments.length; ++i) {
    const segment = cloneSpeakerSegment(segments[i]);
    if (unstableSpeakers.has(segment.speaker)) {
      const previous = merged.length > 0 ? merged[merged.length - 1] : undefined;
      const next = findNextStableSegment(segments, i + 1, unstableSpeakers);
      const previousGap = previous ? segment.start - previous.end : Number.MAX_VALUE;
      const nextGap = next ? next.start - segment.end : Number.MAX_VALUE;
      if (previous && previousGap <= DIARIZATION_UNSTABLE_SPEAKER_MERGE_GAP_SECONDS &&
        (!next || previousGap <= nextGap || nextGap > DIARIZATION_UNSTABLE_SPEAKER_MERGE_GAP_SECONDS)) {
        segment.speaker = previous.speaker;
      } else if (next && nextGap <= DIARIZATION_UNSTABLE_SPEAKER_MERGE_GAP_SECONDS) {
        segment.speaker = next.speaker;
      }
    }
    pushSpeakerSegment(merged, segment);
  }

  return merged;
}

function normalizeSpeakerSegments(segments: OfflineSpeakerDiarizationSegment[]): NormalizedSpeakerSegment[] {
  const sorted = segments
    .filter((segment: OfflineSpeakerDiarizationSegment) => segment.end > segment.start)
    .sort((a: OfflineSpeakerDiarizationSegment, b: OfflineSpeakerDiarizationSegment) => a.start - b.start);

  const initial: NormalizedSpeakerSegment[] = [];
  for (let i = 0; i < sorted.length; ++i) {
    const segment = sorted[i];
    const duration = segment.end - segment.start;
    if (duration < DIARIZATION_RAW_MIN_SEGMENT_SECONDS) {
      continue;
    }

    initial.push({
      start: segment.start,
      end: segment.end,
      speaker: segment.speaker
    });
  }

  const merged = mergeUnstableSpeakers(smoothShortSpeakerIslands(resolveSpeakerOverlaps(initial)));
  const speakerMap: Map<number, number> = new Map();
  let nextSpeaker = 0;
  for (let i = 0; i < merged.length; ++i) {
    const originalSpeaker = merged[i].speaker;
    let speaker = speakerMap.get(originalSpeaker);
    if (speaker === undefined) {
      speaker = nextSpeaker;
      speakerMap.set(originalSpeaker, speaker);
      nextSpeaker += 1;
    }
    merged[i].speaker = speaker;
  }

  return merged;
}

function splitTextBySegmentDuration(text: string, segments: NormalizedSpeakerSegment[]): string[] {
  const cleanText = text.trim();
  if (cleanText.length === 0 || segments.length === 0) {
    return [];
  }

  let totalDuration = 0;
  for (let i = 0; i < segments.length; ++i) {
    totalDuration += Math.max(0, segments[i].end - segments[i].start);
  }
  if (totalDuration <= 0) {
    return [cleanText];
  }

  const parts: string[] = [];
  let textOffset = 0;
  for (let i = 0; i < segments.length; ++i) {
    if (textOffset >= cleanText.length) {
      parts.push('');
      continue;
    }

    let count = 0;
    if (i === segments.length - 1) {
      count = cleanText.length - textOffset;
    } else {
      const ratio = Math.max(0, segments[i].end - segments[i].start) / totalDuration;
      count = Math.max(1, Math.round(cleanText.length * ratio));
      count = Math.min(count, cleanText.length - textOffset);
    }
    parts.push(cleanText.slice(textOffset, textOffset + count));
    textOffset += count;
  }
  return parts;
}

function countSpeakerIds(segments: NormalizedSpeakerSegment[]): number {
  const speakerIds: Set<number> = new Set();
  for (let i = 0; i < segments.length; ++i) {
    speakerIds.add(segments[i].speaker);
  }
  return speakerIds.size;
}

function formatSpeakerSegmentLabel(
  segment: NormalizedSpeakerSegment,
  candidates: Map<number, SpeakerCandidate>
): string {
  const speaker = segment.speaker + 1;
  const candidate: SpeakerCandidate | undefined = candidates.get(segment.speaker);
  return candidate && candidate.displayName ? `${candidate.displayName}(S${speaker})` :
    `未知说话人(S${speaker})`;
}

function shouldRescoreDiarizationSegments(
  samples: Float32Array,
  sampleRate: number,
  limit: number,
  enableSegmentRescore: boolean
): boolean {
  if (enableSegmentRescore) {
    return true;
  }
  const duration = samples.length / sampleRate;
  return limit <= DIARIZATION_AUTO_RESCORE_MAX_SEGMENTS &&
    duration <= DIARIZATION_AUTO_RESCORE_MAX_SECONDS;
}

function pcm16ToFloat32(buffer: ArrayBuffer): Float32Array {
  const input = new Int16Array(buffer);
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; ++i) {
    output[i] = input[i] / 32768.0;
  }
  return output;
}

function float32ToPcm16Buffer(samples: Float32Array, start: number, end: number): ArrayBuffer {
  const length = Math.max(0, end - start);
  const buffer = new ArrayBuffer(length * 2);
  const output = new Int16Array(buffer);
  for (let i = 0; i < length; ++i) {
    const sample = Math.max(-1, Math.min(1, samples[start + i]));
    output[i] = Math.max(-32768, Math.min(32767, Math.round(sample * 32767)));
  }
  return buffer;
}

function ensureStream(): OnlineStream {
  if (!recognizer) {
    throw new Error('recognizer is not initialized');
  }
  if (!micStream) {
    micStream = recognizer.createStream();
  }
  return micStream;
}

function decodeReady(stream: OnlineStream): void {
  if (!recognizer) {
    return;
  }

  const t0 = Date.now();
  let emittedPartial = false;
  let partialIntervalMs = 0;

  while (recognizer.isReady(stream)) {
    recognizer.decode(stream);
    const result: OnlineRecognizerResult = recognizer.getResult(stream);
    const text = result.text.trim();
    if (text !== '' && text !== latestText) {
      const now = Date.now();
      if (lastPartialEmitMs > 0) {
        partialIntervalMs = now - lastPartialEmitMs;
      }
      lastPartialEmitMs = now;
      latestText = text;
      emittedPartial = true;
      const response: WorkerResponse = { msgType: 'sherpa-asr-partial', text };
      postResponse(response);
    }

  }

  if (emittedPartial) {
    const chunkDecodeMs = Date.now() - t0;
    const now = Date.now();
    // Throttle: at most one chunk-telemetry message per interval.
    if (now - lastChunkTelemetryMs >= TELEMETRY_CHUNK_MIN_INTERVAL_MS) {
      lastChunkTelemetryMs = now;
      postTelemetry({ chunkDecodeMs, partialIntervalMs });
    }
  }
}

function acceptWaveform(samples: Float32Array): void {
  const stream = ensureStream();
  const wave: Samples = { samples, sampleRate: SAMPLE_RATE };
  stream.acceptWaveform(wave);
  decodeReady(stream);
}

function decodeSamplesDirect(samples: Float32Array, sampleRate: number): string {
  if (!recognizer) {
    throw new Error('recognizer is not initialized');
  }

  const stream = recognizer.createStream();
  const wave: Samples = { samples, sampleRate };
  stream.acceptWaveform(wave);
  const tailPadding = new Float32Array(DIRECT_TAIL_PADDING_SAMPLES);
  const tailWave: Samples = { samples: tailPadding, sampleRate };
  stream.acceptWaveform(tailWave);
  stream.inputFinished();
  while (recognizer.isReady(stream)) {
    recognizer.decode(stream);
  }
  return cleanDecodedText(recognizer.getResult(stream).text);
}

function decodeSamplesDirectWithPunctuation(samples: Float32Array, sampleRate: number): string {
  return addPunctuation(decodeSamplesDirect(samples, sampleRate));
}

function decodeSamplesWithOfflineRecognizer(
  offlineAsr: OfflineRecognizer,
  samples: Float32Array,
  sampleRate: number
): string {
  const stream: OfflineStream = offlineAsr.createStream();
  const wave: Samples = { samples, sampleRate };
  stream.acceptWaveform(wave);
  offlineAsr.decode(stream);
  const result: OfflineRecognizerResult = offlineAsr.getResult(stream);
  return cleanDecodedText(result.text);
}

function decodeSamplesFinalDirect(samples: Float32Array, sampleRate: number): string {
  return decodeSamplesWithOfflineRecognizer(ensureFinalRecognizer(), samples, sampleRate);
}

function decodeSamplesFinalDirectWithPunctuation(samples: Float32Array, sampleRate: number): string {
  return addPunctuation(decodeSamplesFinalDirect(samples, sampleRate));
}

function decodeSamplesOfflineParaformerDirect(samples: Float32Array, sampleRate: number): string {
  return decodeSamplesWithOfflineRecognizer(ensureOfflineParaformerRecognizer(), samples, sampleRate);
}

function decodeSamplesOfflineParaformerDirectWithPunctuation(samples: Float32Array, sampleRate: number): string {
  return addPunctuation(decodeSamplesOfflineParaformerDirect(samples, sampleRate));
}

type SegmentDecoder = (samples: Float32Array, sampleRate: number) => string;

/** Plain concatenation, used when the plan guarantees the segments do not overlap. */
function appendSegmentText(base: string, next: string): string {
  const clean = cleanDecodedText(next);
  if (clean.length === 0) {
    return base;
  }
  return base.length === 0 ? clean : base + clean;
}

function joinSegmentText(base: string, next: string, mergeTextOverlap: boolean): string {
  return mergeTextOverlap ? appendTextWithOverlap(base, next) : appendSegmentText(base, next);
}

/**
 * Decode a long recording block by block, using the shared decode plan so every decoder
 * (streaming, SenseVoice, offline Paraformer) cuts the audio the same way.
 */
/**
 * Ask a second recogniser whether a block really said that.
 *
 * Fails open. A block is only dropped when the second model ran and confirmed nothing; if
 * it cannot be loaded or throws, its silence means nothing and the text stands. Losing a
 * transcript because a corroborating model failed would be a worse bug than the one this
 * is here to catch.
 */
function corroboratedByModel(
  text: string,
  samples: Float32Array,
  sampleRate: number,
  corroborate: SegmentDecoder
): boolean {
  try {
    return isCorroborated(text, corroborate(samples, sampleRate));
  } catch (error) {
    hilog.warn(DOMAIN, TAG, `corroboration unavailable, keeping block: ${error}`);
    return true;
  }
}

function decodeSamplesWithPlan(
  samples: Float32Array,
  sampleRate: number,
  decodeSegment: SegmentDecoder,
  corroborate?: SegmentDecoder
): string {
  const plan = buildDecodePlan(samples, sampleRate);
  let rawText = '';
  for (let i = 0; i < plan.segments.length; ++i) {
    const segment = plan.segments[i];
    const start = Math.max(0, segment.start - plan.padSamples);
    const end = Math.min(samples.length, segment.end + plan.padSamples);
    if (end <= start) {
      continue;
    }

    const block = samples.subarray(start, end);
    const segmentText = decodeSegment(block, sampleRate);
    // Blocks now cover silence as well as speech, and this model answers silence with
    // fluent text rather than with nothing. Dropped here rather than at the end so one
    // such block cannot contaminate the rest. See isDecodeRepetitionCollapse.
    if (isDecodeRepetitionCollapse(segmentText)) {
      continue;
    }
    // Repetition only catches the crude version. The fluent version — an ordinary
    // sentence about a courtroom, from a block containing no such thing — is invisible
    // to every text statistic tried, and needs a second opinion on the audio itself.
    if (corroborate !== undefined && segmentText.trim().length > 0 &&
      !corroboratedByModel(segmentText, block, sampleRate, corroborate)) {
      continue;
    }
    rawText = joinSegmentText(rawText, segmentText, plan.mergeTextOverlap);
  }

  return addPunctuation(cleanDecodedText(rawText));
}

function decodeSamplesSegmentedWithPunctuation(samples: Float32Array, sampleRate: number): string {
  if (!isLongAudio(samples, sampleRate)) {
    return decodeSamplesDirectWithPunctuation(samples, sampleRate);
  }
  return decodeSamplesWithPlan(samples, sampleRate, decodeSamplesDirect);
}

function decodeSamplesFinalSegmentedWithPunctuation(samples: Float32Array, sampleRate: number): string {
  if (!isLongAudio(samples, sampleRate)) {
    return decodeSamplesFinalDirectWithPunctuation(samples, sampleRate);
  }
  return decodeSamplesWithPlan(samples, sampleRate, decodeSamplesFinalDirect);
}

function decodeSamplesOfflineParaformerSegmentedWithPunctuation(samples: Float32Array, sampleRate: number): string {
  if (!isLongAudio(samples, sampleRate)) {
    return decodeSamplesOfflineParaformerDirectWithPunctuation(samples, sampleRate);
  }
  // SenseVoice is the corroborator: it is already in the package, it is a different
  // architecture trained on different data, and it answers silence and noise with nothing
  // where paraformer answers them with sentences.
  return decodeSamplesWithPlan(samples, sampleRate,
    decodeSamplesOfflineParaformerDirect, decodeSamplesFinalDirect);
}

function decodeSamplesForFileModeWithPunctuation(
  samples: Float32Array,
  sampleRate: number,
  fileAsrMode?: string
): string {
  if (fileAsrMode === FILE_ASR_MODE_OFFLINE_PARAFORMER) {
    return decodeSamplesOfflineParaformerSegmentedWithPunctuation(samples, sampleRate);
  }
  if (fileAsrMode === FILE_ASR_MODE_SENSEVOICE) {
    return decodeSamplesFinalSegmentedWithPunctuation(samples, sampleRate);
  }
  if (fileAsrMode === FILE_ASR_MODE_STREAMING) {
    return decodeSamplesSegmentedWithPunctuation(samples, sampleRate);
  }

  return decodeSamplesBestWithPunctuation(samples, sampleRate);
}

function decodeSamplesForFileModeRaw(
  samples: Float32Array,
  sampleRate: number,
  fileAsrMode?: string
): string {
  if (fileAsrMode === FILE_ASR_MODE_OFFLINE_PARAFORMER) {
    return decodeSamplesOfflineParaformerDirect(samples, sampleRate);
  }
  if (fileAsrMode === FILE_ASR_MODE_SENSEVOICE) {
    return decodeSamplesFinalDirect(samples, sampleRate);
  }
  if (fileAsrMode === FILE_ASR_MODE_STREAMING) {
    return decodeSamplesDirect(samples, sampleRate);
  }

  try {
    const finalText = decodeSamplesFinalDirect(samples, sampleRate);
    if (finalText.trim().length > 0) {
      return finalText;
    }
  } catch (_error) {
    // Keep imported audio progress available if the larger model cannot be loaded.
  }
  return decodeSamplesDirect(samples, sampleRate);
}

function decodeSamplesBestWithPunctuation(samples: Float32Array, sampleRate: number): string {
  if (isLongAudio(samples, sampleRate)) {
    return decodeSamplesSegmentedWithPunctuation(samples, sampleRate);
  }

  try {
    const finalText = decodeSamplesFinalDirectWithPunctuation(samples, sampleRate);
    if (finalText.trim().length > 0) {
      return finalText;
    }
  } catch (_error) {
    // Keep the app usable if the larger final-pass model cannot be loaded.
  }
  return decodeSamplesSegmentedWithPunctuation(samples, sampleRate);
}

function decodeSamplesBestDirectWithPunctuation(samples: Float32Array, sampleRate: number): string {
  // Speaker-segment rescoring used to start from SenseVoice while the main transcript came
  // from offline Paraformer, so enabling diarization silently swapped in a different (and
  // on our benchmark weaker: 6.61% vs 5.01% CER) model. Try the main model first and keep
  // SenseVoice / streaming as fallbacks.
  try {
    const paraformerText = decodeSamplesOfflineParaformerDirectWithPunctuation(samples, sampleRate);
    if (paraformerText.trim().length > 0) {
      return paraformerText;
    }
  } catch (_error) {
    // Fall through to the next model.
  }

  try {
    const finalText = decodeSamplesFinalDirectWithPunctuation(samples, sampleRate);
    if (finalText.trim().length > 0) {
      return finalText;
    }
  } catch (_error) {
    // Segment-level rescoring should fall back to the streaming model.
  }
  return decodeSamplesDirectWithPunctuation(samples, sampleRate);
}

function finishCurrentSegment(): void {
  if (!recognizer || !micStream) {
    return;
  }

  const t0 = Date.now();
  finishingSegment = true;
  try {
    const stream = micStream;
    const tailPadding = new Float32Array(FINAL_TAIL_PADDING_SAMPLES);
    const tailWave: Samples = { samples: tailPadding, sampleRate: SAMPLE_RATE };
    stream.acceptWaveform(tailWave);
    stream.inputFinished();
    decodeReady(stream);

    const result = recognizer.getResult(stream);
    let rawText = result.text.trim() !== '' ? result.text.trim() : latestText.trim();
    if (rawText === '' && micLength > 0) {
      rawText = decodeSamplesDirect(concatChunks(micChunks), SAMPLE_RATE);
    }
    const text = addPunctuation(rawText);
    if (text !== '') {
      const response: WorkerResponse = { msgType: 'sherpa-asr-final', text };
      postResponse(response);
    }

    recognizer.reset(stream);
    micStream = recognizer.createStream();
    latestText = '';
    lastPartialEmitMs = 0;
    inSpeech = false;
    silenceSamples = 0;
    resetPreRoll();
    resetMicChunks();
    postTelemetry({ finalSegmentMs: Date.now() - t0 });
  } finally {
    finishingSegment = false;
  }
}

function vadSawSpeech(samples: Float32Array): boolean {
  if (!vad) {
    return true;
  }

  let detected = false;
  const windowSize = vad.config.sileroVad.windowSize;
  for (let i = 0; i < samples.length; i += windowSize) {
    vad.acceptWaveform(samples.subarray(i, Math.min(i + windowSize, samples.length)));
    detected = vad.isDetected() || detected;
    while (!vad.isEmpty()) {
      vad.pop();
    }
  }
  return detected;
}

function processMicPcm(buffer: ArrayBuffer): void {
  const samples = pcm16ToFloat32(buffer);
  appendSessionChunks(samples);
  appendMicChunks(samples);
  appendPreRoll(samples);

  const detected = vadSawSpeech(samples);
  if (detected) {
    silenceSamples = 0;
    if (!inSpeech) {
      inSpeech = true;
      acceptWaveform(concatChunks(preRollChunks));
      return;
    }
    acceptWaveform(samples);
    return;
  }

  if (inSpeech) {
    silenceSamples += samples.length;
    acceptWaveform(samples);
    if (silenceSamples >= TRAILING_SILENCE_SAMPLES) {
      finishCurrentSegment();
    }
  }
}

function processVoiceprintPcm(buffer: ArrayBuffer): void {
  const samples = pcm16ToFloat32(buffer);
  appendVoiceprintChunks(samples);
}

function startMic(): void {
  if (!recognizer) {
    throw new Error('recognizer is not initialized');
  }
  vad?.reset();
  micStream = recognizer.createStream();
  latestText = '';
  inSpeech = false;
  silenceSamples = 0;
  resetPreRoll();
  resetMicChunks();
  resetSessionChunks();
}

function startVoiceprintRegistration(): void {
  ensureSpeakerEmbeddingExtractor();
  resetVoiceprintChunks();
}

function stopVoiceprintRegistration(request: WorkerRequest): void {
  const samples = trimVoiceprintSilence(concatChunks(voiceprintChunks));
  try {
    const extraction: VoiceprintExtractionResult = extractVoiceprint(samples, true);
    if (!extraction.quality.accepted) {
      const rejected: WorkerResponse = {
        msgType: 'sherpa-voiceprint-rejected',
        speakerName: cleanSpeakerName(request.speakerName),
        quality: extraction.quality,
        voiceprintReady: true,
      };
      postResponse(rejected);
      return;
    }
    const response: WorkerResponse = {
      msgType: 'sherpa-voiceprint-registered',
      speakerName: cleanSpeakerName(request.speakerName),
      speakerNames: getRegisteredSpeakerNames(),
      embedding: extraction.embedding,
      quality: extraction.quality,
      voiceprintReady: true,
    };
    postResponse(response);
  } catch (error) {
    postError(`${error}`, 'sherpa-voiceprint-error');
  } finally {
    resetVoiceprintChunks();
  }
}

function cancelMic(): void {
  vad?.reset();
  latestText = '';
  inSpeech = false;
  silenceSamples = 0;
  resetPreRoll();
  resetMicChunks();
  resetSessionChunks();
  if (recognizer) {
    micStream = recognizer.createStream();
  }
}

function cancelVoiceprintRegistration(): void {
  resetVoiceprintChunks();
}

function flushVadAndFinishSegment(): void {
  vad?.flush();
  while (vad && !vad.isEmpty()) {
    vad.pop();
  }
  if (!inSpeech && latestText.trim() === '' && micLength > 0) {
    acceptWaveform(concatChunks(micChunks));
  }
  finishCurrentSegment();
}

function shouldConsiderMicFallback(samples: Float32Array, baselineText: string): boolean {
  const duration = samples.length / SAMPLE_RATE;
  if (duration < MIC_FALLBACK_MIN_SECONDS) {
    return false;
  }
  if (computeRms(samples) < MIC_FALLBACK_MIN_RMS) {
    return false;
  }

  const cleanBaseline = cleanDecodedText(baselineText);
  if (cleanBaseline.length === 0) {
    return true;
  }
  if (duration >= MIC_FALLBACK_LONG_SESSION_SECONDS) {
    return true;
  }
  return cleanBaseline.length / duration < MIC_FALLBACK_MIN_CHARS_PER_SECOND;
}

function chooseMicFinalText(samples: Float32Array, baselineText: string): string {
  const t0 = Date.now();
  const cleanBaseline = cleanDecodedText(baselineText);
  let result = cleanBaseline;

  if (samples.length > 0) {
    try {
      const secondPassText = decodeSamplesOfflineParaformerSegmentedWithPunctuation(samples, SAMPLE_RATE);
      if (secondPassText.trim().length > 0) {
        result = secondPassText.trim();
      }
    } catch (_error) {
      // Keep the real-time result usable if the second-pass model cannot be loaded.
    }
  }

  // Only run the fallback path if the second pass did not produce text.
  if (result === cleanBaseline && !isLongAudio(samples, SAMPLE_RATE) &&
    shouldConsiderMicFallback(samples, cleanBaseline)) {
    const fallbackText = decodeSamplesBestWithPunctuation(samples, SAMPLE_RATE);
    if (fallbackText.length > 0 &&
      (cleanBaseline.length === 0 || fallbackText.length >= cleanBaseline.length)) {
      result = fallbackText;
    }
  }

  // result already passed through addPunctuation (which applies the dictionary and
  // number-width normalization) via the *WithPunctuation decoders / cleanBaseline.
  const audioDurationSec = samples.length / SAMPLE_RATE;
  const finalRedecodeMs = Date.now() - t0;
  const finalRedecodeRtf = audioDurationSec > 0 ? (finalRedecodeMs / 1000) / audioDurationSec : 0;
  postTelemetry({ finalRedecodeMs, finalRedecodeRtf, audioDurationSec });
  return result;
}

function buildEnhancedText(baselineText: string, enhancedText: string, enhancedLabel: string, compareEnhanced: boolean): string {
  const cleanEnhanced = enhancedText.trim().length > 0 ? enhancedText.trim() : '(空)';
  if (!compareEnhanced) {
    return cleanEnhanced;
  }

  const cleanBaseline = baselineText.trim().length > 0 ? baselineText.trim() : '(空)';
  return `【基线】${cleanBaseline}\n${enhancedLabel}${cleanEnhanced}`;
}

function stopMic(request: WorkerRequest): void {
  const enableSpeakerDiarization = request.enableSpeakerDiarization === true;
  const enableSegmentRescore = request.enableSegmentRescore === true;
  const compareEnhanced = request.compareEnhanced === true;
  const sessionSamples = sessionLength > 0 ? concatChunks(sessionChunks) : new Float32Array(0);

  if (enableSpeakerDiarization || enableSegmentRescore) {
    const previousCollector = collectedFinalSegments;
    collectedFinalSegments = [];
    try {
      flushVadAndFinishSegment();
      const baselineText = collectedFinalSegments.join('');
      collectedFinalSegments = previousCollector;
      const finalText = chooseMicFinalText(sessionSamples, baselineText);
      if (finalText.trim().length > 0) {
        const baselineResponse: WorkerResponse = {
          msgType: 'sherpa-asr-final',
          text: finalText.trim()
        };
        postResponse(baselineResponse);
      }

      let enhancedText = '';
      let enhancedTurns: SpeakerTurn[] = [];
      let enhancedCandidates: SpeakerCandidate[] = [];
      let enhancedLabel = '【增强:最终识别】';
      if (sessionSamples.length > 0 && enableSpeakerDiarization) {
        try {
          const enhancedFallbackText = finalText;
          const diarizationResult: DiarizationDecodeResult = decodeSamplesWithDiarizationResult(
            sessionSamples,
            SAMPLE_RATE,
            enableSegmentRescore,
            enhancedFallbackText,
            request.meetingId
          );
          enhancedText = diarizationResult.text;
          enhancedTurns = diarizationResult.turns;
          enhancedCandidates = diarizationResult.candidates;
          enhancedLabel = enableSegmentRescore ? '【增强:说话人分离+最终识别】' : '【增强:说话人分离】';
        } catch (error) {
          enhancedText = `说话人分离失败: ${error}\n【最终识别】${decodeSamplesBestWithPunctuation(sessionSamples, SAMPLE_RATE)}`;
        }
      } else if (sessionSamples.length > 0) {
        enhancedText = finalText;
      } else {
        enhancedText = finalText;
      }
      const response: WorkerResponse = {
        msgType: 'sherpa-asr-final',
        text: compareEnhanced ?
          `${enhancedLabel}${enhancedText.trim().length > 0 ? enhancedText.trim() : '(空)'}` :
          buildEnhancedText('', enhancedText, enhancedLabel, false),
        meetingId: request.meetingId,
        speakerTurns: enhancedTurns,
        speakerCandidates: enhancedCandidates,
      };
      postResponse(response);
    } finally {
      collectedFinalSegments = previousCollector;
      resetSessionChunks();
    }
  } else {
    const previousCollector = collectedFinalSegments;
    collectedFinalSegments = [];
    try {
      flushVadAndFinishSegment();
      const baselineText = collectedFinalSegments.join('');
      collectedFinalSegments = previousCollector;
      const finalText = chooseMicFinalText(sessionSamples, baselineText);
      if (finalText.trim().length > 0) {
        const finalResponse: WorkerResponse = {
          msgType: 'sherpa-asr-final',
          text: finalText.trim()
        };
        postResponse(finalResponse);
      }
    } finally {
      collectedFinalSegments = previousCollector;
      resetSessionChunks();
    }
  }

  const response: WorkerResponse = { msgType: 'sherpa-asr-stopped' };
  postResponse(response);
}

function isWavBytes(data: Uint8Array): boolean {
  return data.length >= 12 &&
    data[0] === 0x52 && data[1] === 0x49 && data[2] === 0x46 && data[3] === 0x46 &&
    data[8] === 0x57 && data[9] === 0x41 && data[10] === 0x56 && data[11] === 0x45;
}

function isMp3Bytes(data: Uint8Array): boolean {
  if (data.length < 3) {
    return false;
  }
  const hasId3 = data[0] === 0x49 && data[1] === 0x44 && data[2] === 0x33;
  const hasFrameSync = data.length >= 2 && data[0] === 0xFF && (data[1] & 0xE0) === 0xE0;
  return hasId3 || hasFrameSync;
}

function isM4aBytes(data: Uint8Array): boolean {
  return data.length >= 12 &&
    data[4] === 0x66 && data[5] === 0x74 && data[6] === 0x79 && data[7] === 0x70;
}

function detectAudioFormat(data: Uint8Array, requestedFormat?: string): string {
  const normalizedFormat = requestedFormat ? requestedFormat.toLowerCase() : '';
  if (isWavBytes(data)) {
    return 'wav';
  }
  if (isMp3Bytes(data)) {
    return 'mp3';
  }
  if (isM4aBytes(data)) {
    return 'm4a';
  }
  if (normalizedFormat === 'wav' || normalizedFormat === 'mp3' || normalizedFormat === 'm4a') {
    throw new Error('音频文件有问题：文件内容不是有效的 MP3/WAV/M4A');
  }
  throw new Error('音频文件有问题：仅支持 MP3、WAV 或 M4A');
}

function normalizeSamplesForAsr(samples: Float32Array, sampleRate: number): AudioSamples {
  if (samples.length === 0 || sampleRate <= 0) {
    throw new Error('音频文件有问题：没有可识别的音频数据');
  }
  if (sampleRate === SAMPLE_RATE) {
    return { samples, sampleRate };
  }
  // Anti-aliased conversion: plain linear interpolation folded everything above 8 kHz
  // back into the speech band when decoding 44.1/48 kHz imports.
  return {
    samples: resampleAntiAliased(samples, sampleRate, SAMPLE_RATE),
    sampleRate: SAMPLE_RATE
  };
}

function decodeWavToSamples(data: Uint8Array): AudioSamples {
  const wave = readWaveFromBinary(data) as Samples;
  return normalizeSamplesForAsr(wave.samples, wave.sampleRate);
}

function decodeMp3ToSamples(data: Uint8Array): AudioSamples {
  const decoded = decodeMp3ToMono16k(data) as AudioDecodeResult;
  return normalizeSamplesForAsr(new Float32Array(decoded.samples), decoded.sampleRate);
}

function decodeM4aToSamples(data: Uint8Array): AudioSamples {
  const decoded = decodeM4aToMono16k(data) as AudioDecodeResult;
  return normalizeSamplesForAsr(new Float32Array(decoded.samples), decoded.sampleRate);
}

function decodeM4aToSamplesRange(data: Uint8Array, startSeconds: number, endSeconds: number): AudioSamples {
  const decoded = decodeM4aToMono16k(data, startSeconds, endSeconds) as AudioDecodeResult;
  return normalizeSamplesForAsr(new Float32Array(decoded.samples), decoded.sampleRate);
}

function decodeAudioBytesToSamples(data: Uint8Array, requestedFormat?: string): AudioSamples {
  const format = detectAudioFormat(data, requestedFormat);
  if (format === 'wav') {
    return decodeWavToSamples(data);
  }
  if (format === 'mp3') {
    return decodeMp3ToSamples(data);
  }
  if (format === 'm4a') {
    return decodeM4aToSamples(data);
  }
  throw new Error('音频文件有问题：仅支持 MP3、WAV 或 M4A');
}

function decodeWaveBytes(data: Uint8Array, fileAsrMode?: string): string {
  const wave = decodeWavToSamples(data);
  return decodeSamplesForFileModeWithPunctuation(wave.samples, wave.sampleRate, fileAsrMode);
}

function decodeSamplesThroughStreamingVad(samples: Float32Array, sampleRate: number): string {
  const wave = normalizeSamplesForAsr(samples, sampleRate);
  const previousCollector = collectedFinalSegments;
  collectedFinalSegments = [];
  try {
    startMic();
    const chunkSamples = 1600;
    for (let offset = 0; offset < wave.samples.length; offset += chunkSamples) {
      const end = Math.min(offset + chunkSamples, wave.samples.length);
      processMicPcm(float32ToPcm16Buffer(wave.samples, offset, end));
    }
    flushVadAndFinishSegment();
    return collectedFinalSegments.join('');
  } finally {
    collectedFinalSegments = previousCollector;
    latestText = '';
    inSpeech = false;
    silenceSamples = 0;
    resetPreRoll();
    resetMicChunks();
    resetSessionChunks();
    if (recognizer) {
      micStream = recognizer.createStream();
    }
  }
}

function decodeWaveBytesThroughStreamingVad(data: Uint8Array): string {
  const wave = decodeWavToSamples(data);
  return decodeSamplesThroughStreamingVad(wave.samples, wave.sampleRate);
}

interface DiarizationDecodeResult {
  text: string;
  turns: SpeakerTurn[];
  candidates: SpeakerCandidate[];
}

function makeSpeakerTurn(meetingId: string, segment: NormalizedSpeakerSegment,
  text: string, candidate?: SpeakerCandidate): SpeakerTurn {
  return {
    turnId: util.generateRandomUUID(true),
    meetingId: meetingId,
    clusterId: `S${segment.speaker + 1}`,
    startMs: Math.round(segment.start * 1000),
    endMs: Math.round(segment.end * 1000),
    text: text,
    identity: candidate && candidate.personId ? 'automatic' : 'unknown',
    personId: candidate ? candidate.personId : undefined,
    displayName: candidate ? candidate.displayName : undefined,
    matchScore: candidate ? candidate.matchScore : undefined,
  };
}

function decodeSamplesWithDiarizationResult(
  samples: Float32Array,
  sampleRate: number,
  enableSegmentRescore: boolean,
  fallbackText: string = '',
  meetingId: string = ''
): DiarizationDecodeResult {
  const resolvedMeetingId: string = meetingId.length > 0 ? meetingId : util.generateRandomUUID(true);
  if (sampleRate !== SAMPLE_RATE) {
    return {
      text: `sample rate ${sampleRate} is not ${SAMPLE_RATE}`,
      turns: [],
      candidates: [],
    };
  }
  const diarization = ensureSpeakerDiarization();
  const rawSegments = diarization.process(samples);
  const speakerOverlaps: SpeakerOverlapRange[] = findSpeakerOverlapRanges(rawSegments);
  const segments = normalizeSpeakerSegments(rawSegments);
  if (segments.length === 0) {
    let cleanFallback = fallbackText.trim();
    if (cleanFallback.length === 0 && samples.length > 0) {
      cleanFallback = decodeSamplesBestDirectWithPunctuation(samples, sampleRate);
    }
    const text: string = cleanFallback.length > 0 ?
      `[S1 ${formatTime(0)}-${formatTime(samples.length / sampleRate)}] ${cleanFallback}` :
      '(未检测到说话人段)';
    return { text: text, turns: [], candidates: [] };
  }

  const lines: string[] = [];
  const limit = Math.min(segments.length, DIARIZATION_MAX_DISPLAY_SEGMENTS);
  let segmentFallbackText = fallbackText.trim();
  if (segmentFallbackText.length === 0 && samples.length > 0) {
    segmentFallbackText = decodeSamplesBestDirectWithPunctuation(samples, sampleRate);
  }
  const candidateBuild: SpeakerCandidateBuild = buildSpeakerCandidates(
    samples, segments, resolvedMeetingId, speakerOverlaps);
  const turns: SpeakerTurn[] = [];
  if (countSpeakerIds(segments) === 1) {
    const firstSegment = segments[0];
    const lastSegment = segments[segments.length - 1];
    const speakerLabel = formatSpeakerSegmentLabel(firstSegment, candidateBuild.bySpeaker);
    const timeRange = `${formatTime(firstSegment.start)}-${formatTime(lastSegment.end)}`;
    const turnText: string = segmentFallbackText.trim().length > 0 ?
      segmentFallbackText.trim() : '(无文本)';
    const mergedSegment: NormalizedSpeakerSegment = {
      start: firstSegment.start,
      end: lastSegment.end,
      speaker: firstSegment.speaker,
    };
    turns.push(makeSpeakerTurn(resolvedMeetingId, mergedSegment, turnText,
      candidateBuild.bySpeaker.get(firstSegment.speaker)));
    return {
      text: `[${speakerLabel} ${timeRange}] ${turnText}`,
      turns: turns,
      candidates: candidateBuild.candidates,
    };
  }

  const fallbackParts = splitTextBySegmentDuration(segmentFallbackText, segments);
  const useSegmentRescore = shouldRescoreDiarizationSegments(samples, sampleRate, limit, enableSegmentRescore);
  for (let i = 0; i < limit; ++i) {
    const segment = segments[i];
    const startSample = Math.max(0, Math.floor((segment.start - DIARIZATION_SEGMENT_PADDING_SECONDS) * SAMPLE_RATE));
    const endSample = Math.min(
      samples.length,
      Math.ceil((segment.end + DIARIZATION_SEGMENT_PADDING_SECONDS) * SAMPLE_RATE)
    );
    let text = '';
    const canRescore = useSegmentRescore &&
      i < DIARIZATION_MAX_RESCORE_SEGMENTS &&
      segment.start < DIARIZATION_MAX_RESCORE_SECONDS;
    if (canRescore && endSample > startSample) {
      text = decodeSamplesBestDirectWithPunctuation(samples.subarray(startSample, endSample), sampleRate);
    }
    if (text.trim().length === 0 && i < fallbackParts.length) {
      text = fallbackParts[i];
    }

    const speakerLabel = formatSpeakerSegmentLabel(segment, candidateBuild.bySpeaker);
    const timeRange = `${formatTime(segment.start)}-${formatTime(segment.end)}`;
    const turnText: string = text.trim().length > 0 ? text.trim() : '(无文本)';
    lines.push(`[${speakerLabel} ${timeRange}] ${turnText}`);
    turns.push(makeSpeakerTurn(resolvedMeetingId, segment, turnText,
      candidateBuild.bySpeaker.get(segment.speaker)));
  }

  if (segments.length > limit) {
    lines.push(`...(余下 ${segments.length - limit} 个说话人片段未显示)`);
  }

  return { text: lines.join('\n'), turns: turns, candidates: candidateBuild.candidates };
}

function decodeSamplesWithDiarization(
  samples: Float32Array,
  sampleRate: number,
  enableSegmentRescore: boolean,
  fallbackText: string = ''
): string {
  return decodeSamplesWithDiarizationResult(
    samples, sampleRate, enableSegmentRescore, fallbackText).text;
}

function decodeWaveBytesWithDiarization(
  data: Uint8Array,
  enableSegmentRescore: boolean,
  fallbackText: string = ''
): string {
  const wave = decodeWavToSamples(data);
  return decodeSamplesWithDiarization(wave.samples, wave.sampleRate, enableSegmentRescore, fallbackText);
}

function decodeSamplesForFileRequest(audio: AudioSamples, request: WorkerRequest): string {
  const finalText = decodeSamplesForFileModeWithPunctuation(audio.samples, audio.sampleRate, request.fileAsrMode);
  return buildFileRequestOutput(audio, request, finalText);
}

function buildFileRequestStructuredOutput(audio: AudioSamples, request: WorkerRequest,
  finalText: string): DiarizationDecodeResult {
  const enableSpeakerDiarization = request.enableSpeakerDiarization === true;
  const enableSegmentRescore = request.enableSegmentRescore === true;
  const compareEnhanced = request.compareEnhanced === true;

  if (!enableSpeakerDiarization && !enableSegmentRescore) {
    return { text: finalText, turns: [], candidates: [] };
  }

  let enhancedResult: DiarizationDecodeResult = { text: '', turns: [], candidates: [] };
  const enhancedFallbackText = finalText;
  if (enableSpeakerDiarization) {
    try {
      enhancedResult = decodeSamplesWithDiarizationResult(
        audio.samples,
        audio.sampleRate,
        enableSegmentRescore,
        enhancedFallbackText,
        request.meetingId
      );
    } catch (error) {
      enhancedResult = {
        text: `说话人分离失败: ${error}\n【最终识别】${enhancedFallbackText}`,
        turns: [],
        candidates: [],
      };
    }
  } else {
    enhancedResult.text = enhancedFallbackText;
  }

  if (!compareEnhanced) {
    return enhancedResult;
  }

  const enhancedLabel = enableSpeakerDiarization ?
    (enableSegmentRescore ? '【增强:说话人分离+最终识别】' : '【增强:说话人分离】') :
    '【增强:最终识别】';
  const baselineText = compareEnhanced ? decodeSamplesThroughStreamingVad(audio.samples, audio.sampleRate) : '';
  enhancedResult.text = `【基线】${baselineText.trim().length > 0 ? baselineText.trim() : '(空)'}\n` +
    `${enhancedLabel}${enhancedResult.text.trim().length > 0 ? enhancedResult.text.trim() : '(空)'}`;
  return enhancedResult;
}

function buildFileRequestOutput(audio: AudioSamples, request: WorkerRequest,
  finalText: string): string {
  return buildFileRequestStructuredOutput(audio, request, finalText).text;
}

function decodeAudioBytesForFileRequest(request: WorkerRequest): string {
  const data = request.data as Uint8Array;
  if (!data || data.length === 0) {
    throw new Error('音频文件有问题：文件为空');
  }
  const audio = decodeAudioBytesToSamples(data, request.audioFormat);
  return decodeSamplesForFileRequest(audio, request);
}

function decodeAudioBytesForFileStructuredRequest(request: WorkerRequest): DiarizationDecodeResult {
  const data = request.data as Uint8Array;
  if (!data || data.length === 0) {
    throw new Error('音频文件有问题：文件为空');
  }
  const audio: AudioSamples = decodeAudioBytesToSamples(data, request.audioFormat);
  const finalText: string = decodeSamplesForFileModeWithPunctuation(
    audio.samples, audio.sampleRate, request.fileAsrMode);
  return buildFileRequestStructuredOutput(audio, request, finalText);
}

function decodeAudioBytesForFileRequestWithProgress(request: WorkerRequest): void {
  const data = request.data as Uint8Array;
  if (!data || data.length === 0) {
    throw new Error('音频文件有问题：文件为空');
  }

  const decodingResponse: WorkerResponse = { msgType: 'sherpa-asr-audio-file-decoding' };
  postResponse(decodingResponse);

  const format = detectAudioFormat(data, request.audioFormat);
  if (format === 'm4a') {
    decodeM4aBytesForFileRequestWithProgress(data, request);
    return;
  }

  const audio = decodeAudioBytesToSamples(data, request.audioFormat);
  const plan = buildDecodePlan(audio.samples, audio.sampleRate);
  const segments = plan.segments;
  const preparedResponse: WorkerResponse = {
    msgType: 'sherpa-asr-audio-file-prepared',
    segmentTotal: segments.length,
    audioDuration: audio.samples.length / audio.sampleRate
  };
  postResponse(preparedResponse);

  // Punctuate only what each segment newly contributed; the full-document pass runs once
  // at the end. Punctuating the whole accumulated text per segment was O(N^2).
  const builder = new TranscriptBuilder(
    (base: string, next: string) => joinSegmentText(base, next, plan.mergeTextOverlap),
    addPunctuation
  );
  for (let i = 0; i < segments.length; ++i) {
    const segment = segments[i];
    const start = Math.max(0, segment.start - plan.padSamples);
    const end = Math.min(audio.samples.length, segment.end + plan.padSamples);
    if (end <= start) {
      continue;
    }

    const segmentText = decodeSamplesForFileModeRaw(
      audio.samples.subarray(start, end),
      audio.sampleRate,
      request.fileAsrMode
    );
    const progressText = builder.addSegment(segmentText);
    const response: WorkerResponse = {
      msgType: 'sherpa-asr-audio-file-segment',
      text: progressText,
      segmentIndex: i + 1,
      segmentTotal: segments.length,
      segmentStart: segment.start / audio.sampleRate,
      segmentEnd: segment.end / audio.sampleRate
    };
    postResponse(response);
  }

  const finalResult: DiarizationDecodeResult = buildFileRequestStructuredOutput(
    audio, request, addPunctuation(builder.getRawText()));
  const response: WorkerResponse = {
    msgType: 'sherpa-asr-audio-file-done',
    text: finalResult.text,
    meetingId: request.meetingId,
    speakerTurns: finalResult.turns,
    speakerCandidates: finalResult.candidates,
  };
  postResponse(response);
}

function buildFixedTimeSegments(durationSeconds: number): SampleRange[] {
  const totalSamples = Math.max(1, Math.ceil(durationSeconds * SAMPLE_RATE));
  const segmentSamples = Math.max(1, Math.floor(LONG_AUDIO_SEGMENT_SECONDS * SAMPLE_RATE));
  const segments: SampleRange[] = [];
  for (let start = 0; start < totalSamples; start += segmentSamples) {
    const end = Math.min(totalSamples, start + segmentSamples);
    if (end > start) {
      segments.push({ start, end });
    }
  }
  return segments.length > 0 ? segments : [{ start: 0, end: totalSamples }];
}

function decodeM4aBytesForFileRequestWithProgress(data: Uint8Array, request: WorkerRequest): void {
  const info = probeM4aInfo(data) as AudioProbeResult;
  if (!info || info.durationSeconds <= 0) {
    const audio: AudioSamples = decodeM4aToSamples(data);
    const rawText: string = decodeSamplesForFileModeWithPunctuation(
      audio.samples, audio.sampleRate, request.fileAsrMode);
    const finalResult: DiarizationDecodeResult = buildFileRequestStructuredOutput(
      audio, request, rawText);
    const response: WorkerResponse = {
      msgType: 'sherpa-asr-audio-file-done',
      text: finalResult.text,
      meetingId: request.meetingId,
      speakerTurns: finalResult.turns,
      speakerCandidates: finalResult.candidates,
    };
    postResponse(response);
    return;
  }

  const segments = buildFixedTimeSegments(info.durationSeconds);
  const preparedResponse: WorkerResponse = {
    msgType: 'sherpa-asr-audio-file-prepared',
    segmentTotal: segments.length,
    audioDuration: info.durationSeconds
  };
  postResponse(preparedResponse);

  // m4a keeps fixed-time blocks: they are demuxed range by range (napi seeks per block), so
  // there is no full waveform to run the VAD over without decoding the file twice.
  // The incremental punctuation fix still applies.
  const builder = new TranscriptBuilder(appendTextWithOverlap, addPunctuation);
  for (let i = 0; i < segments.length; ++i) {
    const segment = segments[i];
    const segmentStart = segment.start / SAMPLE_RATE;
    const segmentEnd = Math.min(info.durationSeconds, segment.end / SAMPLE_RATE);
    const decodeStart = Math.max(0, segmentStart - LONG_AUDIO_OVERLAP_SECONDS);
    const decodeEnd = Math.min(info.durationSeconds, segmentEnd + LONG_AUDIO_OVERLAP_SECONDS);
    const audio = decodeM4aToSamplesRange(data, decodeStart, decodeEnd);
    const segmentText = decodeSamplesForFileModeRaw(audio.samples, audio.sampleRate, request.fileAsrMode);
    const progressText = builder.addSegment(segmentText);
    const response: WorkerResponse = {
      msgType: 'sherpa-asr-audio-file-segment',
      text: progressText,
      segmentIndex: i + 1,
      segmentTotal: segments.length,
      segmentStart,
      segmentEnd
    };
    postResponse(response);
  }

  const finalText = addPunctuation(builder.getRawText());
  const needsFullAudioForFinal =
    request.enableSpeakerDiarization === true ||
    request.compareEnhanced === true;
  if (!needsFullAudioForFinal) {
    const response: WorkerResponse = {
      msgType: 'sherpa-asr-audio-file-done',
      text: finalText,
      meetingId: request.meetingId,
      speakerTurns: [],
      speakerCandidates: [],
    };
    postResponse(response);
    return;
  }

  const finalizingResponse: WorkerResponse = {
    msgType: 'sherpa-asr-audio-file-prepared',
    segmentTotal: segments.length,
    audioDuration: info.durationSeconds
  };
  postResponse(finalizingResponse);
  const fullAudio = decodeM4aToSamples(data);
  const structured: DiarizationDecodeResult = buildFileRequestStructuredOutput(
    fullAudio, request, finalText);
  const response: WorkerResponse = {
    msgType: 'sherpa-asr-audio-file-done',
    text: structured.text,
    meetingId: request.meetingId,
    speakerTurns: structured.turns,
    speakerCandidates: structured.candidates,
  };
  postResponse(response);
}

function decodeWaveBytesForSelfTest(request: WorkerRequest): string {
  const data = request.data as Uint8Array;
  if (!data || data.length === 0) {
    throw new Error('音频文件有问题：文件为空');
  }
  const audio = decodeWavToSamples(data);
  return decodeSamplesForFileRequest(audio, request);
}

/**
 * Load and run the offline models once on a short silent buffer.
 *
 * They used to be created lazily on first use, so the first press-to-talk release or the
 * first imported file paid for model load plus first-inference allocation on top of the
 * decode itself. Best effort: a failure here must never make the worker unusable, the
 * lazy `ensure*` paths still work.
 */
function warmUpOfflineModels(): void {
  const silence = new Float32Array(WARM_UP_SAMPLES);
  try {
    decodeSamplesWithOfflineRecognizer(ensureOfflineParaformerRecognizer(), silence, SAMPLE_RATE);
  } catch (_error) {
    // Offline model unavailable; the decode paths already fall back.
  }
  try {
    ensureFileVad();
  } catch (_error) {
    // Segmentation already falls back to fixed-length blocks without a VAD.
  }
  if (punctuation) {
    try {
      punctuation.addPunct('预热');
    } catch (_error) {
      // Punctuation stays optional.
    }
  }
}

workerPort.onmessage = (e: MessageEvents) => {
  const request = e.data as WorkerRequest;
  const msgType = request.msgType;

  try {
    if (msgType === 'sherpa-asr-init' && !recognizer) {
      const context = request.context as Context;
      appContext = context;
      loadReplacementDictionary(context);
      loadHotwordsBuf(context);
      recognizer = initRecognizer(context);
      vad = initVad(context);
      try {
        punctuation = initPunctuation(context);
      } catch (_error) {
        punctuation = undefined;
      }
      micStream = recognizer.createStream();
      const response: WorkerResponse = { msgType: 'sherpa-asr-ready', punctuationReady: !!punctuation };
      postResponse(response);
      // Ready is posted first so the UI unblocks; the warm-up then moves the offline model
      // load and its first (slowest) inference off the user's first recognition.
      warmUpOfflineModels();
      return;
    }

    if (msgType === 'sherpa-asr-start') {
      startMic();
      const response: WorkerResponse = { msgType: 'sherpa-asr-started' };
      postResponse(response);
      return;
    }

    if (msgType === 'sherpa-asr-stop') {
      stopMic(request);
      return;
    }

    if (msgType === 'sherpa-asr-cancel') {
      cancelMic();
      return;
    }

    if (msgType === 'sherpa-asr-pcm') {
      processMicPcm(request.pcm as ArrayBuffer);
      return;
    }

    if (msgType === 'sherpa-voiceprint-restore') {
      restoreVoiceprints(request.voiceprints);
      const response: WorkerResponse = {
        msgType: 'sherpa-voiceprint-restored',
        speakerNames: getRegisteredSpeakerNames(),
        voiceprintReady: true,
      };
      postResponse(response);
      return;
    }

    if (msgType === 'sherpa-voiceprint-start') {
      try {
        startVoiceprintRegistration();
        const response: WorkerResponse = { msgType: 'sherpa-voiceprint-started' };
        postResponse(response);
      } catch (error) {
        postError(`${error}`, 'sherpa-voiceprint-error');
      }
      return;
    }

    if (msgType === 'sherpa-voiceprint-pcm') {
      processVoiceprintPcm(request.pcm as ArrayBuffer);
      return;
    }

    if (msgType === 'sherpa-voiceprint-stop') {
      stopVoiceprintRegistration(request);
      return;
    }

    if (msgType === 'sherpa-voiceprint-cancel') {
      cancelVoiceprintRegistration();
      return;
    }

    if (msgType === 'sherpa-asr-decode-wav') {
      const result: DiarizationDecodeResult = decodeAudioBytesForFileStructuredRequest(request);
      const response: WorkerResponse = {
        msgType: 'sherpa-asr-self-test-done',
        text: result.text,
        meetingId: request.meetingId,
        speakerTurns: result.turns,
        speakerCandidates: result.candidates,
      };
      postResponse(response);
      return;
    }

    /*
     * Deferred transcription of a stored recording segment.
     *
     * Kept separate from 'sherpa-asr-decode-wav' even though the decoding is identical:
     * that path answers with 'sherpa-asr-self-test-done', and the queue needs a reply it
     * can match to the request it sent. requestId does that, so several segments can be
     * in flight across a restart without their answers being confused for each other or
     * for a self-test the user happened to trigger.
     */
    if (msgType === 'sherpa-asr-transcribe-segment') {
      const requestId = request.requestId ?? '';
      try {
        const result: DiarizationDecodeResult = decodeAudioBytesForFileStructuredRequest(request);
        postResponse({
          msgType: 'sherpa-asr-segment-done',
          requestId: requestId,
          text: result.text,
          meetingId: request.meetingId,
          speakerTurns: result.turns,
          speakerCandidates: result.candidates,
        });
      } catch (error) {
        postResponse({
          msgType: 'sherpa-asr-segment-failed',
          requestId,
          message: `${error}`,
        });
      }
      return;
    }

    if (msgType === 'sherpa-asr-decode-audio-file') {
      try {
        decodeAudioBytesForFileRequestWithProgress(request);
      } catch (error) {
        postError(`${error}`, 'sherpa-asr-audio-file-error');
      }
      return;
    }
  } catch (error) {
    postError(`${error}`);
  }
};

workerPort.onmessageerror = (_e: MessageEvents) => {
  postError('worker message deserialize failed');
};

workerPort.onerror = (e: ErrorEvent) => {
  postError(`${e.message}`);
};
