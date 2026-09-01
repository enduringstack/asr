import {
  ACOUSTIC_SCENE_LABELS,
  AcousticSceneLabel,
} from './AcousticSceneTypes';

export interface SceneMotionEvidence {
  speedKmh?: number;
  speedAccuracyKmh?: number;
  locationAgeMs?: number;
}

export interface AcousticSceneFusionResult {
  classIndex: number;
  label: AcousticSceneLabel;
  confidence: number;
  probabilities: number[];
  usedContext: boolean;
  evidence: string[];
}

const MIN_PROBABILITY: number = 0.0001;
const MAX_LOCATION_AGE_MS: number = 15_000;
const MAX_SPEED_ACCURACY_KMH: number = 20;
const HIGH_SPEED_RAIL_KMH: number = 120;

function containsAny(text: string, words: string[]): boolean {
  const normalized: string = text.toLowerCase();
  for (let i: number = 0; i < words.length; i++) {
    if (normalized.includes(words[i].toLowerCase())) {
      return true;
    }
  }
  return false;
}

function normalizedAudioProbabilities(values: number[]): number[] {
  const result: number[] = [];
  let total: number = 0;
  for (let i: number = 0; i < ACOUSTIC_SCENE_LABELS.length; i++) {
    const value: number = i < values.length && Number.isFinite(values[i]) ?
      Math.max(0, values[i]) : 0;
    result.push(value);
    total += value;
  }
  if (total <= 0) {
    return [0, 0, 0, 0, 0, 1];
  }
  for (let i: number = 0; i < result.length; i++) {
    result[i] /= total;
  }
  return result;
}

function softmax(scores: number[]): number[] {
  let maximum: number = scores[0];
  for (let i: number = 1; i < scores.length; i++) {
    maximum = Math.max(maximum, scores[i]);
  }
  const result: number[] = [];
  let denominator: number = 0;
  for (let i: number = 0; i < scores.length; i++) {
    const value: number = Math.exp(scores[i] - maximum);
    result.push(value);
    denominator += value;
  }
  for (let i: number = 0; i < result.length; i++) {
    result[i] /= denominator;
  }
  return result;
}

function addEvidence(scores: number[], index: number, weight: number,
  message: string, evidence: string[]): void {
  scores[index] += weight;
  evidence.push(message);
}

export function fuseAcousticScene(audioProbabilities: number[], motion: SceneMotionEvidence,
  transcript: string, eventText: string): AcousticSceneFusionResult {
  const audio: number[] = normalizedAudioProbabilities(audioProbabilities);
  const scores: number[] = [];
  for (let i: number = 0; i < audio.length; i++) {
    scores.push(Math.log(Math.max(MIN_PROBABILITY, audio[i])));
  }
  const evidence: string[] = [];

  const speedReliable: boolean = motion.speedKmh !== undefined &&
    motion.speedAccuracyKmh !== undefined && motion.locationAgeMs !== undefined &&
    motion.speedKmh >= 0 && motion.speedAccuracyKmh <= MAX_SPEED_ACCURACY_KMH &&
    motion.locationAgeMs <= MAX_LOCATION_AGE_MS;
  if (speedReliable && (motion.speedKmh as number) >= HIGH_SPEED_RAIL_KMH) {
    addEvidence(scores, 1, 5.0, `可靠速度 ${(motion.speedKmh as number).toFixed(0)} km/h`, evidence);
    scores[0] -= 2.0;
  }

  if (containsAny(transcript, ['高铁', '动车', '城际列车', 'bullet train', 'shinkansen',
    'tgv', 'eurostar', 'ice train'])) {
    addEvidence(scores, 1, 3.5, 'ASR 检出高速铁路关键词', evidence);
  }
  if (containsAny(transcript, ['地铁', '号线', '换乘', 'subway', 'metro line', 'underground'])) {
    addEvidence(scores, 0, 3.5, 'ASR 检出地铁 / 换乘关键词', evidence);
  }
  if (containsAny(transcript, ['购物中心', '商场', '百货', '楼层', 'shopping mall',
    'department store'])) {
    addEvidence(scores, 2, 2.6, 'ASR 检出商场关键词', evidence);
  }
  if (containsAny(transcript, ['咖啡', '餐厅', '菜单', '点单', '买单', 'coffee',
    'restaurant', 'menu'])) {
    addEvidence(scores, 3, 2.6, 'ASR 检出餐饮关键词', evidence);
  }
  if (containsAny(transcript, ['演唱会', '音乐会', '音乐节', '现场演出', 'concert',
    'music festival', 'live show'])) {
    addEvidence(scores, 4, 3.2, 'ASR 检出演出关键词', evidence);
  }

  if (containsAny(eventText, ['applause', 'cheering', '欢呼', '掌声'])) {
    addEvidence(scores, 4, 3.0, '声音事件检出掌声 / 欢呼', evidence);
  }
  if (containsAny(eventText, ['dishes', 'cutlery', 'clink', '餐具', '碰杯'])) {
    addEvidence(scores, 3, 2.2, '声音事件检出餐具声', evidence);
  }

  const probabilities: number[] = softmax(scores);
  let classIndex: number = 0;
  for (let i: number = 1; i < probabilities.length; i++) {
    if (probabilities[i] > probabilities[classIndex]) {
      classIndex = i;
    }
  }
  return {
    classIndex,
    label: ACOUSTIC_SCENE_LABELS[classIndex],
    confidence: probabilities[classIndex],
    probabilities,
    usedContext: evidence.length > 0,
    evidence,
  };
}
