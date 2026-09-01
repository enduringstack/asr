export type AcousticSceneLabel = 'metro' | 'high_speed_train' | 'shopping_mall' |
  'cafe_restaurant' | 'concert' | 'other';

export type AcousticSceneState = 'waiting' | 'running' | 'completed' | 'failed';
export type AcousticSceneFilter = 'all' | 'correct' | 'incorrect';

export const ACOUSTIC_SCENE_LABELS: AcousticSceneLabel[] = [
  'metro',
  'high_speed_train',
  'shopping_mall',
  'cafe_restaurant',
  'concert',
  'other',
];

export const ACOUSTIC_SCENE_CONFIDENCE_THRESHOLD: number = 0.40;

export interface AcousticScenePrediction {
  rawLabel: AcousticSceneLabel;
  label: AcousticSceneLabel;
  displayName: string;
  confidence: number;
  accepted: boolean;
  probabilities: number[];
  windowCount: number;
  durationSeconds: number;
  elapsedMs: number;
}

export interface AcousticSceneSummaryRow {
  state: AcousticSceneState;
  correct: boolean;
}

export interface AcousticSceneSummary {
  total: number;
  correct: number;
  failed: number;
  accuracy: number;
}

export function acousticSceneDisplayName(label: string): string {
  if (label === 'metro') {
    return '地铁';
  }
  if (label === 'high_speed_train') {
    return '高铁 / 城际列车';
  }
  if (label === 'shopping_mall') {
    return '商场';
  }
  if (label === 'cafe_restaurant') {
    return '咖啡厅 / 餐厅';
  }
  if (label === 'concert') {
    return '音乐现场';
  }
  return '其他';
}

export function acousticSceneLabelAt(index: number): AcousticSceneLabel {
  if (index < 0 || index >= ACOUSTIC_SCENE_LABELS.length) {
    return 'other';
  }
  return ACOUSTIC_SCENE_LABELS[index];
}

export function buildAcousticScenePrediction(classIndex: number, confidence: number,
  probabilities: number[], windowCount: number, durationSeconds: number,
  elapsedMs: number): AcousticScenePrediction {
  const safeConfidence: number = Math.max(0, Math.min(1, confidence));
  const label: AcousticSceneLabel = acousticSceneLabelAt(classIndex);
  const accepted: boolean = label === 'other' || safeConfidence >= ACOUSTIC_SCENE_CONFIDENCE_THRESHOLD;
  return {
    rawLabel: label,
    label: accepted ? label : 'other',
    displayName: accepted ? acousticSceneDisplayName(label) : '其他（低置信度）',
    confidence: safeConfidence,
    accepted,
    probabilities,
    windowCount,
    durationSeconds,
    elapsedMs,
  };
}

export function buildAcousticSceneSummary(rows: AcousticSceneSummaryRow[]): AcousticSceneSummary {
  let total: number = 0;
  let correct: number = 0;
  let failed: number = 0;
  for (let i: number = 0; i < rows.length; i++) {
    if (rows[i].state === 'completed') {
      total += 1;
      if (rows[i].correct) {
        correct += 1;
      }
    } else if (rows[i].state === 'failed') {
      failed += 1;
    }
  }
  return {
    total,
    correct,
    failed,
    accuracy: total > 0 ? correct / total : 0,
  };
}

export function acousticSceneResultMatchesFilter(filter: AcousticSceneFilter,
  state: AcousticSceneState, correct: boolean): boolean {
  if (filter === 'all') {
    return true;
  }
  if (filter === 'correct') {
    return state === 'completed' && correct;
  }
  return state === 'failed' || (state === 'completed' && !correct);
}
