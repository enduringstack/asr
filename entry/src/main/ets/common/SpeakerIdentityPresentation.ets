export type IdentityResultFilter = 'all' | 'known' | 'unknown' | 'errors';

export interface IdentityOutcomeInput {
  state: string;
  accepted: boolean;
  correct: boolean;
  knownTruth: boolean;
}

export function buildIdentityOutcomeLabel(input: IdentityOutcomeInput): string {
  if (input.state === '等待') {
    return '等待运行';
  }
  if (input.state === '运行中') {
    return '正在识别';
  }
  if (input.state === '失败') {
    return '处理失败';
  }
  if (input.correct) {
    return input.knownTruth ? '正确识别' : '正确拒识';
  }
  if (input.knownTruth) {
    return input.accepted ? '误识别' : '错误拒识';
  }
  return input.accepted ? '未知误接纳' : '结果异常';
}

export function buildIdentityDecisionText(accepted: boolean, score: number,
  threshold: number): string {
  if (score < 0) {
    return '等待模型输出';
  }
  const scoreText: string = score.toFixed(3);
  const thresholdText: string = threshold.toFixed(2);
  return accepted ?
    `${scoreText} ≥ ${thresholdText}，高于阈值，接纳为已知人物` :
    `${scoreText} < ${thresholdText}，低于阈值，拒绝所有已注册人物`;
}

export function identityResultMatchesFilter(filter: IdentityResultFilter,
  knownTruth: boolean, state: string, correct: boolean): boolean {
  if (filter === 'known') {
    return knownTruth;
  }
  if (filter === 'unknown') {
    return !knownTruth;
  }
  if (filter === 'errors') {
    return state === '失败' || (state === '未通过' && !correct);
  }
  return true;
}
