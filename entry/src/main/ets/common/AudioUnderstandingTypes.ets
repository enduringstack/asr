export type AudioUnderstandingTarget = 'emotion' | 'event';
export type AudioUnderstandingFilter = 'all' | 'emotion' | 'event' | 'errors';
export type AudioUnderstandingState = 'waiting' | 'running' | 'completed' | 'failed';

export interface AudioUnderstandingPrediction {
  text: string;
  lang: string;
  emotion: string;
  event: string;
}

export interface AudioUnderstandingCaseResult {
  target: AudioUnderstandingTarget;
  state: AudioUnderstandingState;
  correct: boolean;
}

export interface AudioUnderstandingSummary {
  correct: number;
  total: number;
  emotionCorrect: number;
  emotionTotal: number;
  eventCorrect: number;
  eventTotal: number;
  failed: number;
}

export function canonicalSenseVoiceLabel(value: string): string {
  let label: string = value.trim().toLowerCase();
  label = label.split('<|').join('');
  label = label.split('|>').join('');
  label = label.split('_').join(' ');
  label = label.trim();
  if (label === 'anger') {
    return 'angry';
  }
  if (label === 'disgust') {
    return 'disgusted';
  }
  if (label === 'fear') {
    return 'fearful';
  }
  if (label === 'sadness') {
    return 'sad';
  }
  return label;
}

export function compareAudioUnderstandingTarget(target: AudioUnderstandingTarget,
  expectedLabel: string, prediction: AudioUnderstandingPrediction): boolean {
  const predicted: string = target === 'emotion' ? prediction.emotion : prediction.event;
  return canonicalSenseVoiceLabel(predicted) === canonicalSenseVoiceLabel(expectedLabel);
}

export function buildAudioUnderstandingSummary(
  rows: AudioUnderstandingCaseResult[]): AudioUnderstandingSummary {
  const summary: AudioUnderstandingSummary = {
    correct: 0,
    total: rows.length,
    emotionCorrect: 0,
    emotionTotal: 0,
    eventCorrect: 0,
    eventTotal: 0,
    failed: 0,
  };
  for (let i: number = 0; i < rows.length; i++) {
    const row: AudioUnderstandingCaseResult = rows[i];
    if (row.target === 'emotion') {
      summary.emotionTotal += 1;
      if (row.state === 'completed' && row.correct) {
        summary.emotionCorrect += 1;
      }
    } else {
      summary.eventTotal += 1;
      if (row.state === 'completed' && row.correct) {
        summary.eventCorrect += 1;
      }
    }
    if (row.state === 'completed' && row.correct) {
      summary.correct += 1;
    }
    if (row.state === 'failed') {
      summary.failed += 1;
    }
  }
  return summary;
}

export function audioUnderstandingResultMatchesFilter(filter: AudioUnderstandingFilter,
  target: AudioUnderstandingTarget, state: AudioUnderstandingState, correct: boolean): boolean {
  if (filter === 'emotion') {
    return target === 'emotion';
  }
  if (filter === 'event') {
    return target === 'event';
  }
  if (filter === 'errors') {
    return state === 'failed' || (state === 'completed' && !correct);
  }
  return true;
}

export function audioUnderstandingLabelText(label: string): string {
  const canonical: string = canonicalSenseVoiceLabel(label);
  if (canonical === 'angry') {
    return '愤怒';
  }
  if (canonical === 'disgusted') {
    return '厌恶';
  }
  if (canonical === 'fearful') {
    return '恐惧';
  }
  if (canonical === 'happy') {
    return '高兴';
  }
  if (canonical === 'neutral') {
    return '中性';
  }
  if (canonical === 'sad') {
    return '悲伤';
  }
  if (canonical === 'cough') {
    return '咳嗽';
  }
  if (canonical === 'laughter') {
    return '笑声';
  }
  if (canonical === 'sneeze') {
    return '喷嚏';
  }
  if (canonical === 'speech') {
    return '语音';
  }
  if (canonical === 'bgm') {
    return '背景音乐';
  }
  if (canonical === 'applause') {
    return '掌声';
  }
  if (canonical === 'cry') {
    return '哭声';
  }
  if (canonical === 'breath') {
    return '呼吸声';
  }
  if (canonical === 'sing') {
    return '歌唱';
  }
  if (canonical === 'speech noise') {
    return '语音噪声';
  }
  if (canonical === 'emo unknown' || canonical === 'event unk') {
    return '未检出';
  }
  return canonical.length > 0 ? label : '未检出';
}

export function audioUnderstandingLanguageText(lang: string): string {
  const canonical: string = canonicalSenseVoiceLabel(lang);
  if (canonical === 'zh') {
    return '中文';
  }
  if (canonical === 'en') {
    return '英语';
  }
  if (canonical === 'yue') {
    return '粤语';
  }
  if (canonical === 'ja') {
    return '日语';
  }
  if (canonical === 'ko') {
    return '韩语';
  }
  if (canonical === 'nospeech') {
    return '无语音';
  }
  return canonical.length > 0 ? lang : '未检出';
}
