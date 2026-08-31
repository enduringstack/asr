export type WorkbenchMode = 'recording' | 'dataset';

export type SpeechModelId = 'asr' | 'punctuation' | 'voiceprint' | 'diarization';

export type DatasetAction = 'run-audio' | 'run-punctuation' |
  'open-voiceprint' | 'open-diarization';

export interface SpeechModelDescriptor {
  id: SpeechModelId;
  name: string;
  subtitle: string;
  accentColor: string;
  softColor: string;
  iconText: string;
}

export interface WorkbenchDatasetSuite {
  id: string;
  modelId: SpeechModelId;
  title: string;
  description: string;
  sampleSummary: string;
  action: DatasetAction;
}

export interface WorkbenchTelemetry {
  chunkDecodeMs?: number;
  partialIntervalMs?: number;
  finalSegmentMs?: number;
  finalRedecodeMs?: number;
  finalRedecodeRtf?: number;
  audioDurationSec?: number;
}

export function buildSpeechModelDescriptors(): SpeechModelDescriptor[] {
  return [
    {
      id: 'asr',
      name: 'ASR',
      subtitle: '离线识别',
      accentColor: '#246BFD',
      softColor: '#EAF1FF',
      iconText: '声',
    },
    {
      id: 'punctuation',
      name: '标点',
      subtitle: '智能断句',
      accentColor: '#0FA9BC',
      softColor: '#E7F9FB',
      iconText: '，',
    },
    {
      id: 'voiceprint',
      name: '声纹',
      subtitle: '识别人物',
      accentColor: '#7758E8',
      softColor: '#F0ECFF',
      iconText: '人',
    },
    {
      id: 'diarization',
      name: '说话人分离',
      subtitle: '多人分段',
      accentColor: '#159A61',
      softColor: '#E8F8F0',
      iconText: '多',
    },
  ];
}

export function buildDatasetSuites(): WorkbenchDatasetSuite[] {
  return [
    {
      id: 'meeting-asr',
      modelId: 'asr',
      title: '会议音频识别',
      description: '使用内置会议语音查看离线识别文本',
      sampleSummary: 'AliMeeting · 14 条',
      action: 'run-audio',
    },
    {
      id: 'meeting-punctuation',
      modelId: 'punctuation',
      title: '会议文本断句',
      description: '对比 ASR 原始文本与标点处理结果',
      sampleSummary: '与 ASR 共用原始音频',
      action: 'run-punctuation',
    },
    {
      id: 'aishell3-identity',
      modelId: 'voiceprint',
      title: 'AISHELL-3 声纹身份',
      description: '评测已知识别和未知说话人拒识',
      sampleSummary: '20 位说话人 · 40 组',
      action: 'open-voiceprint',
    },
    {
      id: 'meeting-diarization',
      modelId: 'diarization',
      title: '真实会议说话人分离',
      description: '查看会议分段稳定性、人物聚类和音频',
      sampleSummary: 'AISHELL-4 + AliMeeting · 17 条',
      action: 'open-diarization',
    },
  ];
}

export function formatTelemetrySummary(telemetry: WorkbenchTelemetry): string {
  const parts: string[] = [];
  if (telemetry.audioDurationSec !== undefined) {
    parts.push(`音频 ${telemetry.audioDurationSec.toFixed(1)} 秒`);
  }
  if (telemetry.finalRedecodeMs !== undefined) {
    parts.push(`处理 ${telemetry.finalRedecodeMs} ms`);
  } else if (telemetry.finalSegmentMs !== undefined) {
    parts.push(`处理 ${telemetry.finalSegmentMs} ms`);
  }
  if (telemetry.finalRedecodeRtf !== undefined) {
    parts.push(`RTF ${telemetry.finalRedecodeRtf.toFixed(3)}`);
  }
  return parts.length > 0 ? parts.join(' · ') : '等待首次运行';
}
