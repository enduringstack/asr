import { AudioUnderstandingTarget } from './AudioUnderstandingTypes';

export interface AudioUnderstandingTestCase {
  id: string;
  dataset: string;
  sourceId: string;
  target: AudioUnderstandingTarget;
  expectedLabel: string;
  expectedDisplayName: string;
  audioPath: string;
  truthBasis: string;
  license: string;
}

function emotionCase(sourceId: string, expectedLabel: string, expectedDisplayName: string,
  agreement: string): AudioUnderstandingTestCase {
  return {
    id: `crema-${sourceId.toLowerCase()}`,
    dataset: 'CREMA-D',
    sourceId,
    target: 'emotion',
    expectedLabel,
    expectedDisplayName,
    audioPath: `test/understanding/crema_d/${sourceId}.wav`,
    truthBasis: `人声多数投票 · 一致率 ${agreement}`,
    license: 'ODbL 1.0 / DbCL 1.0',
  };
}

function eventCase(row: number, speakerId: string, expectedLabel: string,
  expectedDisplayName: string): AudioUnderstandingTestCase {
  const rowText: string = row < 10 ? `000${row}` : (row < 100 ? `00${row}` : `${row}`);
  return {
    id: `vocalsound-${rowText}-${expectedLabel}`,
    dataset: 'VocalSound',
    sourceId: `test/${row} · ${speakerId}`,
    target: 'event',
    expectedLabel,
    expectedDisplayName,
    audioPath: `test/understanding/vocalsound/test_${rowText}_${speakerId}_${expectedLabel}.wav`,
    truthBasis: '官方 test split 文件级事件标签',
    license: 'CC BY-SA 4.0',
  };
}

export function buildAudioUnderstandingTestCases(): AudioUnderstandingTestCase[] {
  return [
    emotionCase('1061_MTI_ANG_XX', 'angry', '愤怒', '97.17%'),
    emotionCase('1029_IEO_ANG_HI', 'angry', '愤怒', '96.80%'),
    emotionCase('1079_IEO_ANG_HI', 'angry', '愤怒', '94.89%'),
    emotionCase('1022_ITH_DIS_XX', 'disgusted', '厌恶', '84.80%'),
    emotionCase('1064_WSI_DIS_XX', 'disgusted', '厌恶', '83.00%'),
    emotionCase('1038_TAI_DIS_XX', 'disgusted', '厌恶', '80.56%'),
    emotionCase('1036_TSI_FEA_XX', 'fearful', '恐惧', '88.62%'),
    emotionCase('1026_IEO_FEA_HI', 'fearful', '恐惧', '87.57%'),
    emotionCase('1008_TSI_FEA_XX', 'fearful', '恐惧', '86.44%'),
    emotionCase('1059_IEO_HAP_HI', 'happy', '高兴', '83.33%'),
    emotionCase('1049_ITS_HAP_XX', 'happy', '高兴', '81.56%'),
    emotionCase('1065_IEO_HAP_MD', 'happy', '高兴', '80.50%'),
    emotionCase('1074_IOM_NEU_XX', 'neutral', '中性', '93.67%'),
    emotionCase('1045_TSI_NEU_XX', 'neutral', '中性', '92.00%'),
    emotionCase('1083_IWL_NEU_XX', 'neutral', '中性', '92.00%'),
    emotionCase('1028_MTI_SAD_XX', 'sad', '悲伤', '91.22%'),
    emotionCase('1058_IEO_SAD_HI', 'sad', '悲伤', '91.22%'),
    emotionCase('1006_WSI_SAD_XX', 'sad', '悲伤', '84.33%'),
    eventCase(0, 'f2750', 'cough', '咳嗽'),
    eventCase(2, 'f1142', 'cough', '咳嗽'),
    eventCase(13, 'm0733', 'cough', '咳嗽'),
    eventCase(45, 'f0866', 'cough', '咳嗽'),
    eventCase(7, 'f1349', 'laughter', '笑声'),
    eventCase(15, 'm2768', 'laughter', '笑声'),
    eventCase(41, 'f0208', 'laughter', '笑声'),
    eventCase(51, 'm3209', 'laughter', '笑声'),
    eventCase(6, 'm0561', 'sneeze', '喷嚏'),
    eventCase(23, 'f3323', 'sneeze', '喷嚏'),
    eventCase(35, 'f1108', 'sneeze', '喷嚏'),
    eventCase(38, 'f3249', 'sneeze', '喷嚏'),
  ];
}
