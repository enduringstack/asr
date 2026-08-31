import { describe, expect, it } from '@ohos/hypium';
import {
  AudioUnderstandingCaseResult,
  AudioUnderstandingSummary,
  audioUnderstandingResultMatchesFilter,
  audioUnderstandingLabelText,
  audioUnderstandingLanguageText,
  buildAudioUnderstandingSummary,
  canonicalSenseVoiceLabel,
  compareAudioUnderstandingTarget,
} from '../main/ets/common/AudioUnderstandingTypes';
import {
  AudioUnderstandingTestCase,
  buildAudioUnderstandingTestCases,
} from '../main/ets/common/AudioUnderstandingTestData';

export default function audioUnderstandingTypesTest() {
  describe('audioUnderstandingTypesTest', () => {
    it('normalizes native SenseVoice labels without inferring from transcript text', 0, () => {
      expect(canonicalSenseVoiceLabel('<|HAPPY|>')).assertEqual('happy');
      expect(canonicalSenseVoiceLabel('FEAR')).assertEqual('fearful');
      expect(canonicalSenseVoiceLabel('Disgust')).assertEqual('disgusted');
      expect(canonicalSenseVoiceLabel('<|Laughter|>')).assertEqual('laughter');
      expect(canonicalSenseVoiceLabel('')).assertEqual('');
      expect(audioUnderstandingLanguageText('<|nospeech|>')).assertEqual('无语音');
      expect(audioUnderstandingLabelText('<|EMO_UNKNOWN|>')).assertEqual('未检出');
    });

    it('compares only the declared target field', 0, () => {
      expect(compareAudioUnderstandingTarget('emotion', 'happy', {
        lang: 'en', emotion: 'HAPPY', event: 'Speech', text: 'hello',
      })).assertTrue();
      expect(compareAudioUnderstandingTarget('event', 'cough', {
        lang: '', emotion: 'NEUTRAL', event: 'Cough', text: '',
      })).assertTrue();
      expect(compareAudioUnderstandingTarget('event', 'laughter', {
        lang: '', emotion: 'HAPPY', event: 'Speech', text: '',
      })).assertFalse();
    });

    it('keeps failed cases in the denominator and reports them separately', 0, () => {
      const rows: AudioUnderstandingCaseResult[] = [
        { target: 'emotion', state: 'completed', correct: true },
        { target: 'emotion', state: 'completed', correct: false },
        { target: 'event', state: 'completed', correct: true },
        { target: 'event', state: 'failed', correct: false },
      ];
      const summary: AudioUnderstandingSummary = buildAudioUnderstandingSummary(rows);
      expect(summary.total).assertEqual(4);
      expect(summary.correct).assertEqual(2);
      expect(summary.emotionTotal).assertEqual(2);
      expect(summary.emotionCorrect).assertEqual(1);
      expect(summary.eventTotal).assertEqual(2);
      expect(summary.eventCorrect).assertEqual(1);
      expect(summary.failed).assertEqual(1);
    });

    it('filters target types and mismatches independently', 0, () => {
      expect(audioUnderstandingResultMatchesFilter('emotion', 'emotion', 'waiting', false)).assertTrue();
      expect(audioUnderstandingResultMatchesFilter('event', 'event', 'completed', true)).assertTrue();
      expect(audioUnderstandingResultMatchesFilter('errors', 'event', 'completed', false)).assertTrue();
      expect(audioUnderstandingResultMatchesFilter('errors', 'emotion', 'failed', false)).assertTrue();
      expect(audioUnderstandingResultMatchesFilter('errors', 'emotion', 'completed', true)).assertFalse();
    });

    it('ships the fixed 18 emotion and 12 event cases with unique audio paths', 0, () => {
      const cases: AudioUnderstandingTestCase[] = buildAudioUnderstandingTestCases();
      expect(cases.length).assertEqual(30);
      let emotionCount: number = 0;
      let eventCount: number = 0;
      const paths: string[] = [];
      for (let i: number = 0; i < cases.length; i++) {
        if (cases[i].target === 'emotion') {
          emotionCount += 1;
        } else {
          eventCount += 1;
        }
        expect(paths.indexOf(cases[i].audioPath)).assertEqual(-1);
        paths.push(cases[i].audioPath);
      }
      expect(emotionCount).assertEqual(18);
      expect(eventCount).assertEqual(12);
    });
  });
}
