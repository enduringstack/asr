import { describe, expect, it } from '@ohos/hypium';
import {
  AsrCaseResult,
  AsrEditScore,
  AsrEvaluationSummary,
  asrResultMatchesFilter,
  buildAsrEvaluationSummary,
  normalizeAsrText,
  scoreAsrText,
} from '../main/ets/common/AsrEvaluationTypes';

export default function asrEvaluationTypesTest() {
  describe('asrEvaluationTypesTest', () => {
    it('normalizes punctuation spaces full-width ASCII and annotation tags', 0, () => {
      expect(normalizeAsrText(' 你好，ＷＯＲＬＤ！<sil> ')).assertEqual('你好world');
      expect(normalizeAsrText('嗯<->这次<_>开会。')).assertEqual('嗯这次开会');
    });

    it('scores exact substitutions deletions and insertions', 0, () => {
      const exact: AsrEditScore = scoreAsrText('你好，世界。', '你好世界');
      expect(exact.exact).assertTrue();
      expect(exact.cer).assertEqual(0);

      const substitution: AsrEditScore = scoreAsrText('今天开会', '今天开水');
      expect(substitution.substitutions).assertEqual(1);
      expect(substitution.deletions).assertEqual(0);
      expect(substitution.insertions).assertEqual(0);

      const deletion: AsrEditScore = scoreAsrText('语音识别', '语音识');
      expect(deletion.deletions).assertEqual(1);
      const insertion: AsrEditScore = scoreAsrText('语音识别', '语音的识别');
      expect(insertion.insertions).assertEqual(1);
    });

    it('aggregates micro and macro CER and counts a failed empty output', 0, () => {
      const rows: AsrCaseResult[] = [
        { state: 'completed', score: scoreAsrText('甲乙', '甲乙'), audioDurationSec: 2, elapsedMs: 100 },
        { state: 'completed', score: scoreAsrText('甲乙丙丁', '甲乙丙'), audioDurationSec: 4, elapsedMs: 200 },
        { state: 'failed', score: scoreAsrText('甲乙', ''), audioDurationSec: 2, elapsedMs: 0 },
        { state: 'waiting', audioDurationSec: 0, elapsedMs: 0 },
      ];
      const summary: AsrEvaluationSummary = buildAsrEvaluationSummary(rows);
      expect(summary.evaluated).assertEqual(3);
      expect(summary.completed).assertEqual(2);
      expect(summary.failed).assertEqual(1);
      expect(summary.exact).assertEqual(1);
      expect(summary.referenceChars).assertEqual(8);
      expect(summary.errors).assertEqual(3);
      expect(summary.microCer).assertEqual(0.375);
      expect(summary.macroCer).assertEqual((0 + 0.25 + 1) / 3);
      expect(summary.rtf).assertEqual(0.0375);
    });

    it('filters only completed mismatches and failures as errors', 0, () => {
      expect(asrResultMatchesFilter('errors', 'completed', scoreAsrText('甲', '甲'))).assertFalse();
      expect(asrResultMatchesFilter('errors', 'completed', scoreAsrText('甲', '乙'))).assertTrue();
      expect(asrResultMatchesFilter('errors', 'failed')).assertTrue();
      expect(asrResultMatchesFilter('all', 'waiting')).assertTrue();
    });
  });
}
