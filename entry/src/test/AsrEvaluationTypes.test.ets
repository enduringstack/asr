import { describe, expect, it } from '@ohos/hypium';
import {
  AsrCaseResult,
  AsrEditScore,
  AsrEvaluationSummary,
  AsrPermutationScore,
  AsrDiarizationScore,
  AsrOverlapDetectionScore,
  AsrTurnCaseResult,
  AsrTurnEvaluationSummary,
  analyzeAsrTurns,
  asrResultMatchesFilter,
  buildAsrEvaluationSummary,
  buildAsrTurnEvaluationSummary,
  groupSpeakerTranscripts,
  normalizeAsrText,
  scoreAsrText,
  scoreSpeakerPermutation,
  scoreSpeakerDiarization,
  scoreSpeakerOverlapDetection,
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

    it('marks only intersecting cross-speaker turns as overlap', 0, () => {
      const turns = analyzeAsrTurns([
        { speakerId: 'A', startSec: 0, endSec: 2, text: '甲' },
        { speakerId: 'B', startSec: 2, endSec: 3, text: '乙' },
        { speakerId: 'A', startSec: 3, endSec: 5, text: '丙' },
        { speakerId: 'B', startSec: 4, endSec: 6, text: '丁' },
      ]);
      expect(turns[0].overlapped).assertFalse();
      expect(turns[1].overlapped).assertFalse();
      expect(turns[2].overlapped).assertTrue();
      expect(turns[3].overlapped).assertTrue();
    });

    it('reports non-overlap and overlap CER independently', 0, () => {
      const rows: AsrTurnCaseResult[] = [
        {
          state: 'completed',
          score: scoreAsrText('甲乙', '甲乙'),
          audioDurationSec: 2,
          elapsedMs: 20,
          overlapped: false,
        },
        {
          state: 'completed',
          score: scoreAsrText('丙丁', ''),
          audioDurationSec: 2,
          elapsedMs: 20,
          overlapped: true,
        },
      ];
      const summary: AsrTurnEvaluationSummary = buildAsrTurnEvaluationSummary(rows);
      expect(summary.nonOverlap.microCer).assertEqual(0);
      expect(summary.overlap.microCer).assertEqual(1);
      expect(summary.all.microCer).assertEqual(0.5);
    });

    it('finds the best speaker permutation and accounts for missing streams', 0, () => {
      const swapped: AsrPermutationScore = scoreSpeakerPermutation(
        [{ speakerId: 'A', text: '你好' }, { speakerId: 'B', text: '世界' }],
        [{ speakerId: 'S1', text: '世界' }, { speakerId: 'S2', text: '你好' }]);
      expect(swapped.score.cer).assertEqual(0);
      expect(swapped.assignments[0].hypothesisSpeakerId).assertEqual('S2');
      expect(swapped.assignments[1].hypothesisSpeakerId).assertEqual('S1');

      const missing: AsrPermutationScore = scoreSpeakerPermutation(
        [{ speakerId: 'A', text: '你好' }, { speakerId: 'B', text: '世界' }],
        [{ speakerId: 'S1', text: '你好' }]);
      expect(missing.score.referenceChars).assertEqual(4);
      expect(missing.score.deletions).assertEqual(2);
      expect(missing.score.cer).assertEqual(0.5);
    });

    it('groups chronological text by anonymous speaker', 0, () => {
      const grouped = groupSpeakerTranscripts([
        { speakerId: 'S1', text: '甲' },
        { speakerId: 'S2', text: '乙' },
        { speakerId: 'S1', text: '丙' },
      ]);
      expect(grouped.length).assertEqual(2);
      expect(grouped[0].speakerId).assertEqual('S1');
      expect(grouped[0].text).assertEqual('甲丙');
      expect(grouped[1].text).assertEqual('乙');
    });

    it('scores diarization after optimal anonymous-speaker mapping', 0, () => {
      const perfect: AsrDiarizationScore = scoreSpeakerDiarization(
        [
          { speakerId: 'A', startSec: 0, endSec: 1 },
          { speakerId: 'B', startSec: 1, endSec: 2 },
        ],
        [
          { speakerId: 'S2', startSec: 0, endSec: 1 },
          { speakerId: 'S1', startSec: 1, endSec: 2 },
        ]);
      expect(perfect.der).assertEqual(0);

      const merged: AsrDiarizationScore = scoreSpeakerDiarization(
        [
          { speakerId: 'A', startSec: 0, endSec: 1 },
          { speakerId: 'B', startSec: 1, endSec: 2 },
        ],
        [{ speakerId: 'S1', startSec: 0, endSec: 2 }]);
      expect(merged.confusionSpeakerSec).assertEqual(1);
      expect(merged.der).assertEqual(0.5);
    });

    it('includes reference overlap instead of hiding it from DER', 0, () => {
      const score: AsrDiarizationScore = scoreSpeakerDiarization(
        [
          { speakerId: 'A', startSec: 0, endSec: 2 },
          { speakerId: 'B', startSec: 1, endSec: 2 },
        ],
        [{ speakerId: 'S1', startSec: 0, endSec: 2 }]);
      expect(score.referenceSpeakerSec).assertEqual(3);
      expect(score.missedSpeakerSec).assertEqual(1);
      expect(score.der).assertEqual(1 / 3);
      expect(score.referenceOverlapSec).assertEqual(1);
      expect(score.predictedOverlapSec).assertEqual(0);
      expect(score.overlapRecall).assertEqual(0);
    });

    it('scores raw model overlap detection independently from final speaker turns', 0, () => {
      const score: AsrOverlapDetectionScore = scoreSpeakerOverlapDetection(
        [
          { speakerId: 'A', startSec: 0, endSec: 2 },
          { speakerId: 'B', startSec: 1, endSec: 2 },
        ],
        [{ startSec: 1.5, endSec: 2.5 }]);
      expect(score.referenceOverlapSec).assertEqual(1);
      expect(score.predictedOverlapSec).assertEqual(1);
      expect(score.truePositiveSec).assertEqual(0.5);
      expect(score.recall).assertEqual(0.5);
    });
  });
}
