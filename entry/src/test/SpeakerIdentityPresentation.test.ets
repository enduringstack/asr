import { describe, expect, it } from '@ohos/hypium';
import {
  buildIdentityDecisionText,
  buildIdentityOutcomeLabel,
  identityResultMatchesFilter,
} from '../main/ets/common/SpeakerIdentityPresentation';

export default function speakerIdentityPresentationTest() {
  describe('speakerIdentityPresentationTest', () => {
    it('labels waiting and running rows without inventing a prediction', 0, () => {
      expect(buildIdentityOutcomeLabel({ state: '等待', accepted: false, correct: false,
        knownTruth: true })).assertEqual('等待运行');
      expect(buildIdentityOutcomeLabel({ state: '运行中', accepted: false, correct: false,
        knownTruth: true })).assertEqual('正在识别');
    });

    it('distinguishes every known-speaker outcome', 0, () => {
      expect(buildIdentityOutcomeLabel({ state: '通过', accepted: true, correct: true,
        knownTruth: true })).assertEqual('正确识别');
      expect(buildIdentityOutcomeLabel({ state: '未通过', accepted: false, correct: false,
        knownTruth: true })).assertEqual('错误拒识');
      expect(buildIdentityOutcomeLabel({ state: '未通过', accepted: true, correct: false,
        knownTruth: true })).assertEqual('误识别');
    });

    it('distinguishes correct rejection from false acceptance', 0, () => {
      expect(buildIdentityOutcomeLabel({ state: '通过', accepted: false, correct: true,
        knownTruth: false })).assertEqual('正确拒识');
      expect(buildIdentityOutcomeLabel({ state: '未通过', accepted: true, correct: false,
        knownTruth: false })).assertEqual('未知误接纳');
    });

    it('explains the threshold decision using real scores', 0, () => {
      expect(buildIdentityDecisionText(true, 0.824, 0.60))
        .assertEqual('0.824 ≥ 0.60，高于阈值，接纳为已知人物');
      expect(buildIdentityDecisionText(false, 0.431, 0.60))
        .assertEqual('0.431 < 0.60，低于阈值，拒绝所有已注册人物');
      expect(buildIdentityDecisionText(false, -1, 0.60)).assertEqual('等待模型输出');
    });

    it('filters known, unknown and error evidence independently', 0, () => {
      expect(identityResultMatchesFilter('known', true, '等待', false)).assertTrue();
      expect(identityResultMatchesFilter('unknown', false, '通过', true)).assertTrue();
      expect(identityResultMatchesFilter('errors', true, '未通过', false)).assertTrue();
      expect(identityResultMatchesFilter('errors', false, '通过', true)).assertFalse();
    });
  });
}
