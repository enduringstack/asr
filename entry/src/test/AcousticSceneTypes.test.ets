import { describe, expect, it } from '@ohos/hypium';
import {
  acousticSceneDisplayName,
  acousticSceneResultMatchesFilter,
  buildAcousticScenePrediction,
  buildAcousticSceneSummary,
} from '../main/ets/common/AcousticSceneTypes';

export default function acousticSceneTypesTest() {
  describe('acousticSceneTypesTest', () => {
    it('keeps low-confidence scene output conservative', 0, () => {
      const prediction = buildAcousticScenePrediction(0, 0.31,
        [0.31, 0.10, 0.12, 0.11, 0.08, 0.28], 1, 10, 35);
      expect(prediction.label).assertEqual('other');
      expect(prediction.rawLabel).assertEqual('metro');
      expect(prediction.accepted).assertFalse();
    });

    it('uses the honest product label for train audio', 0, () => {
      expect(acousticSceneDisplayName('high_speed_train')).assertEqual('高铁 / 城际列车');
      const prediction = buildAcousticScenePrediction(1, 0.82,
        [0.03, 0.82, 0.04, 0.03, 0.03, 0.05], 2, 15, 42);
      expect(prediction.label).assertEqual('high_speed_train');
      expect(prediction.accepted).assertTrue();
    });

    it('summarizes only completed classifications', 0, () => {
      const summary = buildAcousticSceneSummary([
        { state: 'completed', correct: true },
        { state: 'completed', correct: false },
        { state: 'failed', correct: false },
        { state: 'waiting', correct: false },
      ]);
      expect(summary.total).assertEqual(2);
      expect(summary.correct).assertEqual(1);
      expect(summary.failed).assertEqual(1);
      expect(summary.accuracy).assertEqual(0.5);
      expect(acousticSceneResultMatchesFilter('incorrect', 'failed', false)).assertTrue();
    });
  });
}
