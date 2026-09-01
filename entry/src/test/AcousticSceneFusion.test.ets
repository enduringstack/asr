import { describe, expect, it } from '@ohos/hypium';
import { fuseAcousticScene } from '../main/ets/common/AcousticSceneFusion';

export default function acousticSceneFusionTest() {
  describe('acousticSceneFusionTest', () => {
    it('uses reliable high speed to separate high speed rail from metro', 0, () => {
      const result = fuseAcousticScene(
        [0.48, 0.34, 0.04, 0.04, 0.02, 0.08],
        { speedKmh: 168, speedAccuracyKmh: 5, locationAgeMs: 1200 },
        '',
        '',
      );
      expect(result.label).assertEqual('high_speed_train');
      expect(result.usedContext).assertTrue();
    });

    it('ignores stale or inaccurate speed evidence', 0, () => {
      const result = fuseAcousticScene(
        [0.62, 0.21, 0.04, 0.04, 0.02, 0.07],
        { speedKmh: 180, speedAccuracyKmh: 60, locationAgeMs: 1200 },
        '',
        '',
      );
      expect(result.label).assertEqual('metro');
      expect(result.usedContext).assertFalse();
    });

    it('uses explicit transcript and event evidence inside confusing groups', 0, () => {
      const metro = fuseAcousticScene(
        [0.31, 0.43, 0.04, 0.04, 0.02, 0.16], {},
        '请换乘地铁二号线，下一站人民广场', '',
      );
      const concert = fuseAcousticScene(
        [0.03, 0.03, 0.06, 0.08, 0.35, 0.45], {}, '',
        'Applause,Music',
      );
      expect(metro.label).assertEqual('metro');
      expect(concert.label).assertEqual('concert');
    });

    it('does not treat missing location as proof of metro', 0, () => {
      const result = fuseAcousticScene(
        [0.18, 0.11, 0.39, 0.16, 0.03, 0.13], {}, '', '',
      );
      expect(result.label).assertEqual('shopping_mall');
      expect(result.usedContext).assertFalse();
    });
  });
}
