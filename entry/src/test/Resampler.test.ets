import { describe, it, expect } from '@ohos/hypium';
import { resampleAntiAliased } from '../main/ets/common/Resampler';

function makeTone(frequency: number, sampleRate: number, lengthSamples: number): Float32Array {
  const samples = new Float32Array(lengthSamples);
  for (let i = 0; i < lengthSamples; ++i) {
    samples[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate);
  }
  return samples;
}

function makeConstant(value: number, lengthSamples: number): Float32Array {
  const samples = new Float32Array(lengthSamples);
  for (let i = 0; i < lengthSamples; ++i) {
    samples[i] = value;
  }
  return samples;
}

/** RMS over the interior only, so kernel edge effects do not pollute the measurement. */
function interiorRms(samples: Float32Array, skip: number): number {
  const start = Math.min(skip, samples.length);
  const end = Math.max(start, samples.length - skip);
  if (end <= start) {
    return 0;
  }
  let sum = 0;
  for (let i = start; i < end; ++i) {
    sum += samples[i] * samples[i];
  }
  return Math.sqrt(sum / (end - start));
}

export default function resamplerTest() {
  describe('resamplerTest', () => {
    it('returns the input untouched when the rate already matches', 0, () => {
      const input = makeTone(1000, 16000, 1600);
      const output = resampleAntiAliased(input, 16000, 16000);
      expect(output.length).assertEqual(input.length);
      expect(output[123]).assertEqual(input[123]);
    });

    it('handles empty and invalid input without throwing', 0, () => {
      expect(resampleAntiAliased(new Float32Array(0), 48000, 16000).length).assertEqual(0);
      const input = makeConstant(0.5, 100);
      expect(resampleAntiAliased(input, 0, 16000).length).assertEqual(100);
      expect(resampleAntiAliased(input, 48000, 0).length).assertEqual(100);
    });

    it('produces the expected output length when decimating', 0, () => {
      const input = makeTone(1000, 48000, 48000);
      const output = resampleAntiAliased(input, 48000, 16000);
      expect(output.length).assertEqual(16000);
    });

    it('produces the expected output length when interpolating', 0, () => {
      const input = makeTone(1000, 8000, 8000);
      const output = resampleAntiAliased(input, 8000, 16000);
      expect(output.length).assertEqual(16000);
    });

    it('has unity DC gain, including at the signal edges', 0, () => {
      const input = makeConstant(0.5, 4800);
      const output = resampleAntiAliased(input, 48000, 16000);
      // Weights are normalised per output sample, so even the truncated kernels at the
      // very first and last output sample must not attenuate.
      expect(Math.abs(output[0] - 0.5) < 1e-3).assertTrue();
      expect(Math.abs(output[800] - 0.5) < 1e-3).assertTrue();
      expect(Math.abs(output[output.length - 1] - 0.5) < 1e-3).assertTrue();
    });

    it('passes a 1 kHz tone through 48 kHz -> 16 kHz essentially unchanged', 0, () => {
      const input = makeTone(1000, 48000, 48000);
      const output = resampleAntiAliased(input, 48000, 16000);
      const gain = interiorRms(output, 200) / interiorRms(input, 200);
      // 1 kHz is far inside the 8 kHz target passband.
      expect(gain > 0.95 && gain < 1.05).assertTrue();
    });

    it('rejects a 12 kHz tone instead of folding it back to 4 kHz', 0, () => {
      // This is the regression the anti-aliased kernel exists for: with plain linear
      // interpolation a 12 kHz component in a 48 kHz file aliases to 4 kHz at nearly full
      // amplitude, straight into the band the acoustic model relies on.
      const input = makeTone(12000, 48000, 48000);
      const output = resampleAntiAliased(input, 48000, 16000);
      const gain = interiorRms(output, 200) / interiorRms(input, 200);
      expect(gain < 0.25).assertTrue();
    });

    it('keeps in-band content while rejecting out-of-band content in the same signal', 0, () => {
      const sampleRate = 48000;
      const length = 48000;
      const mixed = new Float32Array(length);
      const inBandOnly = new Float32Array(length);
      for (let i = 0; i < length; ++i) {
        const inBand = Math.sin(2 * Math.PI * 1000 * i / sampleRate);
        const outOfBand = Math.sin(2 * Math.PI * 15000 * i / sampleRate);
        mixed[i] = 0.5 * inBand + 0.5 * outOfBand;
        inBandOnly[i] = 0.5 * inBand;
      }

      const fromMixed = resampleAntiAliased(mixed, sampleRate, 16000);
      const fromClean = resampleAntiAliased(inBandOnly, sampleRate, 16000);
      let maxDelta = 0;
      for (let i = 200; i < fromMixed.length - 200; ++i) {
        const delta = Math.abs(fromMixed[i] - fromClean[i]);
        if (delta > maxDelta) {
          maxDelta = delta;
        }
      }
      // The 15 kHz component must not survive as an audible alias in the 16 kHz signal.
      expect(maxDelta < 0.05).assertTrue();
    });

    it('preserves a constant when upsampling', 0, () => {
      const input = makeConstant(-0.25, 800);
      const output = resampleAntiAliased(input, 8000, 16000);
      expect(output.length).assertEqual(1600);
      expect(Math.abs(output[900] + 0.25) < 1e-3).assertTrue();
    });
  });
}
