/**
 * Anti-aliased sample-rate conversion.
 *
 * The previous implementation interpolated linearly between neighbouring input samples.
 * Linear interpolation has no stop-band: when decimating (48 kHz -> 16 kHz) every
 * component above the *target* Nyquist frequency folds back into the speech band. A
 * 12 kHz component in a 48 kHz file lands at 4 kHz with almost full amplitude, right in
 * the middle of the band the acoustic model depends on. That is invisible on a 16 kHz
 * benchmark corpus such as AISHELL (no resampling happens there) but degrades every
 * imported 44.1/48 kHz recording.
 *
 * This module uses a Blackman-windowed sinc kernel whose cut-off follows the lower of
 * the two rates, so the band above the target Nyquist is attenuated before decimation.
 * The kernel is tabulated once and read with linear interpolation, which keeps the cost
 * to a couple of multiply-adds per tap instead of a Math.sin() call per tap.
 *
 * This file must stay free of @kit imports so it can run in local (non-device) tests.
 */

const SINC_ZERO_CROSSINGS: number = 6;
const KERNEL_STEPS_PER_UNIT: number = 512;
const KERNEL_TABLE_SIZE: number = SINC_ZERO_CROSSINGS * KERNEL_STEPS_PER_UNIT + 2;

function windowedSinc(x: number): number {
  if (x >= SINC_ZERO_CROSSINGS) {
    return 0;
  }
  // Blackman window over [-SINC_ZERO_CROSSINGS, SINC_ZERO_CROSSINGS]; equals 1 at x == 0.
  const t = (x + SINC_ZERO_CROSSINGS) / (2 * SINC_ZERO_CROSSINGS);
  const window = 0.42 - 0.5 * Math.cos(2 * Math.PI * t) + 0.08 * Math.cos(4 * Math.PI * t);
  if (x === 0) {
    return window;
  }
  const piX = Math.PI * x;
  return (Math.sin(piX) / piX) * window;
}

function buildKernelTable(): Float32Array {
  const table = new Float32Array(KERNEL_TABLE_SIZE);
  for (let i = 0; i < KERNEL_TABLE_SIZE; ++i) {
    table[i] = windowedSinc(i / KERNEL_STEPS_PER_UNIT);
  }
  return table;
}

const KERNEL_TABLE: Float32Array = buildKernelTable();

// The kernel is even, so it is tabulated for |x| only.
function kernelAt(x: number): number {
  const ax = x < 0 ? -x : x;
  if (ax >= SINC_ZERO_CROSSINGS) {
    return 0;
  }
  const position = ax * KERNEL_STEPS_PER_UNIT;
  const index = Math.floor(position);
  const frac = position - index;
  const low = KERNEL_TABLE[index];
  const high = KERNEL_TABLE[index + 1];
  return low + (high - low) * frac;
}

/**
 * Resample `samples` from `sourceRate` to `targetRate`.
 *
 * Weights are normalised per output sample, so a constant input maps to the same
 * constant (unity DC gain) and partial kernels at the signal edges do not attenuate.
 */
export function resampleAntiAliased(
  samples: Float32Array,
  sourceRate: number,
  targetRate: number
): Float32Array {
  if (samples.length === 0 || sourceRate <= 0 || targetRate <= 0 || sourceRate === targetRate) {
    return samples;
  }

  const ratio = targetRate / sourceRate;
  const outLength = Math.max(1, Math.round(samples.length * ratio));
  const out = new Float32Array(outLength);
  // When decimating, stretch the kernel in the source domain so its cut-off follows the
  // target Nyquist frequency. When interpolating, the source band is already narrow enough.
  const filterScale = ratio < 1 ? ratio : 1;
  const halfWidth = Math.ceil(SINC_ZERO_CROSSINGS / filterScale);

  for (let n = 0; n < outLength; ++n) {
    const center = n / ratio;
    const base = Math.floor(center);
    const first = Math.max(0, base - halfWidth + 1);
    const last = Math.min(samples.length - 1, base + halfWidth);
    let acc = 0;
    let weightSum = 0;
    for (let i = first; i <= last; ++i) {
      const weight = kernelAt((i - center) * filterScale);
      if (weight === 0) {
        continue;
      }
      acc += samples[i] * weight;
      weightSum += weight;
    }
    out[n] = weightSum !== 0 ? acc / weightSum : 0;
  }

  return out;
}
