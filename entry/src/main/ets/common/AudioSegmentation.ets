/**
 * Segmentation planning for the offline (file / second-pass) recognition path.
 *
 * The previous plan cut the audio every 10 s and only nudged the cut by up to 0.6 s
 * towards the quietest 0.16 s window. In continuous speech, or with steady background
 * noise, the quietest window is frequently *inside* a word. Paraformer is a
 * non-autoregressive CIF model, so a cut inside a word damages both the tail of the
 * preceding block and the head of the following one, and the 0.25 s audio overlap then
 * has to be undone again by a text-level de-duplication heuristic.
 *
 * The plan is now driven by the VAD that the project already ships (silero): blocks
 * start and end in real silence, pure-silence stretches are skipped entirely, and no
 * audio overlap (and therefore no text de-duplication) is needed. The fixed-length
 * planner is kept as the fallback for when the VAD is unavailable or detects nothing.
 *
 * This file must stay free of @kit imports so it can run in local (non-device) tests.
 */

export interface SampleRange {
  start: number;
  end: number;
}

/**
 * A decode plan. `padSamples` is added around each segment before decoding.
 * `mergeTextOverlap` tells the caller whether decoded segment texts can overlap and
 * therefore need the text-level de-duplication heuristic (true only for the fallback
 * fixed-length planner; VAD blocks are disjoint in speech).
 */
export interface DecodePlan {
  segments: SampleRange[];
  padSamples: number;
  mergeTextOverlap: boolean;
}

export interface SpeechRegionOptions {
  /** Speech runs shorter than this are treated as noise blips and dropped. */
  minSpeechSamples: number;
  /** Silence shorter than this does not split a region (keeps words of one clause together). */
  minSilenceSamples: number;
  /** Symmetric padding kept around every detected region so onsets are not clipped. */
  padSamples: number;
}

export interface DecodeBlockOptions {
  /** Hard cap on one decode call. */
  maxBlockSamples: number;
  /**
   * Two speech regions are decoded together only if the silence between them is at most
   * this long. Grouping neighbouring clauses gives the acoustic model and the punctuation
   * model more context; grouping across a long pause would just put that pause back into
   * the decoded audio, which is exactly the work VAD planning is meant to avoid.
   */
  maxGapSamples: number;
}

export interface FixedSegmentOptions {
  segmentSamples: number;
  searchRadiusSamples: number;
  boundaryWindowSamples: number;
  minSegmentSamples: number;
}

export function computeRms(samples: Float32Array): number {
  if (samples.length === 0) {
    return 0;
  }

  let sum = 0;
  for (let i = 0; i < samples.length; ++i) {
    sum += samples[i] * samples[i];
  }
  return Math.sqrt(sum / samples.length);
}

export function scoreAudioWindow(samples: Float32Array, center: number, windowSamples: number): number {
  const halfWindow = Math.max(1, Math.floor(windowSamples / 2));
  const start = Math.max(0, center - halfWindow);
  const end = Math.min(samples.length, center + halfWindow);
  if (end <= start) {
    return Number.MAX_VALUE;
  }

  let sum = 0;
  for (let i = start; i < end; ++i) {
    sum += samples[i] * samples[i];
  }
  return sum / (end - start);
}

export function findQuietBoundary(
  samples: Float32Array,
  target: number,
  searchRadiusSamples: number,
  windowSamples: number,
  minBoundary: number,
  maxBoundary: number
): number {
  const lower = Math.max(minBoundary, target - searchRadiusSamples);
  const upper = Math.min(maxBoundary, target + searchRadiusSamples);
  if (upper <= lower) {
    return Math.max(minBoundary, Math.min(maxBoundary, target));
  }

  const step = Math.max(1, Math.floor(windowSamples / 4));
  let best = lower;
  let bestScore = Number.MAX_VALUE;
  for (let center = lower; center <= upper; center += step) {
    const score = scoreAudioWindow(samples, center, windowSamples);
    if (score < bestScore) {
      bestScore = score;
      best = center;
    }
  }
  return best;
}

/**
 * Fallback planner: fixed-length blocks nudged towards the quietest nearby window.
 * The returned segments tile [0, samples.length) without gaps.
 */
export function buildFixedLengthSegments(
  samples: Float32Array,
  options: FixedSegmentOptions
): SampleRange[] {
  const segmentSamples = Math.max(1, options.segmentSamples);
  if (samples.length <= segmentSamples) {
    const whole: SampleRange = { start: 0, end: samples.length };
    return [whole];
  }

  const minSegmentSamples = Math.max(1, options.minSegmentSamples);
  const boundaries: number[] = [0];

  let previous = 0;
  let target = segmentSamples;
  while (target < samples.length - minSegmentSamples) {
    const minBoundary = Math.min(samples.length, previous + minSegmentSamples);
    const maxBoundary = Math.min(samples.length - minSegmentSamples, target + options.searchRadiusSamples);
    if (maxBoundary <= minBoundary) {
      break;
    }

    const boundary = findQuietBoundary(
      samples,
      target,
      options.searchRadiusSamples,
      options.boundaryWindowSamples,
      minBoundary,
      maxBoundary
    );
    if (boundary <= previous) {
      break;
    }
    boundaries.push(boundary);
    previous = boundary;
    target = previous + segmentSamples;
  }
  boundaries.push(samples.length);

  const segments: SampleRange[] = [];
  for (let i = 0; i < boundaries.length - 1; ++i) {
    const start = boundaries[i];
    const end = boundaries[i + 1];
    if (end > start) {
      const range: SampleRange = { start, end };
      segments.push(range);
    }
  }
  return segments;
}

function mergeAdjacent(ranges: SampleRange[], maxGap: number): SampleRange[] {
  const merged: SampleRange[] = [];
  for (let i = 0; i < ranges.length; ++i) {
    const range = ranges[i];
    if (range.end <= range.start) {
      continue;
    }
    const previous: SampleRange | undefined = merged.length > 0 ? merged[merged.length - 1] : undefined;
    if (previous && range.start - previous.end <= maxGap) {
      previous.end = Math.max(previous.end, range.end);
      continue;
    }
    const copy: SampleRange = { start: range.start, end: range.end };
    merged.push(copy);
  }
  return merged;
}

/**
 * Turn per-window VAD decisions into padded speech regions.
 *
 * `flags[i]` is the decision for the window starting at `i * windowSamples`.
 */
export function speechFlagsToRegions(
  flags: boolean[],
  windowSamples: number,
  totalSamples: number,
  options: SpeechRegionOptions
): SampleRange[] {
  if (flags.length === 0 || windowSamples <= 0 || totalSamples <= 0) {
    return [];
  }

  const raw: SampleRange[] = [];
  let runStart = -1;
  for (let i = 0; i < flags.length; ++i) {
    if (flags[i]) {
      if (runStart < 0) {
        runStart = i;
      }
      continue;
    }
    if (runStart >= 0) {
      const range: SampleRange = {
        start: runStart * windowSamples,
        end: Math.min(totalSamples, i * windowSamples)
      };
      raw.push(range);
      runStart = -1;
    }
  }
  if (runStart >= 0) {
    const tail: SampleRange = { start: runStart * windowSamples, end: totalSamples };
    raw.push(tail);
  }

  // Bridge short pauses first, so a clause split by a 200 ms breath stays one region and
  // is not then discarded as two sub-minimum blips.
  const bridged = mergeAdjacent(raw, options.minSilenceSamples);

  const kept: SampleRange[] = [];
  for (let i = 0; i < bridged.length; ++i) {
    if (bridged[i].end - bridged[i].start >= options.minSpeechSamples) {
      kept.push(bridged[i]);
    }
  }

  const padded: SampleRange[] = [];
  for (let i = 0; i < kept.length; ++i) {
    const range: SampleRange = {
      start: Math.max(0, kept[i].start - options.padSamples),
      end: Math.min(totalSamples, kept[i].end + options.padSamples)
    };
    padded.push(range);
  }

  // Padding can make neighbours touch or overlap; collapse them again.
  return mergeAdjacent(padded, 0);
}

function splitOverlongRange(range: SampleRange, maxBlockSamples: number): SampleRange[] {
  const length = range.end - range.start;
  if (length <= maxBlockSamples) {
    const single: SampleRange = { start: range.start, end: range.end };
    return [single];
  }

  // Even parts, so the last piece is never a sliver.
  const parts = Math.ceil(length / maxBlockSamples);
  const partLength = Math.ceil(length / parts);
  const pieces: SampleRange[] = [];
  for (let start = range.start; start < range.end; start += partLength) {
    const piece: SampleRange = { start, end: Math.min(range.end, start + partLength) };
    pieces.push(piece);
  }
  return pieces;
}

/**
 * Pack speech regions into decode blocks of at most `maxBlockSamples`.
 *
 * A region joins the current block when it is close enough (`maxGapSamples`) and the
 * block still fits, so neighbouring clauses are decoded together instead of one call per
 * clause. Anything further away starts a new block, which is what keeps long silence out
 * of the decoded audio.
 */
export function planDecodeBlocks(
  regions: SampleRange[],
  options: DecodeBlockOptions
): SampleRange[] {
  const blocks: SampleRange[] = [];
  for (let i = 0; i < regions.length; ++i) {
    if (regions[i].end <= regions[i].start) {
      continue;
    }
    const pieces = splitOverlongRange(regions[i], options.maxBlockSamples);
    for (let j = 0; j < pieces.length; ++j) {
      const piece = pieces[j];
      const current: SampleRange | undefined = blocks.length > 0 ? blocks[blocks.length - 1] : undefined;
      if (current &&
        piece.start - current.end <= options.maxGapSamples &&
        piece.end - current.start <= options.maxBlockSamples) {
        current.end = piece.end;
        continue;
      }
      const block: SampleRange = { start: piece.start, end: piece.end };
      blocks.push(block);
    }
  }

  return blocks;
}
