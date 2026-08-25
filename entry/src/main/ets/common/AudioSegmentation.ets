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
 * The plan is now driven by the VAD that the project already ships (silero): blocks start
 * and end in real silence, and no audio overlap (and therefore no text de-duplication) is
 * needed. The fixed-length planner is kept as the fallback for when the VAD is
 * unavailable.
 *
 * Blocks used to skip pure-silence stretches as well, which turned out to be a way of
 * losing speech: see coverWholeSignal. The VAD now decides where blocks are cut and no
 * longer decides which audio is decoded at all.
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

/**
 * Grow speech regions until they tile the whole signal, keeping their boundaries.
 *
 * The planner used to hand `planDecodeBlocks` the speech regions as they came, so audio
 * the VAD did not mark was never decoded. That is safe only if the VAD hears everything a
 * listener would, and on far-field office audio it does not: measured on a real recording,
 * silero found 2.2 s of speech in one 30 s segment and 1.8 s in another, and the device
 * stored 47 and 53 characters where decoding those same files in full yields 142 and 143.
 * Two thirds of what was said went missing, silently, with a plausible-looking transcript
 * to cover for it. Whole-segment blindness was already handled — nothing detected meant
 * decode everything — so it was only partial blindness that lost text, which is the case
 * nobody thinks to check.
 *
 * The VAD is still worth using: it knows where the silences are, and a block that starts
 * and ends in one gives the acoustic model clean edges. So it keeps choosing the cuts and
 * stops choosing the omissions. Each gap is split down the middle between its neighbours,
 * the first region is pulled back to the start and the last pushed out to the end.
 *
 * Decoding the silence too is close to free here: the device reports a real-time factor
 * around 0.03, so even decoding several times more audio stays far ahead of capture.
 */
export function coverWholeSignal(regions: SampleRange[], totalSamples: number): SampleRange[] {
  if (totalSamples <= 0) {
    return [];
  }
  if (regions.length === 0) {
    const whole: SampleRange = { start: 0, end: totalSamples };
    return [whole];
  }

  const covered: SampleRange[] = [];
  for (let i = 0; i < regions.length; ++i) {
    const start = i === 0
      ? 0
      : Math.floor((regions[i - 1].end + regions[i].start) / 2);
    const end = i === regions.length - 1
      ? totalSamples
      : Math.floor((regions[i].end + regions[i + 1].start) / 2);
    if (end > start) {
      const range: SampleRange = { start, end };
      covered.push(range);
    }
  }
  return covered;
}

/** Runs of at least this many identical characters are what collapse looks like. */
const REPETITION_MIN_RUN: number = 3;
/** A block is discarded once this share of it sits inside such runs. */
const REPETITION_LIMIT: number = 0.30;

/** Han, digits and latin only: spacing and punctuation would dilute the measurement. */
function repetitionCore(text: string): string {
  let core: string = '';
  for (let i = 0; i < text.length; ++i) {
    const code = text.charCodeAt(i);
    const keep = (code >= 48 && code <= 57) || (code >= 65 && code <= 90) ||
      (code >= 97 && code <= 122) || (code >= 0x4e00 && code <= 0x9fff);
    if (keep) {
      core += text.charAt(i);
    }
  }
  return core;
}

/**
 * True when a decoded block is the recogniser talking to itself rather than transcribing.
 *
 * Decoding the whole signal means silence and room noise now reach the model, and this one
 * does not answer them with nothing. Digital silence comes back as 啦啦啦啦啦啦啦啦啦; white
 * noise comes back as 被告被有被有的诉的的的据据据证证证证证证证证证证的意见, which is fluent
 * enough to survive into a transcript and be read as something someone said. The model has
 * a strong prior towards courtroom vocabulary and falls back on it whenever the acoustic
 * evidence is too weak to argue.
 *
 * What separates that from recognition is repetition. Measured on decodes of the reference
 * project's article reading, where the words are on record, at most 3% of characters sit
 * inside a run of three or more identical ones, and usually none do. Measured on decodes of
 * synthesised silence and noise: 59% to 100%. The threshold is placed well up that gap and
 * nearer the noise side, so a real decode is never the borderline case — applied to all
 * fourteen segments it discarded no block and cost no matched character, while suppressing
 * every synthetic case.
 *
 * It cannot catch the subtler failure, where real but faint speech is decoded into fluent
 * wrong words. That needs better audio, not a filter.
 */
export function isDecodeRepetitionCollapse(text: string): boolean {
  const core = repetitionCore(text);
  if (core.length === 0) {
    return false;
  }
  let inside = 0;
  let start = 0;
  for (let i = 1; i <= core.length; ++i) {
    if (i === core.length || core.charAt(i) !== core.charAt(start)) {
      if (i - start >= REPETITION_MIN_RUN) {
        inside += i - start;
      }
      start = i;
    }
  }
  return inside >= core.length * REPETITION_LIMIT;
}

/** Below this share of confirmed characters a block counts as unheard by the second model. */
const CORROBORATION_LIMIT: number = 0.10;

/**
 * How much of one decode the other model confirms, from 0 to 1.
 *
 * Longest common subsequence rather than shared runs, because it is cheap to write, cheap
 * to run on strings this short, and scores at least as high on any pair — which errs
 * towards keeping text rather than discarding it.
 *
 * Normalised by the length of `text` rather than by the shorter of the two, because the
 * question is whether *this* transcript is corroborated. Dividing by the shorter one lets
 * three characters vouch for a whole invented sentence; on a real recording that scored
 * 1.00 for a block where the two models had almost nothing in common.
 */
export function corroborationShare(text: string, other: string): number {
  const a = repetitionCore(text);
  const b = repetitionCore(other);
  if (a.length === 0) {
    return 1;
  }
  if (b.length === 0) {
    return 0;
  }
  let previous: number[] = new Array<number>(b.length + 1).fill(0);
  for (let i = 0; i < a.length; ++i) {
    const current: number[] = new Array<number>(b.length + 1).fill(0);
    for (let j = 0; j < b.length; ++j) {
      current[j + 1] = a.charAt(i) === b.charAt(j)
        ? previous[j] + 1
        : Math.max(previous[j + 1], current[j]);
    }
    previous = current;
  }
  return previous[b.length] / a.length;
}

/**
 * True when a second recogniser heard enough of the same thing to believe the first.
 *
 * A model's hallucination is its own prior talking, and two models trained separately do
 * not share one. Measured on decodes of synthesised silence and noise, the second model
 * confirms nothing at all — 0.00 in every case, because it returns no text where this one
 * returns courtroom sentences. Measured on decodes of a person reading a known article,
 * the worst of twenty-six pieces is 0.19 and the median is 0.96.
 *
 * The threshold sits between those, nearer the invention side. It is deliberately close to
 * "confirmed nothing whatsoever": that is the case worth acting on, and the cost of being
 * wrong is asymmetric — a discarded block loses words that were said, a kept one puts words
 * nobody said into a transcript someone will read as a record.
 */
export function isCorroborated(text: string, other: string): boolean {
  return corroborationShare(text, other) >= CORROBORATION_LIMIT;
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
