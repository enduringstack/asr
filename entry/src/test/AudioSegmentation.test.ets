import { describe, it, expect } from '@ohos/hypium';
import {
  buildFixedLengthSegments,
  computeRms,
  DecodeBlockOptions,
  FixedSegmentOptions,
  planDecodeBlocks,
  SampleRange,
  speechFlagsToRegions,
  SpeechRegionOptions
} from '../main/ets/common/AudioSegmentation';

const WINDOW = 512;

function regionOptions(minSpeech: number, minSilence: number, pad: number): SpeechRegionOptions {
  return { minSpeechSamples: minSpeech, minSilenceSamples: minSilence, padSamples: pad };
}

function blockOptions(maxBlock: number, maxGap: number): DecodeBlockOptions {
  return { maxBlockSamples: maxBlock, maxGapSamples: maxGap };
}

function range(start: number, end: number): SampleRange {
  return { start, end };
}

function flagsFrom(pattern: string): boolean[] {
  const flags: boolean[] = [];
  for (let i = 0; i < pattern.length; ++i) {
    flags.push(pattern.charAt(i) === '1');
  }
  return flags;
}

function totalLength(ranges: SampleRange[]): number {
  let total = 0;
  for (let i = 0; i < ranges.length; ++i) {
    total += ranges[i].end - ranges[i].start;
  }
  return total;
}

export default function audioSegmentationTest() {
  describe('audioSegmentationTest', () => {
    it('computes RMS, and reports zero for an empty buffer', 0, () => {
      const constant = new Float32Array(100);
      for (let i = 0; i < constant.length; ++i) {
        constant[i] = 0.5;
      }
      expect(Math.abs(computeRms(constant) - 0.5) < 1e-6).assertTrue();
      expect(computeRms(new Float32Array(0))).assertEqual(0);
    });

    it('returns no speech regions for empty input', 0, () => {
      expect(speechFlagsToRegions([], WINDOW, 10 * WINDOW, regionOptions(0, 0, 0)).length).assertEqual(0);
      expect(speechFlagsToRegions(flagsFrom('111'), WINDOW, 0, regionOptions(0, 0, 0)).length).assertEqual(0);
    });

    it('converts a contiguous run of speech windows into one region', 0, () => {
      const regions = speechFlagsToRegions(flagsFrom('0110'), WINDOW, 4 * WINDOW, regionOptions(0, 0, 0));
      expect(regions.length).assertEqual(1);
      expect(regions[0].start).assertEqual(WINDOW);
      expect(regions[0].end).assertEqual(3 * WINDOW);
    });

    it('clamps the last region to the real sample count', 0, () => {
      const total = 3 * WINDOW - 100;
      const regions = speechFlagsToRegions(flagsFrom('111'), WINDOW, total, regionOptions(0, 0, 0));
      expect(regions.length).assertEqual(1);
      expect(regions[0].end).assertEqual(total);
    });

    it('drops noise blips shorter than the minimum speech duration', 0, () => {
      const regions = speechFlagsToRegions(flagsFrom('01000'), WINDOW, 5 * WINDOW, regionOptions(3 * WINDOW, 0, 0));
      expect(regions.length).assertEqual(0);
    });

    it('bridges a short pause so one clause stays a single region', 0, () => {
      // Without bridging this is two 512-sample runs, each below a 1024-sample minimum,
      // and both would be discarded as blips.
      const regions = speechFlagsToRegions(
        flagsFrom('10100'),
        WINDOW,
        5 * WINDOW,
        regionOptions(2 * WINDOW, 2 * WINDOW, 0)
      );
      expect(regions.length).assertEqual(1);
      expect(regions[0].start).assertEqual(0);
      expect(regions[0].end).assertEqual(3 * WINDOW);
    });

    it('does not bridge a pause longer than the bridge threshold', 0, () => {
      const regions = speechFlagsToRegions(
        flagsFrom('100001'),
        WINDOW,
        6 * WINDOW,
        regionOptions(0, WINDOW, 0)
      );
      expect(regions.length).assertEqual(2);
    });

    it('pads regions, clamping at the signal edges and re-merging overlaps', 0, () => {
      const total = 6 * WINDOW;
      const regions = speechFlagsToRegions(flagsFrom('101010'), WINDOW, total, regionOptions(0, 0, WINDOW));
      // Every padded region touches its neighbour, so they collapse into one.
      expect(regions.length).assertEqual(1);
      expect(regions[0].start).assertEqual(0);
      expect(regions[0].end).assertEqual(total);
    });

    it('plans no blocks when there is no speech', 0, () => {
      expect(planDecodeBlocks([], blockOptions(1000, 500)).length).assertEqual(0);
    });

    it('packs neighbouring regions into one block while they fit', 0, () => {
      const blocks = planDecodeBlocks([range(0, 400), range(600, 900)], blockOptions(1000, 500));
      expect(blocks.length).assertEqual(1);
      expect(blocks[0].start).assertEqual(0);
      expect(blocks[0].end).assertEqual(900);
    });

    it('starts a new block instead of exceeding the cap', 0, () => {
      const blocks = planDecodeBlocks([range(0, 400), range(900, 1400)], blockOptions(1000, 500));
      expect(blocks.length).assertEqual(2);
      expect(blocks[0].end).assertEqual(400);
      expect(blocks[1].start).assertEqual(900);
      expect(blocks[1].end).assertEqual(1400);
    });

    it('starts a new block across a long pause even when the cap allows merging', 0, () => {
      // Merging here would fit under the cap but would drag 4600 samples of silence into
      // the decoded audio, which is exactly what VAD planning is supposed to remove.
      const blocks = planDecodeBlocks([range(0, 400), range(5000, 5400)], blockOptions(16000, 500));
      expect(blocks.length).assertEqual(2);
      expect(blocks[0].end).assertEqual(400);
      expect(blocks[1].start).assertEqual(5000);
    });

    it('splits an over-long region into even parts that all respect the cap', 0, () => {
      const blocks = planDecodeBlocks([range(0, 2500)], blockOptions(1000, 500));
      expect(blocks.length).assertEqual(3);
      for (let i = 0; i < blocks.length; ++i) {
        expect(blocks[i].end - blocks[i].start <= 1000).assertTrue();
        // Even parts: no sliver at the end.
        expect(blocks[i].end - blocks[i].start >= 300).assertTrue();
      }
      expect(blocks[0].start).assertEqual(0);
      expect(blocks[blocks.length - 1].end).assertEqual(2500);
    });

    it('covers every speech region exactly once', 0, () => {
      const regions = [range(0, 400), range(600, 900), range(5000, 5400)];
      const blocks = planDecodeBlocks(regions, blockOptions(1000, 500));
      for (let i = 0; i < regions.length; ++i) {
        let covered = false;
        for (let j = 0; j < blocks.length; ++j) {
          if (blocks[j].start <= regions[i].start && blocks[j].end >= regions[i].end) {
            covered = true;
          }
        }
        expect(covered).assertTrue();
      }
    });

    it('skips silence, so a sparse recording decodes far less audio than its duration', 0, () => {
      const total = 16000 * 60;
      // One second of speech per ten seconds of recording.
      const regions: SampleRange[] = [];
      for (let i = 0; i < 6; ++i) {
        regions.push(range(i * 16000 * 10, i * 16000 * 10 + 16000));
      }
      const blocks = planDecodeBlocks(regions, blockOptions(16000 * 24, 16000 * 3 / 2));
      expect(blocks.length).assertEqual(6);
      expect(totalLength(blocks)).assertEqual(6 * 16000);
      expect(totalLength(blocks) < total / 2).assertTrue();
    });

    it('tiles the whole signal with the fixed-length fallback planner', 0, () => {
      const samples = new Float32Array(16000 * 35);
      for (let i = 0; i < samples.length; ++i) {
        samples[i] = Math.sin(i / 20) * 0.2;
      }
      const options: FixedSegmentOptions = {
        segmentSamples: 16000 * 10,
        searchRadiusSamples: 16000,
        boundaryWindowSamples: 2560,
        minSegmentSamples: 16000 * 6
      };
      const segments = buildFixedLengthSegments(samples, options);
      expect(segments.length > 1).assertTrue();
      expect(segments[0].start).assertEqual(0);
      expect(segments[segments.length - 1].end).assertEqual(samples.length);
      // Contiguous: the fallback planner must not drop audio between segments.
      for (let i = 1; i < segments.length; ++i) {
        expect(segments[i].start).assertEqual(segments[i - 1].end);
      }
    });

    it('returns a single segment when the audio is shorter than one block', 0, () => {
      const samples = new Float32Array(16000 * 3);
      const options: FixedSegmentOptions = {
        segmentSamples: 16000 * 10,
        searchRadiusSamples: 16000,
        boundaryWindowSamples: 2560,
        minSegmentSamples: 16000 * 6
      };
      const segments = buildFixedLengthSegments(samples, options);
      expect(segments.length).assertEqual(1);
      expect(segments[0].start).assertEqual(0);
      expect(segments[0].end).assertEqual(samples.length);
    });
  });
}
