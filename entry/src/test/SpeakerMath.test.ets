import { describe, it, expect } from '@ohos/hypium';
import {
  averageEmbeddings,
  cosineSimilarity,
  normalizeEmbedding,
  prototypeSimilarityFloor,
} from '../main/ets/services/speaker/SpeakerMath';
import { VoicePrototype } from '../main/ets/common/SpeakerTypes';

function prototype(id: string, embedding: number[]): VoicePrototype {
  return {
    prototypeId: id,
    embedding: embedding,
    speechMs: 5000,
    qualityScore: 1,
    source: 'activeEnrollment',
    createdAt: 1,
  };
}

export default function speakerMathTest() {
  describe('speakerMathTest', () => {
    it('normalizes embeddings before cosine comparison', 0, () => {
      const value: number[] = normalizeEmbedding([3, 4]);
      expect(Math.abs(value[0] - 0.6) < 1e-6).assertTrue();
      expect(Math.abs(value[1] - 0.8) < 1e-6).assertTrue();
      expect(Math.abs(cosineSimilarity([3, 4], [6, 8]) - 1) < 1e-6).assertTrue();
    });

    it('averages normalized rounds and returns a normalized center', 0, () => {
      const center: number[] = averageEmbeddings([[1, 0], [0.8, 0.2], [0.9, 0.1]]);
      expect(center.length).assertEqual(2);
      expect(center[0] > 0.98).assertTrue();
      expect(Math.abs(cosineSimilarity(center, center) - 1) < 1e-6).assertTrue();
    });

    it('uses the weakest pair to reject an inconsistent enrollment round', 0, () => {
      const consistent: VoicePrototype[] = [
        prototype('a', [1, 0]), prototype('b', [0.9, 0.1]), prototype('c', [0.8, 0.2]),
      ];
      const inconsistent: VoicePrototype[] = [
        prototype('a', [1, 0]), prototype('b', [0.9, 0.1]), prototype('c', [-1, 0]),
      ];
      expect(prototypeSimilarityFloor(consistent) > 0.9).assertTrue();
      expect(prototypeSimilarityFloor(inconsistent) < 0).assertTrue();
    });

    it('never treats dimension mismatch as a valid match', 0, () => {
      expect(cosineSimilarity([1, 0], [1, 0, 0])).assertEqual(-1);
      expect(averageEmbeddings([[1, 0], [1, 0, 0]]).length).assertEqual(2);
    });
  });
}
