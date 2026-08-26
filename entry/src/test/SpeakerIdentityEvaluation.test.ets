import { describe, expect, it } from '@ohos/hypium';
import {
  buildAishell3IdentityEnrollments,
  buildAishell3IdentityQueries,
  SpeakerIdentityEnrollmentSpec,
  SpeakerIdentityQuerySpec,
} from '../main/ets/common/SpeakerIdentityTestData';
import { SPEAKER_EMBEDDING_DIM, SpeakerSearchProfile } from '../main/ets/common/SpeakerTypes';
import {
  buildTestSearchProfile,
  scoreSpeakerIdentityCase,
  SpeakerIdentityScoredCase,
  SpeakerIdentitySummary,
  summarizeSpeakerIdentityCases,
} from '../main/ets/services/speaker/SpeakerIdentityEvaluation';

function axis(index: number): number[] {
  const vector: number[] = [];
  for (let i: number = 0; i < SPEAKER_EMBEDDING_DIM; i++) {
    vector.push(i === index ? 1 : 0);
  }
  return vector;
}

export default function speakerIdentityEvaluationTest() {
  describe('speakerIdentityEvaluationTest', () => {
    it('keeps enrollment and query fixtures disjoint', 0, () => {
      const enrollments: SpeakerIdentityEnrollmentSpec[] = buildAishell3IdentityEnrollments();
      const queries: SpeakerIdentityQuerySpec[] = buildAishell3IdentityQueries();
      expect(enrollments.length).assertEqual(10);
      expect(queries.length).assertEqual(40);
      const enrollmentPaths: string[] = [];
      for (let i: number = 0; i < enrollments.length; i++) {
        expect(enrollments[i].samplePaths.length).assertEqual(5);
        enrollmentPaths.push(...enrollments[i].samplePaths);
      }
      const queryPaths: string[] = [];
      for (let i: number = 0; i < queries.length; i++) {
        expect(queries[i].samplePaths.length).assertEqual(3);
        queryPaths.push(...queries[i].samplePaths);
      }
      let overlap: number = 0;
      for (let i: number = 0; i < queryPaths.length; i++) {
        if (enrollmentPaths.indexOf(queryPaths[i]) >= 0) {
          overlap += 1;
        }
      }
      expect(overlap).assertEqual(0);
    });

    it('correctly accepts a known aggregate query', 0, () => {
      const first: SpeakerSearchProfile = buildTestSearchProfile(
        'person-1', '人物一', [axis(0), axis(0), axis(0)]);
      const second: SpeakerSearchProfile = buildTestSearchProfile(
        'person-2', '人物二', [axis(1), axis(1), axis(1)]);
      const result: SpeakerIdentityScoredCase = scoreSpeakerIdentityCase(
        'person-1', [axis(0), axis(0), axis(0)], [first, second], 0.60);
      expect(result.accepted).assertTrue();
      expect(result.correct).assertTrue();
      expect(result.predictedPersonId).assertEqual('person-1');
      expect(result.topScore > 0.99).assertTrue();
    });

    it('rejects an unknown query below threshold', 0, () => {
      const first: SpeakerSearchProfile = buildTestSearchProfile(
        'person-1', '人物一', [axis(0), axis(0), axis(0)]);
      const second: SpeakerSearchProfile = buildTestSearchProfile(
        'person-2', '人物二', [axis(1), axis(1), axis(1)]);
      const result: SpeakerIdentityScoredCase = scoreSpeakerIdentityCase(
        undefined, [axis(2), axis(2), axis(2)], [first, second], 0.60);
      expect(result.accepted).assertFalse();
      expect(result.correct).assertTrue();
    });

    it('reports false rejects, misidentification and false accepts separately', 0, () => {
      const cases: SpeakerIdentityScoredCase[] = [
        { expectedPersonId: 'a', predictedPersonId: 'a', accepted: true,
          correct: true, topScore: 0.9, secondScore: 0.2 },
        { expectedPersonId: 'a', accepted: false,
          correct: false, topScore: 0.5, secondScore: 0.2 },
        { expectedPersonId: 'a', predictedPersonId: 'b', accepted: true,
          correct: false, topScore: 0.8, secondScore: 0.7 },
        { accepted: false, correct: true, topScore: 0.4, secondScore: 0.3 },
        { predictedPersonId: 'b', accepted: true,
          correct: false, topScore: 0.7, secondScore: 0.4 },
      ];
      const summary: SpeakerIdentitySummary = summarizeSpeakerIdentityCases(cases);
      expect(summary.knownTotal).assertEqual(3);
      expect(summary.knownCorrect).assertEqual(1);
      expect(summary.knownFalseReject).assertEqual(1);
      expect(summary.knownMisidentified).assertEqual(1);
      expect(summary.unknownTotal).assertEqual(2);
      expect(summary.unknownRejected).assertEqual(1);
      expect(summary.unknownFalseAccept).assertEqual(1);
    });
  });
}
