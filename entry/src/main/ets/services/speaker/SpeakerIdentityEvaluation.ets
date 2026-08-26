import {
  SPEAKER_EMBEDDING_DIM,
  SPEAKER_IDENTITY_THRESHOLD,
  SPEAKER_MODEL_ID,
  SPEAKER_MODEL_SHA256,
  SpeakerSearchProfile,
} from '../../common/SpeakerTypes';
import { averageEmbeddings, cosineSimilarity } from './SpeakerMath';

export interface SpeakerIdentityPrediction {
  accepted: boolean;
  personId?: string;
  displayName?: string;
  topScore: number;
  secondScore: number;
}

export interface SpeakerIdentityScoredCase {
  expectedPersonId?: string;
  predictedPersonId?: string;
  accepted: boolean;
  correct: boolean;
  topScore: number;
  secondScore: number;
}

export interface SpeakerIdentitySummary {
  knownTotal: number;
  knownCorrect: number;
  knownFalseReject: number;
  knownMisidentified: number;
  unknownTotal: number;
  unknownRejected: number;
  unknownFalseAccept: number;
  knownCorrectRate: number;
  unknownRejectRate: number;
  balancedAccuracy: number;
}

/** Builds one in-memory gallery entry without touching SpeakerProfileStore. */
export function buildTestSearchProfile(personId: string, displayName: string,
  enrollmentEmbeddings: number[][]): SpeakerSearchProfile {
  const centroid: number[] = averageEmbeddings(enrollmentEmbeddings);
  if (centroid.length !== SPEAKER_EMBEDDING_DIM) {
    throw new Error('测试注册样本维度与当前声纹模型不一致');
  }
  return {
    personId: personId,
    displayName: displayName,
    modelId: SPEAKER_MODEL_ID,
    modelSha256: SPEAKER_MODEL_SHA256,
    embeddingDim: SPEAKER_EMBEDDING_DIM,
    centroid: centroid,
  };
}

/**
 * Open-set prediction shared by production inference and the labeled test center.
 * Keeping one implementation prevents the diagnostic path from passing with logic the
 * app does not actually use for meetings.
 */
export function predictSpeakerIdentity(queryEmbedding: number[],
  profiles: SpeakerSearchProfile[],
  threshold: number = SPEAKER_IDENTITY_THRESHOLD): SpeakerIdentityPrediction {
  if (queryEmbedding.length !== SPEAKER_EMBEDDING_DIM) {
    return { accepted: false, topScore: -1, secondScore: -1 };
  }
  let best: SpeakerSearchProfile | undefined = undefined;
  let bestScore: number = -1;
  let secondScore: number = -1;
  for (let i: number = 0; i < profiles.length; i++) {
    const profile: SpeakerSearchProfile = profiles[i];
    if (profile.modelId !== SPEAKER_MODEL_ID ||
      profile.modelSha256 !== SPEAKER_MODEL_SHA256 ||
      profile.embeddingDim !== SPEAKER_EMBEDDING_DIM ||
      profile.centroid.length !== SPEAKER_EMBEDDING_DIM) {
      continue;
    }
    const score: number = cosineSimilarity(queryEmbedding, profile.centroid);
    if (score > bestScore) {
      secondScore = bestScore;
      bestScore = score;
      best = profile;
    } else if (score > secondScore) {
      secondScore = score;
    }
  }
  if (!best || bestScore < threshold) {
    return { accepted: false, topScore: bestScore, secondScore: secondScore };
  }
  return {
    accepted: true,
    personId: best.personId,
    displayName: best.displayName,
    topScore: bestScore,
    secondScore: secondScore,
  };
}

export function scoreSpeakerIdentityCase(expectedPersonId: string | undefined,
  queryEmbeddings: number[][], profiles: SpeakerSearchProfile[],
  threshold: number = SPEAKER_IDENTITY_THRESHOLD): SpeakerIdentityScoredCase {
  const query: number[] = averageEmbeddings(queryEmbeddings);
  if (query.length !== SPEAKER_EMBEDDING_DIM) {
    throw new Error('测试查询样本维度与当前声纹模型不一致');
  }
  const prediction: SpeakerIdentityPrediction = predictSpeakerIdentity(query, profiles, threshold);
  const correct: boolean = expectedPersonId ?
    prediction.accepted && prediction.personId === expectedPersonId : !prediction.accepted;
  return {
    expectedPersonId: expectedPersonId,
    predictedPersonId: prediction.personId,
    accepted: prediction.accepted,
    correct: correct,
    topScore: prediction.topScore,
    secondScore: prediction.secondScore,
  };
}

export function summarizeSpeakerIdentityCases(
  cases: SpeakerIdentityScoredCase[]): SpeakerIdentitySummary {
  let knownTotal: number = 0;
  let knownCorrect: number = 0;
  let knownFalseReject: number = 0;
  let knownMisidentified: number = 0;
  let unknownTotal: number = 0;
  let unknownRejected: number = 0;
  let unknownFalseAccept: number = 0;
  for (let i: number = 0; i < cases.length; i++) {
    const result: SpeakerIdentityScoredCase = cases[i];
    if (result.expectedPersonId) {
      knownTotal += 1;
      if (!result.accepted) {
        knownFalseReject += 1;
      } else if (result.predictedPersonId === result.expectedPersonId) {
        knownCorrect += 1;
      } else {
        knownMisidentified += 1;
      }
    } else {
      unknownTotal += 1;
      if (result.accepted) {
        unknownFalseAccept += 1;
      } else {
        unknownRejected += 1;
      }
    }
  }
  const knownCorrectRate: number = knownTotal > 0 ? knownCorrect / knownTotal : 0;
  const unknownRejectRate: number = unknownTotal > 0 ? unknownRejected / unknownTotal : 0;
  return {
    knownTotal: knownTotal,
    knownCorrect: knownCorrect,
    knownFalseReject: knownFalseReject,
    knownMisidentified: knownMisidentified,
    unknownTotal: unknownTotal,
    unknownRejected: unknownRejected,
    unknownFalseAccept: unknownFalseAccept,
    knownCorrectRate: knownCorrectRate,
    unknownRejectRate: unknownRejectRate,
    balancedAccuracy: (knownCorrectRate + unknownRejectRate) / 2,
  };
}
