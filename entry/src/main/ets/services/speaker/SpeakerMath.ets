import { util } from '@kit.ArkTS';
import {
  SPEAKER_EMBEDDING_DIM,
  SPEAKER_MAX_PROTOTYPES,
  SPEAKER_MODEL_ID,
  SPEAKER_MODEL_SHA256,
  SPEAKER_PROFILE_SCHEMA_VERSION,
  SpeakerProfile,
  SpeakerProfileState,
  SpeakerRole,
  VoicePrototype,
  VoicePrototypeSource,
} from '../../common/SpeakerTypes';

const EPSILON: number = 1e-8;

export function newSpeakerId(): string {
  return util.generateRandomUUID(true);
}

export function cleanDisplayName(value: string): string {
  const name: string = value.trim();
  if (name.length === 0) {
    return '未命名人物';
  }
  return name.length > 24 ? name.substring(0, 24) : name;
}

export function normalizeEmbedding(values: number[]): number[] {
  let sum: number = 0;
  for (let i: number = 0; i < values.length; i++) {
    const value: number = Number.isFinite(values[i]) ? values[i] : 0;
    sum += value * value;
  }
  const norm: number = Math.sqrt(sum);
  const normalized: number[] = [];
  for (let i: number = 0; i < values.length; i++) {
    normalized.push(norm > EPSILON ? values[i] / norm : 0);
  }
  return normalized;
}

export function averageEmbeddings(embeddings: number[][]): number[] {
  if (embeddings.length === 0) {
    return [];
  }
  const dim: number = embeddings[0].length;
  const sum: number[] = [];
  for (let i: number = 0; i < dim; i++) {
    sum.push(0);
  }
  let accepted: number = 0;
  for (let i: number = 0; i < embeddings.length; i++) {
    const embedding: number[] = embeddings[i];
    if (embedding.length !== dim) {
      continue;
    }
    const normalized: number[] = normalizeEmbedding(embedding);
    for (let j: number = 0; j < dim; j++) {
      sum[j] += normalized[j];
    }
    accepted += 1;
  }
  if (accepted === 0) {
    return [];
  }
  return normalizeEmbedding(sum);
}

export function cosineSimilarity(left: number[], right: number[]): number {
  if (left.length === 0 || left.length !== right.length) {
    return -1;
  }
  let dot: number = 0;
  let leftNorm: number = 0;
  let rightNorm: number = 0;
  for (let i: number = 0; i < left.length; i++) {
    dot += left[i] * right[i];
    leftNorm += left[i] * left[i];
    rightNorm += right[i] * right[i];
  }
  const denominator: number = Math.sqrt(leftNorm) * Math.sqrt(rightNorm);
  return denominator > EPSILON ? dot / denominator : -1;
}

export function prototypeSimilarityFloor(prototypes: VoicePrototype[]): number {
  if (prototypes.length < 2) {
    return 1;
  }
  let floor: number = 1;
  for (let i: number = 0; i < prototypes.length; i++) {
    for (let j: number = i + 1; j < prototypes.length; j++) {
      floor = Math.min(floor,
        cosineSimilarity(prototypes[i].embedding, prototypes[j].embedding));
    }
  }
  return floor;
}

export function createVoicePrototype(
  embedding: number[],
  speechMs: number,
  qualityScore: number,
  source: VoicePrototypeSource,
): VoicePrototype {
  return {
    prototypeId: newSpeakerId(),
    embedding: normalizeEmbedding(embedding),
    speechMs: Math.max(0, Math.round(speechMs)),
    qualityScore: Math.max(0, Math.min(1, qualityScore)),
    source: source,
    createdAt: Date.now(),
  };
}

export function createSpeakerProfile(
  displayName: string,
  role: SpeakerRole,
  prototypes: VoicePrototype[],
  state: SpeakerProfileState,
): SpeakerProfile {
  const now: number = Date.now();
  const selected: VoicePrototype[] = selectPrototypes(prototypes);
  return {
    schemaVersion: SPEAKER_PROFILE_SCHEMA_VERSION,
    personId: newSpeakerId(),
    displayName: cleanDisplayName(displayName),
    role: role,
    state: state,
    modelId: SPEAKER_MODEL_ID,
    modelSha256: SPEAKER_MODEL_SHA256,
    embeddingDim: SPEAKER_EMBEDDING_DIM,
    enrollmentEmbeddings: selected,
    centroid: centroidOf(selected),
    sampleCount: selected.length,
    totalSpeechMs: totalSpeechOf(selected),
    sourceSessionCount: countSessions(selected),
    createdAt: now,
    updatedAt: now,
  };
}

export function refreshSpeakerProfile(profile: SpeakerProfile,
  prototypes: VoicePrototype[]): SpeakerProfile {
  const selected: VoicePrototype[] = selectPrototypes(prototypes);
  const sourceSessions: number = countSessions(selected);
  let state: SpeakerProfileState = profile.state;
  if (selected.length > 0 && profile.state === 'needsReRegister') {
    state = profile.role === 'self' && selected.length >= 3 ? 'stable' : 'provisional';
  }
  if (profile.role === 'participant' && sourceSessions >= 2) {
    state = 'stable';
  }
  return {
    schemaVersion: SPEAKER_PROFILE_SCHEMA_VERSION,
    personId: profile.personId,
    displayName: cleanDisplayName(profile.displayName),
    role: profile.role,
    state: state,
    modelId: SPEAKER_MODEL_ID,
    modelSha256: SPEAKER_MODEL_SHA256,
    embeddingDim: SPEAKER_EMBEDDING_DIM,
    enrollmentEmbeddings: selected,
    centroid: centroidOf(selected),
    sampleCount: selected.length,
    totalSpeechMs: totalSpeechOf(selected),
    sourceSessionCount: sourceSessions,
    createdAt: profile.createdAt,
    updatedAt: Date.now(),
  };
}

export function isSearchableProfile(profile: SpeakerProfile): boolean {
  return profile.state !== 'needsReRegister' &&
    profile.modelId === SPEAKER_MODEL_ID &&
    profile.modelSha256 === SPEAKER_MODEL_SHA256 &&
    profile.embeddingDim === SPEAKER_EMBEDDING_DIM &&
    profile.centroid.length === SPEAKER_EMBEDDING_DIM;
}

function centroidOf(prototypes: VoicePrototype[]): number[] {
  const embeddings: number[][] = [];
  for (let i: number = 0; i < prototypes.length; i++) {
    if (prototypes[i].embedding.length === SPEAKER_EMBEDDING_DIM) {
      embeddings.push(prototypes[i].embedding);
    }
  }
  return averageEmbeddings(embeddings);
}

function totalSpeechOf(prototypes: VoicePrototype[]): number {
  let total: number = 0;
  for (let i: number = 0; i < prototypes.length; i++) {
    total += Math.max(0, prototypes[i].speechMs);
  }
  return total;
}

function countSessions(prototypes: VoicePrototype[]): number {
  const sessions: Set<string> = new Set<string>();
  let activeEnrollment: boolean = false;
  for (let i: number = 0; i < prototypes.length; i++) {
    const prototype: VoicePrototype = prototypes[i];
    if (prototype.source === 'activeEnrollment') {
      activeEnrollment = true;
    } else if (prototype.evidence && prototype.evidence.meetingId.length > 0) {
      sessions.add(prototype.evidence.meetingId);
    }
  }
  return sessions.size + (activeEnrollment ? 1 : 0);
}

/** Keep the most useful bounded set, preferring quality and cross-session diversity. */
function selectPrototypes(prototypes: VoicePrototype[]): VoicePrototype[] {
  const valid: VoicePrototype[] = [];
  for (let i: number = 0; i < prototypes.length; i++) {
    const item: VoicePrototype = prototypes[i];
    if (item.embedding.length === SPEAKER_EMBEDDING_DIM && item.qualityScore > 0) {
      valid.push(item);
    }
  }
  valid.sort((a: VoicePrototype, b: VoicePrototype): number => {
    if (a.qualityScore !== b.qualityScore) {
      return b.qualityScore - a.qualityScore;
    }
    return b.createdAt - a.createdAt;
  });
  const selected: VoicePrototype[] = [];
  const sessionKeys: Set<string> = new Set<string>();
  // Reserve one strong template from every available session before filling the rest
  // by quality. This keeps a later phone/room from being crowded out by one old meeting.
  for (let i: number = 0; i < valid.length && selected.length < SPEAKER_MAX_PROTOTYPES; i++) {
    const item: VoicePrototype = valid[i];
    const key: string = item.source === 'activeEnrollment' ? 'activeEnrollment' :
      item.evidence ? item.evidence.meetingId : item.prototypeId;
    if (sessionKeys.has(key)) {
      continue;
    }
    sessionKeys.add(key);
    selected.push(item);
  }
  for (let i: number = 0; i < valid.length && selected.length < SPEAKER_MAX_PROTOTYPES; i++) {
    let exists: boolean = false;
    for (let j: number = 0; j < selected.length; j++) {
      if (selected[j].prototypeId === valid[i].prototypeId) {
        exists = true;
        break;
      }
    }
    if (!exists) {
      selected.push(valid[i]);
    }
  }
  return selected;
}
