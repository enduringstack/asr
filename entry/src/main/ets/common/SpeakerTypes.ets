/**
 * Shared, serializable contracts for the on-device speaker/voiceprint feature.
 *
 * These types deliberately contain no UI or sherpa-onnx objects. They cross the
 * ThreadWorker boundary, are stored on disk, and are also used by recording export.
 */

export const SPEAKER_PROFILE_SCHEMA_VERSION: number = 2;
export const SPEAKER_MODEL_ID: string = 'speech_campplus_sv_zh_en_16k-common_advanced';
export const SPEAKER_MODEL_SHA256: string =
  'aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2';
export const SPEAKER_EMBEDDING_DIM: number = 192;
export const SPEAKER_EMBEDDING_THREADS: number = 2;
export const SPEAKER_IDENTITY_THRESHOLD: number = 0.60;
export const SPEAKER_CLUSTER_THRESHOLD: number = 0.85;
export const SPEAKER_IDENTIFY_MIN_SEGMENTS: number = 3;
export const SPEAKER_IDENTIFY_MIN_SPEECH_MS: number = 4000;
export const SPEAKER_MEETING_ENROLL_MIN_SPEECH_MS: number = 8000;
export const SPEAKER_ACTIVE_ENROLL_ROUNDS: number = 3;
export const SPEAKER_ACTIVE_ENROLL_MIN_MS: number = 4000;
export const SPEAKER_ACTIVE_ENROLL_MAX_MS: number = 6500;
export const SPEAKER_MAX_PROTOTYPES: number = 12;
export const SPEAKER_MAX_PROFILES: number = 10;

export type SpeakerRole = 'self' | 'participant';
export type SpeakerProfileState = 'provisional' | 'stable' | 'needsReRegister';
export type VoicePrototypeSource = 'activeEnrollment' | 'meetingLabel';
export type SpeakerIdentityState = 'unknown' | 'automatic' | 'confirmed';
export type MeetingAssignmentSource = 'automatic' | 'user';

export interface VoiceEvidence {
  meetingId: string;
  segmentId: string;
  startMs: number;
  endMs: number;
}

export interface VoicePrototype {
  prototypeId: string;
  embedding: number[];
  speechMs: number;
  qualityScore: number;
  source: VoicePrototypeSource;
  evidence?: VoiceEvidence;
  createdAt: number;
}

export interface SpeakerProfile {
  schemaVersion: number;
  personId: string;
  displayName: string;
  role: SpeakerRole;
  state: SpeakerProfileState;
  modelId: string;
  modelSha256: string;
  embeddingDim: number;
  enrollmentEmbeddings: VoicePrototype[];
  centroid: number[];
  sampleCount: number;
  totalSpeechMs: number;
  sourceSessionCount: number;
  createdAt: number;
  updatedAt: number;
}

/** Minimal profile payload sent to the inference worker. */
export interface SpeakerSearchProfile {
  personId: string;
  displayName: string;
  modelId: string;
  modelSha256: string;
  embeddingDim: number;
  centroid: number[];
}

export interface SpeakerTurn {
  turnId: string;
  meetingId: string;
  clusterId: string;
  startMs: number;
  endMs: number;
  text: string;
  identity: SpeakerIdentityState;
  personId?: string;
  displayName?: string;
  matchScore?: number;
}

/**
 * One cluster-level query produced by the worker. queryEmbedding is never exported;
 * it only reaches the local editor so an explicitly confirmed cluster can be enrolled.
 */
export interface SpeakerCandidate {
  meetingId: string;
  clusterId: string;
  segmentCount: number;
  speechMs: number;
  qualityScore: number;
  queryEmbedding?: number[];
  personId?: string;
  displayName?: string;
  matchScore?: number;
}

export interface SpeakerDecodeResult {
  text: string;
  meetingId: string;
  turns: SpeakerTurn[];
  candidates: SpeakerCandidate[];
}

export interface MeetingSpeakerAssignment {
  meetingId: string;
  clusterId: string;
  personId?: string;
  displayNameOverride?: string;
  source: MeetingAssignmentSource;
  updatedAt: number;
}

export interface SpeakerProfileDocument {
  schemaVersion: number;
  modelId: string;
  modelSha256: string;
  embeddingDim: number;
  profiles: SpeakerProfile[];
  updatedAt: number;
}

export interface EnrollmentQuality {
  accepted: boolean;
  reason: string;
  speechMs: number;
  rms: number;
  clippingRatio: number;
  voiceActivityRatio: number;
  qualityScore: number;
}

export interface VoiceprintExtractionResult {
  embedding: number[];
  quality: EnrollmentQuality;
}

/** Shape used only while migrating the old clear-text Preferences value. */
export interface LegacyVoiceprintProfile {
  name: string;
  embedding: number[];
  modelId?: string;
}
