export interface SpeakerIdentityEnrollmentSpec {
  truthSpeakerId: string;
  personId: string;
  displayName: string;
  samplePaths: string[];
}

export interface SpeakerIdentityQuerySpec {
  id: string;
  truthSpeakerId: string;
  expectedPersonId?: string;
  expectedDisplayName: string;
  samplePaths: string[];
  playbackPath: string;
}

const ROOT: string = 'test/aishell3_voiceprint';

const KNOWN_SPEAKER_IDS: string[] = [
  'SSB0395', 'SSB0407', 'SSB0427', 'SSB0434', 'SSB0534',
  'SSB0539', 'SSB0544', 'SSB0565', 'SSB0578', 'SSB0588',
];

const UNKNOWN_SPEAKER_IDS: string[] = [
  'SSB0601', 'SSB0606', 'SSB0607', 'SSB0623', 'SSB0686',
  'SSB0693', 'SSB0700', 'SSB0702', 'SSB0711', 'SSB0716',
];

function pad2(value: number): string {
  return value < 10 ? `0${value}` : `${value}`;
}

function enrollmentPaths(ordinal: number): string[] {
  const paths: string[] = [];
  for (let sample: number = 1; sample <= 5; sample++) {
    paths.push(`${ROOT}/known${pad2(ordinal)}-enroll${pad2(sample)}.wav`);
  }
  return paths;
}

function queryPaths(kind: string, ordinal: number, query: number): string[] {
  const paths: string[] = [];
  for (let part: number = 1; part <= 3; part++) {
    paths.push(`${ROOT}/${kind}${pad2(ordinal)}-query${pad2(query)}-part${pad2(part)}.wav`);
  }
  return paths;
}

function queryPlaybackPath(kind: string, ordinal: number, query: number): string {
  return `${ROOT}/${kind}${pad2(ordinal)}-query${pad2(query)}-playback.wav`;
}

export function buildAishell3IdentityEnrollments(): SpeakerIdentityEnrollmentSpec[] {
  const result: SpeakerIdentityEnrollmentSpec[] = [];
  for (let i: number = 0; i < KNOWN_SPEAKER_IDS.length; i++) {
    const ordinal: number = i + 1;
    result.push({
      truthSpeakerId: KNOWN_SPEAKER_IDS[i],
      personId: `aishell3-test-known-${pad2(ordinal)}`,
      displayName: `AISHELL 已知 ${pad2(ordinal)}`,
      samplePaths: enrollmentPaths(ordinal),
    });
  }
  return result;
}

export function buildAishell3IdentityQueries(): SpeakerIdentityQuerySpec[] {
  const result: SpeakerIdentityQuerySpec[] = [];
  for (let i: number = 0; i < KNOWN_SPEAKER_IDS.length; i++) {
    const ordinal: number = i + 1;
    for (let query: number = 1; query <= 2; query++) {
      result.push({
        id: `known-${pad2(ordinal)}-${query}`,
        truthSpeakerId: KNOWN_SPEAKER_IDS[i],
        expectedPersonId: `aishell3-test-known-${pad2(ordinal)}`,
        expectedDisplayName: `AISHELL 已知 ${pad2(ordinal)}`,
        samplePaths: queryPaths('known', ordinal, query),
        playbackPath: queryPlaybackPath('known', ordinal, query),
      });
    }
  }
  for (let i: number = 0; i < UNKNOWN_SPEAKER_IDS.length; i++) {
    const ordinal: number = i + 1;
    for (let query: number = 1; query <= 2; query++) {
      result.push({
        id: `unknown-${pad2(ordinal)}-${query}`,
        truthSpeakerId: UNKNOWN_SPEAKER_IDS[i],
        expectedDisplayName: '未知说话人',
        samplePaths: queryPaths('unknown', ordinal, query),
        playbackPath: queryPlaybackPath('unknown', ordinal, query),
      });
    }
  }
  return result;
}
