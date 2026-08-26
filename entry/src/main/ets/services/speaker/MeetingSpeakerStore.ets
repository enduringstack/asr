import { common } from '@kit.AbilityKit';
import { preferences } from '@kit.ArkData';
import {
  MeetingSpeakerAssignment,
  SpeakerProfile,
  SpeakerTurn,
} from '../../common/SpeakerTypes';
import { cleanDisplayName } from './SpeakerMath';

const PREFERENCES_NAME: string = 'asr_meeting_speakers_v2';
const ASSIGNMENTS_KEY: string = 'assignments';

/** Stores user edits separately from biometric templates. */
export class MeetingSpeakerStore {
  private context: common.UIAbilityContext;

  constructor(context: common.UIAbilityContext) {
    this.context = context;
  }

  async assignmentsFor(meetingId: string): Promise<MeetingSpeakerAssignment[]> {
    const all: MeetingSpeakerAssignment[] = await this.load();
    const result: MeetingSpeakerAssignment[] = [];
    for (let i: number = 0; i < all.length; i++) {
      if (all[i].meetingId === meetingId) {
        result.push(all[i]);
      }
    }
    return result;
  }

  async assign(assignment: MeetingSpeakerAssignment): Promise<void> {
    const meetingId: string = assignment.meetingId.trim().substring(0, 128);
    const clusterId: string = assignment.clusterId.trim().substring(0, 128);
    if (meetingId.length === 0 || clusterId.length === 0) {
      throw new Error('会议或说话人标识为空');
    }
    if (assignment.personId && assignment.personId.length > 128) {
      throw new Error('人物标识无效');
    }
    const normalized: MeetingSpeakerAssignment = {
      meetingId: meetingId,
      clusterId: clusterId,
      personId: assignment.personId,
      displayNameOverride: assignment.displayNameOverride ?
        cleanDisplayName(assignment.displayNameOverride) : undefined,
      source: assignment.source,
      updatedAt: assignment.updatedAt,
    };
    const all: MeetingSpeakerAssignment[] = await this.load();
    const kept: MeetingSpeakerAssignment[] = [];
    for (let i: number = 0; i < all.length; i++) {
      if (all[i].meetingId !== normalized.meetingId ||
        all[i].clusterId !== normalized.clusterId) {
        kept.push(all[i]);
      }
    }
    kept.push(normalized);
    await this.save(kept);
  }

  async clearPerson(personId: string): Promise<void> {
    const all: MeetingSpeakerAssignment[] = await this.load();
    const kept: MeetingSpeakerAssignment[] = [];
    for (let i: number = 0; i < all.length; i++) {
      const displayNameOverride: string | undefined = all[i].displayNameOverride;
      if (all[i].personId !== personId) {
        kept.push(all[i]);
      } else if (displayNameOverride && displayNameOverride.length > 0) {
        kept.push({
          meetingId: all[i].meetingId,
          clusterId: all[i].clusterId,
          displayNameOverride: displayNameOverride,
          source: 'user',
          updatedAt: Date.now(),
        });
      }
    }
    await this.save(kept);
  }

  apply(turns: SpeakerTurn[], assignments: MeetingSpeakerAssignment[],
    profiles: SpeakerProfile[]): SpeakerTurn[] {
    const result: SpeakerTurn[] = [];
    for (let i: number = 0; i < turns.length; i++) {
      const source: SpeakerTurn = turns[i];
      let assignment: MeetingSpeakerAssignment | undefined = undefined;
      for (let j: number = 0; j < assignments.length; j++) {
        if (assignments[j].clusterId === source.clusterId) {
          assignment = assignments[j];
          break;
        }
      }
      if (!assignment) {
        let renamed: string | undefined = source.displayName;
        if (source.personId) {
          for (let j: number = 0; j < profiles.length; j++) {
            if (profiles[j].personId === source.personId) {
              renamed = profiles[j].displayName;
              break;
            }
          }
        }
        if (renamed === source.displayName) {
          result.push(source);
        } else {
          result.push({
            turnId: source.turnId,
            meetingId: source.meetingId,
            clusterId: source.clusterId,
            startMs: source.startMs,
            endMs: source.endMs,
            text: source.text,
            identity: source.identity,
            personId: source.personId,
            displayName: renamed,
            matchScore: source.matchScore,
          });
        }
        continue;
      }
      let displayName: string | undefined = assignment.displayNameOverride;
      if (assignment.personId) {
        for (let j: number = 0; j < profiles.length; j++) {
          if (profiles[j].personId === assignment.personId) {
            displayName = profiles[j].displayName;
            break;
          }
        }
      }
      result.push({
        turnId: source.turnId,
        meetingId: source.meetingId,
        clusterId: source.clusterId,
        startMs: source.startMs,
        endMs: source.endMs,
        text: source.text,
        identity: 'confirmed',
        personId: assignment.personId,
        displayName: displayName,
        matchScore: source.matchScore,
      });
    }
    return result;
  }

  private async load(): Promise<MeetingSpeakerAssignment[]> {
    const store = await preferences.getPreferences(this.context, PREFERENCES_NAME);
    const value = await store.get(ASSIGNMENTS_KEY, '[]');
    try {
      return JSON.parse(`${value}`) as MeetingSpeakerAssignment[];
    } catch (_error) {
      return [];
    }
  }

  private async save(assignments: MeetingSpeakerAssignment[]): Promise<void> {
    const store = await preferences.getPreferences(this.context, PREFERENCES_NAME);
    await store.put(ASSIGNMENTS_KEY, JSON.stringify(assignments));
    await store.flush();
  }
}
