import { util } from '@kit.ArkTS';
import { SpeakerCandidate, SpeakerTurn } from '../../common/SpeakerTypes';

export interface SpeakerEditSession {
  sessionId: string;
  meetingId: string;
  title: string;
  turns: SpeakerTurn[];
  candidates: SpeakerCandidate[];
}

/**
 * Process-memory handoff for the speaker editor.
 *
 * Candidate embeddings are biometric material. Keeping the single active editor
 * payload here avoids serializing them into router parameters or any persistent store.
 */
export class SpeakerEditSessionStore {
  private static instance: SpeakerEditSessionStore | undefined = undefined;

  static getInstance(): SpeakerEditSessionStore {
    if (!SpeakerEditSessionStore.instance) {
      SpeakerEditSessionStore.instance = new SpeakerEditSessionStore();
    }
    return SpeakerEditSessionStore.instance;
  }

  private activeSession: SpeakerEditSession | undefined = undefined;

  put(meetingId: string, title: string, turns: SpeakerTurn[],
    candidates: SpeakerCandidate[]): string {
    const sessionId: string = util.generateRandomUUID(true);
    this.activeSession = {
      sessionId: sessionId,
      meetingId: meetingId,
      title: title,
      turns: turns.slice(),
      candidates: candidates.slice(),
    };
    return sessionId;
  }

  get(sessionId: string): SpeakerEditSession | undefined {
    if (!this.activeSession || this.activeSession.sessionId !== sessionId) {
      return undefined;
    }
    return this.activeSession;
  }

  clear(sessionId: string): void {
    if (this.activeSession && this.activeSession.sessionId === sessionId) {
      this.activeSession = undefined;
    }
  }
}
