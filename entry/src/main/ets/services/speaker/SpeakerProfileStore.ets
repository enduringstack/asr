import { common } from '@kit.AbilityKit';
import { preferences } from '@kit.ArkData';
import { fileIo } from '@kit.CoreFileKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import {
  LegacyVoiceprintProfile,
  SPEAKER_EMBEDDING_DIM,
  SPEAKER_MAX_PROFILES,
  SPEAKER_MODEL_ID,
  SPEAKER_MODEL_SHA256,
  SPEAKER_PROFILE_SCHEMA_VERSION,
  SpeakerProfile,
  SpeakerProfileDocument,
  SpeakerSearchProfile,
  VoicePrototype,
} from '../../common/SpeakerTypes';
import { HuksEncryptedFile } from './HuksEncryptedFile';
import {
  cleanDisplayName,
  cosineSimilarity,
  createSpeakerProfile,
  isSearchableProfile,
  refreshSpeakerProfile,
} from './SpeakerMath';

const DOMAIN: number = 0x0001;
const TAG: string = 'SpeakerProfileStore';
const PROFILE_DIR: string = 'voiceprints';
const PROFILE_FILE: string = 'profiles.v2.huks';
const LEGACY_PREFERENCES: string = 'asr_voiceprints';
const LEGACY_PROFILES_KEY: string = 'profiles';

export interface SpeakerMigrationResult {
  migratedNames: number;
  needsReRegister: number;
}

/** Single source of truth for encrypted, model-bound voiceprint profiles. */
export class SpeakerProfileStore {
  private static instance: SpeakerProfileStore | undefined = undefined;

  static getInstance(): SpeakerProfileStore {
    if (!SpeakerProfileStore.instance) {
      SpeakerProfileStore.instance = new SpeakerProfileStore();
    }
    return SpeakerProfileStore.instance;
  }

  private context: common.UIAbilityContext | undefined = undefined;
  private path: string = '';
  private profiles: SpeakerProfile[] = [];
  private encryptedFile: HuksEncryptedFile = new HuksEncryptedFile();
  private initialized: boolean = false;
  private needsEncryptedRewrite: boolean = false;

  async init(context: common.UIAbilityContext): Promise<SpeakerMigrationResult> {
    this.context = context;
    const dir: string = `${context.filesDir}/${PROFILE_DIR}`;
    if (!fileIo.accessSync(dir)) {
      fileIo.mkdirSync(dir, true);
    }
    this.path = `${dir}/${PROFILE_FILE}`;

    if (this.encryptedFile.exists(this.path)) {
      this.needsEncryptedRewrite = false;
      this.profiles = await this.readDocument();
      if (this.needsEncryptedRewrite) {
        await this.persist();
      }
      this.initialized = true;
      await this.clearLegacyProfiles(context);
      return { migratedNames: 0, needsReRegister: this.countNeedsReRegister() };
    }

    const migration: SpeakerMigrationResult = await this.migrateLegacy(context);
    await this.persist();
    await this.clearLegacyProfiles(context);
    this.initialized = true;
    return migration;
  }

  list(): SpeakerProfile[] {
    this.requireInitialized();
    return this.profiles.slice().sort((a: SpeakerProfile, b: SpeakerProfile): number => {
      if (a.role !== b.role) {
        return a.role === 'self' ? -1 : 1;
      }
      return b.updatedAt - a.updatedAt;
    });
  }

  find(personId: string): SpeakerProfile | undefined {
    this.requireInitialized();
    for (let i: number = 0; i < this.profiles.length; i++) {
      if (this.profiles[i].personId === personId) {
        return this.profiles[i];
      }
    }
    return undefined;
  }

  searchProfiles(): SpeakerSearchProfile[] {
    this.requireInitialized();
    const result: SpeakerSearchProfile[] = [];
    for (let i: number = 0; i < this.profiles.length && result.length < SPEAKER_MAX_PROFILES; i++) {
      const profile: SpeakerProfile = this.profiles[i];
      if (!isSearchableProfile(profile)) {
        continue;
      }
      result.push({
        personId: profile.personId,
        displayName: profile.displayName,
        modelId: profile.modelId,
        modelSha256: profile.modelSha256,
        embeddingDim: profile.embeddingDim,
        centroid: profile.centroid,
      });
    }
    return result;
  }

  async saveActiveEnrollment(displayName: string,
    prototypes: VoicePrototype[]): Promise<SpeakerProfile> {
    this.requireInitialized();
    let selfIndex: number = -1;
    for (let i: number = 0; i < this.profiles.length; i++) {
      if (this.profiles[i].role === 'self') {
        selfIndex = i;
        break;
      }
    }
    for (let i: number = 0; i < prototypes.length; i++) {
      for (let j: number = 0; j < this.profiles.length; j++) {
        if (j === selfIndex || !isSearchableProfile(this.profiles[j])) {
          continue;
        }
        if (cosineSimilarity(prototypes[i].embedding, this.profiles[j].centroid) >= 0.75) {
          throw new Error(`本次声音与 ${this.profiles[j].displayName} 很接近，请检查人物资料`);
        }
      }
    }
    let profile: SpeakerProfile;
    if (selfIndex >= 0) {
      const current: SpeakerProfile = this.profiles[selfIndex];
      current.displayName = cleanDisplayName(displayName);
      current.state = 'stable';
      profile = refreshSpeakerProfile(current, prototypes);
      profile.state = 'stable';
      this.profiles[selfIndex] = profile;
    } else {
      this.ensureCapacity();
      profile = createSpeakerProfile(displayName, 'self', prototypes, 'stable');
      this.profiles.push(profile);
    }
    await this.persist();
    return profile;
  }

  async saveMeetingEnrollment(displayName: string, prototype: VoicePrototype,
    personId?: string): Promise<SpeakerProfile> {
    this.requireInitialized();
    const index: number = personId ? this.indexOf(personId) : -1;
    this.assertNoIdentityConflict(prototype, index);
    let profile: SpeakerProfile;
    if (index >= 0) {
      const current: SpeakerProfile = this.profiles[index];
      const combined: VoicePrototype[] = current.enrollmentEmbeddings.slice();
      combined.push(prototype);
      profile = refreshSpeakerProfile(current, combined);
      this.profiles[index] = profile;
    } else {
      this.ensureCapacity();
      profile = createSpeakerProfile(displayName, 'participant', [prototype], 'provisional');
      this.profiles.push(profile);
    }
    await this.persist();
    return profile;
  }

  async rename(personId: string, displayName: string): Promise<void> {
    this.requireInitialized();
    const index: number = this.indexOf(personId);
    if (index < 0) {
      throw new Error('speaker profile does not exist');
    }
    this.profiles[index].displayName = cleanDisplayName(displayName);
    this.profiles[index].updatedAt = Date.now();
    await this.persist();
  }

  async remove(personId: string): Promise<void> {
    this.requireInitialized();
    const kept: SpeakerProfile[] = [];
    for (let i: number = 0; i < this.profiles.length; i++) {
      if (this.profiles[i].personId !== personId) {
        kept.push(this.profiles[i]);
      }
    }
    this.profiles = kept;
    await this.persist();
  }

  private async readDocument(): Promise<SpeakerProfile[]> {
    const raw: string = await this.encryptedFile.readText(this.path);
    const document: SpeakerProfileDocument = JSON.parse(raw) as SpeakerProfileDocument;
    if (document.schemaVersion !== SPEAKER_PROFILE_SCHEMA_VERSION) {
      throw new Error('unsupported voiceprint profile schema');
    }
    const profiles: SpeakerProfile[] = [];
    for (let i: number = 0; i < document.profiles.length; i++) {
      const profile: SpeakerProfile = document.profiles[i];
      if (!profile.personId || !profile.displayName) {
        continue;
      }
      if (profile.modelId !== SPEAKER_MODEL_ID ||
        profile.modelSha256 !== SPEAKER_MODEL_SHA256 ||
        profile.embeddingDim !== SPEAKER_EMBEDDING_DIM) {
        if (profile.state !== 'needsReRegister' ||
          profile.enrollmentEmbeddings.length > 0 || profile.centroid.length > 0 ||
          profile.sampleCount !== 0 || profile.totalSpeechMs !== 0) {
          this.needsEncryptedRewrite = true;
        }
        profile.state = 'needsReRegister';
        profile.enrollmentEmbeddings = [];
        profile.centroid = [];
        profile.sampleCount = 0;
        profile.totalSpeechMs = 0;
      }
      profiles.push(profile);
    }
    return profiles;
  }

  private async migrateLegacy(context: common.UIAbilityContext): Promise<SpeakerMigrationResult> {
    const store = await preferences.getPreferences(context, LEGACY_PREFERENCES);
    const rawValue = await store.get(LEGACY_PROFILES_KEY, '');
    const raw: string = `${rawValue}`;
    if (raw.trim().length === 0) {
      this.profiles = [];
      return { migratedNames: 0, needsReRegister: 0 };
    }

    let oldProfiles: LegacyVoiceprintProfile[] = [];
    try {
      oldProfiles = JSON.parse(raw) as LegacyVoiceprintProfile[];
    } catch (error) {
      hilog.warn(DOMAIN, TAG, `legacy profile JSON ignored: ${error}`);
    }
    const migrated: SpeakerProfile[] = [];
    const names: Set<string> = new Set<string>();
    for (let i: number = 0; i < oldProfiles.length &&
      migrated.length < SPEAKER_MAX_PROFILES; i++) {
      const displayName: string = cleanDisplayName(oldProfiles[i].name || '未命名人物');
      if (names.has(displayName)) {
        continue;
      }
      names.add(displayName);
      // The old screen had no role field. A single legacy profile was the normal
      // "register myself" case even when the user entered a real name instead of “我”.
      const role = displayName === '我' || oldProfiles.length === 1 ? 'self' : 'participant';
      migrated.push(createSpeakerProfile(displayName, role, [], 'needsReRegister'));
    }
    this.profiles = migrated;
    return { migratedNames: migrated.length, needsReRegister: migrated.length };
  }

  /** Clear old clear-text vectors only after the encrypted document is readable/written. */
  private async clearLegacyProfiles(context: common.UIAbilityContext): Promise<void> {
    try {
      const store = await preferences.getPreferences(context, LEGACY_PREFERENCES);
      const value = await store.get(LEGACY_PROFILES_KEY, '');
      if (`${value}`.trim().length === 0) {
        return;
      }
      await store.delete(LEGACY_PROFILES_KEY);
      await store.flush();
    } catch (error) {
      // The encrypted store remains authoritative. Retry this best-effort cleanup on the
      // next init rather than making the biometric feature unavailable.
      hilog.warn(DOMAIN, TAG, `legacy profile cleanup deferred: ${error}`);
    }
  }

  private async persist(): Promise<void> {
    const document: SpeakerProfileDocument = {
      schemaVersion: SPEAKER_PROFILE_SCHEMA_VERSION,
      modelId: SPEAKER_MODEL_ID,
      modelSha256: SPEAKER_MODEL_SHA256,
      embeddingDim: SPEAKER_EMBEDDING_DIM,
      profiles: this.profiles,
      updatedAt: Date.now(),
    };
    await this.encryptedFile.writeText(this.path, JSON.stringify(document));
  }

  private indexOf(personId: string): number {
    for (let i: number = 0; i < this.profiles.length; i++) {
      if (this.profiles[i].personId === personId) {
        return i;
      }
    }
    return -1;
  }

  private ensureCapacity(): void {
    if (this.profiles.length >= SPEAKER_MAX_PROFILES) {
      throw new Error(`最多保存 ${SPEAKER_MAX_PROFILES} 位人物，请先删除不再使用的人物`);
    }
  }

  private assertNoIdentityConflict(prototype: VoicePrototype, targetIndex: number): void {
    if (prototype.embedding.length !== SPEAKER_EMBEDDING_DIM) {
      throw new Error('声纹样本维度与当前模型不一致');
    }
    if (targetIndex >= 0) {
      const target: SpeakerProfile = this.profiles[targetIndex];
      if (target.centroid.length === SPEAKER_EMBEDDING_DIM &&
        cosineSimilarity(prototype.embedding, target.centroid) < 0.50) {
        throw new Error(`所选语音与 ${target.displayName} 的已有声纹差异过大，请复核`);
      }
    }
    for (let i: number = 0; i < this.profiles.length; i++) {
      if (i === targetIndex || !isSearchableProfile(this.profiles[i])) {
        continue;
      }
      const score: number = cosineSimilarity(
        prototype.embedding, this.profiles[i].centroid);
      if (score >= 0.75) {
        throw new Error(`所选语音与 ${this.profiles[i].displayName} 很接近，请直接关联该人物`);
      }
    }
  }

  private countNeedsReRegister(): number {
    let count: number = 0;
    for (let i: number = 0; i < this.profiles.length; i++) {
      if (this.profiles[i].state === 'needsReRegister') {
        count += 1;
      }
    }
    return count;
  }

  private requireInitialized(): void {
    if (!this.initialized) {
      throw new Error('speaker profile store is not initialized');
    }
  }
}
