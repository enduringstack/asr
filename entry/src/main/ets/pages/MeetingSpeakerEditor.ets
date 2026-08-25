import { common } from '@kit.AbilityKit';
import { router } from '@kit.ArkUI';
import {
  MeetingSpeakerAssignment,
  SPEAKER_EMBEDDING_DIM,
  SPEAKER_IDENTITY_THRESHOLD,
  SPEAKER_MEETING_ENROLL_MIN_SPEECH_MS,
  SpeakerCandidate,
  SpeakerProfile,
  SpeakerTurn,
  VoicePrototype,
} from '../common/SpeakerTypes';
import { MeetingSpeakerStore } from '../services/speaker/MeetingSpeakerStore';
import {
  averageEmbeddings,
  cosineSimilarity,
  createVoicePrototype,
} from '../services/speaker/SpeakerMath';
import {
  SpeakerEditSession,
  SpeakerEditSessionStore,
} from '../services/speaker/SpeakerEditSessionStore';
import { SpeakerProfileStore } from '../services/speaker/SpeakerProfileStore';

interface MeetingEditorParams {
  sessionId: string;
}

interface SpeakerClusterRow {
  clusterId: string;
  displayName: string;
  personId?: string;
  identityText: string;
  turnCount: number;
  speechMs: number;
}

@Entry
@Component
struct MeetingSpeakerEditorPage {
  @State title: string = '说话人编辑';
  @State turns: SpeakerTurn[] = [];
  @State candidates: SpeakerCandidate[] = [];
  @State profiles: SpeakerProfile[] = [];
  @State clusterRows: SpeakerClusterRow[] = [];
  @State selectedClusterId: string = '';
  @State editName: string = '';
  @State selectedPersonId: string = '';
  @State enrollmentClusterIds: string[] = [];
  @State statusMessage: string = '正在读取说话人信息...';

  private meetingId: string = '';
  private profileStore: SpeakerProfileStore = SpeakerProfileStore.getInstance();
  private meetingStore: MeetingSpeakerStore | undefined = undefined;
  private editSessions: SpeakerEditSessionStore = SpeakerEditSessionStore.getInstance();
  private sessionId: string = '';

  aboutToAppear(): void {
    const params: MeetingEditorParams = router.getParams() as MeetingEditorParams;
    this.sessionId = params && params.sessionId ? params.sessionId : '';
    const session: SpeakerEditSession | undefined = this.editSessions.get(this.sessionId);
    if (!session) {
      this.turns = [];
      this.candidates = [];
      this.statusMessage = '编辑会话已失效，请返回识别结果后重新进入';
      return;
    }
    this.meetingId = session.meetingId;
    this.title = session.title;
    this.turns = session.turns;
    this.candidates = session.candidates;
    const context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;
    this.meetingStore = new MeetingSpeakerStore(context);
    this.profileStore.init(context)
      .then((): Promise<MeetingSpeakerAssignment[]> => {
        this.profiles = this.profileStore.list();
        return this.meetingStore ? this.meetingStore.assignmentsFor(this.meetingId) :
          Promise.resolve<MeetingSpeakerAssignment[]>([]);
      })
      .then((assignments: MeetingSpeakerAssignment[]): void => {
        if (this.meetingStore) {
          this.turns = this.meetingStore.apply(this.turns, assignments, this.profiles);
        }
        this.rebuildRows();
        this.statusMessage = this.candidates.length > 0
          ? '点击一个说话人后，可选择本次改名、关联或记住'
          : '当前没有可用声纹样本，仍可仅修改本次名称或关联人物';
      })
      .catch((error: Error): void => {
        this.statusMessage = `读取失败: ${error.message}`;
        this.rebuildRows();
      });
  }

  aboutToDisappear(): void {
    this.editSessions.clear(this.sessionId);
  }

  private rebuildRows(): void {
    const rows: SpeakerClusterRow[] = [];
    const ids: string[] = [];
    for (let i: number = 0; i < this.turns.length; i++) {
      if (ids.indexOf(this.turns[i].clusterId) < 0) {
        ids.push(this.turns[i].clusterId);
      }
    }
    for (let i: number = 0; i < ids.length; i++) {
      const clusterId: string = ids[i];
      let turnCount: number = 0;
      let speechMs: number = 0;
      let displayName: string = `未知说话人 ${clusterId}`;
      let personId: string | undefined = undefined;
      let automatic: boolean = false;
      for (let j: number = 0; j < this.turns.length; j++) {
        const turn: SpeakerTurn = this.turns[j];
        if (turn.clusterId !== clusterId) {
          continue;
        }
        turnCount += 1;
        speechMs += Math.max(0, turn.endMs - turn.startMs);
        if (turn.displayName && turn.displayName.length > 0) {
          displayName = turn.displayName;
        }
        if (turn.personId) {
          personId = turn.personId;
        }
        automatic = automatic || turn.identity === 'automatic';
      }
      const row: SpeakerClusterRow = {
        clusterId: clusterId,
        displayName: displayName,
        personId: personId,
        identityText: personId ? (automatic ? '自动识别' : '已确认') : '未知',
        turnCount: turnCount,
        speechMs: speechMs,
      };
      rows.push(row);
    }
    this.clusterRows = rows;
  }

  private selectCluster(row: SpeakerClusterRow): void {
    this.selectedClusterId = row.clusterId;
    const defaultUnknown: boolean = row.displayName === '未知说话人' ||
      row.displayName === `未知说话人 ${row.clusterId}`;
    this.editName = row.personId || defaultUnknown ? '' :
      row.displayName.replace(` ${row.clusterId}`, '');
    this.selectedPersonId = row.personId || '';
    this.statusMessage = `正在编辑 ${row.clusterId}`;
  }

  private toggleEnrollmentCluster(): void {
    if (this.selectedClusterId.length === 0) {
      return;
    }
    const selected: string[] = this.enrollmentClusterIds.slice();
    const index: number = selected.indexOf(this.selectedClusterId);
    if (index >= 0) {
      selected.splice(index, 1);
    } else {
      selected.push(this.selectedClusterId);
    }
    this.enrollmentClusterIds = selected;
    this.statusMessage = selected.length > 0
      ? `已选择 ${selected.length} 组片段用于声纹确认`
      : '已清空声纹选段';
  }

  private enrollmentTargets(): string[] {
    if (this.enrollmentClusterIds.length > 0) {
      return this.enrollmentClusterIds.slice();
    }
    return this.selectedClusterId.length > 0 ? [this.selectedClusterId] : [];
  }

  private selectedEnrollmentSpeechMs(): number {
    const ids: string[] = this.enrollmentTargets();
    let total: number = 0;
    for (let i: number = 0; i < this.candidates.length; i++) {
      const candidate: SpeakerCandidate = this.candidates[i];
      if (ids.indexOf(candidate.clusterId) >= 0 && candidate.queryEmbedding &&
        candidate.queryEmbedding.length === SPEAKER_EMBEDDING_DIM) {
        total += candidate.speechMs;
      }
    }
    return total;
  }

  private canRememberSelected(): boolean {
    return this.selectedEnrollmentSpeechMs() >= SPEAKER_MEETING_ENROLL_MIN_SPEECH_MS;
  }

  private async saveAssignment(assignment: MeetingSpeakerAssignment): Promise<void> {
    if (!this.meetingStore) {
      throw new Error('meeting speaker store is not ready');
    }
    await this.meetingStore.assign(assignment);
    const assignments: MeetingSpeakerAssignment[] = await this.meetingStore.assignmentsFor(this.meetingId);
    this.profiles = this.profileStore.list();
    this.turns = this.meetingStore.apply(this.turns, assignments, this.profiles);
    this.rebuildRows();
    this.enrollmentClusterIds = [];
  }

  private async assignClusters(clusterIds: string[], personId?: string,
    displayNameOverride?: string): Promise<void> {
    if (!this.meetingStore) {
      throw new Error('meeting speaker store is not ready');
    }
    for (let i: number = 0; i < clusterIds.length; i++) {
      const assignment: MeetingSpeakerAssignment = {
        meetingId: this.meetingId,
        clusterId: clusterIds[i],
        personId: personId,
        displayNameOverride: displayNameOverride,
        source: 'user',
        updatedAt: Date.now(),
      };
      await this.meetingStore.assign(assignment);
    }
    const assignments: MeetingSpeakerAssignment[] =
      await this.meetingStore.assignmentsFor(this.meetingId);
    this.profiles = this.profileStore.list();
    this.turns = this.meetingStore.apply(this.turns, assignments, this.profiles);
    this.rebuildRows();
    this.enrollmentClusterIds = [];
  }

  private renameForMeeting(): void {
    const name: string = this.editName.trim();
    if (this.selectedClusterId.length === 0 || name.length === 0) {
      return;
    }
    const assignment: MeetingSpeakerAssignment = {
      meetingId: this.meetingId,
      clusterId: this.selectedClusterId,
      displayNameOverride: name,
      source: 'user',
      updatedAt: Date.now(),
    };
    this.saveAssignment(assignment)
      .then((): void => {
        this.statusMessage = `本次会议已改为“${name}”，不会修改声纹`;
      })
      .catch((error: Error): void => {
        this.statusMessage = `保存失败: ${error.message}`;
      });
  }

  private linkExisting(rememberVoice: boolean): void {
    if (this.selectedClusterId.length === 0 || this.selectedPersonId.length === 0) {
      return;
    }
    const profile: SpeakerProfile | undefined = this.profileStore.find(this.selectedPersonId);
    if (!profile) {
      this.statusMessage = '选择的人物不存在';
      return;
    }
    const proceed = async (): Promise<void> => {
      const targetClusters: string[] = rememberVoice
        ? this.enrollmentTargets() : [this.selectedClusterId];
      if (rememberVoice) {
        const prototype: VoicePrototype = this.prototypeFromSelected();
        await this.profileStore.saveMeetingEnrollment(
          profile.displayName, prototype, profile.personId);
      }
      await this.assignClusters(targetClusters, profile.personId);
    };
    proceed()
      .then((): void => {
        this.statusMessage = rememberVoice
          ? `已确认并更新 ${profile.displayName} 的声纹`
          : `本次会议已关联到 ${profile.displayName}，声纹未改变`;
      })
      .catch((error: Error): void => {
        this.statusMessage = `操作失败: ${error.message}`;
      });
  }

  private createAndRemember(): void {
    const name: string = this.editName.trim();
    if (name.length === 0 || this.selectedClusterId.length === 0) {
      return;
    }
    let prototype: VoicePrototype;
    try {
      prototype = this.prototypeFromSelected();
    } catch (error) {
      this.statusMessage = `${error}`.replace('Error: ', '');
      return;
    }
    this.profileStore.saveMeetingEnrollment(name, prototype)
      .then((profile: SpeakerProfile): Promise<void> => {
        return this.assignClusters(this.enrollmentTargets(), profile.personId);
      })
      .then((): void => {
        this.statusMessage = `已新建并记住 ${name}，后续会议可自动识别`;
      })
      .catch((error: Error): void => {
        this.statusMessage = `建档失败: ${error.message}`;
      });
  }

  private prototypeFromSelected(): VoicePrototype {
    const clusterIds: string[] = this.enrollmentTargets();
    const embeddings: number[][] = [];
    let speechMs: number = 0;
    let weightedQuality: number = 0;
    for (let i: number = 0; i < this.candidates.length; i++) {
      const candidate: SpeakerCandidate = this.candidates[i];
      if (clusterIds.indexOf(candidate.clusterId) < 0 ||
        !candidate.queryEmbedding ||
        candidate.queryEmbedding.length !== SPEAKER_EMBEDDING_DIM) {
        continue;
      }
      embeddings.push(candidate.queryEmbedding);
      speechMs += candidate.speechMs;
      weightedQuality += candidate.qualityScore * candidate.speechMs;
    }
    if (embeddings.length === 0) {
      throw new Error('所选片段没有质量合格的声纹样本');
    }
    if (speechMs < SPEAKER_MEETING_ENROLL_MIN_SPEECH_MS) {
      throw new Error(`至少需要 8 秒有效语音，当前 ${(speechMs / 1000).toFixed(1)} 秒`);
    }
    for (let i: number = 0; i < embeddings.length; i++) {
      for (let j: number = i + 1; j < embeddings.length; j++) {
        if (cosineSimilarity(embeddings[i], embeddings[j]) < SPEAKER_IDENTITY_THRESHOLD) {
          throw new Error('所选片段不像同一个人，请取消可疑片段后重试');
        }
      }
    }
    const prototype: VoicePrototype = createVoicePrototype(
      averageEmbeddings(embeddings),
      speechMs,
      weightedQuality / Math.max(1, speechMs),
      'meetingLabel');
    prototype.evidence = {
      meetingId: this.meetingId,
      segmentId: clusterIds.join(','),
      startMs: this.clustersStartMs(clusterIds),
      endMs: this.clustersEndMs(clusterIds),
    };
    return prototype;
  }

  private markUnknown(): void {
    if (this.selectedClusterId.length === 0) {
      return;
    }
    const assignment: MeetingSpeakerAssignment = {
      meetingId: this.meetingId,
      clusterId: this.selectedClusterId,
      displayNameOverride: '未知说话人',
      source: 'user',
      updatedAt: Date.now(),
    };
    this.saveAssignment(assignment)
      .then((): void => {
        this.selectedPersonId = '';
        this.statusMessage = '已撤销本次自动身份，保留为未知说话人';
      })
      .catch((error: Error): void => {
        this.statusMessage = `保存失败: ${error.message}`;
      });
  }

  private clusterStartMs(clusterId: string): number {
    let start: number = Number.MAX_VALUE;
    for (let i: number = 0; i < this.turns.length; i++) {
      if (this.turns[i].clusterId === clusterId) {
        start = Math.min(start, this.turns[i].startMs);
      }
    }
    return start === Number.MAX_VALUE ? 0 : start;
  }

  private clustersStartMs(clusterIds: string[]): number {
    let start: number = Number.MAX_VALUE;
    for (let i: number = 0; i < clusterIds.length; i++) {
      start = Math.min(start, this.clusterStartMs(clusterIds[i]));
    }
    return start === Number.MAX_VALUE ? 0 : start;
  }

  private clusterEndMs(clusterId: string): number {
    let end: number = 0;
    for (let i: number = 0; i < this.turns.length; i++) {
      if (this.turns[i].clusterId === clusterId) {
        end = Math.max(end, this.turns[i].endMs);
      }
    }
    return end;
  }

  private clustersEndMs(clusterIds: string[]): number {
    let end: number = 0;
    for (let i: number = 0; i < clusterIds.length; i++) {
      end = Math.max(end, this.clusterEndMs(clusterIds[i]));
    }
    return end;
  }

  @Builder
  private ClusterCard(row: SpeakerClusterRow) {
    Column() {
      Row() {
        Column() {
          Text(row.displayName)
            .fontSize(16)
            .fontWeight(FontWeight.Medium)
            .fontColor('#1F2937')
          Text(`${row.clusterId} · ${row.identityText}`)
            .fontSize(12)
            .fontColor('#6B7280')
            .margin({ top: 3 })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
        Text(`${row.turnCount} 段 · ${(row.speechMs / 1000).toFixed(1)} 秒`)
          .fontSize(12)
          .fontColor('#6B7280')
      }
    }
    .width('100%')
    .padding(13)
    .backgroundColor(this.selectedClusterId === row.clusterId ? '#EAF2FF' :
      this.enrollmentClusterIds.indexOf(row.clusterId) >= 0 ? '#F5F3FF' : '#FFFFFF')
    .borderRadius(10)
    .margin({ bottom: 8 })
    .onClick((): void => {
      this.selectCluster(row);
    })
  }

  build() {
    Column() {
      Row() {
        Button('返回')
          .fontSize(13)
          .height(34)
          .backgroundColor('#F3F4F6')
          .fontColor('#374151')
          .onClick((): void => {
            router.back();
          })
        Column() {
          Text('编辑说话人')
            .fontSize(20)
            .fontWeight(FontWeight.Bold)
            .fontColor('#111827')
          Text(this.title)
            .fontSize(12)
            .fontColor('#6B7280')
        }
        .alignItems(HorizontalAlign.Start)
        .margin({ left: 12 })
        Blank()
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 16, bottom: 10 })

      Text(this.statusMessage)
        .fontSize(13)
        .fontColor('#6B7280')
        .width('100%')
        .padding({ left: 18, right: 18, bottom: 8 })

      List() {
        ForEach(this.clusterRows, (row: SpeakerClusterRow): void => {
          ListItem() {
            this.ClusterCard(row)
          }
        }, (row: SpeakerClusterRow): string => row.clusterId)
      }
      .layoutWeight(1)
      .width('100%')
      .padding({ left: 16, right: 16 })

      if (this.selectedClusterId.length > 0) {
        Column() {
          Text(`编辑 ${this.selectedClusterId}`)
            .fontSize(15)
            .fontWeight(FontWeight.Medium)
            .fontColor('#1F2937')
            .width('100%')

          TextInput({ text: this.editName, placeholder: '输入本次显示名或新人物姓名' })
            .height(40)
            .backgroundColor('#FFFFFF')
            .onChange((value: string): void => {
              this.editName = value;
            })
            .margin({ top: 8 })

          Row() {
            Text(`声纹选段 ${this.enrollmentTargets().length} 组 · ` +
              `${(this.selectedEnrollmentSpeechMs() / 1000).toFixed(1)} 秒`)
              .fontSize(12)
              .fontColor('#6B7280')
              .layoutWeight(1)
            Button(this.enrollmentClusterIds.indexOf(this.selectedClusterId) >= 0
              ? '移出声纹选段' : '加入声纹选段')
              .fontSize(11)
              .height(30)
              .backgroundColor('#F2EEFF')
              .fontColor('#5B3FB5')
              .onClick((): void => {
                this.toggleEnrollmentCluster();
              })
          }
          .width('100%')
          .margin({ top: 6 })

          Row() {
            Button('仅本次改名')
              .layoutWeight(1)
              .height(38)
              .fontSize(13)
              .backgroundColor('#F3F4F6')
              .fontColor('#374151')
              .enabled(this.editName.trim().length > 0)
              .onClick((): void => {
                this.renameForMeeting();
              })
            Button('新建并记住')
              .layoutWeight(1)
              .height(38)
              .fontSize(13)
              .backgroundColor('#0066CC')
              .fontColor('#FFFFFF')
              .enabled(this.editName.trim().length > 0 && this.canRememberSelected())
              .margin({ left: 8 })
              .onClick((): void => {
                this.createAndRemember();
              })
          }
          .width('100%')
          .margin({ top: 8 })

          if (this.profiles.length > 0) {
            Text('关联已有人物')
              .fontSize(12)
              .fontColor('#6B7280')
              .width('100%')
              .margin({ top: 10, bottom: 5 })
            Scroll() {
              Row() {
                ForEach(this.profiles, (profile: SpeakerProfile): void => {
                  Button(profile.displayName)
                    .fontSize(12)
                    .height(32)
                    .backgroundColor(this.selectedPersonId === profile.personId ? '#DDD6FE' : '#FFFFFF')
                    .fontColor('#4C1D95')
                    .margin({ right: 6 })
                    .onClick((): void => {
                      this.selectedPersonId = profile.personId;
                    })
                }, (profile: SpeakerProfile): string => profile.personId)
              }
            }
            .scrollable(ScrollDirection.Horizontal)
            .width('100%')

            Row() {
              Button('仅关联本次')
                .layoutWeight(1)
                .height(38)
                .fontSize(13)
                .backgroundColor('#F3F4F6')
                .fontColor('#374151')
                .enabled(this.selectedPersonId.length > 0)
                .onClick((): void => {
                  this.linkExisting(false);
                })
              Button('确认并更新声纹')
                .layoutWeight(1)
                .height(38)
                .fontSize(13)
                .backgroundColor('#5B3FB5')
                .fontColor('#FFFFFF')
                .enabled(this.selectedPersonId.length > 0 && this.canRememberSelected())
                .margin({ left: 8 })
                .onClick((): void => {
                  this.linkExisting(true);
                })
            }
            .width('100%')
            .margin({ top: 7 })
          }

          Button('撤销身份，标为未知说话人')
            .width('100%')
            .height(36)
            .fontSize(12)
            .backgroundColor('#FEF2F2')
            .fontColor('#B91C1C')
            .margin({ top: 9 })
            .onClick((): void => {
              this.markUnknown();
            })

          Text('“仅本次”不会改变声纹；只有明确确认“记住/更新声纹”才影响后续会议。')
            .fontSize(11)
            .fontColor('#6B7280')
            .margin({ top: 7 })

          if (!this.canRememberSelected()) {
            Text('记住人物需要累计至少 8 秒质量合格的语音；可在上方继续加入片段。')
              .fontSize(11)
              .fontColor('#B45309')
              .margin({ top: 4 })
          }
        }
        .width('100%')
        .padding(14)
        .backgroundColor('#F8FAFC')
        .borderRadius({ topLeft: 16, topRight: 16 })
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#F7F8FA')
  }
}
