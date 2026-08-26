import { common } from '@kit.AbilityKit';
import { router } from '@kit.ArkUI';
import { ErrorEvent, MessageEvents, util, worker } from '@kit.ArkTS';
import { hilog } from '@kit.PerformanceAnalysisKit';
import {
  buildAishell3IdentityEnrollments,
  buildAishell3IdentityQueries,
  SpeakerIdentityEnrollmentSpec,
  SpeakerIdentityQuerySpec,
} from '../common/SpeakerIdentityTestData';
import { SPEAKER_IDENTITY_THRESHOLD, SpeakerSearchProfile } from '../common/SpeakerTypes';
import { RawPlaybackState, RawResourcePlayer } from '../services/audio/RawResourcePlayer';
import {
  buildTestSearchProfile,
  scoreSpeakerIdentityCase,
  SpeakerIdentityScoredCase,
  SpeakerIdentitySummary,
  summarizeSpeakerIdentityCases,
} from '../services/speaker/SpeakerIdentityEvaluation';

const LOG_DOMAIN: number = 0x0001;
const LOG_TAG: string = 'SpeakerIdentityTest';

interface IdentityWorkerRequest {
  msgType: string;
  context?: common.UIAbilityContext;
  data?: Uint8Array;
  audioFormat?: string;
  requestId?: string;
}

interface IdentityWorkerResponse {
  msgType: string;
  message?: string;
  requestId?: string;
  embedding?: number[];
  audioDuration?: number;
}

interface IdentityTestRow {
  id: string;
  state: string;
  truthSpeakerId: string;
  expectedName: string;
  predictedName: string;
  score: number;
  correct: boolean;
  elapsedMs: number;
  error: string;
}

@Entry
@Component
struct SpeakerIdentityTestCenterPage {
  @State ready: boolean = false;
  @State running: boolean = false;
  @State statusMessage: string = '正在加载 CAM++ 声纹模型...';
  @State progressText: string = '0 / 40';
  @State summaryText: string = '尚未运行有真值身份测试';
  @State knownMetric: string = '—';
  @State unknownMetric: string = '—';
  @State balancedMetric: string = '—';
  @State playingId: string = '';
  @State playPositionMs: number = 0;
  @State playDurationMs: number = 0;
  @State playbackBusy: boolean = false;
  @State playbackError: string = '';
  @State rows: IdentityTestRow[] = [];

  private workerInstance: worker.ThreadWorker | undefined = undefined;
  private audioPlayer: RawResourcePlayer = new RawResourcePlayer();
  private enrollments: SpeakerIdentityEnrollmentSpec[] = buildAishell3IdentityEnrollments();
  private queries: SpeakerIdentityQuerySpec[] = buildAishell3IdentityQueries();
  private testProfiles: SpeakerSearchProfile[] = [];
  private scoredCases: SpeakerIdentityScoredCase[] = [];
  private phase: string = 'idle';
  private enrollmentIndex: number = 0;
  private queryIndex: number = 0;
  private sampleIndex: number = 0;
  private currentEmbeddings: number[][] = [];
  private currentAudioSeconds: number = 0;
  private currentStartedAt: number = 0;
  private pendingRequestId: string = '';
  private runToken: number = 0;

  aboutToAppear(): void {
    this.rows = this.emptyRows();
    this.audioPlayer.setCallback((state: RawPlaybackState): void => {
      this.playingId = state.audioId;
      this.playPositionMs = state.positionMs;
      this.playDurationMs = state.durationMs;
      if (state.errorMessage.length > 0) {
        this.playbackError = `播放失败：${state.errorMessage}`;
      }
    });
    const context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;
    const instance: worker.ThreadWorker = new worker.ThreadWorker(
      'entry/ets/workers/StreamingSherpaWorker.ets', { name: 'Labeled speaker identity test' });
    instance.onmessage = (event: MessageEvents): void => {
      this.onWorkerMessage(event);
    };
    instance.onerror = (event: ErrorEvent): void => {
      this.abortRun(`测试引擎异常：${event.message}`);
    };
    instance.onmessageerror = (_event: MessageEvents): void => {
      this.abortRun('测试引擎消息解析失败');
    };
    this.workerInstance = instance;
    instance.postMessage({ msgType: 'sherpa-speaker-test-init', context: context } as IdentityWorkerRequest);
  }

  aboutToDisappear(): void {
    this.runToken += 1;
    this.running = false;
    this.phase = 'idle';
    this.testProfiles = [];
    this.currentEmbeddings = [];
    this.audioPlayer.setCallback(null);
    this.audioPlayer.stop().catch((error: Error): void => {
      hilog.warn(LOG_DOMAIN, LOG_TAG, `stop playback failed: ${error.message}`);
    });
    if (this.workerInstance) {
      this.workerInstance.terminate();
      this.workerInstance = undefined;
    }
  }

  private emptyRows(): IdentityTestRow[] {
    const result: IdentityTestRow[] = [];
    for (let i: number = 0; i < this.queries.length; i++) {
      result.push({
        id: this.queries[i].id,
        state: '等待',
        truthSpeakerId: this.queries[i].truthSpeakerId,
        expectedName: this.queries[i].expectedDisplayName,
        predictedName: '',
        score: -1,
        correct: false,
        elapsedMs: 0,
        error: '',
      });
    }
    return result;
  }

  private onWorkerMessage(event: MessageEvents): void {
    const response: IdentityWorkerResponse = event.data as IdentityWorkerResponse;
    if (response.msgType === 'sherpa-speaker-test-ready') {
      this.ready = true;
      this.statusMessage = '模型已就绪，可运行 AISHELL-3 已知/未知身份测试';
      return;
    }
    if (response.msgType === 'sherpa-speaker-embedding-test-done') {
      if (!this.running || !response.requestId || response.requestId !== this.pendingRequestId) {
        return;
      }
      this.acceptEmbedding(response);
      return;
    }
    if (response.msgType === 'sherpa-speaker-embedding-test-failed') {
      if (!this.running || !response.requestId || response.requestId !== this.pendingRequestId) {
        return;
      }
      this.handleEmbeddingFailure(response.message || '声纹提取失败');
      return;
    }
    if (response.msgType === 'sherpa-asr-error') {
      this.abortRun(`测试引擎错误：${response.message || '未知错误'}`);
    }
  }

  private startTest(): void {
    if (!this.ready || this.running || this.playbackBusy || this.playingId.length > 0) {
      return;
    }
    this.runToken += 1;
    this.rows = this.emptyRows();
    this.testProfiles = [];
    this.scoredCases = [];
    this.enrollmentIndex = 0;
    this.queryIndex = 0;
    this.sampleIndex = 0;
    this.currentEmbeddings = [];
    this.currentAudioSeconds = 0;
    this.playbackError = '';
    this.knownMetric = '—';
    this.unknownMetric = '—';
    this.balancedMetric = '—';
    this.summaryText = '正在建立隔离测试声纹库，不会读取或修改个人声纹';
    this.running = true;
    this.phase = 'enrollment';
    this.requestCurrentSample();
  }

  private currentSamplePath(): string {
    if (this.phase === 'enrollment') {
      return this.enrollments[this.enrollmentIndex].samplePaths[this.sampleIndex];
    }
    return this.queries[this.queryIndex].samplePaths[this.sampleIndex];
  }

  private requestCurrentSample(): void {
    if (!this.running || !this.workerInstance) {
      return;
    }
    const token: number = this.runToken;
    const path: string = this.currentSamplePath();
    if (this.phase === 'enrollment') {
      this.progressText = `注册 ${this.enrollmentIndex + 1} / ${this.enrollments.length}`;
      this.statusMessage = `正在建立 ${this.enrollments[this.enrollmentIndex].displayName} · ` +
        `样本 ${this.sampleIndex + 1} / 5`;
    } else {
      this.progressText = `${this.queryIndex + 1} / ${this.queries.length}`;
      this.statusMessage = `正在识别 ${this.queries[this.queryIndex].truthSpeakerId} · ` +
        `样本 ${this.sampleIndex + 1} / 3`;
      if (this.sampleIndex === 0) {
        const updated: IdentityTestRow[] = this.rows.slice();
        updated[this.queryIndex].state = '运行中';
        this.rows = updated;
        this.currentStartedAt = Date.now();
      }
    }
    const context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;
    context.resourceManager.getRawFileContent(path)
      .then((data: Uint8Array): void => {
        if (!this.running || token !== this.runToken || !this.workerInstance) {
          return;
        }
        const requestId: string = util.generateRandomUUID(true);
        this.pendingRequestId = requestId;
        const request: IdentityWorkerRequest = {
          msgType: 'sherpa-speaker-embedding-test',
          requestId: requestId,
          data: data,
          audioFormat: 'wav',
        };
        this.workerInstance.postMessage(request, [data.buffer]);
      })
      .catch((error: Error): void => {
        if (this.running && token === this.runToken) {
          this.handleEmbeddingFailure(`读取 ${path} 失败：${error.message}`);
        }
      });
  }

  private acceptEmbedding(response: IdentityWorkerResponse): void {
    const embedding: number[] | undefined = response.embedding;
    if (!embedding || embedding.length === 0) {
      this.handleEmbeddingFailure('模型返回了空声纹');
      return;
    }
    this.currentEmbeddings.push(embedding);
    this.currentAudioSeconds += response.audioDuration || 0;
    this.sampleIndex += 1;
    const expectedSamples: number = this.phase === 'enrollment' ? 5 : 3;
    if (this.sampleIndex < expectedSamples) {
      this.requestCurrentSample();
      return;
    }
    if (this.phase === 'enrollment') {
      this.finishEnrollmentSpeaker();
    } else {
      this.finishQueryCase();
    }
  }

  private finishEnrollmentSpeaker(): void {
    const spec: SpeakerIdentityEnrollmentSpec = this.enrollments[this.enrollmentIndex];
    try {
      this.testProfiles.push(buildTestSearchProfile(
        spec.personId, spec.displayName, this.currentEmbeddings));
    } catch (error) {
      this.abortRun(`${spec.displayName} 建档失败：${error}`.replace('Error: ', ''));
      return;
    }
    this.enrollmentIndex += 1;
    this.sampleIndex = 0;
    this.currentEmbeddings = [];
    this.currentAudioSeconds = 0;
    if (this.enrollmentIndex < this.enrollments.length) {
      this.requestCurrentSample();
      return;
    }
    this.phase = 'query';
    this.queryIndex = 0;
    this.progressText = `0 / ${this.queries.length}`;
    this.summaryText = `已建立 ${this.testProfiles.length} 位 AISHELL 测试人物，开始开放集识别`;
    this.requestCurrentSample();
  }

  private finishQueryCase(): void {
    const spec: SpeakerIdentityQuerySpec = this.queries[this.queryIndex];
    try {
      const scored: SpeakerIdentityScoredCase = scoreSpeakerIdentityCase(
        spec.expectedPersonId, this.currentEmbeddings, this.testProfiles,
        SPEAKER_IDENTITY_THRESHOLD);
      this.scoredCases.push(scored);
      const result: IdentityTestRow = {
        id: spec.id,
        state: scored.correct ? '通过' : '未通过',
        truthSpeakerId: spec.truthSpeakerId,
        expectedName: spec.expectedDisplayName,
        predictedName: scored.accepted ? this.profileName(scored.predictedPersonId) : '未知说话人',
        score: scored.topScore,
        correct: scored.correct,
        elapsedMs: Date.now() - this.currentStartedAt,
        error: '',
      };
      const updated: IdentityTestRow[] = this.rows.slice();
      updated[this.queryIndex] = result;
      this.rows = updated;
      hilog.info(LOG_DOMAIN, LOG_TAG,
        `RESULT id=${result.id} truth=${result.truthSpeakerId} expected=${result.expectedName} ` +
        `predicted=${result.predictedName} score=${result.score.toFixed(4)} correct=${result.correct}`);
    } catch (error) {
      this.markQueryFailure(`${error}`.replace('Error: ', ''));
      return;
    }
    this.advanceQuery();
  }

  private profileName(personId?: string): string {
    if (!personId) {
      return '未知说话人';
    }
    for (let i: number = 0; i < this.testProfiles.length; i++) {
      if (this.testProfiles[i].personId === personId) {
        return this.testProfiles[i].displayName;
      }
    }
    return personId;
  }

  private handleEmbeddingFailure(message: string): void {
    if (this.phase === 'enrollment') {
      const label: string = this.enrollmentIndex < this.enrollments.length ?
        this.enrollments[this.enrollmentIndex].displayName : '测试人物';
      this.abortRun(`${label} 声纹提取失败：${message}`);
      return;
    }
    this.markQueryFailure(message);
  }

  private markQueryFailure(message: string): void {
    if (this.queryIndex >= 0 && this.queryIndex < this.rows.length) {
      const updated: IdentityTestRow[] = this.rows.slice();
      updated[this.queryIndex].state = '失败';
      updated[this.queryIndex].error = message;
      updated[this.queryIndex].elapsedMs = Date.now() - this.currentStartedAt;
      this.rows = updated;
      hilog.error(LOG_DOMAIN, LOG_TAG,
        `RESULT id=${updated[this.queryIndex].id} state=failed error=${message}`);
    }
    this.advanceQuery();
  }

  private advanceQuery(): void {
    this.queryIndex += 1;
    this.sampleIndex = 0;
    this.currentEmbeddings = [];
    this.currentAudioSeconds = 0;
    if (this.queryIndex < this.queries.length) {
      this.requestCurrentSample();
      return;
    }
    this.finishTest();
  }

  private finishTest(): void {
    const summary: SpeakerIdentitySummary = summarizeSpeakerIdentityCases(this.scoredCases);
    let failed: number = 0;
    for (let i: number = 0; i < this.rows.length; i++) {
      if (this.rows[i].state === '失败') {
        failed += 1;
      }
    }
    this.knownMetric = `${summary.knownCorrect} / ${summary.knownTotal} ` +
      `(${(summary.knownCorrectRate * 100).toFixed(1)}%)`;
    this.unknownMetric = `${summary.unknownRejected} / ${summary.unknownTotal} ` +
      `(${(summary.unknownRejectRate * 100).toFixed(1)}%)`;
    this.balancedMetric = `${(summary.balancedAccuracy * 100).toFixed(1)}%`;
    this.summaryText = `已知识别错误 ${summary.knownMisidentified}，拒识 ${summary.knownFalseReject}；` +
      `未知误接纳 ${summary.unknownFalseAccept}；处理失败 ${failed}`;
    this.progressText = `${this.queries.length} / ${this.queries.length}`;
    this.statusMessage = 'AISHELL-3 有真值身份测试完成，隔离测试声纹未写入个人资料';
    this.running = false;
    this.phase = 'idle';
    this.pendingRequestId = '';
    this.testProfiles = [];
    this.scoredCases = [];
    hilog.info(LOG_DOMAIN, LOG_TAG,
      `SUMMARY knownCorrect=${summary.knownCorrect}/${summary.knownTotal} ` +
      `unknownRejected=${summary.unknownRejected}/${summary.unknownTotal} ` +
      `balanced=${summary.balancedAccuracy.toFixed(4)} failed=${failed}`);
  }

  private abortRun(message: string): void {
    this.runToken += 1;
    this.running = false;
    this.phase = 'idle';
    this.pendingRequestId = '';
    this.testProfiles = [];
    this.currentEmbeddings = [];
    this.statusMessage = message;
    hilog.error(LOG_DOMAIN, LOG_TAG, message);
  }

  private togglePlayback(spec: SpeakerIdentityQuerySpec): void {
    if (this.running || this.playbackBusy) {
      return;
    }
    this.playbackError = '';
    this.playbackBusy = true;
    const context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;
    this.audioPlayer.toggle(spec.id, spec.playbackPath, context.resourceManager)
      .then((): void => {
        this.playbackBusy = false;
      })
      .catch((error: Error): void => {
        this.playbackBusy = false;
        this.playbackError = `播放失败：${error.message}`;
      });
  }

  private formatPlaybackTime(positionMs: number, durationMs: number): string {
    const format = (valueMs: number): string => {
      const totalSeconds: number = Math.max(0, Math.floor(valueMs / 1000));
      const minutes: number = Math.floor(totalSeconds / 60);
      const seconds: number = totalSeconds % 60;
      return `${minutes}:${seconds < 10 ? `0${seconds}` : `${seconds}`}`;
    };
    return `${format(positionMs)} / ${format(durationMs)}`;
  }

  @Builder
  private MetricCard(label: string, value: string, color: string) {
    Column() {
      Text(value)
        .fontSize(18)
        .fontWeight(FontWeight.Bold)
        .fontColor(color)
      Text(label)
        .fontSize(10)
        .fontColor('#6B7280')
        .margin({ top: 3 })
    }
    .layoutWeight(1)
    .padding({ top: 10, bottom: 10 })
    .backgroundColor('#FFFFFF')
    .borderRadius(9)
  }

  @Builder
  private ResultRow(spec: SpeakerIdentityQuerySpec, result: IdentityTestRow) {
    Column() {
      Row() {
        Column() {
          Text(`${result.truthSpeakerId} · ${spec.id}`)
            .fontSize(14)
            .fontWeight(FontWeight.Medium)
            .fontColor('#1F2937')
          Text(`真值：${result.expectedName}`)
            .fontSize(11)
            .fontColor('#6B7280')
            .margin({ top: 3 })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
        Text(result.state)
          .fontSize(12)
          .fontColor(result.state === '通过' ? '#047857' :
            result.state === '等待' || result.state === '运行中' ? '#6B7280' : '#B91C1C')
      }
      .width('100%')

      if (result.state === '通过' || result.state === '未通过') {
        Text(`预测：${result.predictedName} · 最高相似度 ${result.score.toFixed(4)} · ` +
          `${result.elapsedMs} ms`)
          .fontSize(11)
          .fontColor(result.correct ? '#047857' : '#B91C1C')
          .width('100%')
          .margin({ top: 7 })
      } else if (result.error.length > 0) {
        Text(result.error)
          .fontSize(11)
          .fontColor('#B91C1C')
          .width('100%')
          .margin({ top: 7 })
      }

      Row() {
        Text(this.playingId === spec.id ?
          '正在播放本组3条测试语音' : '试听参与本次评分的3条原始语音')
          .fontSize(11)
          .fontColor(this.playingId === spec.id ? '#047857' : '#6B7280')
          .layoutWeight(1)
        Button(this.playingId === spec.id ?
          `■ 停止 ${this.formatPlaybackTime(this.playPositionMs, this.playDurationMs)}` : '▶ 播放')
          .fontSize(11)
          .height(30)
          .backgroundColor(this.playingId === spec.id ? '#047857' : '#EAF2FF')
          .fontColor(this.playingId === spec.id ? '#FFFFFF' : '#0066CC')
          .enabled(!this.running && !this.playbackBusy)
          .onClick((): void => {
            this.togglePlayback(spec);
          })
      }
      .width('100%')
      .margin({ top: 8 })
    }
    .width('100%')
    .padding(12)
    .backgroundColor('#FFFFFF')
    .borderRadius(10)
    .margin({ bottom: 8 })
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
          Text('声纹身份准确率')
            .fontSize(20)
            .fontWeight(FontWeight.Bold)
          Text(this.statusMessage)
            .fontSize(11)
            .fontColor('#6B7280')
            .maxLines(2)
          if (this.playbackError.length > 0) {
            Text(this.playbackError)
              .fontSize(10)
              .fontColor('#B91C1C')
          }
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
        .margin({ left: 12 })
        Text(this.progressText)
          .fontSize(13)
          .fontColor('#0066CC')
      }
      .width('100%')
      .padding(16)

      Column() {
        Text('AISHELL-3 标注真值 · 10 名已知 + 10 名未知')
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .fontColor('#4C1D95')
        Text('每名已知人物使用5条语音建档；每次测试聚合另外3条语音，注册集与测试集完全分离。')
          .fontSize(11)
          .fontColor('#6D28D9')
          .margin({ top: 4 })
        Text('测试使用隔离的内存声纹库，不读取、不覆盖、不写入你的个人声纹资料。')
          .fontSize(11)
          .fontColor('#6D28D9')
          .margin({ top: 3 })
        Row({ space: 8 }) {
          this.MetricCard('已知识别', this.knownMetric, '#047857')
          this.MetricCard('未知拒识', this.unknownMetric, '#B45309')
          this.MetricCard('综合准确率', this.balancedMetric, '#0066CC')
        }
        .width('100%')
        .margin({ top: 10 })
        Text(this.summaryText)
          .fontSize(11)
          .fontColor('#4B5563')
          .margin({ top: 7 })
      }
      .alignItems(HorizontalAlign.Start)
      .width('100%')
      .padding(13)
      .backgroundColor('#F5F3FF')
      .borderRadius(12)
      .margin({ left: 16, right: 16, bottom: 10 })

      List() {
        ForEach(this.rows, (result: IdentityTestRow, index: number): void => {
          ListItem() {
            this.ResultRow(this.queries[index], result)
          }
        }, (result: IdentityTestRow): string =>
          `${result.id}-${result.state}-${result.elapsedMs}`)
      }
      .layoutWeight(1)
      .width('100%')
      .padding({ left: 16, right: 16 })

      Button(this.running ? `正在运行 ${this.progressText}` : '运行 AISHELL-3 身份测试')
        .width('80%')
        .height(48)
        .backgroundColor('#0066CC')
        .fontColor('#FFFFFF')
        .enabled(this.ready && !this.running && !this.playbackBusy && this.playingId.length === 0)
        .onClick((): void => {
          this.startTest();
        })
        .margin({ top: 8, bottom: 20 })
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#F7F8FA')
  }
}
