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
import {
  buildIdentityDecisionText,
  buildIdentityOutcomeLabel,
  IdentityResultFilter,
  identityResultMatchesFilter,
} from '../common/SpeakerIdentityPresentation';
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
  accepted: boolean;
  score: number;
  secondScore: number;
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
  @State resultFilter: IdentityResultFilter = 'all';
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
        accepted: false,
        score: -1,
        secondScore: -1,
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
        accepted: scored.accepted,
        score: scored.topScore,
        secondScore: scored.secondScore,
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

  private isKnownTruth(spec: SpeakerIdentityQuerySpec): boolean {
    return spec.expectedPersonId !== undefined;
  }

  private hasModelResult(result: IdentityTestRow): boolean {
    return result.state === '通过' || result.state === '未通过';
  }

  private outcomeLabel(spec: SpeakerIdentityQuerySpec, result: IdentityTestRow): string {
    return buildIdentityOutcomeLabel({
      state: result.state,
      accepted: result.accepted,
      correct: result.correct,
      knownTruth: this.isKnownTruth(spec),
    });
  }

  private outcomeColor(result: IdentityTestRow): string {
    if (result.state === '通过') {
      return '#137A4C';
    }
    if (result.state === '未通过' || result.state === '失败') {
      return '#B42318';
    }
    if (result.state === '运行中') {
      return '#246BFD';
    }
    return '#667085';
  }

  private outcomeSoftColor(result: IdentityTestRow): string {
    if (result.state === '通过') {
      return '#E9F8F0';
    }
    if (result.state === '未通过' || result.state === '失败') {
      return '#FFF0EE';
    }
    if (result.state === '运行中') {
      return '#EAF1FF';
    }
    return '#F2F4F7';
  }

  private evidenceBorderColor(result: IdentityTestRow): string {
    if (result.state === '通过') {
      return '#8ED8B4';
    }
    if (result.state === '未通过' || result.state === '失败') {
      return '#F5A39A';
    }
    if (result.state === '运行中') {
      return '#8EB3FF';
    }
    return '#DCE2EA';
  }

  private modelOutputName(result: IdentityTestRow): string {
    if (result.state === '运行中') {
      return '正在聚合 3 段语音';
    }
    if (!this.hasModelResult(result)) {
      return '等待模型输出';
    }
    return result.predictedName;
  }

  private filterCount(filter: IdentityResultFilter): number {
    let count: number = 0;
    for (let i: number = 0; i < this.rows.length; i++) {
      if (identityResultMatchesFilter(filter, this.isKnownTruth(this.queries[i]),
        this.rows[i].state, this.rows[i].correct)) {
        count += 1;
      }
    }
    return count;
  }

  private shouldShowResult(spec: SpeakerIdentityQuerySpec, result: IdentityTestRow): boolean {
    return identityResultMatchesFilter(this.resultFilter, this.isKnownTruth(spec),
      result.state, result.correct);
  }

  private metricValue(metricId: string): string {
    if (metricId === 'known') {
      return this.knownMetric;
    }
    if (metricId === 'unknown') {
      return this.unknownMetric;
    }
    return this.balancedMetric;
  }

  @Builder
  private MetricCard(metricId: string, label: string, color: string) {
    Column() {
      Text(label)
        .fontSize(10)
        .fontColor('#667085')
      Text(this.metricValue(metricId))
        .fontSize(17)
        .fontWeight(FontWeight.Bold)
        .fontColor(color)
        .margin({ top: 4 })
    }
    .layoutWeight(1)
    .padding({ top: 11, bottom: 11 })
    .backgroundColor('#FFFFFF')
    .border({ width: 0.5, color: '#E4E9F0' })
    .borderRadius(10)
  }

  @Builder
  private FilterButton(filter: IdentityResultFilter, label: string) {
    Button(`${label} ${this.filterCount(filter)}`)
      .layoutWeight(1)
      .height(34)
      .fontSize(11)
      .fontWeight(this.resultFilter === filter ? FontWeight.Bold : FontWeight.Normal)
      .backgroundColor(this.resultFilter === filter ? '#EAF1FF' : '#FFFFFF')
      .fontColor(this.resultFilter === filter ? '#246BFD' : '#667085')
      .border({
        width: this.resultFilter === filter ? 1 : 0.5,
        color: this.resultFilter === filter ? '#246BFD' : '#DCE2EA',
      })
      .onClick((): void => {
        this.resultFilter = filter;
      })
  }

  @Builder
  private EvidenceCard(spec: SpeakerIdentityQuerySpec, result: IdentityTestRow, index: number) {
    Column() {
      Row() {
        Text(`${index + 1} / ${this.rows.length}`)
          .fontSize(11)
          .fontColor('#246BFD')
          .padding({ left: 9, right: 9, top: 4, bottom: 4 })
          .backgroundColor('#EEF4FF')
          .borderRadius(10)
        Text(this.isKnownTruth(spec) ? '已知人物' : '未知人物')
          .fontSize(11)
          .fontColor('#6941C6')
          .margin({ left: 7 })
        Blank()
        Text(this.outcomeLabel(spec, result))
          .fontSize(11)
          .fontWeight(FontWeight.Bold)
          .fontColor(this.outcomeColor(result))
          .padding({ left: 9, right: 9, top: 4, bottom: 4 })
          .backgroundColor(this.outcomeSoftColor(result))
          .borderRadius(10)
      }
      .width('100%')

      Row() {
        Column() {
          Text('数据集真值')
            .fontSize(10)
            .fontColor('#667085')
          Text(result.truthSpeakerId)
            .fontSize(17)
            .fontWeight(FontWeight.Bold)
            .fontColor('#172033')
            .margin({ top: 6 })
          Text(result.expectedName)
            .fontSize(12)
            .fontColor('#246BFD')
            .margin({ top: 4 })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
        .padding(11)
        .backgroundColor('#F5F7FA')
        .borderRadius(9)

        Text('→')
          .fontSize(20)
          .fontColor('#98A2B3')
          .margin({ left: 7, right: 7 })

        Column() {
          Text('CAM++ 模型输出')
            .fontSize(10)
            .fontColor('#6941C6')
          Text(this.modelOutputName(result))
            .fontSize(15)
            .fontWeight(FontWeight.Bold)
            .fontColor(this.hasModelResult(result) ? '#5B3FB5' : '#98A2B3')
            .maxLines(2)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
            .margin({ top: 9 })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
        .padding(11)
        .backgroundColor('#F7F4FF')
        .borderRadius(9)
      }
      .width('100%')
      .margin({ top: 11 })

      if (this.hasModelResult(result)) {
        Row() {
          Text(`最高相似度 ${result.score.toFixed(3)}`)
            .fontSize(11)
            .fontWeight(FontWeight.Medium)
            .fontColor('#172033')
          Blank()
          Text(`接纳阈值 ${SPEAKER_IDENTITY_THRESHOLD.toFixed(2)}`)
            .fontSize(11)
            .fontColor('#667085')
        }
        .width('100%')
        .margin({ top: 11 })
        Text(buildIdentityDecisionText(result.accepted, result.score, SPEAKER_IDENTITY_THRESHOLD))
          .fontSize(11)
          .fontColor(this.outcomeColor(result))
          .width('100%')
          .margin({ top: 5 })
        Text(`次高相似度 ${result.secondScore.toFixed(3)} · 处理 ${result.elapsedMs} ms`)
          .fontSize(10)
          .fontColor('#98A2B3')
          .width('100%')
          .margin({ top: 4 })
      } else if (result.error.length > 0) {
        Text(result.error)
          .fontSize(11)
          .fontColor('#B42318')
          .width('100%')
          .padding(9)
          .backgroundColor('#FFF0EE')
          .borderRadius(8)
          .margin({ top: 10 })
      } else {
        Text(result.state === '运行中' ?
          '正在提取并聚合本组 3 段测试语音的声纹向量。' :
          '真值已加载；运行后显示预测身份、相似度与阈值判定。')
          .fontSize(11)
          .fontColor(result.state === '运行中' ? '#246BFD' : '#667085')
          .width('100%')
          .margin({ top: 10 })
      }

      Row() {
        Column() {
          Text('参与评分的原始音频')
            .fontSize(11)
            .fontWeight(FontWeight.Medium)
            .fontColor('#344054')
          Text(this.playingId === spec.id ? '正在播放 3 段查询语音' : '3 段查询语音 · 与注册集分离')
            .fontSize(10)
            .fontColor(this.playingId === spec.id ? '#137A4C' : '#667085')
            .margin({ top: 2 })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
        Button(this.playingId === spec.id ?
          `停止 ${this.formatPlaybackTime(this.playPositionMs, this.playDurationMs)}` : '试听')
          .fontSize(11)
          .height(32)
          .backgroundColor(this.playingId === spec.id ? '#137A4C' : '#EAF1FF')
          .fontColor(this.playingId === spec.id ? '#FFFFFF' : '#246BFD')
          .enabled(!this.running && !this.playbackBusy)
          .onClick((): void => {
            this.togglePlayback(spec);
          })
      }
      .width('100%')
      .padding({ top: 9 })
      .border({ width: { top: 0.5 }, color: '#E4E9F0' })
      .margin({ top: 11 })
    }
    .width('100%')
    .padding(13)
    .backgroundColor('#FFFFFF')
    .border({ width: 1, color: this.evidenceBorderColor(result) })
    .borderRadius(14)
    .margin({ bottom: 10 })
  }

  build() {
    Column() {
      Row() {
        Button('返回')
          .fontSize(12)
          .height(34)
          .backgroundColor('#F2F4F7')
          .fontColor('#344054')
          .onClick((): void => {
            router.back();
          })
        Column() {
          Text('声纹身份评测')
            .fontSize(21)
            .fontWeight(FontWeight.Bold)
            .fontColor('#172033')
          Text(this.statusMessage)
            .fontSize(10)
            .fontColor('#667085')
            .maxLines(2)
            .margin({ top: 2 })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
        .margin({ left: 10 })
        Column() {
          Text('CAM++')
            .fontSize(11)
            .fontWeight(FontWeight.Bold)
            .fontColor('#6941C6')
          Text(this.progressText)
            .fontSize(11)
            .fontColor('#246BFD')
            .margin({ top: 3 })
        }
        .alignItems(HorizontalAlign.End)
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 14, bottom: 10 })

      Column() {
        Row() {
          Text('AISHELL-3 标注数据')
            .fontSize(15)
            .fontWeight(FontWeight.Bold)
            .fontColor('#3B2A86')
          Blank()
          Text('开放集身份识别')
            .fontSize(10)
            .fontColor('#6941C6')
        }
        .width('100%')
        Text('10 名已知 · 10 名未知 · 40 组测试')
          .fontSize(12)
          .fontColor('#5B3FB5')
          .width('100%')
          .margin({ top: 4 })
        Text('每名已知人物用 5 条语音建档；每组测试聚合另外 3 条语音。注册集与测试集完全分离。')
          .fontSize(10)
          .fontColor('#6D6590')
          .width('100%')
          .margin({ top: 5 })
        Row({ space: 7 }) {
          this.MetricCard('known', '已知识别', '#137A4C')
          this.MetricCard('unknown', '未知拒识', '#B54708')
          this.MetricCard('balanced', '综合准确率', '#246BFD')
        }
        .width('100%')
        .margin({ top: 10 })
        Text(this.summaryText)
          .fontSize(10)
          .fontColor('#667085')
          .width('100%')
          .margin({ top: 7 })
        if (this.playbackError.length > 0) {
          Text(this.playbackError)
            .fontSize(10)
            .fontColor('#B42318')
            .width('100%')
            .margin({ top: 5 })
        }
      }
      .alignItems(HorizontalAlign.Start)
      .width('100%')
      .padding(13)
      .backgroundColor('#F7F4FF')
      .border({ width: 0.5, color: '#DDD5FF' })
      .borderRadius(14)
      .margin({ left: 16, right: 16, bottom: 9 })

      Row({ space: 6 }) {
        this.FilterButton('all', '全部')
        this.FilterButton('known', '已知')
        this.FilterButton('unknown', '未知')
        this.FilterButton('errors', '错误')
      }
      .width('100%')
      .padding({ left: 16, right: 16, bottom: 8 })

      List() {
        ForEach(this.rows, (result: IdentityTestRow, index: number): void => {
          if (this.shouldShowResult(this.queries[index], result)) {
            ListItem() {
              this.EvidenceCard(this.queries[index], result, index)
            }
          }
        }, (result: IdentityTestRow): string =>
          `${result.id}-${result.state}-${result.elapsedMs}-${this.resultFilter}`)
      }
      .layoutWeight(1)
      .width('100%')
      .padding({ left: 16, right: 16 })
      .scrollBar(BarState.Off)

      Column() {
        Text('测试声纹仅保存在隔离内存中，不读取或写入个人声纹资料。')
          .fontSize(9)
          .fontColor('#667085')
          .margin({ bottom: 6 })
        Button(this.running ? `正在运行 ${this.progressText}` : '运行 AISHELL-3 身份测试')
          .width('100%')
          .height(46)
          .backgroundColor('#246BFD')
          .fontColor('#FFFFFF')
          .enabled(this.ready && !this.running && !this.playbackBusy && this.playingId.length === 0)
          .onClick((): void => {
            this.startTest();
          })
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 7, bottom: 16 })
      .backgroundColor('#F6F8FC')
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#F6F8FC')
  }
}
