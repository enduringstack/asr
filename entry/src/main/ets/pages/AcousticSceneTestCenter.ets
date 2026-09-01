import { common } from '@kit.AbilityKit';
import { router } from '@kit.ArkUI';
import { ErrorEvent, MessageEvents, util, worker } from '@kit.ArkTS';
import { hilog } from '@kit.PerformanceAnalysisKit';
import {
  ACOUSTIC_SCENE_LABELS,
  AcousticSceneFilter,
  AcousticSceneLabel,
  AcousticScenePrediction,
  AcousticSceneState,
  AcousticSceneSummary,
  acousticSceneDisplayName,
  acousticSceneResultMatchesFilter,
  buildAcousticScenePrediction,
  buildAcousticSceneSummary,
} from '../common/AcousticSceneTypes';
import {
  AcousticSceneTestCase,
  buildAcousticSceneTestCases,
} from '../common/AcousticSceneTestData';
import { RawPlaybackState, RawResourcePlayer } from '../services/audio/RawResourcePlayer';

const LOG_DOMAIN: number = 0x0001;
const LOG_TAG: string = 'SceneTest';

interface SceneWorkerRequest {
  msgType: string;
  context?: common.UIAbilityContext;
  data?: Uint8Array;
  audioFormat?: string;
  requestId?: string;
}

interface SceneWorkerResponse {
  msgType: string;
  message?: string;
  requestId?: string;
  sceneClassIndex?: number;
  sceneConfidence?: number;
  sceneProbabilities?: number[];
  sceneWindowCount?: number;
  audioDuration?: number;
  elapsedMs?: number;
}

interface SceneTestRow {
  id: string;
  state: AcousticSceneState;
  correct: boolean;
  prediction: AcousticScenePrediction | undefined;
  error: string;
}

@Entry
@Component
struct AcousticSceneTestCenterPage {
  @State ready: boolean = false;
  @State running: boolean = false;
  @State statusMessage: string = '正在加载端侧场景模型...';
  @State progressText: string = '0 / 18';
  @State resultFilter: AcousticSceneFilter = 'all';
  @State rows: SceneTestRow[] = [];
  @State playingId: string = '';
  @State playPositionMs: number = 0;
  @State playDurationMs: number = 0;
  @State playbackBusy: boolean = false;
  @State playbackError: string = '';

  private workerInstance: worker.ThreadWorker | undefined = undefined;
  private audioPlayer: RawResourcePlayer = new RawResourcePlayer();
  private cases: AcousticSceneTestCase[] = buildAcousticSceneTestCases();
  private pendingRequestId: string = '';
  private pendingIndex: number = -1;
  private batchMode: boolean = false;
  private runToken: number = 0;

  aboutToAppear(): void {
    this.rows = this.emptyRows();
    this.progressText = `0 / ${this.cases.length}`;
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
      'entry/ets/workers/StreamingSherpaWorker.ets', { name: 'Acoustic scene evaluation' });
    instance.onmessage = (event: MessageEvents): void => {
      this.onWorkerMessage(event);
    };
    instance.onerror = (event: ErrorEvent): void => {
      this.abortRun(`场景评测引擎异常：${event.message}`);
    };
    instance.onmessageerror = (_event: MessageEvents): void => {
      this.abortRun('场景评测引擎消息解析失败');
    };
    this.workerInstance = instance;
    instance.postMessage({ msgType: 'sherpa-scene-test-init', context } as SceneWorkerRequest);
  }

  aboutToDisappear(): void {
    this.runToken += 1;
    this.running = false;
    this.audioPlayer.setCallback(null);
    this.audioPlayer.stop().catch((error: Error): void => {
      hilog.warn(LOG_DOMAIN, LOG_TAG, `stop playback failed: ${error.message}`);
    });
    if (this.workerInstance) {
      this.workerInstance.terminate();
      this.workerInstance = undefined;
    }
  }

  private emptyRows(): SceneTestRow[] {
    const result: SceneTestRow[] = [];
    for (let i: number = 0; i < this.cases.length; i++) {
      result.push({ id: this.cases[i].id, state: 'waiting', correct: false,
        prediction: undefined, error: '' });
    }
    return result;
  }

  private onWorkerMessage(event: MessageEvents): void {
    const response: SceneWorkerResponse = event.data as SceneWorkerResponse;
    if (response.msgType === 'sherpa-scene-test-ready') {
      this.ready = true;
      this.statusMessage = '模型已就绪，可试听并核对真实标签与端侧输出';
      return;
    }
    if (response.msgType === 'sherpa-scene-test-done') {
      if (this.running && response.requestId === this.pendingRequestId) {
        this.acceptResult(response);
      }
      return;
    }
    if (response.msgType === 'sherpa-scene-test-failed') {
      if (this.running && response.requestId === this.pendingRequestId) {
        this.acceptFailure(response.message || '场景分类失败');
      }
      return;
    }
    if (response.msgType === 'sherpa-asr-error') {
      this.abortRun(response.message || '场景模型异常');
    }
  }

  private startBatch(): void {
    if (!this.ready || this.running || this.playingId.length > 0 || this.playbackBusy) {
      return;
    }
    this.rows = this.emptyRows();
    this.batchMode = true;
    this.runToken += 1;
    this.running = true;
    this.playbackError = '';
    this.progressText = `0 / ${this.cases.length}`;
    this.requestCase(0);
  }

  private startSingle(index: number): void {
    if (!this.ready || this.running || this.playingId.length > 0 || this.playbackBusy ||
      index < 0 || index >= this.cases.length) {
      return;
    }
    this.batchMode = false;
    this.runToken += 1;
    this.running = true;
    this.playbackError = '';
    this.progressText = '0 / 1';
    this.requestCase(index);
  }

  private requestCase(index: number): void {
    if (!this.running || !this.workerInstance || index < 0 || index >= this.cases.length) {
      return;
    }
    const token: number = this.runToken;
    const spec: AcousticSceneTestCase = this.cases[index];
    const updated: SceneTestRow[] = this.rows.slice();
    updated[index] = { id: spec.id, state: 'running', correct: false,
      prediction: undefined, error: '' };
    this.rows = updated;
    this.pendingIndex = index;
    this.statusMessage = `正在识别 ${spec.dataset} · ${spec.expectedDisplayName}`;
    const context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;
    context.resourceManager.getRawFileContent(spec.audioPath)
      .then((data: Uint8Array): void => {
        if (!this.running || token !== this.runToken || !this.workerInstance) {
          return;
        }
        const requestId: string = util.generateRandomUUID(true);
        this.pendingRequestId = requestId;
        this.workerInstance.postMessage({
          msgType: 'sherpa-scene-test', requestId, data, audioFormat: 'wav',
        } as SceneWorkerRequest, [data.buffer]);
      })
      .catch((error: Error): void => {
        if (this.running && token === this.runToken) {
          this.acceptFailure(`读取固定音频失败：${error.message}`);
        }
      });
  }

  private acceptResult(response: SceneWorkerResponse): void {
    if (this.pendingIndex < 0 || this.pendingIndex >= this.cases.length) {
      this.abortRun('场景结果缺少对应样本');
      return;
    }
    const spec: AcousticSceneTestCase = this.cases[this.pendingIndex];
    const prediction: AcousticScenePrediction = buildAcousticScenePrediction(
      response.sceneClassIndex !== undefined ? response.sceneClassIndex : 5,
      response.sceneConfidence !== undefined ? response.sceneConfidence : 0,
      response.sceneProbabilities || [],
      response.sceneWindowCount || 0,
      response.audioDuration || 0,
      response.elapsedMs || 0,
    );
    const correct: boolean = prediction.label === spec.expectedLabel;
    const updated: SceneTestRow[] = this.rows.slice();
    updated[this.pendingIndex] = { id: spec.id, state: 'completed', correct,
      prediction, error: '' };
    this.rows = updated;
    hilog.info(LOG_DOMAIN, LOG_TAG,
      `RESULT id=${spec.id} truth=${spec.expectedLabel} prediction=${prediction.label} ` +
      `confidence=${prediction.confidence.toFixed(4)} correct=${correct}`);
    this.advanceAfterResult();
  }

  private acceptFailure(message: string): void {
    if (this.pendingIndex >= 0 && this.pendingIndex < this.rows.length) {
      const updated: SceneTestRow[] = this.rows.slice();
      updated[this.pendingIndex] = { id: this.cases[this.pendingIndex].id,
        state: 'failed', correct: false, prediction: undefined, error: message };
      this.rows = updated;
    }
    this.advanceAfterResult();
  }

  private advanceAfterResult(): void {
    const completedIndex: number = this.pendingIndex;
    this.pendingRequestId = '';
    if (this.batchMode && completedIndex + 1 < this.cases.length) {
      this.progressText = `${completedIndex + 1} / ${this.cases.length}`;
      this.requestCase(completedIndex + 1);
      return;
    }
    this.running = false;
    this.progressText = this.batchMode ?
      `${this.cases.length} / ${this.cases.length}` : '1 / 1';
    this.statusMessage = this.batchMode ?
      '固定校准集运行完成，可筛选未匹配样本逐条试听' : '单条场景识别完成';
    const summary: AcousticSceneSummary = this.summary();
    hilog.info(LOG_DOMAIN, LOG_TAG,
      `SUMMARY correct=${summary.correct}/${summary.total} failed=${summary.failed}`);
    this.pendingIndex = -1;
  }

  private abortRun(message: string): void {
    this.runToken += 1;
    this.running = false;
    this.pendingIndex = -1;
    this.pendingRequestId = '';
    this.statusMessage = message;
    hilog.error(LOG_DOMAIN, LOG_TAG, message);
  }

  private summary(): AcousticSceneSummary {
    return buildAcousticSceneSummary(this.rows);
  }

  private hasAnyResult(): boolean {
    return this.summary().total > 0 || this.summary().failed > 0;
  }

  private shouldShow(index: number): boolean {
    return acousticSceneResultMatchesFilter(this.resultFilter,
      this.rows[index].state, this.rows[index].correct);
  }

  private filterCount(filter: AcousticSceneFilter): number {
    let result: number = 0;
    for (let i: number = 0; i < this.rows.length; i++) {
      if (acousticSceneResultMatchesFilter(filter, this.rows[i].state, this.rows[i].correct)) {
        result += 1;
      }
    }
    return result;
  }

  private outcomeText(row: SceneTestRow): string {
    if (row.state === 'waiting') {
      return '等待运行';
    }
    if (row.state === 'running') {
      return '正在识别';
    }
    if (row.state === 'failed') {
      return '处理失败';
    }
    return row.correct ? '匹配' : '不匹配';
  }

  private classMetricText(label: AcousticSceneLabel): string {
    let total: number = 0;
    let correct: number = 0;
    for (let i: number = 0; i < this.cases.length; i++) {
      if (this.cases[i].expectedLabel === label && this.rows[i].state === 'completed') {
        total += 1;
        if (this.rows[i].correct) {
          correct += 1;
        }
      }
    }
    return total > 0 ? `${correct}/${total}` : '—';
  }

  private outcomeColor(row: SceneTestRow): string {
    if (row.state === 'completed' && row.correct) {
      return '#137A4C';
    }
    if (row.state === 'completed' || row.state === 'failed') {
      return '#B42318';
    }
    if (row.state === 'running') {
      return '#C05A13';
    }
    return '#667085';
  }

  private probabilityAt(row: SceneTestRow, index: number): number {
    if (!row.prediction || index < 0 || index >= row.prediction.probabilities.length) {
      return 0;
    }
    return row.prediction.probabilities[index];
  }

  private togglePlayback(spec: AcousticSceneTestCase): void {
    if (this.running || this.playbackBusy) {
      return;
    }
    this.playbackError = '';
    this.playbackBusy = true;
    const context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;
    this.audioPlayer.toggle(spec.id, spec.audioPath, context.resourceManager)
      .then((): void => {
        this.playbackBusy = false;
      })
      .catch((error: Error): void => {
        this.playbackBusy = false;
        this.playbackError = `播放失败：${error.message}`;
      });
  }

  private formatPlaybackTime(positionMs: number, durationMs: number): string {
    const seconds: number = Math.max(0, Math.floor(positionMs / 1000));
    const total: number = Math.max(0, Math.floor(durationMs / 1000));
    return `0:${seconds < 10 ? `0${seconds}` : `${seconds}`} / ` +
      `0:${total < 10 ? `0${total}` : `${total}`}`;
  }

  @Builder
  private FilterButton(filter: AcousticSceneFilter, label: string) {
    Button(`${label} ${this.filterCount(filter)}`)
      .layoutWeight(1)
      .height(34)
      .fontSize(10)
      .fontWeight(this.resultFilter === filter ? FontWeight.Bold : FontWeight.Normal)
      .backgroundColor(this.resultFilter === filter ? '#FFF3E8' : '#FFFFFF')
      .fontColor(this.resultFilter === filter ? '#C05A13' : '#667085')
      .border({ width: 0.7, color: this.resultFilter === filter ? '#D98242' : '#DCE2EA' })
      .onClick((): void => {
        this.resultFilter = filter;
      })
  }

  @Builder
  private ProbabilityRow(row: SceneTestRow, label: AcousticSceneLabel, index: number) {
    Row({ space: 7 }) {
      Text(acousticSceneDisplayName(label))
        .fontSize(9)
        .fontColor('#667085')
        .width(82)
      Progress({ value: this.probabilityAt(row, index) * 100, total: 100,
        type: ProgressType.Linear })
        .color(row.prediction && row.prediction.rawLabel === label ? '#C05A13' : '#C9D0DA')
        .layoutWeight(1)
        .height(5)
      Text(`${(this.probabilityAt(row, index) * 100).toFixed(1)}%`)
        .fontSize(9)
        .fontColor('#667085')
        .width(42)
        .textAlign(TextAlign.End)
    }
    .width('100%')
  }

  @Builder
  private ClassMetric(label: AcousticSceneLabel) {
    Column() {
      Text(acousticSceneDisplayName(label))
        .fontSize(9)
        .fontColor('#667085')
        .maxLines(1)
      Text(this.classMetricText(label))
        .fontSize(13)
        .fontWeight(FontWeight.Bold)
        .fontColor('#8A3F0B')
        .margin({ top: 3 })
    }
    .layoutWeight(1)
    .padding({ top: 8, bottom: 8 })
    .backgroundColor('#FFFFFF')
    .borderRadius(8)
  }

  @Builder
  private EvidenceCard(spec: AcousticSceneTestCase, row: SceneTestRow, index: number) {
    Column() {
      Row() {
        Text(`${index + 1}`)
          .fontSize(10)
          .fontColor('#C05A13')
          .padding({ left: 8, right: 8, top: 4, bottom: 4 })
          .backgroundColor('#FFF3E8')
          .borderRadius(10)
        Column() {
          Text(`${spec.dataset} · ${spec.sourceId}`)
            .fontSize(12)
            .fontWeight(FontWeight.Bold)
            .fontColor('#172033')
            .maxLines(1)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
          Text(spec.license)
            .fontSize(9)
            .fontColor('#98A2B3')
            .margin({ top: 2 })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
        .margin({ left: 8 })
        Text(this.outcomeText(row))
          .fontSize(10)
          .fontWeight(FontWeight.Bold)
          .fontColor(this.outcomeColor(row))
      }
      .width('100%')

      Row({ space: 8 }) {
        Column() {
          Text('真实标签')
            .fontSize(10)
            .fontColor('#667085')
          Text(spec.expectedDisplayName)
            .fontSize(17)
            .fontWeight(FontWeight.Bold)
            .fontColor('#1C6B52')
            .margin({ top: 6 })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
        .padding(10)
        .backgroundColor('#EAF8F2')
        .borderRadius(9)
        Column() {
          Text('模型输出')
            .fontSize(10)
            .fontColor('#667085')
          Text(row.prediction ? row.prediction.displayName :
            (row.state === 'failed' ? '处理失败' : '等待模型输出'))
            .fontSize(17)
            .fontWeight(FontWeight.Bold)
            .fontColor(row.prediction ? this.outcomeColor(row) : '#98A2B3')
            .margin({ top: 6 })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
        .padding(10)
        .backgroundColor('#FFF3E8')
        .borderRadius(9)
      }
      .width('100%')
      .margin({ top: 10 })

      if (row.prediction) {
        Column({ space: 4 }) {
          ForEach(ACOUSTIC_SCENE_LABELS,
            (label: AcousticSceneLabel, probabilityIndex: number): void => {
              this.ProbabilityRow(row, label, probabilityIndex)
            }, (label: AcousticSceneLabel): string => `${row.id}-${label}`)
        }
        .width('100%')
        .padding(9)
        .backgroundColor('#F8F9FB')
        .borderRadius(9)
        .margin({ top: 9 })
        Text(`置信度 ${(row.prediction.confidence * 100).toFixed(1)}% · ` +
          `${row.prediction.windowCount} 个窗口 · ${row.prediction.elapsedMs} ms`)
          .fontSize(9)
          .fontColor('#667085')
          .width('100%')
          .margin({ top: 6 })
      } else if (row.state === 'failed') {
        Text(row.error)
          .fontSize(10)
          .fontColor('#B42318')
          .width('100%')
          .margin({ top: 8 })
      }

      Text(spec.truthBasis)
        .fontSize(9)
        .fontColor('#98A2B3')
        .width('100%')
        .margin({ top: 8 })
      Row() {
        Text(this.playingId === spec.id ?
          `正在播放 ${this.formatPlaybackTime(this.playPositionMs, this.playDurationMs)}` :
          '可播放原始评分音频')
          .fontSize(9)
          .fontColor(this.playingId === spec.id ? '#137A4C' : '#667085')
          .layoutWeight(1)
        Button(this.playingId === spec.id ? '停止' : '播放')
          .height(32)
          .fontSize(10)
          .backgroundColor(this.playingId === spec.id ? '#137A4C' : '#EAF1FF')
          .fontColor(this.playingId === spec.id ? '#FFFFFF' : '#246BFD')
          .enabled(!this.running && !this.playbackBusy)
          .onClick((): void => {
            this.togglePlayback(spec);
          })
        Button('运行')
          .height(32)
          .fontSize(10)
          .backgroundColor('#FFF3E8')
          .fontColor('#C05A13')
          .enabled(this.ready && !this.running && !this.playbackBusy && this.playingId.length === 0)
          .margin({ left: 6 })
          .onClick((): void => {
            this.startSingle(index);
          })
      }
      .width('100%')
      .padding({ top: 9 })
      .border({ width: { top: 0.5 }, color: '#E4E9F0' })
      .margin({ top: 9 })
    }
    .width('100%')
    .padding(13)
    .backgroundColor('#FFFFFF')
    .border({ width: 0.7, color: row.state === 'completed' ?
      (row.correct ? '#9BDDBB' : '#F2A69E') : '#E1E6ED' })
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
          Text('环境场景评测')
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
          Text('EfficientAT DyMN04')
            .fontSize(10)
            .fontWeight(FontWeight.Bold)
            .fontColor('#C05A13')
          Text(this.progressText)
            .fontSize(10)
            .fontColor('#667085')
            .margin({ top: 3 })
        }
        .alignItems(HorizontalAlign.End)
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 14, bottom: 10 })

      Column() {
        Row() {
          Text('带真实标签的固定校准集')
            .fontSize(14)
            .fontWeight(FontWeight.Bold)
            .fontColor('#8A3F0B')
          Blank()
          Text(`${this.cases.length} 条 · 每条 30 秒 · 可播放`)
            .fontSize(10)
            .fontColor('#C05A13')
        }
        .width('100%')
        Text('六类各 3 条连续实地录音；真实标签与模型输出并排显示。盲测集未放入 App。')
          .fontSize(10)
          .fontColor('#667085')
          .width('100%')
          .margin({ top: 5 })
        if (this.hasAnyResult()) {
          Text(`准确率 ${this.summary().correct}/${this.summary().total}  ` +
            `${(this.summary().accuracy * 100).toFixed(1)}% · 失败 ${this.summary().failed}`)
            .fontSize(18)
            .fontWeight(FontWeight.Bold)
            .fontColor('#C05A13')
            .margin({ top: 10 })
        } else {
          Text('准确率 —')
            .fontSize(18)
            .fontWeight(FontWeight.Bold)
            .fontColor('#98A2B3')
            .margin({ top: 10 })
        }
        Row({ space: 5 }) {
          this.ClassMetric('metro')
          this.ClassMetric('high_speed_train')
          this.ClassMetric('shopping_mall')
        }
        .width('100%')
        .margin({ top: 9 })
        Row({ space: 5 }) {
          this.ClassMetric('cafe_restaurant')
          this.ClassMetric('concert')
          this.ClassMetric('other')
        }
        .width('100%')
        .margin({ top: 5 })
        if (this.playbackError.length > 0) {
          Text(this.playbackError)
            .fontSize(9)
            .fontColor('#B42318')
            .margin({ top: 5 })
        }
      }
      .alignItems(HorizontalAlign.Start)
      .width('100%')
      .padding(13)
      .backgroundColor('#FFF8F2')
      .border({ width: 0.5, color: '#F2D3BB' })
      .borderRadius(14)
      .margin({ left: 16, right: 16, bottom: 9 })

      Row({ space: 6 }) {
        this.FilterButton('all', '全部')
        this.FilterButton('correct', '匹配')
        this.FilterButton('incorrect', '未匹配')
      }
      .width('100%')
      .padding({ left: 16, right: 16, bottom: 8 })

      List() {
        ForEach(this.rows, (row: SceneTestRow, index: number): void => {
          if (this.shouldShow(index)) {
            ListItem() {
              this.EvidenceCard(this.cases[index], row, index)
            }
          }
        }, (row: SceneTestRow): string =>
          `${row.id}-${row.state}-${row.prediction ? row.prediction.elapsedMs : 0}-${this.resultFilter}`)
      }
      .layoutWeight(1)
      .width('100%')
      .padding({ left: 16, right: 16 })
      .scrollBar(BarState.Off)

      Column() {
        Text('模型在手机本地运行；这是可重复校准 Demo，不代表 95% 盲测结论。')
          .fontSize(9)
          .fontColor('#667085')
          .margin({ bottom: 6 })
        Button(this.running && this.batchMode ?
          `正在运行 ${this.progressText}` : `批量运行 ${this.cases.length} 条`)
          .width('100%')
          .height(46)
          .backgroundColor('#C05A13')
          .fontColor('#FFFFFF')
          .enabled(this.ready && !this.running && !this.playbackBusy && this.playingId.length === 0)
          .onClick((): void => {
            this.startBatch();
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
