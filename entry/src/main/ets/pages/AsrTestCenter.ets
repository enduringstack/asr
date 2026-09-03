import { common } from '@kit.AbilityKit';
import { router } from '@kit.ArkUI';
import { ErrorEvent, MessageEvents, util, worker } from '@kit.ArkTS';
import { hilog } from '@kit.PerformanceAnalysisKit';
import {
  AsrCaseResult,
  AsrEditScore,
  AsrEvaluationFilter,
  AsrEvaluationState,
  AsrEvaluationSummary,
  asrResultMatchesFilter,
  buildAsrEvaluationSummary,
  scoreAsrText,
} from '../common/AsrEvaluationTypes';
import { AsrTestCase, buildAsrTestCases } from '../common/AsrTestData';
import { RawPlaybackState, RawResourcePlayer } from '../services/audio/RawResourcePlayer';

const LOG_DOMAIN: number = 0x0001;
const LOG_TAG: string = 'AsrTestCenter';

interface AsrWorkerRequest {
  msgType: string;
  context?: common.UIAbilityContext;
  data?: Uint8Array;
  audioFormat?: string;
  requestId?: string;
}

interface AsrWorkerResponse {
  msgType: string;
  message?: string;
  requestId?: string;
  rawText?: string;
  audioDuration?: number;
  elapsedMs?: number;
}

interface AsrTestRow {
  id: string;
  state: AsrEvaluationState;
  hypothesis: string;
  score?: AsrEditScore;
  audioDurationSec: number;
  elapsedMs: number;
  error: string;
}

@Entry
@Component
struct AsrTestCenterPage {
  @State ready: boolean = false;
  @State running: boolean = false;
  @State statusMessage: string = '正在加载端侧 Paraformer 模型...';
  @State selectedDataset: string = 'aishell1';
  @State progressText: string = '0 / 20';
  @State resultFilter: AsrEvaluationFilter = 'all';
  @State rows: AsrTestRow[] = [];
  @State playingId: string = '';
  @State playPositionMs: number = 0;
  @State playDurationMs: number = 0;
  @State playbackBusy: boolean = false;
  @State playbackError: string = '';

  private workerInstance: worker.ThreadWorker | undefined = undefined;
  private audioPlayer: RawResourcePlayer = new RawResourcePlayer();
  private cases: AsrTestCase[] = buildAsrTestCases();
  private runQueue: number[] = [];
  private queuePosition: number = -1;
  private pendingIndex: number = -1;
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
      'entry/ets/workers/StreamingSherpaWorker.ets', { name: 'ASR ground truth evaluation' });
    instance.onmessage = (event: MessageEvents): void => {
      this.onWorkerMessage(event);
    };
    instance.onerror = (event: ErrorEvent): void => {
      this.abortRun(`评测引擎异常：${event.message}`);
    };
    instance.onmessageerror = (_event: MessageEvents): void => {
      this.abortRun('评测引擎消息解析失败');
    };
    this.workerInstance = instance;
    instance.postMessage({
      msgType: 'sherpa-asr-eval-init',
      context,
    } as AsrWorkerRequest);
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

  private emptyRows(): AsrTestRow[] {
    const result: AsrTestRow[] = [];
    for (let index: number = 0; index < this.cases.length; index++) {
      result.push({
        id: this.cases[index].id,
        state: 'waiting',
        hypothesis: '',
        audioDurationSec: 0,
        elapsedMs: 0,
        error: '',
      });
    }
    return result;
  }

  private onWorkerMessage(event: MessageEvents): void {
    const response: AsrWorkerResponse = event.data as AsrWorkerResponse;
    if (response.msgType === 'sherpa-asr-eval-ready') {
      this.ready = true;
      this.statusMessage = '模型已就绪；预测不随数据打包，由手机现场计算';
      return;
    }
    if (response.msgType === 'sherpa-asr-eval-done') {
      if (!this.running || response.requestId !== this.pendingRequestId) {
        return;
      }
      this.acceptResult(response);
      return;
    }
    if (response.msgType === 'sherpa-asr-eval-failed') {
      if (!this.running || response.requestId !== this.pendingRequestId) {
        return;
      }
      this.acceptFailure(response.message || '模型处理失败');
      return;
    }
    if (response.msgType === 'sherpa-asr-error') {
      this.abortRun(response.message || 'ASR 模型异常');
    }
  }

  private datasetIndices(datasetId: string): number[] {
    const result: number[] = [];
    for (let index: number = 0; index < this.cases.length; index++) {
      if (this.cases[index].datasetId === datasetId) {
        result.push(index);
      }
    }
    return result;
  }

  private selectDataset(datasetId: string): void {
    if (this.running || this.playingId.length > 0) {
      return;
    }
    this.selectedDataset = datasetId;
    this.resultFilter = 'all';
    this.progressText = `${this.evaluatedCount(datasetId)} / ${this.datasetIndices(datasetId).length}`;
    this.statusMessage = '已切换数据集，可运行 20 条或逐条试听';
  }

  private startBatch(): void {
    if (!this.ready || this.running || this.playbackBusy || this.playingId.length > 0) {
      return;
    }
    this.runQueue = this.datasetIndices(this.selectedDataset);
    if (this.runQueue.length === 0) {
      this.statusMessage = '当前数据集没有测试音频';
      return;
    }
    const updated: AsrTestRow[] = this.rows.slice();
    for (let position: number = 0; position < this.runQueue.length; position++) {
      const index: number = this.runQueue[position];
      updated[index] = {
        id: this.cases[index].id,
        state: 'waiting',
        hypothesis: '',
        audioDurationSec: 0,
        elapsedMs: 0,
        error: '',
      };
    }
    this.rows = updated;
    this.beginQueue();
  }

  private startSingle(index: number): void {
    if (!this.ready || this.running || this.playbackBusy || this.playingId.length > 0 ||
      index < 0 || index >= this.cases.length) {
      return;
    }
    this.runQueue = [index];
    this.beginQueue();
  }

  private beginQueue(): void {
    this.runToken += 1;
    this.running = true;
    this.playbackError = '';
    this.queuePosition = 0;
    this.progressText = `0 / ${this.runQueue.length}`;
    this.requestCase(this.runQueue[0]);
  }

  private requestCase(index: number): void {
    if (!this.running || !this.workerInstance || index < 0 || index >= this.cases.length) {
      return;
    }
    const token: number = this.runToken;
    const spec: AsrTestCase = this.cases[index];
    const updated: AsrTestRow[] = this.rows.slice();
    updated[index] = {
      id: spec.id,
      state: 'running',
      hypothesis: '',
      audioDurationSec: 0,
      elapsedMs: 0,
      error: '',
    };
    this.rows = updated;
    this.pendingIndex = index;
    this.statusMessage = `正在识别 ${spec.dataset} · ${spec.sourceId}`;
    const context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;
    context.resourceManager.getRawFileContent(spec.audioPath)
      .then((data: Uint8Array): void => {
        if (!this.running || token !== this.runToken || !this.workerInstance) {
          return;
        }
        const requestId: string = util.generateRandomUUID(true);
        this.pendingRequestId = requestId;
        this.workerInstance.postMessage({
          msgType: 'sherpa-asr-eval',
          requestId,
          data,
          audioFormat: 'wav',
        } as AsrWorkerRequest, [data.buffer]);
      })
      .catch((error: Error): void => {
        if (this.running && token === this.runToken) {
          this.acceptFailure(`读取测试音频失败：${error.message}`);
        }
      });
  }

  private acceptResult(response: AsrWorkerResponse): void {
    if (this.pendingIndex < 0 || this.pendingIndex >= this.cases.length) {
      this.abortRun('评测结果缺少对应样本');
      return;
    }
    const spec: AsrTestCase = this.cases[this.pendingIndex];
    const hypothesis: string = response.rawText || '';
    const score: AsrEditScore = scoreAsrText(spec.reference, hypothesis);
    const updated: AsrTestRow[] = this.rows.slice();
    updated[this.pendingIndex] = {
      id: spec.id,
      state: 'completed',
      hypothesis,
      score,
      audioDurationSec: response.audioDuration || spec.durationSec,
      elapsedMs: response.elapsedMs || 0,
      error: '',
    };
    this.rows = updated;
    hilog.info(LOG_DOMAIN, LOG_TAG,
      `RESULT id=${spec.id} cer=${score.cer.toFixed(4)} s=${score.substitutions} ` +
      `d=${score.deletions} i=${score.insertions} hypothesis=${hypothesis}`);
    this.advanceAfterResult();
  }

  private acceptFailure(message: string): void {
    if (this.pendingIndex >= 0 && this.pendingIndex < this.rows.length) {
      const spec: AsrTestCase = this.cases[this.pendingIndex];
      const updated: AsrTestRow[] = this.rows.slice();
      updated[this.pendingIndex] = {
        id: spec.id,
        state: 'failed',
        hypothesis: '',
        audioDurationSec: spec.durationSec,
        elapsedMs: 0,
        error: message,
      };
      this.rows = updated;
      hilog.error(LOG_DOMAIN, LOG_TAG, `RESULT id=${spec.id} state=failed error=${message}`);
    }
    this.advanceAfterResult();
  }

  private advanceAfterResult(): void {
    this.pendingRequestId = '';
    this.queuePosition += 1;
    this.progressText = `${this.queuePosition} / ${this.runQueue.length}`;
    if (this.queuePosition < this.runQueue.length) {
      this.requestCase(this.runQueue[this.queuePosition]);
      return;
    }
    this.running = false;
    this.pendingIndex = -1;
    const summary: AsrEvaluationSummary = this.summary(this.selectedDataset);
    this.statusMessage =
      `评测完成：微平均 CER ${(summary.microCer * 100).toFixed(2)}%，可筛选错误逐条试听`;
    hilog.info(LOG_DOMAIN, LOG_TAG,
      `SUMMARY dataset=${this.selectedDataset} completed=${summary.completed} failed=${summary.failed} ` +
      `cer=${summary.microCer.toFixed(4)} rtf=${summary.rtf.toFixed(4)}`);
  }

  private abortRun(message: string): void {
    this.runToken += 1;
    this.running = false;
    this.pendingIndex = -1;
    this.pendingRequestId = '';
    this.statusMessage = message;
    hilog.error(LOG_DOMAIN, LOG_TAG, message);
  }

  private summary(datasetId: string): AsrEvaluationSummary {
    const resultRows: AsrCaseResult[] = [];
    for (let index: number = 0; index < this.cases.length; index++) {
      if (this.cases[index].datasetId !== datasetId) {
        continue;
      }
      resultRows.push({
        state: this.rows[index].state,
        score: this.rows[index].score,
        audioDurationSec: this.rows[index].audioDurationSec,
        elapsedMs: this.rows[index].elapsedMs,
      });
    }
    return buildAsrEvaluationSummary(resultRows);
  }

  private evaluatedCount(datasetId: string): number {
    return this.summary(datasetId).evaluated;
  }

  private filterCount(filter: AsrEvaluationFilter): number {
    let count: number = 0;
    for (let index: number = 0; index < this.cases.length; index++) {
      if (this.cases[index].datasetId === this.selectedDataset &&
        asrResultMatchesFilter(filter, this.rows[index].state, this.rows[index].score)) {
        count += 1;
      }
    }
    return count;
  }

  private shouldShow(index: number): boolean {
    return this.cases[index].datasetId === this.selectedDataset &&
      asrResultMatchesFilter(this.resultFilter, this.rows[index].state, this.rows[index].score);
  }

  private outcomeText(row: AsrTestRow): string {
    if (row.state === 'running') {
      return '识别中';
    }
    if (row.state === 'failed') {
      return '失败';
    }
    if (row.state === 'completed' && row.score !== undefined) {
      return row.score.exact ? '完全一致' : `CER ${(row.score.cer * 100).toFixed(1)}%`;
    }
    return '等待运行';
  }

  private outcomeColor(row: AsrTestRow): string {
    if (row.state === 'completed' && row.score !== undefined && row.score.exact) {
      return '#137A4C';
    }
    if (row.state === 'completed' || row.state === 'failed') {
      return '#B42318';
    }
    return row.state === 'running' ? '#246BFD' : '#667085';
  }

  private metricText(metric: string): string {
    const summary: AsrEvaluationSummary = this.summary(this.selectedDataset);
    if (summary.evaluated === 0) {
      return '—';
    }
    if (metric === 'cer') {
      return `${(summary.microCer * 100).toFixed(2)}%`;
    }
    if (metric === 'exact') {
      return `${summary.exact}/${summary.evaluated}`;
    }
    return summary.rtf.toFixed(3);
  }

  private rowRenderKey(row: AsrTestRow): string {
    const errors: number = row.score !== undefined ? row.score.errors : -1;
    return `${row.id}:${row.state}:${errors}:${row.hypothesis}:${row.error}`;
  }

  private taskText(spec: AsrTestCase): string {
    let text: string = 'ASR';
    if (spec.supportsVoiceprint) {
      text += ' · 声纹';
    }
    if (spec.supportsDiarization) {
      text += ' · 说话人分离';
    }
    return text;
  }

  private datasetOrdinal(index: number): number {
    let ordinal: number = 0;
    const datasetId: string = this.cases[index].datasetId;
    for (let position: number = 0; position <= index; position++) {
      if (this.cases[position].datasetId === datasetId) {
        ordinal += 1;
      }
    }
    return ordinal;
  }

  private togglePlayback(spec: AsrTestCase): void {
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

  @Builder
  private DatasetTab(datasetId: string, label: string) {
    Button(`${label}  ${this.evaluatedCount(datasetId)}/20`)
      .height(34)
      .fontSize(10)
      .fontWeight(this.selectedDataset === datasetId ? FontWeight.Bold : FontWeight.Normal)
      .backgroundColor(this.selectedDataset === datasetId ? '#EAF1FF' : '#FFFFFF')
      .fontColor(this.selectedDataset === datasetId ? '#246BFD' : '#667085')
      .border({ width: 0.5, color: this.selectedDataset === datasetId ? '#246BFD' : '#DCE2EA' })
      .enabled(!this.running && this.playingId.length === 0)
      .onClick((): void => {
        this.selectDataset(datasetId);
      })
  }

  @Builder
  private MetricCard(metric: string, label: string, color: string) {
    Column() {
      Text(label).fontSize(9).fontColor('#667085')
      Text(this.metricText(metric))
        .fontSize(15).fontWeight(FontWeight.Bold).fontColor(color).margin({ top: 5 })
    }
    .layoutWeight(1)
    .padding({ top: 11, bottom: 11 })
    .backgroundColor('#F7F9FC')
    .borderRadius(10)
  }

  @Builder
  private EvidenceCard(spec: AsrTestCase, row: AsrTestRow, index: number) {
    Column() {
      Row() {
        Text(`${this.datasetOrdinal(index)}`)
          .fontSize(10).fontColor('#246BFD')
          .padding({ left: 8, right: 8, top: 4, bottom: 4 })
          .backgroundColor('#EAF1FF').borderRadius(10)
        Column() {
          Text(`${spec.dataset} · ${spec.sourceId}`)
            .fontSize(13).fontWeight(FontWeight.Bold).fontColor('#172033')
          Text(`${spec.condition} · ${spec.durationSec.toFixed(1)} 秒`)
            .fontSize(9).fontColor('#98A2B3').margin({ top: 2 })
        }
        .alignItems(HorizontalAlign.Start).layoutWeight(1).margin({ left: 8 })
        Text(this.outcomeText(row))
          .fontSize(10).fontWeight(FontWeight.Bold).fontColor(this.outcomeColor(row))
          .padding({ left: 8, right: 8, top: 4, bottom: 4 })
          .backgroundColor(row.state === 'completed' || row.state === 'failed' ? '#FFF0EE' : '#F2F4F7')
          .borderRadius(9)
      }
      .width('100%')

      Row() {
        Text(this.taskText(spec)).fontSize(9).fontColor('#7758E8')
        Blank()
        Text(`说话人 ${spec.speakerIds.join('、')}`)
          .fontSize(9).fontColor('#475467').maxLines(1)
      }
      .width('100%').margin({ top: 8 })

      Column() {
        Text('真实标签').fontSize(9).fontColor('#667085')
        Text(spec.reference).fontSize(13).fontColor('#172033').margin({ top: 4 })
      }
      .alignItems(HorizontalAlign.Start).width('100%').padding(10)
      .backgroundColor('#F1F6FF').borderRadius(9).margin({ top: 8 })

      Column() {
        Text('端侧模型输出').fontSize(9).fontColor('#667085')
        Text(row.state === 'completed' ? (row.hypothesis.length > 0 ? row.hypothesis : '（空结果）') :
          (row.state === 'running' ? '正在识别...' : '等待运行'))
          .fontSize(13).fontColor(row.state === 'completed' ? '#172033' : '#98A2B3').margin({ top: 4 })
      }
      .alignItems(HorizontalAlign.Start).width('100%').padding(10)
      .backgroundColor('#F7F9FC').borderRadius(9).margin({ top: 7 })

      if (row.state === 'completed' && row.score !== undefined) {
        Text(`S ${row.score.substitutions} · D ${row.score.deletions} · I ${row.score.insertions} · ` +
          `处理 ${row.elapsedMs} ms · RTF ${row.audioDurationSec > 0 ?
            ((row.elapsedMs / 1000) / row.audioDurationSec).toFixed(3) : '—'}`)
          .fontSize(9).fontColor('#667085').width('100%').margin({ top: 7 })
      } else if (row.state === 'failed') {
        Text(row.error).fontSize(9).fontColor('#B42318').width('100%').margin({ top: 7 })
      }
      if (spec.speakerTurns.length > 0) {
        Text(`逐段真值 ${spec.speakerTurns.length} 段 · 可继续用于说话人分离评分`)
          .fontSize(9).fontColor('#159A61').width('100%').margin({ top: 6 })
      }

      Row() {
        Text(this.playingId === spec.id ? '正在播放，可直接核对标签' : `${spec.license} · 原始音频`)
          .fontSize(9).fontColor(this.playingId === spec.id ? '#137A4C' : '#667085').layoutWeight(1)
        Button(this.playingId === spec.id ? '停止' : '播放')
          .height(31).fontSize(10)
          .backgroundColor(this.playingId === spec.id ? '#137A4C' : '#EAF1FF')
          .fontColor(this.playingId === spec.id ? '#FFFFFF' : '#246BFD')
          .enabled(!this.running && !this.playbackBusy)
          .onClick((): void => { this.togglePlayback(spec); })
        Button('运行')
          .height(31).fontSize(10).backgroundColor('#EEF0FF').fontColor('#4E55E6')
          .enabled(this.ready && !this.running && !this.playbackBusy && this.playingId.length === 0)
          .margin({ left: 6 })
          .onClick((): void => { this.startSingle(index); })
      }
      .width('100%').padding({ top: 8 }).margin({ top: 8 })
      .border({ width: { top: 0.5 }, color: '#E4E9F0' })
    }
    .width('100%').padding(12).backgroundColor('#FFFFFF')
    .border({ width: 0.7, color: '#E1E6ED' }).borderRadius(13).margin({ bottom: 9 })
  }

  build() {
    Column() {
      Row() {
        Button('返回')
          .fontSize(12).height(34).backgroundColor('#F2F4F7').fontColor('#344054')
          .onClick((): void => { router.back(); })
        Column() {
          Text('中文 ASR 真值评测')
            .fontSize(20).fontWeight(FontWeight.Bold).fontColor('#172033')
          Text(this.statusMessage).fontSize(9).fontColor('#667085').maxLines(2).margin({ top: 2 })
        }
        .alignItems(HorizontalAlign.Start).layoutWeight(1).margin({ left: 10 })
        Column() {
          Text('Paraformer').fontSize(10).fontWeight(FontWeight.Bold).fontColor('#246BFD')
          Text('INT8 · 端侧').fontSize(8).fontColor('#667085').margin({ top: 2 })
        }
      }
      .width('100%').padding({ left: 14, right: 14, top: 12, bottom: 9 })
      .backgroundColor('#FFFFFF')

      Scroll() {
        Column() {
          Scroll() {
            Row({ space: 7 }) {
              this.DatasetTab('aishell1', 'A1')
              this.DatasetTab('aishell2', 'A2')
              this.DatasetTab('aishell3', 'A3')
              this.DatasetTab('aishell4', 'A4')
              this.DatasetTab('aishell5', 'A5')
            }
          }
          .scrollable(ScrollDirection.Horizontal).scrollBar(BarState.Off).width('100%')

          Row({ space: 7 }) {
            this.MetricCard('cer', '微平均 CER', '#246BFD')
            this.MetricCard('exact', '完全一致', '#137A4C')
            this.MetricCard('rtf', '实时系数', '#7758E8')
          }
          .width('100%').margin({ top: 10 })

          Row() {
            Text(`进度 ${this.progressText}`).fontSize(10).fontColor('#475467').layoutWeight(1)
            Button('全部')
              .height(30).fontSize(9)
              .backgroundColor(this.resultFilter === 'all' ? '#EAF1FF' : '#FFFFFF')
              .fontColor(this.resultFilter === 'all' ? '#246BFD' : '#667085')
              .onClick((): void => { this.resultFilter = 'all'; })
            Button(`错误 ${this.filterCount('errors')}`)
              .height(30).fontSize(9).margin({ left: 5 })
              .backgroundColor(this.resultFilter === 'errors' ? '#FFF0EE' : '#FFFFFF')
              .fontColor(this.resultFilter === 'errors' ? '#B42318' : '#667085')
              .onClick((): void => { this.resultFilter = 'errors'; })
            Button(this.running ? '正在运行' : '运行当前 20 条')
              .height(34).fontSize(10).margin({ left: 7 })
              .backgroundColor(this.ready && !this.running ? '#246BFD' : '#B9C8E8')
              .fontColor('#FFFFFF')
              .enabled(this.ready && !this.running && !this.playbackBusy && this.playingId.length === 0)
              .onClick((): void => { this.startBatch(); })
          }
          .width('100%').margin({ top: 10, bottom: 10 })

          if (this.playbackError.length > 0) {
            Text(this.playbackError).fontSize(9).fontColor('#B42318').width('100%').margin({ bottom: 7 })
          }
          ForEach(this.rows, (row: AsrTestRow, index: number) => {
            if (this.shouldShow(index)) {
              this.EvidenceCard(this.cases[index], row, index)
            }
          }, (row: AsrTestRow) => this.rowRenderKey(row))
        }
        .width('100%').padding({ left: 12, right: 12, top: 8, bottom: 28 })
      }
      .layoutWeight(1).width('100%').backgroundColor('#F5F7FA')
    }
    .width('100%').height('100%').backgroundColor('#F5F7FA')
  }
}
