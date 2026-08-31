import { common } from '@kit.AbilityKit';
import { router } from '@kit.ArkUI';
import { ErrorEvent, MessageEvents, util, worker } from '@kit.ArkTS';
import { hilog } from '@kit.PerformanceAnalysisKit';
import {
  AudioUnderstandingFilter,
  AudioUnderstandingPrediction,
  AudioUnderstandingState,
  AudioUnderstandingSummary,
  audioUnderstandingLabelText,
  audioUnderstandingLanguageText,
  audioUnderstandingResultMatchesFilter,
  buildAudioUnderstandingSummary,
  compareAudioUnderstandingTarget,
} from '../common/AudioUnderstandingTypes';
import {
  AudioUnderstandingTestCase,
  buildAudioUnderstandingTestCases,
} from '../common/AudioUnderstandingTestData';
import { RawPlaybackState, RawResourcePlayer } from '../services/audio/RawResourcePlayer';

const LOG_DOMAIN: number = 0x0001;
const LOG_TAG: string = 'UnderstandingTest';

interface UnderstandingWorkerRequest {
  msgType: string;
  context?: common.UIAbilityContext;
  data?: Uint8Array;
  audioFormat?: string;
  requestId?: string;
}

interface UnderstandingWorkerResponse {
  msgType: string;
  message?: string;
  requestId?: string;
  text?: string;
  lang?: string;
  emotion?: string;
  event?: string;
  audioDuration?: number;
  elapsedMs?: number;
}

interface UnderstandingTestRow {
  id: string;
  target: 'emotion' | 'event';
  state: AudioUnderstandingState;
  correct: boolean;
  text: string;
  lang: string;
  emotion: string;
  event: string;
  audioDuration: number;
  elapsedMs: number;
  error: string;
}

@Entry
@Component
struct AudioUnderstandingTestCenterPage {
  @State ready: boolean = false;
  @State running: boolean = false;
  @State statusMessage: string = '正在加载 SenseVoice 声音理解模型...';
  @State progressText: string = '0 / 30';
  @State resultFilter: AudioUnderstandingFilter = 'all';
  @State rows: UnderstandingTestRow[] = [];
  @State playingId: string = '';
  @State playPositionMs: number = 0;
  @State playDurationMs: number = 0;
  @State playbackBusy: boolean = false;
  @State playbackError: string = '';

  private workerInstance: worker.ThreadWorker | undefined = undefined;
  private audioPlayer: RawResourcePlayer = new RawResourcePlayer();
  private cases: AudioUnderstandingTestCase[] = buildAudioUnderstandingTestCases();
  private pendingRequestId: string = '';
  private pendingIndex: number = -1;
  private batchMode: boolean = false;
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
      'entry/ets/workers/StreamingSherpaWorker.ets', { name: 'Audio understanding evaluation' });
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
    instance.postMessage({
      msgType: 'sherpa-understanding-test-init',
      context,
    } as UnderstandingWorkerRequest);
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

  private emptyRows(): UnderstandingTestRow[] {
    const result: UnderstandingTestRow[] = [];
    for (let i: number = 0; i < this.cases.length; i++) {
      result.push({
        id: this.cases[i].id,
        target: this.cases[i].target,
        state: 'waiting',
        correct: false,
        text: '',
        lang: '',
        emotion: '',
        event: '',
        audioDuration: 0,
        elapsedMs: 0,
        error: '',
      });
    }
    return result;
  }

  private onWorkerMessage(event: MessageEvents): void {
    const response: UnderstandingWorkerResponse = event.data as UnderstandingWorkerResponse;
    if (response.msgType === 'sherpa-understanding-test-ready') {
      this.ready = true;
      this.statusMessage = '模型已就绪，可播放音频并查看真实标签与模型输出';
      return;
    }
    if (response.msgType === 'sherpa-understanding-test-done') {
      if (!this.running || !response.requestId || response.requestId !== this.pendingRequestId) {
        return;
      }
      this.acceptResult(response);
      return;
    }
    if (response.msgType === 'sherpa-understanding-test-failed') {
      if (!this.running || !response.requestId || response.requestId !== this.pendingRequestId) {
        return;
      }
      this.acceptFailure(response.message || '模型处理失败');
      return;
    }
    if (response.msgType === 'sherpa-asr-error') {
      this.abortRun(response.message || '声音理解模型异常');
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
    const spec: AudioUnderstandingTestCase = this.cases[index];
    const updated: UnderstandingTestRow[] = this.rows.slice();
    updated[index] = {
      id: spec.id,
      target: spec.target,
      state: 'running',
      correct: false,
      text: '',
      lang: '',
      emotion: '',
      event: '',
      audioDuration: 0,
      elapsedMs: 0,
      error: '',
    };
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
          msgType: 'sherpa-understanding-test',
          requestId,
          data,
          audioFormat: 'wav',
        } as UnderstandingWorkerRequest, [data.buffer]);
      })
      .catch((error: Error): void => {
        if (this.running && token === this.runToken) {
          this.acceptFailure(`读取测试音频失败：${error.message}`);
        }
      });
  }

  private acceptResult(response: UnderstandingWorkerResponse): void {
    if (this.pendingIndex < 0 || this.pendingIndex >= this.cases.length) {
      this.abortRun('测试结果缺少对应样本');
      return;
    }
    const spec: AudioUnderstandingTestCase = this.cases[this.pendingIndex];
    const prediction: AudioUnderstandingPrediction = {
      text: response.text || '',
      lang: response.lang || '',
      emotion: response.emotion || '',
      event: response.event || '',
    };
    const correct: boolean = compareAudioUnderstandingTarget(
      spec.target, spec.expectedLabel, prediction);
    const updated: UnderstandingTestRow[] = this.rows.slice();
    updated[this.pendingIndex] = {
      id: spec.id,
      target: spec.target,
      state: 'completed',
      correct,
      text: prediction.text,
      lang: prediction.lang,
      emotion: prediction.emotion,
      event: prediction.event,
      audioDuration: response.audioDuration || 0,
      elapsedMs: response.elapsedMs || 0,
      error: '',
    };
    this.rows = updated;
    hilog.info(LOG_DOMAIN, LOG_TAG,
      `RESULT id=${spec.id} target=${spec.target} truth=${spec.expectedLabel} ` +
      `lang=${prediction.lang} emotion=${prediction.emotion} event=${prediction.event} correct=${correct}`);
    this.advanceAfterResult();
  }

  private acceptFailure(message: string): void {
    if (this.pendingIndex >= 0 && this.pendingIndex < this.rows.length) {
      const spec: AudioUnderstandingTestCase = this.cases[this.pendingIndex];
      const updated: UnderstandingTestRow[] = this.rows.slice();
      updated[this.pendingIndex] = {
        id: spec.id,
        target: spec.target,
        state: 'failed',
        correct: false,
        text: '',
        lang: '',
        emotion: '',
        event: '',
        audioDuration: 0,
        elapsedMs: 0,
        error: message,
      };
      this.rows = updated;
      hilog.error(LOG_DOMAIN, LOG_TAG, `RESULT id=${spec.id} state=failed error=${message}`);
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
      '带真值声音理解评测完成，可筛选未匹配结果逐条试听' :
      '单条声音理解完成，结果已与真实标签对比';
    const summary: AudioUnderstandingSummary = this.summary();
    hilog.info(LOG_DOMAIN, LOG_TAG,
      `SUMMARY correct=${summary.correct}/${summary.total} emotion=` +
      `${summary.emotionCorrect}/${summary.emotionTotal} event=` +
      `${summary.eventCorrect}/${summary.eventTotal} failed=${summary.failed}`);
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

  private summary(): AudioUnderstandingSummary {
    return buildAudioUnderstandingSummary(this.rows);
  }

  private hasAnyResult(): boolean {
    for (let i: number = 0; i < this.rows.length; i++) {
      if (this.rows[i].state === 'completed' || this.rows[i].state === 'failed') {
        return true;
      }
    }
    return false;
  }

  private metricText(metric: string): string {
    if (!this.hasAnyResult()) {
      return '—';
    }
    const summary: AudioUnderstandingSummary = this.summary();
    if (metric === 'emotion') {
      return this.ratioText(summary.emotionCorrect, summary.emotionTotal);
    }
    if (metric === 'event') {
      return this.ratioText(summary.eventCorrect, summary.eventTotal);
    }
    return this.ratioText(summary.correct, summary.total);
  }

  private ratioText(correct: number, total: number): string {
    const percentage: number = total > 0 ? (correct / total) * 100 : 0;
    return `${correct}/${total}  ${percentage.toFixed(1)}%`;
  }

  private filterCount(filter: AudioUnderstandingFilter): number {
    let count: number = 0;
    for (let i: number = 0; i < this.rows.length; i++) {
      if (audioUnderstandingResultMatchesFilter(filter, this.cases[i].target,
        this.rows[i].state, this.rows[i].correct)) {
        count += 1;
      }
    }
    return count;
  }

  private shouldShow(index: number): boolean {
    return audioUnderstandingResultMatchesFilter(this.resultFilter, this.cases[index].target,
      this.rows[index].state, this.rows[index].correct);
  }

  private outcomeText(row: UnderstandingTestRow): string {
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

  private outcomeColor(row: UnderstandingTestRow): string {
    if (row.state === 'completed' && row.correct) {
      return '#137A4C';
    }
    if (row.state === 'completed' || row.state === 'failed') {
      return '#B42318';
    }
    if (row.state === 'running') {
      return '#4E55E6';
    }
    return '#667085';
  }

  private outcomeSoftColor(row: UnderstandingTestRow): string {
    if (row.state === 'completed' && row.correct) {
      return '#E9F8F0';
    }
    if (row.state === 'completed' || row.state === 'failed') {
      return '#FFF0EE';
    }
    if (row.state === 'running') {
      return '#EEF0FF';
    }
    return '#F2F4F7';
  }

  private targetAccent(spec: AudioUnderstandingTestCase): string {
    return spec.target === 'emotion' ? '#EA7B22' : '#148A70';
  }

  private targetSoftColor(spec: AudioUnderstandingTestCase): string {
    return spec.target === 'emotion' ? '#FFF4EA' : '#EAF8F4';
  }

  private targetName(spec: AudioUnderstandingTestCase): string {
    return spec.target === 'emotion' ? '情感' : '事件';
  }

  private predictedTargetText(spec: AudioUnderstandingTestCase,
    row: UnderstandingTestRow): string {
    if (row.state === 'waiting') {
      return '等待模型输出';
    }
    if (row.state === 'running') {
      return '正在识别';
    }
    if (row.state === 'failed') {
      return '处理失败';
    }
    const label: string = spec.target === 'emotion' ? row.emotion : row.event;
    return audioUnderstandingLabelText(label);
  }

  private togglePlayback(spec: AudioUnderstandingTestCase): void {
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
    const format = (valueMs: number): string => {
      const totalSeconds: number = Math.max(0, Math.floor(valueMs / 1000));
      const seconds: number = totalSeconds % 60;
      return `0:${seconds < 10 ? `0${seconds}` : `${seconds}`}`;
    };
    return `${format(positionMs)} / ${format(durationMs)}`;
  }

  @Builder
  private MetricCard(metric: string, label: string, color: string, softColor: string) {
    Column() {
      Text(label)
        .fontSize(10)
        .fontColor('#667085')
      Text(this.metricText(metric))
        .fontSize(16)
        .fontWeight(FontWeight.Bold)
        .fontColor(color)
        .margin({ top: 5 })
    }
    .layoutWeight(1)
    .padding({ top: 12, bottom: 12 })
    .backgroundColor(softColor)
    .border({ width: 0.5, color: '#E2E7EF' })
    .borderRadius(11)
  }

  @Builder
  private FilterButton(filter: AudioUnderstandingFilter, label: string) {
    Button(`${label} ${this.filterCount(filter)}`)
      .layoutWeight(1)
      .height(34)
      .fontSize(10)
      .fontWeight(this.resultFilter === filter ? FontWeight.Bold : FontWeight.Normal)
      .backgroundColor(this.resultFilter === filter ? '#EEF0FF' : '#FFFFFF')
      .fontColor(this.resultFilter === filter ? '#4E55E6' : '#667085')
      .border({
        width: this.resultFilter === filter ? 1 : 0.5,
        color: this.resultFilter === filter ? '#6570F2' : '#DCE2EA',
      })
      .onClick((): void => {
        this.resultFilter = filter;
      })
  }

  @Builder
  private EvidenceCard(spec: AudioUnderstandingTestCase, row: UnderstandingTestRow, index: number) {
    Column() {
      Row() {
        Text(`${index + 1}`)
          .fontSize(10)
          .fontColor('#4E55E6')
          .padding({ left: 8, right: 8, top: 4, bottom: 4 })
          .backgroundColor('#EEF0FF')
          .borderRadius(10)
        Column() {
          Text(`${spec.dataset} · ${spec.expectedDisplayName}`)
            .fontSize(14)
            .fontWeight(FontWeight.Bold)
            .fontColor('#172033')
          Text(spec.sourceId)
            .fontSize(9)
            .fontColor('#98A2B3')
            .margin({ top: 2 })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
        .margin({ left: 8 })
        Text(this.targetName(spec))
          .fontSize(10)
          .fontColor(this.targetAccent(spec))
          .padding({ left: 8, right: 8, top: 4, bottom: 4 })
          .backgroundColor(this.targetSoftColor(spec))
          .borderRadius(9)
        Text(this.outcomeText(row))
          .fontSize(10)
          .fontWeight(FontWeight.Bold)
          .fontColor(this.outcomeColor(row))
          .padding({ left: 8, right: 8, top: 4, bottom: 4 })
          .backgroundColor(this.outcomeSoftColor(row))
          .borderRadius(9)
          .margin({ left: 6 })
      }
      .width('100%')

      Row({ space: 8 }) {
        Column() {
          Text('真实标签')
            .fontSize(10)
            .fontColor('#667085')
          Text(spec.expectedDisplayName)
            .fontSize(20)
            .fontWeight(FontWeight.Bold)
            .fontColor(this.targetAccent(spec))
            .margin({ top: 7 })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
        .padding(11)
        .backgroundColor(this.targetSoftColor(spec))
        .borderRadius(10)

        Column() {
          Text('SenseVoice 输出')
            .fontSize(10)
            .fontColor('#667085')
          Text(this.predictedTargetText(spec, row))
            .fontSize(18)
            .fontWeight(FontWeight.Bold)
            .fontColor(row.state === 'completed' ? this.outcomeColor(row) : '#98A2B3')
            .maxLines(2)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
            .margin({ top: 7 })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
        .padding(11)
        .backgroundColor(row.state === 'completed' && !row.correct ? '#FFF0EE' : '#F5F7FA')
        .borderRadius(10)
      }
      .width('100%')
      .margin({ top: 10 })

      if (row.state === 'completed') {
        Row({ space: 7 }) {
          Text(`语种 ${audioUnderstandingLanguageText(row.lang)}`)
            .fontSize(10)
            .fontColor('#475467')
          Text(`情感 ${audioUnderstandingLabelText(row.emotion)}`)
            .fontSize(10)
            .fontColor('#475467')
          Text(`事件 ${audioUnderstandingLabelText(row.event)}`)
            .fontSize(10)
            .fontColor('#475467')
        }
        .width('100%')
        .margin({ top: 10 })
        if (row.text.length > 0) {
          Text(`识别文本：${row.text}`)
            .fontSize(10)
            .fontColor('#667085')
            .maxLines(2)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
            .width('100%')
            .margin({ top: 6 })
        }
        Text(`处理 ${row.elapsedMs} ms · 音频 ${row.audioDuration.toFixed(1)} 秒`)
          .fontSize(9)
          .fontColor('#98A2B3')
          .width('100%')
          .margin({ top: 5 })
      } else if (row.state === 'failed') {
        Text(row.error)
          .fontSize(10)
          .fontColor('#B42318')
          .width('100%')
          .padding(8)
          .backgroundColor('#FFF0EE')
          .borderRadius(8)
          .margin({ top: 9 })
      }

      Text(spec.truthBasis)
        .fontSize(9)
        .fontColor('#98A2B3')
        .width('100%')
        .margin({ top: 8 })

      Row() {
        Text(this.playingId === spec.id ?
          `正在播放 ${this.formatPlaybackTime(this.playPositionMs, this.playDurationMs)}` :
          `${spec.license} · 原始评分音频`)
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
          .backgroundColor('#EEF0FF')
          .fontColor('#4E55E6')
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
          Text('声音理解评测')
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
          Text('SenseVoice')
            .fontSize(11)
            .fontWeight(FontWeight.Bold)
            .fontColor('#4E55E6')
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
          Text('带真实标签的固定评测')
            .fontSize(14)
            .fontWeight(FontWeight.Bold)
            .fontColor('#303682')
          Blank()
          Text('30 条 · 可播放')
            .fontSize(10)
            .fontColor('#4E55E6')
        }
        .width('100%')
        Text('CREMA-D 评测 6 类情感；VocalSound 评测咳嗽、笑声和喷嚏。只比较目标字段。')
          .fontSize(10)
          .fontColor('#667085')
          .width('100%')
          .margin({ top: 5 })
        Row({ space: 7 }) {
          this.MetricCard('all', '总体', '#4E55E6', '#F1F2FF')
          this.MetricCard('emotion', '情感', '#D96816', '#FFF4EA')
          this.MetricCard('event', '事件', '#137A62', '#EAF8F4')
        }
        .width('100%')
        .margin({ top: 10 })
        if (this.hasAnyResult()) {
          Text(`处理失败 ${this.summary().failed} 条；失败样本保留在准确率分母中。`)
            .fontSize(9)
            .fontColor('#667085')
            .width('100%')
            .margin({ top: 7 })
        }
        if (this.playbackError.length > 0) {
          Text(this.playbackError)
            .fontSize(9)
            .fontColor('#B42318')
            .width('100%')
            .margin({ top: 5 })
        }
      }
      .alignItems(HorizontalAlign.Start)
      .width('100%')
      .padding(13)
      .backgroundColor('#F8F8FF')
      .border({ width: 0.5, color: '#DDE0FF' })
      .borderRadius(14)
      .margin({ left: 16, right: 16, bottom: 9 })

      Row({ space: 6 }) {
        this.FilterButton('all', '全部')
        this.FilterButton('emotion', '情感')
        this.FilterButton('event', '事件')
        this.FilterButton('errors', '未匹配')
      }
      .width('100%')
      .padding({ left: 16, right: 16, bottom: 8 })

      List() {
        ForEach(this.rows, (row: UnderstandingTestRow, index: number): void => {
          if (this.shouldShow(index)) {
            ListItem() {
              this.EvidenceCard(this.cases[index], row, index)
            }
          }
        }, (row: UnderstandingTestRow): string =>
          `${row.id}-${row.state}-${row.elapsedMs}-${this.resultFilter}`)
      }
      .layoutWeight(1)
      .width('100%')
      .padding({ left: 16, right: 16 })
      .scrollBar(BarState.Off)

      Column() {
        Text('模型输出来自端侧 SenseVoice；测试不会写入个人声纹或会议数据。')
          .fontSize(9)
          .fontColor('#667085')
          .margin({ bottom: 6 })
        Button(this.running && this.batchMode ?
          `正在运行 ${this.progressText}` : '批量运行 30 条')
          .width('100%')
          .height(46)
          .backgroundColor('#4E55E6')
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
