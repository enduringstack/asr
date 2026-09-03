import { common } from '@kit.AbilityKit';
import { router } from '@kit.ArkUI';
import { ErrorEvent, MessageEvents, util, worker } from '@kit.ArkTS';
import { hilog } from '@kit.PerformanceAnalysisKit';
import {
  AsrAnalyzedTurn,
  AsrCaseResult,
  AsrDiarizationScore,
  AsrDiarizationTurn,
  AsrEditScore,
  AsrEvaluationFilter,
  AsrEvaluationProtocol,
  AsrEvaluationState,
  AsrEvaluationSummary,
  AsrOverlapDetectionScore,
  AsrPermutationScore,
  AsrSpeakerTranscript,
  AsrTurnCaseResult,
  AsrTurnEvaluationSummary,
  analyzeAsrTurns,
  asrResultMatchesFilter,
  buildAsrEvaluationSummary,
  buildAsrTurnEvaluationSummary,
  groupSpeakerTranscripts,
  scoreAsrText,
  scoreSpeakerDiarization,
  scoreSpeakerOverlapDetection,
  scoreSpeakerPermutation,
} from '../common/AsrEvaluationTypes';
import { AsrTestCase, GroundTruthSpeakerTurn, buildAsrTestCases } from '../common/AsrTestData';
import { SpeakerSearchProfile } from '../common/SpeakerTypes';
import { RawPlaybackState, RawResourcePlayer } from '../services/audio/RawResourcePlayer';
import { SpeakerProfileStore } from '../services/speaker/SpeakerProfileStore';

const LOG_DOMAIN: number = 0x0001;
const LOG_TAG: string = 'AsrTestCenter';

interface AsrWorkerRequest {
  msgType: string;
  context?: common.UIAbilityContext;
  data?: Uint8Array;
  audioFormat?: string;
  requestId?: string;
  evaluationSegments?: AsrEvaluationSegmentRequest[];
  meetingId?: string;
  voiceprints?: SpeakerSearchProfile[];
}

interface AsrEvaluationSegmentRequest {
  segmentId: string;
  startSec: number;
  endSec: number;
}

interface AsrEvaluationSegmentResponse {
  segmentId: string;
  rawText: string;
  audioDuration: number;
  elapsedMs: number;
  error?: string;
}

interface AsrDiarizationSegmentResponse extends AsrEvaluationSegmentResponse {
  clusterId: string;
  startSec: number;
  endSec: number;
  overlapped: boolean;
}

interface AsrModelOverlapRange {
  startSec: number;
  endSec: number;
}

interface AsrEvaluationSpeakerStream {
  clusterId: string;
  rawText: string;
  text: string;
  personId?: string;
  displayName?: string;
  matchScore?: number;
}

interface AsrWorkerResponse {
  msgType: string;
  message?: string;
  requestId?: string;
  rawText?: string;
  audioDuration?: number;
  elapsedMs?: number;
  evaluationSegmentResults?: AsrEvaluationSegmentResponse[];
  diarizationSegmentResults?: AsrDiarizationSegmentResponse[];
  modelOverlapRanges?: AsrModelOverlapRange[];
  diarizationElapsedMs?: number;
  evaluationSpeakerStreams?: AsrEvaluationSpeakerStream[];
}

interface AsrTurnRow {
  segmentId: string;
  speakerId: string;
  startSec: number;
  endSec: number;
  reference: string;
  overlapped: boolean;
  state: AsrEvaluationState;
  hypothesis: string;
  score?: AsrEditScore;
  audioDurationSec: number;
  elapsedMs: number;
  error: string;
}

interface AsrPredictedTurnRow {
  segmentId: string;
  clusterId: string;
  startSec: number;
  endSec: number;
  overlapped: boolean;
  hypothesis: string;
  audioDurationSec: number;
  elapsedMs: number;
  error: string;
}

interface AsrTestRow {
  id: string;
  protocol: AsrEvaluationProtocol;
  state: AsrEvaluationState;
  hypothesis: string;
  score?: AsrEditScore;
  audioDurationSec: number;
  elapsedMs: number;
  error: string;
  turnRows: AsrTurnRow[];
  predictedTurns?: AsrPredictedTurnRow[];
  speakerStreams?: AsrEvaluationSpeakerStream[];
  diarizationScore?: AsrDiarizationScore;
  diarizationElapsedMs?: number;
  modelOverlapRanges?: AsrModelOverlapRange[];
  overlapDetectionScore?: AsrOverlapDetectionScore;
}

@Entry
@Component
struct AsrTestCenterPage {
  @State ready: boolean = false;
  @State running: boolean = false;
  @State statusMessage: string = '正在加载 Paraformer、Pyannote 和 CAM++ 模型...';
  @State selectedDataset: string = 'aishell1';
  @State evaluationProtocol: AsrEvaluationProtocol = 'full_window';
  @State progressText: string = '0 / 20';
  @State resultFilter: AsrEvaluationFilter = 'all';
  @State rows: AsrTestRow[] = [];
  @State playingId: string = '';
  @State playPositionMs: number = 0;
  @State playDurationMs: number = 0;
  @State playbackBusy: boolean = false;
  @State playbackError: string = '';

  private workerInstance: worker.ThreadWorker | undefined = undefined;
  private profileStore: SpeakerProfileStore = SpeakerProfileStore.getInstance();
  private audioPlayer: RawResourcePlayer = new RawResourcePlayer();
  private cases: AsrTestCase[] = buildAsrTestCases();
  private runQueue: number[] = [];
  private queuePosition: number = -1;
  private pendingIndex: number = -1;
  private pendingRequestId: string = '';
  private runToken: number = 0;
  private protocolByDataset: Map<string, AsrEvaluationProtocol> = new Map();

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
    this.initializeWorker(context, instance);
  }

  private async initializeWorker(context: common.UIAbilityContext,
    instance: worker.ThreadWorker): Promise<void> {
    let voiceprints: SpeakerSearchProfile[] = [];
    try {
      await this.profileStore.init(context);
      voiceprints = this.profileStore.searchProfiles();
    } catch (error) {
      hilog.warn(LOG_DOMAIN, LOG_TAG, `load voiceprints for evaluation failed: ${error}`);
    }
    if (this.workerInstance !== instance) {
      return;
    }
    instance.postMessage({
      msgType: 'sherpa-asr-eval-init',
      context,
      voiceprints,
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
        protocol: this.defaultProtocolForDataset(this.cases[index].datasetId),
        state: 'waiting',
        hypothesis: '',
        audioDurationSec: 0,
        elapsedMs: 0,
        error: '',
        turnRows: [],
      });
    }
    return result;
  }

  private defaultProtocolForDataset(datasetId: string): AsrEvaluationProtocol {
    return datasetId === 'aishell4' || datasetId === 'aishell5' ? 'speaker_pipeline' : 'full_window';
  }

  private protocolForDataset(datasetId: string): AsrEvaluationProtocol {
    return this.protocolByDataset.get(datasetId) ?? this.defaultProtocolForDataset(datasetId);
  }

  private supportsOracleTurns(datasetId: string): boolean {
    const indices: number[] = this.datasetIndices(datasetId);
    for (let i: number = 0; i < indices.length; i++) {
      if (this.cases[indices[i]].speakerTurns.length > 0) {
        return true;
      }
    }
    return false;
  }

  private createTurnRows(spec: AsrTestCase, state: AsrEvaluationState,
    error: string = ''): AsrTurnRow[] {
    const turns: AsrAnalyzedTurn[] = analyzeAsrTurns(spec.speakerTurns);
    const rows: AsrTurnRow[] = [];
    for (let i: number = 0; i < turns.length; i++) {
      rows.push({
        segmentId: `${spec.id}-turn-${i}`,
        speakerId: turns[i].speakerId,
        startSec: turns[i].startSec,
        endSec: turns[i].endSec,
        reference: turns[i].text,
        overlapped: turns[i].overlapped,
        state,
        hypothesis: '',
        score: state === 'failed' ? scoreAsrText(turns[i].text, '') : undefined,
        audioDurationSec: Math.max(0, turns[i].endSec - turns[i].startSec),
        elapsedMs: 0,
        error,
      });
    }
    return rows;
  }

  private selectProtocol(protocol: AsrEvaluationProtocol): void {
    if (this.running || this.playingId.length > 0 || protocol === this.evaluationProtocol) {
      return;
    }
    if ((protocol === 'oracle_turns' || protocol === 'speaker_pipeline') &&
      !this.supportsOracleTurns(this.selectedDataset)) {
      return;
    }
    this.evaluationProtocol = protocol;
    this.protocolByDataset.set(this.selectedDataset, protocol);
    const updated: AsrTestRow[] = this.rows.slice();
    const indices: number[] = this.datasetIndices(this.selectedDataset);
    for (let i: number = 0; i < indices.length; i++) {
      const index: number = indices[i];
      updated[index] = {
        id: this.cases[index].id,
        protocol,
        state: 'waiting',
        hypothesis: '',
        audioDurationSec: 0,
        elapsedMs: 0,
        error: '',
        turnRows: [],
      };
    }
    this.rows = updated;
    this.resultFilter = 'all';
    this.progressText = `0 / ${indices.length}`;
    if (protocol === 'speaker_pipeline') {
      this.statusMessage = '完整音频将由端侧模型预测说话人边界与说话人簇';
    } else if (protocol === 'oracle_turns') {
      this.statusMessage = 'ASR 上限诊断只向模型发送真实时间戳，不代表真实会议链路';
    } else {
      this.statusMessage = '整段转写保留原始单路 ASR 基线，适合单说话人数据';
    }
  }

  private onWorkerMessage(event: MessageEvents): void {
    const response: AsrWorkerResponse = event.data as AsrWorkerResponse;
    if (response.msgType === 'sherpa-asr-eval-ready') {
      this.ready = true;
      this.statusMessage = '模型已就绪；分段、聚类、识别与声纹匹配均由手机现场计算';
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
    this.evaluationProtocol = this.protocolForDataset(datasetId);
    this.resultFilter = 'all';
    this.progressText = `${this.evaluatedSampleCount(datasetId)} / ${this.datasetIndices(datasetId).length}`;
    this.statusMessage = this.evaluationProtocol === 'speaker_pipeline' ?
      '已切换数据集；默认运行完整音频的端到端多人会议链路' :
      '已切换数据集，可运行 20 条或逐条试听';
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
        protocol: this.evaluationProtocol,
        state: 'waiting',
        hypothesis: '',
        audioDurationSec: 0,
        elapsedMs: 0,
        error: '',
        turnRows: [],
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
      protocol: this.evaluationProtocol,
      state: 'running',
      hypothesis: '',
      audioDurationSec: 0,
      elapsedMs: 0,
      error: '',
      turnRows: this.evaluationProtocol === 'oracle_turns' ?
        this.createTurnRows(spec, 'running') : [],
    };
    this.rows = updated;
    this.pendingIndex = index;
    this.statusMessage = this.evaluationProtocol === 'speaker_pipeline' ?
      `正在对完整音频分段、聚类和识别 ${spec.dataset} · ${spec.sourceId}` :
      `正在识别 ${spec.dataset} · ${spec.sourceId}`;
    const context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;
    context.resourceManager.getRawFileContent(spec.audioPath)
      .then((data: Uint8Array): void => {
        if (!this.running || token !== this.runToken || !this.workerInstance) {
          return;
        }
        const requestId: string = util.generateRandomUUID(true);
        this.pendingRequestId = requestId;
        const segments: AsrEvaluationSegmentRequest[] = [];
        if (this.evaluationProtocol === 'oracle_turns') {
          const analyzedTurns: AsrAnalyzedTurn[] = analyzeAsrTurns(spec.speakerTurns);
          for (let i: number = 0; i < analyzedTurns.length; i++) {
            segments.push({
              segmentId: `${spec.id}-turn-${i}`,
              startSec: analyzedTurns[i].startSec,
              endSec: analyzedTurns[i].endSec,
            });
          }
        }
        this.workerInstance.postMessage({
          msgType: this.evaluationProtocol === 'speaker_pipeline' ?
            'sherpa-asr-eval-diarization' : 'sherpa-asr-eval',
          requestId,
          data,
          audioFormat: 'wav',
          evaluationSegments: segments,
          meetingId: spec.id,
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
    if (this.evaluationProtocol === 'speaker_pipeline' && spec.speakerTurns.length > 0) {
      this.acceptSpeakerPipelineResult(spec, response);
      return;
    }
    if (this.evaluationProtocol === 'oracle_turns' && spec.speakerTurns.length > 0) {
      this.acceptOracleTurnResult(spec, response);
      return;
    }
    const hypothesis: string = response.rawText || '';
    const score: AsrEditScore = scoreAsrText(spec.reference, hypothesis);
    const updated: AsrTestRow[] = this.rows.slice();
    updated[this.pendingIndex] = {
      id: spec.id,
      protocol: this.evaluationProtocol,
      state: 'completed',
      hypothesis,
      score,
      audioDurationSec: response.audioDuration || spec.durationSec,
      elapsedMs: response.elapsedMs || 0,
      error: '',
      turnRows: [],
    };
    this.rows = updated;
    hilog.info(LOG_DOMAIN, LOG_TAG,
      `RESULT id=${spec.id} cer=${score.cer.toFixed(4)} s=${score.substitutions} ` +
      `d=${score.deletions} i=${score.insertions} hypothesis=${hypothesis}`);
    this.advanceAfterResult();
  }

  private findSegmentResult(results: AsrEvaluationSegmentResponse[],
    segmentId: string): AsrEvaluationSegmentResponse | undefined {
    for (let i: number = 0; i < results.length; i++) {
      if (results[i].segmentId === segmentId) {
        return results[i];
      }
    }
    return undefined;
  }

  private acceptOracleTurnResult(spec: AsrTestCase, response: AsrWorkerResponse): void {
    const baseRows: AsrTurnRow[] = this.createTurnRows(spec, 'running');
    const results: AsrEvaluationSegmentResponse[] = response.evaluationSegmentResults ?? [];
    const turnRows: AsrTurnRow[] = [];
    let combinedHypothesis: string = '';
    let totalDuration: number = 0;
    let totalElapsedMs: number = 0;
    for (let i: number = 0; i < baseRows.length; i++) {
      const base: AsrTurnRow = baseRows[i];
      const segment: AsrEvaluationSegmentResponse | undefined =
        this.findSegmentResult(results, base.segmentId);
      const error: string = segment?.error ?? (segment === undefined ? '模型未返回该分段' : '');
      const hypothesis: string = segment?.rawText ?? '';
      const duration: number = segment?.audioDuration ?? base.audioDurationSec;
      const elapsedMs: number = segment?.elapsedMs ?? 0;
      const score: AsrEditScore = scoreAsrText(base.reference, hypothesis);
      combinedHypothesis += hypothesis;
      totalDuration += duration;
      totalElapsedMs += elapsedMs;
      turnRows.push({
        segmentId: base.segmentId,
        speakerId: base.speakerId,
        startSec: base.startSec,
        endSec: base.endSec,
        reference: base.reference,
        overlapped: base.overlapped,
        state: error.length > 0 ? 'failed' : 'completed',
        hypothesis,
        score,
        audioDurationSec: duration,
        elapsedMs,
        error,
      });
    }
    const score: AsrEditScore = scoreAsrText(spec.reference, combinedHypothesis);
    const updated: AsrTestRow[] = this.rows.slice();
    updated[this.pendingIndex] = {
      id: spec.id,
      protocol: 'oracle_turns',
      state: 'completed',
      hypothesis: combinedHypothesis,
      score,
      audioDurationSec: totalDuration,
      elapsedMs: totalElapsedMs,
      error: '',
      turnRows,
    };
    this.rows = updated;
    const turnSummary: AsrTurnEvaluationSummary = this.turnSummaryForRows(turnRows);
    hilog.info(LOG_DOMAIN, LOG_TAG,
      `RESULT id=${spec.id} protocol=oracle_turns non_overlap_cer=` +
      `${turnSummary.nonOverlap.microCer.toFixed(4)} overlap_cer=${turnSummary.overlap.microCer.toFixed(4)}`);
    this.advanceAfterResult();
  }

  private acceptSpeakerPipelineResult(spec: AsrTestCase, response: AsrWorkerResponse): void {
    const modelSegments: AsrDiarizationSegmentResponse[] = response.diarizationSegmentResults ?? [];
    const predictedTurns: AsrPredictedTurnRow[] = [];
    const referenceDiarization: AsrDiarizationTurn[] = [];
    const hypothesisDiarization: AsrDiarizationTurn[] = [];
    const referenceTexts: AsrSpeakerTranscript[] = [];
    const hypothesisTexts: AsrSpeakerTranscript[] = [];
    for (let i: number = 0; i < spec.speakerTurns.length; i++) {
      referenceDiarization.push({
        speakerId: spec.speakerTurns[i].speakerId,
        startSec: spec.speakerTurns[i].startSec,
        endSec: spec.speakerTurns[i].endSec,
      });
      referenceTexts.push({
        speakerId: spec.speakerTurns[i].speakerId,
        text: spec.speakerTurns[i].text,
      });
    }
    for (let i: number = 0; i < modelSegments.length; i++) {
      const segment: AsrDiarizationSegmentResponse = modelSegments[i];
      const error: string = segment.error ?? '';
      predictedTurns.push({
        segmentId: segment.segmentId,
        clusterId: segment.clusterId,
        startSec: segment.startSec,
        endSec: segment.endSec,
        overlapped: segment.overlapped,
        hypothesis: segment.rawText,
        audioDurationSec: segment.audioDuration,
        elapsedMs: segment.elapsedMs,
        error,
      });
      hypothesisDiarization.push({
        speakerId: segment.clusterId,
        startSec: segment.startSec,
        endSec: segment.endSec,
      });
      hypothesisTexts.push({ speakerId: segment.clusterId, text: segment.rawText });
    }
    const referenceStreams: AsrSpeakerTranscript[] = groupSpeakerTranscripts(referenceTexts);
    const hypothesisStreams: AsrSpeakerTranscript[] = groupSpeakerTranscripts(hypothesisTexts);
    const cpCer: AsrPermutationScore = scoreSpeakerPermutation(referenceStreams, hypothesisStreams);
    const diarizationScore: AsrDiarizationScore =
      scoreSpeakerDiarization(referenceDiarization, hypothesisDiarization);
    const modelOverlapRanges: AsrModelOverlapRange[] = response.modelOverlapRanges ?? [];
    const overlapDetectionScore: AsrOverlapDetectionScore =
      scoreSpeakerOverlapDetection(referenceDiarization, modelOverlapRanges);
    const speakerStreams: AsrEvaluationSpeakerStream[] = response.evaluationSpeakerStreams ?? [];
    let displayText: string = '';
    for (let i: number = 0; i < speakerStreams.length; i++) {
      const identity: string = speakerStreams[i].displayName ?? speakerStreams[i].clusterId;
      displayText += `${displayText.length > 0 ? '\n' : ''}[${identity}] ${speakerStreams[i].text}`;
    }
    if (displayText.length === 0) {
      displayText = response.rawText ?? '';
    }
    const updated: AsrTestRow[] = this.rows.slice();
    updated[this.pendingIndex] = {
      id: spec.id,
      protocol: 'speaker_pipeline',
      state: 'completed',
      hypothesis: displayText,
      score: cpCer.score,
      audioDurationSec: response.audioDuration ?? spec.durationSec,
      elapsedMs: response.elapsedMs ?? 0,
      error: modelSegments.length === 0 ? '模型未检测到说话人片段' : '',
      turnRows: [],
      predictedTurns,
      speakerStreams,
      diarizationScore,
      diarizationElapsedMs: response.diarizationElapsedMs ?? 0,
      modelOverlapRanges,
      overlapDetectionScore,
    };
    this.rows = updated;
    hilog.info(LOG_DOMAIN, LOG_TAG,
      `RESULT id=${spec.id} protocol=speaker_pipeline cpcer=${cpCer.score.cer.toFixed(4)} ` +
      `der=${diarizationScore.der.toFixed(4)} turns=${predictedTurns.length}`);
    this.advanceAfterResult();
  }

  private acceptFailure(message: string): void {
    if (this.pendingIndex >= 0 && this.pendingIndex < this.rows.length) {
      const spec: AsrTestCase = this.cases[this.pendingIndex];
      const updated: AsrTestRow[] = this.rows.slice();
      updated[this.pendingIndex] = {
        id: spec.id,
        protocol: this.evaluationProtocol,
        state: 'failed',
        hypothesis: '',
        audioDurationSec: spec.durationSec,
        elapsedMs: 0,
        error: message,
        turnRows: this.evaluationProtocol === 'oracle_turns' ?
          this.createTurnRows(spec, 'failed', message) : [],
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
    if (this.evaluationProtocol === 'speaker_pipeline') {
      const diarization: AsrDiarizationScore = this.pipelineDiarizationSummary(this.selectedDataset);
      const overlap: AsrOverlapDetectionScore = this.pipelineOverlapSummary(this.selectedDataset);
      this.statusMessage = `评测完成：cpCER ${(summary.microCer * 100).toFixed(2)}%，` +
        `DER ${(diarization.der * 100).toFixed(2)}%，重叠检测召回 ${(overlap.recall * 100).toFixed(1)}%`;
    } else if (this.evaluationProtocol === 'oracle_turns') {
      const turnSummary: AsrTurnEvaluationSummary = this.turnSummary(this.selectedDataset);
      const overlapText: string = turnSummary.overlap.evaluated > 0 ?
        `${(turnSummary.overlap.microCer * 100).toFixed(2)}%` : '无重叠段';
      this.statusMessage = `评测完成：非重叠 CER ` +
        `${(turnSummary.nonOverlap.microCer * 100).toFixed(2)}%，重叠 CER ${overlapText}`;
    } else {
      this.statusMessage =
        `评测完成：微平均 CER ${(summary.microCer * 100).toFixed(2)}%，可筛选错误逐条试听`;
    }
    hilog.info(LOG_DOMAIN, LOG_TAG,
      `SUMMARY dataset=${this.selectedDataset} completed=${summary.completed} failed=${summary.failed} ` +
      `protocol=${this.evaluationProtocol} cer=${summary.microCer.toFixed(4)} rtf=${summary.rtf.toFixed(4)}`);
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
    const protocol: AsrEvaluationProtocol = datasetId === this.selectedDataset ?
      this.evaluationProtocol : this.protocolForDataset(datasetId);
    if (protocol === 'oracle_turns') {
      return this.turnSummary(datasetId).all;
    }
    const resultRows: AsrCaseResult[] = [];
    for (let index: number = 0; index < this.cases.length; index++) {
      if (this.cases[index].datasetId !== datasetId || this.rows[index].protocol !== protocol) {
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

  private pipelineDiarizationSummary(datasetId: string): AsrDiarizationScore {
    const result: AsrDiarizationScore = {
      der: 0,
      referenceSpeakerSec: 0,
      missedSpeakerSec: 0,
      falseAlarmSpeakerSec: 0,
      confusionSpeakerSec: 0,
      referenceOverlapSec: 0,
      predictedOverlapSec: 0,
      overlapTruePositiveSec: 0,
      overlapRecall: 0,
      assignments: [],
    };
    for (let i: number = 0; i < this.cases.length; i++) {
      if (this.cases[i].datasetId !== datasetId ||
        this.rows[i].protocol !== 'speaker_pipeline' ||
        this.rows[i].diarizationScore === undefined) {
        continue;
      }
      const score: AsrDiarizationScore = this.rows[i].diarizationScore as AsrDiarizationScore;
      result.referenceSpeakerSec += score.referenceSpeakerSec;
      result.missedSpeakerSec += score.missedSpeakerSec;
      result.falseAlarmSpeakerSec += score.falseAlarmSpeakerSec;
      result.confusionSpeakerSec += score.confusionSpeakerSec;
      result.referenceOverlapSec += score.referenceOverlapSec;
      result.predictedOverlapSec += score.predictedOverlapSec;
      result.overlapTruePositiveSec += score.overlapTruePositiveSec;
    }
    const errors: number = result.missedSpeakerSec + result.falseAlarmSpeakerSec +
      result.confusionSpeakerSec;
    result.der = result.referenceSpeakerSec > 0 ? errors / result.referenceSpeakerSec : 0;
    result.overlapRecall = result.referenceOverlapSec > 0 ?
      result.overlapTruePositiveSec / result.referenceOverlapSec : 0;
    return result;
  }

  private pipelineOverlapSummary(datasetId: string): AsrOverlapDetectionScore {
    const result: AsrOverlapDetectionScore = {
      referenceOverlapSec: 0,
      predictedOverlapSec: 0,
      truePositiveSec: 0,
      recall: 0,
    };
    for (let i: number = 0; i < this.cases.length; i++) {
      if (this.cases[i].datasetId !== datasetId ||
        this.rows[i].protocol !== 'speaker_pipeline' ||
        this.rows[i].overlapDetectionScore === undefined) {
        continue;
      }
      const score: AsrOverlapDetectionScore =
        this.rows[i].overlapDetectionScore as AsrOverlapDetectionScore;
      result.referenceOverlapSec += score.referenceOverlapSec;
      result.predictedOverlapSec += score.predictedOverlapSec;
      result.truePositiveSec += score.truePositiveSec;
    }
    result.recall = result.referenceOverlapSec > 0 ?
      result.truePositiveSec / result.referenceOverlapSec : 0;
    return result;
  }

  private turnSummaryForRows(rows: AsrTurnRow[]): AsrTurnEvaluationSummary {
    const resultRows: AsrTurnCaseResult[] = [];
    for (let i: number = 0; i < rows.length; i++) {
      resultRows.push({
        state: rows[i].state,
        score: rows[i].score,
        audioDurationSec: rows[i].audioDurationSec,
        elapsedMs: rows[i].elapsedMs,
        overlapped: rows[i].overlapped,
      });
    }
    return buildAsrTurnEvaluationSummary(resultRows);
  }

  private turnSummary(datasetId: string): AsrTurnEvaluationSummary {
    const turnRows: AsrTurnRow[] = [];
    for (let index: number = 0; index < this.cases.length; index++) {
      if (this.cases[index].datasetId !== datasetId ||
        this.rows[index].protocol !== 'oracle_turns') {
        continue;
      }
      for (let turnIndex: number = 0; turnIndex < this.rows[index].turnRows.length; turnIndex++) {
        turnRows.push(this.rows[index].turnRows[turnIndex]);
      }
    }
    return this.turnSummaryForRows(turnRows);
  }

  private evaluatedSampleCount(datasetId: string): number {
    const protocol: AsrEvaluationProtocol = datasetId === this.selectedDataset ?
      this.evaluationProtocol : this.protocolForDataset(datasetId);
    let count: number = 0;
    for (let i: number = 0; i < this.cases.length; i++) {
      if (this.cases[i].datasetId === datasetId && this.rows[i].protocol === protocol &&
        (this.rows[i].state === 'completed' || this.rows[i].state === 'failed')) {
        count += 1;
      }
    }
    return count;
  }

  private rowMatchesFilter(row: AsrTestRow, filter: AsrEvaluationFilter): boolean {
    if (filter === 'all') {
      return true;
    }
    if (row.protocol === 'oracle_turns' && row.turnRows.length > 0) {
      for (let i: number = 0; i < row.turnRows.length; i++) {
        const turn: AsrTurnRow = row.turnRows[i];
        if (turn.state === 'failed' ||
          (turn.state === 'completed' && turn.score !== undefined && !turn.score.exact)) {
          return true;
        }
      }
      return false;
    }
    return asrResultMatchesFilter(filter, row.state, row.score);
  }

  private filterCount(filter: AsrEvaluationFilter): number {
    let count: number = 0;
    for (let index: number = 0; index < this.cases.length; index++) {
      if (this.cases[index].datasetId === this.selectedDataset &&
        this.rowMatchesFilter(this.rows[index], filter)) {
        count += 1;
      }
    }
    return count;
  }

  private shouldShow(index: number): boolean {
    return this.cases[index].datasetId === this.selectedDataset &&
      this.rowMatchesFilter(this.rows[index], this.resultFilter);
  }

  private outcomeText(row: AsrTestRow): string {
    if (row.state === 'running') {
      return '识别中';
    }
    if (row.state === 'failed') {
      return '失败';
    }
    if (row.state === 'completed' && row.protocol === 'speaker_pipeline' && row.score !== undefined) {
      return `cpCER ${(row.score.cer * 100).toFixed(1)}%`;
    }
    if (row.state === 'completed' && row.protocol === 'oracle_turns' && row.turnRows.length > 0) {
      const summary: AsrTurnEvaluationSummary = this.turnSummaryForRows(row.turnRows);
      return summary.nonOverlap.evaluated > 0 ?
        `非重叠 ${(summary.nonOverlap.microCer * 100).toFixed(1)}%` : '仅含重叠段';
    }
    if (row.state === 'completed' && row.score !== undefined) {
      return row.score.exact ? '完全一致' : `CER ${(row.score.cer * 100).toFixed(1)}%`;
    }
    return '等待运行';
  }

  private outcomeColor(row: AsrTestRow): string {
    if (row.state === 'completed' && row.protocol === 'oracle_turns' && row.turnRows.length > 0) {
      const summary: AsrTurnEvaluationSummary = this.turnSummaryForRows(row.turnRows);
      return summary.nonOverlap.microCer === 0 ? '#137A4C' : '#B42318';
    }
    if (row.state === 'completed' && row.score !== undefined && row.score.exact) {
      return '#137A4C';
    }
    if (row.state === 'completed' || row.state === 'failed') {
      return '#B42318';
    }
    return row.state === 'running' ? '#246BFD' : '#667085';
  }

  private rowTurnCerText(row: AsrTestRow, overlap: boolean): string {
    const summary: AsrTurnEvaluationSummary = this.turnSummaryForRows(row.turnRows);
    const selected: AsrEvaluationSummary = overlap ? summary.overlap : summary.nonOverlap;
    if (selected.evaluated === 0) {
      return overlap ? '无重叠段' : '无非重叠段';
    }
    return `${overlap ? '重叠' : '非重叠'} ${(selected.microCer * 100).toFixed(1)}%`;
  }

  private pipelineDerText(row: AsrTestRow): string {
    if (row.diarizationScore === undefined) {
      return 'DER —';
    }
    return `DER ${(row.diarizationScore.der * 100).toFixed(1)}% · ` +
      `漏检 ${row.diarizationScore.missedSpeakerSec.toFixed(1)}s · ` +
      `误检 ${row.diarizationScore.falseAlarmSpeakerSec.toFixed(1)}s · ` +
      `混淆 ${row.diarizationScore.confusionSpeakerSec.toFixed(1)}s`;
  }

  private pipelineOverlapText(row: AsrTestRow): string {
    if (row.overlapDetectionScore === undefined || row.overlapDetectionScore.referenceOverlapSec <= 0) {
      return '真值无重叠语音';
    }
    return `重叠检测召回 ${(row.overlapDetectionScore.recall * 100).toFixed(1)}% · ` +
      `真值 ${row.overlapDetectionScore.referenceOverlapSec.toFixed(1)}s · ` +
      `模型预测 ${row.overlapDetectionScore.predictedOverlapSec.toFixed(1)}s · 尚未拆分波形`;
  }

  private streamIdentityText(stream: AsrEvaluationSpeakerStream): string {
    if (stream.displayName !== undefined && stream.displayName.length > 0) {
      return `${stream.displayName} · ${stream.clusterId} · 匹配 ${(stream.matchScore ?? 0).toFixed(3)}`;
    }
    return `${stream.clusterId} · 未匹配已注册声纹`;
  }

  private metricText(metric: string): string {
    const summary: AsrEvaluationSummary = this.summary(this.selectedDataset);
    if (summary.evaluated === 0) {
      return '—';
    }
    if (metric === 'cer') {
      return `${(summary.microCer * 100).toFixed(2)}%`;
    }
    if (metric === 'overlap') {
      const overlap: AsrEvaluationSummary = this.turnSummary(this.selectedDataset).overlap;
      return overlap.evaluated > 0 ? `${(overlap.microCer * 100).toFixed(2)}%` : '—';
    }
    if (metric === 'der') {
      const diarization: AsrDiarizationScore = this.pipelineDiarizationSummary(this.selectedDataset);
      return `${(diarization.der * 100).toFixed(2)}%`;
    }
    if (metric === 'overlap_recall') {
      const overlap: AsrOverlapDetectionScore = this.pipelineOverlapSummary(this.selectedDataset);
      return `${(overlap.recall * 100).toFixed(1)}%`;
    }
    if (metric === 'exact') {
      return `${summary.exact}/${summary.evaluated}`;
    }
    return summary.rtf.toFixed(3);
  }

  private rowRenderKey(row: AsrTestRow): string {
    const errors: number = row.score !== undefined ? row.score.errors : -1;
    let turnEvidence: string = '';
    for (let i: number = 0; i < row.turnRows.length; i++) {
      turnEvidence += `:${row.turnRows[i].state}:${row.turnRows[i].hypothesis}`;
    }
    const predictedTurns: AsrPredictedTurnRow[] = row.predictedTurns ?? [];
    for (let i: number = 0; i < predictedTurns.length; i++) {
      turnEvidence += `:${predictedTurns[i].clusterId}:${predictedTurns[i].hypothesis}`;
    }
    return `${row.id}:${row.protocol}:${row.state}:${errors}:${row.hypothesis}:${row.error}${turnEvidence}`;
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
    Button(`${label}  ${this.evaluatedSampleCount(datasetId)}/20`)
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
  private ProtocolButton(protocol: AsrEvaluationProtocol, label: string, enabled: boolean) {
    Button(label)
      .height(32)
      .fontSize(10)
      .fontWeight(this.evaluationProtocol === protocol ? FontWeight.Bold : FontWeight.Normal)
      .backgroundColor(this.evaluationProtocol === protocol ? '#172033' : '#FFFFFF')
      .fontColor(this.evaluationProtocol === protocol ? '#FFFFFF' : (enabled ? '#475467' : '#B8C0CC'))
      .border({ width: 0.5, color: this.evaluationProtocol === protocol ? '#172033' : '#DCE2EA' })
      .enabled(enabled && !this.running && this.playingId.length === 0)
      .onClick((): void => {
        this.selectProtocol(protocol);
      })
  }

  @Builder
  private TurnEvidence(turn: AsrTurnRow, index: number) {
    Column() {
      Row() {
        Text(`${index + 1} · ${turn.speakerId}`)
          .fontSize(10).fontWeight(FontWeight.Bold).fontColor('#344054')
        Text(`${turn.startSec.toFixed(2)}–${turn.endSec.toFixed(2)} 秒`)
          .fontSize(9).fontColor('#98A2B3').margin({ left: 7 })
        Blank()
        Text(turn.overlapped ? '重叠语音' : '非重叠')
          .fontSize(9).fontColor(turn.overlapped ? '#B54708' : '#137A4C')
          .padding({ left: 7, right: 7, top: 3, bottom: 3 })
          .backgroundColor(turn.overlapped ? '#FFF4E8' : '#EAF8F0').borderRadius(8)
      }
      .width('100%')
      Text(`真值：${turn.reference}`)
        .fontSize(11).fontColor('#172033').width('100%').margin({ top: 6 })
      Text(`输出：${turn.state === 'completed' ?
        (turn.hypothesis.length > 0 ? turn.hypothesis : '（空结果）') :
        (turn.state === 'failed' ? '（处理失败）' : '正在识别...')}`)
        .fontSize(11).fontColor(turn.state === 'failed' ? '#B42318' : '#475467')
        .width('100%').margin({ top: 4 })
      if (turn.score !== undefined) {
        Text(`CER ${(turn.score.cer * 100).toFixed(1)}% · S ${turn.score.substitutions} · ` +
          `D ${turn.score.deletions} · I ${turn.score.insertions} · ${turn.elapsedMs} ms`)
          .fontSize(9).fontColor('#667085').width('100%').margin({ top: 5 })
      } else if (turn.error.length > 0) {
        Text(turn.error).fontSize(9).fontColor('#B42318').width('100%').margin({ top: 5 })
      }
    }
    .width('100%').padding(9).backgroundColor(turn.overlapped ? '#FFFAF5' : '#F8FAFC')
    .border({ width: 0.5, color: turn.overlapped ? '#F2C99D' : '#E1E6ED' })
    .borderRadius(9).margin({ top: 6 })
  }

  @Builder
  private TruthTurnEvidence(turn: GroundTruthSpeakerTurn, index: number) {
    Row() {
      Text(`${index + 1}`).fontSize(9).fontColor('#246BFD').width(20)
      Text(turn.speakerId).fontSize(9).fontWeight(FontWeight.Bold).fontColor('#344054').width(54)
      Text(`${turn.startSec.toFixed(2)}–${turn.endSec.toFixed(2)}`)
        .fontSize(8).fontColor('#98A2B3').width(76)
      Text(turn.text).fontSize(10).fontColor('#172033').layoutWeight(1).maxLines(3)
    }
    .width('100%').padding({ top: 5, bottom: 5 })
    .border({ width: { bottom: 0.5 }, color: '#E4E9F0' })
  }

  @Builder
  private PredictedTurnEvidence(turn: AsrPredictedTurnRow, index: number) {
    Column() {
      Row() {
        Text(`${index + 1} · ${turn.clusterId}`)
          .fontSize(10).fontWeight(FontWeight.Bold).fontColor('#344054')
        Text(`${turn.startSec.toFixed(2)}–${turn.endSec.toFixed(2)} 秒`)
          .fontSize(9).fontColor('#98A2B3').margin({ left: 7 })
        Blank()
        if (turn.overlapped) {
          Text('模型检测到重叠').fontSize(8).fontColor('#B54708')
        }
      }
      .width('100%')
      Text(turn.hypothesis.length > 0 ? turn.hypothesis : '（空结果）')
        .fontSize(11).fontColor('#172033').width('100%').margin({ top: 5 })
      Text(`${turn.elapsedMs} ms${turn.error.length > 0 ? ` · ${turn.error}` : ''}`)
        .fontSize(8).fontColor(turn.error.length > 0 ? '#B42318' : '#667085')
        .width('100%').margin({ top: 4 })
    }
    .width('100%').padding(9).backgroundColor('#F8FAFC')
    .border({ width: 0.5, color: '#E1E6ED' }).borderRadius(9).margin({ top: 6 })
  }

  @Builder
  private SpeakerStreamEvidence(stream: AsrEvaluationSpeakerStream) {
    Column() {
      Text(this.streamIdentityText(stream))
        .fontSize(9).fontWeight(FontWeight.Bold)
        .fontColor(stream.displayName !== undefined ? '#137A4C' : '#475467').width('100%')
      Text(stream.text.length > 0 ? stream.text : '（空结果）')
        .fontSize(11).fontColor('#172033').width('100%').margin({ top: 5 })
    }
    .width('100%').padding(9).backgroundColor('#F1F6FF').borderRadius(9).margin({ top: 6 })
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

      if (row.protocol === 'speaker_pipeline' && spec.speakerTurns.length > 0) {
        Column() {
          Text('真实说话人时间线 · 仅在模型完成后用于评分')
            .fontSize(9).fontWeight(FontWeight.Bold).fontColor('#246BFD').width('100%')
          ForEach(spec.speakerTurns, (turn: GroundTruthSpeakerTurn, turnIndex: number) => {
            this.TruthTurnEvidence(turn, turnIndex)
          }, (turn: GroundTruthSpeakerTurn, turnIndex: number) =>
            `${turn.speakerId}:${turn.startSec}:${turnIndex}`)
        }
        .alignItems(HorizontalAlign.Start).width('100%').padding(9)
        .backgroundColor('#FAFCFF').borderRadius(9).margin({ top: 7 })
      }

      Column() {
        Text(row.protocol === 'oracle_turns' ? '端侧模型逐段合并输出' : '端侧模型输出')
          .fontSize(9).fontColor('#667085')
        Text(row.state === 'completed' ? (row.hypothesis.length > 0 ? row.hypothesis : '（空结果）') :
          (row.state === 'running' ? '正在识别...' : '等待运行'))
          .fontSize(13).fontColor(row.state === 'completed' ? '#172033' : '#98A2B3').margin({ top: 4 })
      }
      .alignItems(HorizontalAlign.Start).width('100%').padding(10)
      .backgroundColor('#F7F9FC').borderRadius(9).margin({ top: 7 })

      if (row.protocol === 'speaker_pipeline' && row.state === 'completed') {
        Text(row.score !== undefined ?
          `cpCER ${(row.score.cer * 100).toFixed(1)}% · S ${row.score.substitutions} · ` +
          `D ${row.score.deletions} · I ${row.score.insertions}` : 'cpCER —')
          .fontSize(9).fontColor('#B42318').width('100%').margin({ top: 8 })
        Text(this.pipelineDerText(row)).fontSize(9).fontColor('#7758E8').width('100%').margin({ top: 4 })
        Text(this.pipelineOverlapText(row)).fontSize(9).fontColor('#B54708').width('100%').margin({ top: 4 })

        Text(`模型预测时间线 ${(row.predictedTurns ?? []).length} 段 · ` +
          `分段耗时 ${row.diarizationElapsedMs ?? 0} ms`)
          .fontSize(10).fontWeight(FontWeight.Bold).fontColor('#344054').width('100%').margin({ top: 10 })
        ForEach(row.predictedTurns ?? [], (turn: AsrPredictedTurnRow, turnIndex: number) => {
          this.PredictedTurnEvidence(turn, turnIndex)
        }, (turn: AsrPredictedTurnRow) => `${turn.segmentId}:${turn.hypothesis}`)

        Text('按说话人合并 → 标点恢复 → 已注册声纹匹配')
          .fontSize(10).fontWeight(FontWeight.Bold).fontColor('#344054').width('100%').margin({ top: 10 })
        ForEach(row.speakerStreams ?? [], (stream: AsrEvaluationSpeakerStream) => {
          this.SpeakerStreamEvidence(stream)
        }, (stream: AsrEvaluationSpeakerStream) => `${stream.clusterId}:${stream.text}`)
      } else if (row.protocol === 'oracle_turns' && row.turnRows.length > 0) {
        Row() {
          Text(`逐段证据 ${row.turnRows.length} 段`).fontSize(10).fontWeight(FontWeight.Bold).fontColor('#344054')
          Blank()
          Text(this.rowTurnCerText(row, false))
            .fontSize(9).fontColor('#137A4C')
          Text(this.rowTurnCerText(row, true))
            .fontSize(9).fontColor('#B54708').margin({ left: 8 })
        }
        .width('100%').margin({ top: 9 })
        ForEach(row.turnRows, (turn: AsrTurnRow, turnIndex: number) => {
          this.TurnEvidence(turn, turnIndex)
        }, (turn: AsrTurnRow) => `${turn.segmentId}:${turn.state}:${turn.hypothesis}`)
      } else if (row.state === 'completed' && row.score !== undefined) {
        Text(`S ${row.score.substitutions} · D ${row.score.deletions} · I ${row.score.insertions} · ` +
          `处理 ${row.elapsedMs} ms · RTF ${row.audioDurationSec > 0 ?
            ((row.elapsedMs / 1000) / row.audioDurationSec).toFixed(3) : '—'}`)
          .fontSize(9).fontColor('#667085').width('100%').margin({ top: 7 })
      } else if (row.state === 'failed') {
        Text(row.error).fontSize(9).fontColor('#B42318').width('100%').margin({ top: 7 })
      }
      if (spec.speakerTurns.length > 0) {
        Text(row.protocol === 'speaker_pipeline' ?
          `完整音频送入模型 · 真值 ${spec.speakerTurns.length} 段仅用于事后评分` :
          (row.protocol === 'oracle_turns' ?
            `ASR 上限诊断使用真实时间戳 ${spec.speakerTurns.length} 段 · 文字标签未发送给模型` :
            `逐段真值 ${spec.speakerTurns.length} 段 · 整段模式不使用时间戳`))
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
          Text(this.evaluationProtocol === 'speaker_pipeline' ? '多人链路' : 'Paraformer')
            .fontSize(10).fontWeight(FontWeight.Bold).fontColor('#246BFD')
          Text(this.evaluationProtocol === 'speaker_pipeline' ? 'Pyannote · CAM++' : 'INT8 · 端侧')
            .fontSize(8).fontColor('#667085').margin({ top: 2 })
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
            this.ProtocolButton('speaker_pipeline', '完整多人链路', this.supportsOracleTurns(this.selectedDataset))
            this.ProtocolButton('full_window', '整段基线', true)
            this.ProtocolButton('oracle_turns', 'ASR 上限诊断', this.supportsOracleTurns(this.selectedDataset))
            Blank()
            Text(this.evaluationProtocol === 'speaker_pipeline' ? '真实入口' : '诊断')
              .fontSize(9).fontColor(this.evaluationProtocol === 'speaker_pipeline' ? '#137A4C' : '#667085')
          }
          .width('100%').margin({ top: 9 })

          Text(this.evaluationProtocol === 'speaker_pipeline' ?
            '完整音频由 Pyannote 预测语音/说话边界，CAM++ 聚类及匹配身份，长段切到不超过 8 秒再逐段 ASR。重叠语音会检测并计入错误，当前不伪装成已完成波形拆分。' :
            (this.evaluationProtocol === 'oracle_turns' ?
              '诊断模式使用真实时间戳测量 ASR 上限；不代表真实会议中的自动分段能力。' :
              '整段音频只产生一条文本；多人同时说话时，该 CER 不代表多人会议能力。'))
            .fontSize(9).fontColor('#667085').width('100%')
            .padding(9).backgroundColor('#F7F9FC').borderRadius(8).margin({ top: 7 })

          if (this.evaluationProtocol === 'speaker_pipeline') {
            Row({ space: 7 }) {
              this.MetricCard('cer', 'cpCER', '#B42318')
              this.MetricCard('der', 'DER', '#7758E8')
              this.MetricCard('overlap_recall', '重叠召回', '#B54708')
            }
            .width('100%').margin({ top: 10 })
          } else if (this.evaluationProtocol === 'oracle_turns') {
            Row({ space: 7 }) {
              this.MetricCard('cer', '非重叠 CER', '#137A4C')
              this.MetricCard('overlap', '重叠 CER', '#B54708')
              this.MetricCard('rtf', '逐段 RTF', '#7758E8')
            }
            .width('100%').margin({ top: 10 })
          } else {
            Row({ space: 7 }) {
              this.MetricCard('cer', '微平均 CER', '#246BFD')
              this.MetricCard('exact', '完全一致', '#137A4C')
              this.MetricCard('rtf', '实时系数', '#7758E8')
            }
            .width('100%').margin({ top: 10 })
          }

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
