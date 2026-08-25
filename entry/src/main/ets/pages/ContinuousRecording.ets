import { common } from '@kit.AbilityKit';
import { router } from '@kit.ArkUI';
import { abilityAccessCtrl, Permissions } from '@kit.AbilityKit';
import { ErrorEvent, MessageEvents, worker } from '@kit.ArkTS';
import { fileIo, picker } from '@kit.CoreFileKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import {
  MeetingSpeakerAssignment,
  SpeakerCandidate,
  SpeakerSearchProfile,
  SpeakerTurn,
} from '../common/SpeakerTypes';
import { ExportResult, RecordingExporter } from '../services/recording/RecordingExporter';
import {
  ContinuousRecordingService,
  RecordingStatus,
  TimelineRow,
} from '../services/recording/ContinuousRecordingService';
import { SegmentRecord } from '../services/recording/SegmentStore';
import { PlaybackState, SegmentPlayer } from '../services/recording/SegmentPlayer';
import {
  TranscriptRecord,
  TranscriptionResult,
} from '../services/recording/TranscriptionQueue';
import { MeetingSpeakerStore } from '../services/speaker/MeetingSpeakerStore';
import { SpeakerEditSessionStore } from '../services/speaker/SpeakerEditSessionStore';
import { SpeakerProfileStore } from '../services/speaker/SpeakerProfileStore';

const DOMAIN = 0x0001;
const TAG = 'ContinuousRecordingPage';
/**
 * Mirrors FILE_ASR_MODE_OFFLINE_PARAFORMER in StreamingSherpaWorker.
 *
 * The worker matches this by string equality and silently falls back when nothing
 * matches, so a typo here degrades recognition instead of failing — which is how it went
 * unnoticed through a whole recording session.
 */
const OFFLINE_PARAFORMER_MODE: string = 'offline_paraformer_large';
const RECORDING_ENGINE: string =
  'offline_paraformer_large+campplus_zh_en_192+structured_speakers_v3';

interface SegmentWorkerRequest {
  msgType: string;
  context?: common.UIAbilityContext;
  requestId?: string;
  data?: Uint8Array;
  audioFormat?: string;
  fileAsrMode?: string;
  enableSpeakerDiarization?: boolean;
  meetingId?: string;
  voiceprints?: SpeakerSearchProfile[];
}

interface SegmentWorkerResponse {
  msgType: string;
  requestId?: string;
  text?: string;
  message?: string;
  meetingId?: string;
  speakerTurns?: SpeakerTurn[];
  speakerCandidates?: SpeakerCandidate[];
}

/** Resolvers for segments currently inside the worker, keyed by request id. */
interface PendingTranscription {
  resolve: (result: TranscriptionResult) => void;
  reject: (error: Error) => void;
  segmentId: string;
  meetingId: string;
}

/**
 * Continuous recording: capture everything, keep the speech, transcribe afterwards.
 *
 * This is a separate page rather than another mode on the main screen because the two
 * have opposite shapes. Live recognition is something a user watches; this is something
 * they leave running and come back to, so what it owes them is a truthful account of
 * what is stored and what is still pending, not a stream of words.
 *
 * The page holds its own worker. Both screens loading the recogniser at once would put
 * two copies of the models in memory, so the main page releases its worker while this
 * one is in front.
 */
@Entry
@Component
struct ContinuousRecording {
  @State status: RecordingStatus = {
    recording: false, segments: 0, transcribed: 0, pending: 0, storedBytes: 0,
    inputLevel: 0, inputThreshold: 0,
  };
  @State rows: TimelineRow[] = [];
  @State statusMessage: string = '正在准备识别引擎...';
  @State engineReady: boolean = false;
  @State busy: boolean = false;
  @State playingId: string = '';
  @State playPositionMs: number = 0;
  @State playDurationMs: number = 0;

  private service: ContinuousRecordingService = ContinuousRecordingService.getInstance();
  private player: SegmentPlayer = new SegmentPlayer();
  private exporter: RecordingExporter = new RecordingExporter();
  private profileStore: SpeakerProfileStore = SpeakerProfileStore.getInstance();
  private meetingSpeakerStore: MeetingSpeakerStore | undefined = undefined;
  private speakerEditSessions: SpeakerEditSessionStore = SpeakerEditSessionStore.getInstance();
  private profileStoreReady: boolean = false;
  private workerInstance: worker.ThreadWorker | undefined = undefined;
  private pending: Map<string, PendingTranscription> =
    new Map<string, PendingTranscription>();
  private requestSeq: number = 0;
  private active: boolean = false;

  aboutToAppear(): void {
    this.active = true;
    this.startEngine();
  }

  onPageShow(): void {
    // A speaker may have been created or updated in the editor while this worker stayed
    // alive. Refresh its search gallery before the next stored segment is decoded.
    this.restoreVoiceprints();
    this.refreshSpeakerAssignments();
  }

  aboutToDisappear(): void {
    this.active = false;
    this.player.setCallback(null);
    this.player.stop().catch((error: Error): void => {
      hilog.warn(DOMAIN, TAG, `stop playback failed: ${error.message}`);
    });
    // Stopping first means the segmenter flushes its open utterance while the worker is
    // still alive to transcribe it.
    this.service.stop()
      .catch((error: Error): void => {
        hilog.warn(DOMAIN, TAG, `stop failed: ${error.message}`);
      })
      .then((): void => {
        this.service.setStatusCallback(null);
        // The worker is about to go away, so the service must stop handing it segments.
        this.service.setEngineReady(false);
        this.shutdownWorker();
      });
  }

  private startEngine(): void {
    const context = getContext(this) as common.UIAbilityContext;
    this.meetingSpeakerStore = new MeetingSpeakerStore(context);
    this.profileStore.init(context)
      .then((): void => {
        this.profileStoreReady = true;
        this.restoreVoiceprints();
        this.refreshSpeakerAssignments();
      })
      .catch((error: Error): void => {
        hilog.warn(DOMAIN, TAG, `speaker profiles unavailable: ${error.message}`);
      });
    try {
      const instance = new worker.ThreadWorker('entry/ets/workers/StreamingSherpaWorker.ets', {
        name: 'Continuous recording ASR worker',
      });
      instance.onmessage = (event: MessageEvents): void => {
        this.onWorkerMessage(event);
      };
      instance.onerror = (event: ErrorEvent): void => {
        this.failAllPending(`识别引擎异常: ${event.message}`);
        this.statusMessage = `识别引擎异常: ${event.message}`;
      };
      instance.onmessageerror = (_event: MessageEvents): void => {
        this.failAllPending('识别引擎消息反序列化失败');
      };
      this.workerInstance = instance;
      const request: SegmentWorkerRequest = { msgType: 'sherpa-asr-init', context };
      instance.postMessage(request);
    } catch (error) {
      this.statusMessage = `识别引擎启动失败: ${error}`;
      hilog.error(DOMAIN, TAG, `worker start failed: ${error}`);
      return;
    }

    this.service.setStatusCallback((status: RecordingStatus): void => {
      if (!this.active) {
        return;
      }
      this.status = status;
      this.rows = this.service.timeline();
    });
    this.player.setCallback((state: PlaybackState): void => {
      if (!this.active) {
        return;
      }
      this.playingId = state.segmentId;
      this.playPositionMs = state.positionMs;
      this.playDurationMs = state.durationMs;
    });
    this.service.init(context.filesDir,
      (record: SegmentRecord, bytes: Uint8Array): Promise<TranscriptionResult> =>
        this.transcribeSegment(record, bytes),
      RECORDING_ENGINE);
    this.rows = this.service.timeline();
    this.status = this.service.status();
  }

  private onWorkerMessage(event: MessageEvents): void {
    const response = event.data as SegmentWorkerResponse;
    const msgType: string = response.msgType;

    if (msgType === 'sherpa-asr-ready') {
      this.engineReady = true;
      // A jump in the pending count right after an upgrade is the queue re-running audio
      // whose text came from an older recogniser, not a stall. Say so.
      const redo: number = this.service.redoCount();
      this.statusMessage = redo > 0
        ? `识别引擎就绪，正在用新模型重转 ${redo} 段旧录音`
        : '识别引擎就绪';
      // Segments recorded during the model load are waiting on exactly this; telling the
      // service lets it clear their failure counts and start working through them.
      this.service.setEngineReady(true);
      this.restoreVoiceprints();
      return;
    }
    if (msgType === 'sherpa-asr-segment-done') {
      const entry = this.pending.get(response.requestId ?? '');
      if (entry !== undefined) {
        this.pending.delete(response.requestId ?? '');
        const result: TranscriptionResult = {
          text: response.text ?? '',
          meetingId: entry.meetingId,
          speakerTurns: this.prefixTurns(response.speakerTurns ?? [],
            entry.segmentId, entry.meetingId),
          speakerCandidates: this.prefixCandidates(response.speakerCandidates ?? [],
            entry.segmentId, entry.meetingId),
        };
        entry.resolve(result);
      }
      return;
    }
    if (msgType === 'sherpa-asr-segment-failed') {
      const entry = this.pending.get(response.requestId ?? '');
      if (entry !== undefined) {
        this.pending.delete(response.requestId ?? '');
        entry.reject(new Error(response.message ?? 'segment transcription failed'));
      }
      return;
    }
    if (msgType === 'sherpa-asr-error') {
      this.statusMessage = response.message ?? '识别引擎报错';
    }
  }

  /**
   * Sends one stored segment to the worker and waits for its reply.
   *
   * The audio buffer is transferred rather than copied — a 30 s segment is about
   * 960 KB, and copying it per segment would be pure waste on a device that is still
   * recording. Replies are matched by request id so a slow segment cannot be mistaken
   * for the answer to a later one.
   */
  private transcribeSegment(record: SegmentRecord,
    bytes: Uint8Array): Promise<TranscriptionResult> {
    const instance = this.workerInstance;
    if (instance === undefined || !this.engineReady) {
      return Promise.reject(new Error('识别引擎尚未就绪'));
    }
    // Re-bound as a non-nullable const: the executor below closes over it, and a
    // narrowing from the guard above does not survive into a closure.
    const active: worker.ThreadWorker = instance;
    this.requestSeq += 1;
    const requestId: string = `seg-${this.requestSeq}`;
    const meetingId: string = this.meetingIdFor(record.startedAt);
    return new Promise<TranscriptionResult>((resolve: (result: TranscriptionResult) => void,
      reject: (error: Error) => void): void => {
      // Spelled out rather than shorthand: the target type here comes from the Map's
      // generic, not from an annotation ArkTS can point at.
      const entry: PendingTranscription = {
        resolve: resolve,
        reject: reject,
        segmentId: record.id,
        meetingId: meetingId,
      };
      this.pending.set(requestId, entry);
      try {
        const request: SegmentWorkerRequest = {
          msgType: 'sherpa-asr-transcribe-segment',
          requestId: requestId,
          data: bytes,
          audioFormat: 'wav',
          // Must match FILE_ASR_MODE_OFFLINE_PARAFORMER in the worker exactly. 'offline'
          // matched no branch and fell through to the catch-all path, which produced
          // markedly worse text: on one 43 s segment the device stored 13 characters
          // where this model decodes several sentences from the same file.
          fileAsrMode: OFFLINE_PARAFORMER_MODE,
          enableSpeakerDiarization: true,
          meetingId: meetingId,
        };
        active.postMessage(request, [bytes.buffer]);
      } catch (error) {
        this.pending.delete(requestId);
        reject(new Error(`${error}`));
      }
    });
  }

  private meetingIdFor(timestamp: number): string {
    const date: Date = new Date(timestamp);
    const pad = (value: number): string => value < 10 ? `0${value}` : `${value}`;
    return `continuous-${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}`;
  }

  private prefixTurns(turns: SpeakerTurn[], segmentId: string,
    meetingId: string): SpeakerTurn[] {
    const result: SpeakerTurn[] = [];
    for (let i: number = 0; i < turns.length; i++) {
      const turn: SpeakerTurn = turns[i];
      result.push({
        turnId: turn.turnId,
        meetingId: meetingId,
        clusterId: `${segmentId}:${turn.clusterId}`,
        startMs: turn.startMs,
        endMs: turn.endMs,
        text: turn.text,
        identity: turn.identity,
        personId: turn.personId,
        displayName: turn.displayName,
        matchScore: turn.matchScore,
      });
    }
    return result;
  }

  private prefixCandidates(candidates: SpeakerCandidate[], segmentId: string,
    meetingId: string): SpeakerCandidate[] {
    const result: SpeakerCandidate[] = [];
    for (let i: number = 0; i < candidates.length; i++) {
      const candidate: SpeakerCandidate = candidates[i];
      result.push({
        meetingId: meetingId,
        clusterId: `${segmentId}:${candidate.clusterId}`,
        segmentCount: candidate.segmentCount,
        speechMs: candidate.speechMs,
        qualityScore: candidate.qualityScore,
        queryEmbedding: candidate.queryEmbedding,
        personId: candidate.personId,
        displayName: candidate.displayName,
        matchScore: candidate.matchScore,
      });
    }
    return result;
  }

  private restoreVoiceprints(): void {
    if (!this.engineReady || !this.profileStoreReady || !this.workerInstance) {
      return;
    }
    const request: SegmentWorkerRequest = {
      msgType: 'sherpa-voiceprint-restore',
      voiceprints: this.profileStore.searchProfiles(),
    };
    this.workerInstance.postMessage(request);
  }

  private failAllPending(message: string): void {
    this.pending.forEach((entry: PendingTranscription): void => {
      entry.reject(new Error(message));
    });
    this.pending.clear();
  }

  private shutdownWorker(): void {
    const instance = this.workerInstance;
    this.workerInstance = undefined;
    this.failAllPending('页面已退出');
    if (instance === undefined) {
      return;
    }
    try {
      instance.onmessage = undefined;
      instance.onerror = undefined;
      instance.onmessageerror = undefined;
      instance.terminate();
    } catch (error) {
      hilog.warn(DOMAIN, TAG, `worker terminate failed: ${error}`);
    }
  }

  private async toggleRecording(): Promise<void> {
    if (this.busy) {
      return;
    }
    this.busy = true;
    try {
      if (this.status.recording) {
        await this.service.stop();
        this.statusMessage = '已停止，正在转录剩余片段';
      } else {
        if (!await this.ensureMicrophonePermission()) {
          this.statusMessage = '未获得麦克风权限';
          return;
        }
        await this.service.start();
        this.statusMessage = '持续录音中，仅存储有人说话的片段';
      }
    } catch (error) {
      this.statusMessage = `操作失败: ${error}`;
      hilog.error(DOMAIN, TAG, `toggle failed: ${error}`);
    } finally {
      this.busy = false;
      this.status = this.service.status();
    }
  }

  /**
   * Packages everything and writes it wherever the user chooses.
   *
   * Recording is stopped first. The open segment holds the audio someone is most likely
   * to be exporting for, and an archive that quietly omits the last thing said would be
   * worse than one that took a moment longer to appear.
   */
  private async exportRecordings(): Promise<void> {
    if (this.busy) {
      return;
    }
    this.busy = true;
    try {
      if (this.status.recording) {
        await this.service.stop();
      }
      const context = getContext(this) as common.UIAbilityContext;
      this.statusMessage = '正在打包录音与转录…';
      const result: ExportResult = await this.exporter.build(
        context.filesDir, this.service.timeline());

      const documentPicker = new picker.DocumentViewPicker(context);
      const options = new picker.DocumentSaveOptions();
      options.newFileNames = [result.fileName];
      const uris: string[] = await documentPicker.save(options);
      if (!uris || uris.length === 0) {
        // Cancelling the dialog is a normal outcome, not a failure; the archive is
        // dropped rather than left to accumulate in the sandbox.
        this.exporter.cleanup(context.filesDir);
        this.statusMessage = '已取消导出';
        return;
      }
      await this.copyToUri(result.archivePath, uris[0]);
      this.exporter.cleanup(context.filesDir);
      this.statusMessage =
        `已导出 ${result.segments} 段（${this.formatBytes(result.bytes)}）`;
    } catch (error) {
      this.statusMessage = `导出失败: ${error}`;
      hilog.error(DOMAIN, TAG, `export failed: ${error}`);
    } finally {
      this.busy = false;
      this.status = this.service.status();
      this.rows = this.service.timeline();
    }
  }

  /** Streams the archive into the picked destination in bounded chunks. */
  private async copyToUri(sourcePath: string, uri: string): Promise<void> {
    const source: fileIo.File = fileIo.openSync(sourcePath, fileIo.OpenMode.READ_ONLY);
    let target: fileIo.File | null = null;
    try {
      target = fileIo.openSync(uri, fileIo.OpenMode.READ_WRITE | fileIo.OpenMode.TRUNC);
      // An export can run to tens of megabytes; reading it whole would spike memory on
      // a device that may still be recording.
      const chunk: ArrayBuffer = new ArrayBuffer(256 * 1024);
      let offset: number = 0;
      while (true) {
        const read: number = fileIo.readSync(source.fd, chunk, { offset: offset });
        if (read <= 0) {
          break;
        }
        fileIo.writeSync(target.fd, chunk.slice(0, read), { offset: offset });
        offset += read;
      }
      fileIo.fsyncSync(target.fd);
    } finally {
      if (target !== null) {
        fileIo.closeSync(target);
      }
      fileIo.closeSync(source);
    }
  }

  private async ensureMicrophonePermission(): Promise<boolean> {
    try {
      const context = getContext(this) as common.UIAbilityContext;
      const manager = abilityAccessCtrl.createAtManager();
      const permissions: Permissions[] = ['ohos.permission.MICROPHONE'];
      const result = await manager.requestPermissionsFromUser(context, permissions);
      const granted: number[] = result.authResults;
      return granted.length > 0 && granted[0] === 0;
    } catch (error) {
      hilog.error(DOMAIN, TAG, `permission request failed: ${error}`);
      return false;
    }
  }

  private formatBytes(bytes: number): string {
    if (bytes < 1024) {
      return `${bytes} B`;
    }
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(0)} KB`;
    }
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }

  private formatClock(timestamp: number): string {
    const date = new Date(timestamp);
    const pad = (value: number): string => value < 10 ? `0${value}` : `${value}`;
    return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  }

  private togglePlayback(row: TimelineRow): void {
    this.player.toggle(row.segment.id, row.segment.path)
      .catch((error: Error): void => {
        this.statusMessage = `播放失败: ${error.message}`;
        hilog.warn(DOMAIN, TAG, `playback toggle failed: ${error.message}`);
      });
  }

  private editSpeakers(row: TimelineRow): void {
    const transcript: TranscriptRecord | null = row.transcript;
    if (transcript === null || !transcript.speakerTurns || transcript.speakerTurns.length === 0) {
      return;
    }
    if (!transcript.speakerCandidates || transcript.speakerCandidates.length === 0) {
      this.recoverSpeakerCandidates(row, transcript);
      return;
    }
    this.openSpeakerEditor(row, transcript);
  }

  private openSpeakerEditor(row: TimelineRow, transcript: TranscriptRecord): void {
    const meetingId: string = transcript.meetingId || row.segment.id;
    const sessionId: string = this.speakerEditSessions.put(
      meetingId,
      `持续录音 ${this.formatClock(row.segment.startedAt)}`,
      transcript.speakerTurns,
      transcript.speakerCandidates || []);
    router.pushUrl({
      url: 'pages/MeetingSpeakerEditor',
      params: {
        sessionId: sessionId,
      },
    }).catch((error: Error): void => {
      this.statusMessage = `打开说话人编辑失败: ${error.message}`;
    });
  }

  /** Re-extracts non-persisted biometric candidates from the still-local recording. */
  private recoverSpeakerCandidates(row: TimelineRow, transcript: TranscriptRecord): void {
    if (!this.engineReady || this.busy) {
      this.statusMessage = '识别引擎尚未就绪，请稍后再编辑说话人';
      return;
    }
    this.busy = true;
    this.statusMessage = '正在从本机录音重新提取声纹样本...';
    this.extractSpeakerCandidates(row, transcript)
      .then((): void => {
        this.rows = this.service.timeline();
        this.busy = false;
        this.statusMessage = '声纹样本已重新提取';
        this.openSpeakerEditor(row, transcript);
      })
      .catch((error: Error): void => {
        this.busy = false;
        this.statusMessage = `重新提取失败: ${error.message}`;
      });
  }

  private async extractSpeakerCandidates(row: TimelineRow,
    transcript: TranscriptRecord): Promise<void> {
    const bytes: Uint8Array = await this.readSegmentBytes(row.segment.path);
    const result: TranscriptionResult = await this.transcribeSegment(row.segment, bytes);
    transcript.meetingId = result.meetingId;
    transcript.speakerTurns = result.speakerTurns;
    transcript.speakerCandidates = result.speakerCandidates;
    transcript.text = result.text;
  }

  private async readSegmentBytes(path: string): Promise<Uint8Array> {
    let audioFile: fileIo.File | undefined = undefined;
    try {
      audioFile = await fileIo.open(path, fileIo.OpenMode.READ_ONLY);
      const stat = await fileIo.stat(audioFile.fd);
      if (stat.size <= 0) {
        throw new Error('录音文件为空');
      }
      const buffer: ArrayBuffer = new ArrayBuffer(stat.size);
      const readBytes: number = await fileIo.read(audioFile.fd, buffer, { length: stat.size });
      if (readBytes <= 0) {
        throw new Error('无法读取录音文件');
      }
      const result: Uint8Array = new Uint8Array(readBytes);
      result.set(new Uint8Array(buffer, 0, readBytes));
      return result;
    } finally {
      if (audioFile) {
        await fileIo.close(audioFile);
      }
    }
  }

  private hasRecordingSpeakerResult(): boolean {
    for (let i: number = 0; i < this.rows.length; i++) {
      const transcript = this.rows[i].transcript;
      if (transcript && transcript.speakerTurns && transcript.speakerTurns.length > 0) {
        return true;
      }
    }
    return false;
  }

  private editAllSpeakers(): void {
    let meetingId: string = '';
    for (let i: number = this.rows.length - 1; i >= 0; i--) {
      const transcript = this.rows[i].transcript;
      if (transcript && transcript.speakerTurns && transcript.speakerTurns.length > 0) {
        meetingId = transcript.meetingId || this.meetingIdFor(this.rows[i].segment.startedAt);
        break;
      }
    }
    if (meetingId.length === 0) {
      return;
    }
    let needsRecovery: boolean = false;
    for (let i: number = 0; i < this.rows.length; i++) {
      const transcript = this.rows[i].transcript;
      if (!transcript || !transcript.speakerTurns || transcript.speakerTurns.length === 0) {
        continue;
      }
      const rowMeetingId: string = transcript.meetingId ||
        this.meetingIdFor(this.rows[i].segment.startedAt);
      if (rowMeetingId === meetingId &&
        (!transcript.speakerCandidates || transcript.speakerCandidates.length === 0)) {
        needsRecovery = true;
        break;
      }
    }
    if (!needsRecovery || !this.engineReady) {
      this.openAllSpeakerEditor(meetingId);
      return;
    }
    this.busy = true;
    this.statusMessage = '正在从本机录音恢复本次会议的声纹样本...';
    this.recoverMeetingCandidates(meetingId)
      .then((): void => {
        this.rows = this.service.timeline();
        this.busy = false;
        this.statusMessage = '本次会议声纹样本已恢复';
        this.openAllSpeakerEditor(meetingId);
      })
      .catch((error: Error): void => {
        this.busy = false;
        this.statusMessage = `声纹样本恢复失败: ${error.message}`;
      });
  }

  private async recoverMeetingCandidates(meetingId: string): Promise<void> {
    for (let i: number = 0; i < this.rows.length; i++) {
      const transcript = this.rows[i].transcript;
      if (!transcript || !transcript.speakerTurns || transcript.speakerTurns.length === 0) {
        continue;
      }
      const rowMeetingId: string = transcript.meetingId ||
        this.meetingIdFor(this.rows[i].segment.startedAt);
      if (rowMeetingId !== meetingId ||
        (transcript.speakerCandidates && transcript.speakerCandidates.length > 0)) {
        continue;
      }
      await this.extractSpeakerCandidates(this.rows[i], transcript);
    }
  }

  private openAllSpeakerEditor(meetingId: string): void {
    const turns: SpeakerTurn[] = [];
    const candidates: SpeakerCandidate[] = [];
    for (let i: number = 0; i < this.rows.length; i++) {
      const transcript = this.rows[i].transcript;
      if (!transcript || !transcript.speakerTurns || transcript.speakerTurns.length === 0) {
        continue;
      }
      const rowMeetingId: string = transcript.meetingId ||
        this.meetingIdFor(this.rows[i].segment.startedAt);
      if (rowMeetingId !== meetingId) {
        continue;
      }
      for (let j: number = 0; j < transcript.speakerTurns.length; j++) {
        turns.push(transcript.speakerTurns[j]);
      }
      const rowCandidates: SpeakerCandidate[] = transcript.speakerCandidates || [];
      for (let j: number = 0; j < rowCandidates.length; j++) {
        candidates.push(rowCandidates[j]);
      }
    }
    if (turns.length === 0) {
      return;
    }
    const sessionId: string = this.speakerEditSessions.put(
      meetingId, '持续录音', turns, candidates);
    router.pushUrl({
      url: 'pages/MeetingSpeakerEditor',
      params: {
        sessionId: sessionId,
      },
    }).catch((error: Error): void => {
      this.statusMessage = `打开说话人编辑失败: ${error.message}`;
    });
  }

  /** Makes persisted meeting labels visible in the timeline and in subsequent exports. */
  private async refreshSpeakerAssignments(): Promise<void> {
    const store: MeetingSpeakerStore | undefined = this.meetingSpeakerStore;
    if (!store || !this.profileStoreReady) {
      return;
    }
    const rows: TimelineRow[] = this.service.timeline();
    const handledMeetingIds: Set<string> = new Set<string>();
    let changed: boolean = false;
    for (let i: number = 0; i < rows.length; i++) {
      const transcript = rows[i].transcript;
      if (!transcript || !transcript.speakerTurns || transcript.speakerTurns.length === 0) {
        continue;
      }
      const meetingId: string = transcript.meetingId ||
        transcript.speakerTurns[0].meetingId || this.meetingIdFor(rows[i].segment.startedAt);
      if (handledMeetingIds.has(meetingId)) {
        continue;
      }
      handledMeetingIds.add(meetingId);
      const assignments: MeetingSpeakerAssignment[] = await store.assignmentsFor(meetingId);
      for (let j: number = 0; j < rows.length; j++) {
        const candidate = rows[j].transcript;
        if (!candidate || !candidate.speakerTurns || candidate.speakerTurns.length === 0) {
          continue;
        }
        const candidateMeetingId: string = candidate.meetingId ||
          candidate.speakerTurns[0].meetingId || this.meetingIdFor(rows[j].segment.startedAt);
        if (candidateMeetingId !== meetingId) {
          continue;
        }
        candidate.meetingId = meetingId;
        candidate.speakerTurns = store.apply(
          candidate.speakerTurns, assignments, this.profileStore.list());
        candidate.text = this.formatSpeakerTurns(candidate.speakerTurns);
        changed = true;
      }
    }
    if (changed && this.active) {
      this.rows = this.service.timeline();
    }
  }

  private formatSpeakerTurns(turns: SpeakerTurn[]): string {
    const lines: string[] = [];
    for (let i: number = 0; i < turns.length; i++) {
      const turn: SpeakerTurn = turns[i];
      const name: string = turn.displayName || `未知说话人 ${turn.clusterId}`;
      lines.push(`[${name} ${this.formatTurnTime(turn.startMs)}-${this.formatTurnTime(turn.endMs)}] ${turn.text}`);
    }
    return lines.join('\n');
  }

  private formatTurnTime(milliseconds: number): string {
    const totalSeconds: number = Math.max(0, Math.floor(milliseconds / 1000));
    const minutes: number = Math.floor(totalSeconds / 60);
    const seconds: number = totalSeconds % 60;
    const minuteText: string = minutes < 10 ? `0${minutes}` : `${minutes}`;
    const secondText: string = seconds < 10 ? `0${seconds}` : `${seconds}`;
    return `${minuteText}:${secondText}`;
  }

  /**
   * One stored segment, whether or not it has been recognised yet.
   *
   * A pending row is shown rather than hidden: its audio already exists and is already
   * playable, and hiding it would make the lag between recording and transcribing look
   * like nothing was captured at all.
   */
  @Builder
  private SegmentRow(row: TimelineRow) {
    Column({ space: 8 }) {
      Row({ space: 8 }) {
        Text(this.formatClock(row.segment.startedAt))
          .fontSize(12)
          .fontColor('#0066CC')
        Text(`${(row.segment.durationMs / 1000).toFixed(1)}s`)
          .fontSize(12)
          .fontColor('#AAAAAA')
        if (row.transcript === null) {
          // Two different states used to share one label. "Waiting" with nothing running
          // is a stall; "waiting" behind a segment that is actively decoding is just a
          // queue, and only one of them is worth reporting as a fault.
          Text(row.transcribing ? '识别中' : '待转录')
            .fontSize(11)
            .fontColor(row.transcribing ? '#0066CC' : '#B36B00')
            .backgroundColor(row.transcribing ? '#E6F0FF' : '#FFF5E6')
            .padding({ left: 6, right: 6, top: 2, bottom: 2 })
            .borderRadius(4)
        }
        Blank()

        if (row.transcript !== null && row.transcript.speakerTurns &&
          row.transcript.speakerTurns.length > 0) {
          Button('说话人')
            .fontSize(11)
            .height(28)
            .backgroundColor('#F2EEFF')
            .fontColor('#5B3FB5')
            .enabled(!this.status.recording && !this.busy)
            .onClick((): void => {
              this.editSpeakers(row);
            })
        }

        Row({ space: 4 }) {
          Text(this.playingId === row.segment.id ? '■' : '▶')
            .fontSize(12)
            .fontColor(this.playingId === row.segment.id ? '#FFFFFF' : '#0066CC')
          Text(this.playingId === row.segment.id
            ? this.formatSeconds(this.playPositionMs, this.playDurationMs)
            : '播放')
            .fontSize(12)
            .fontColor(this.playingId === row.segment.id ? '#FFFFFF' : '#0066CC')
        }
        .padding({ left: 10, right: 10, top: 5, bottom: 5 })
        .backgroundColor(this.playingId === row.segment.id ? '#0066CC' : '#EAF2FF')
        .borderRadius(999)
        .onClick((): void => {
          this.togglePlayback(row);
        })
      }
      .width('100%')

      Text(this.rowText(row))
        .fontSize(15)
        .fontColor(row.transcript !== null && row.transcript.text.length > 0
          ? '#333333' : '#BBBBBB')
        .width('100%')
    }
    .width('100%')
    .padding(12)
    .backgroundColor('#FFFFFF')
    .borderRadius(10)
  }

  private rowText(row: TimelineRow): string {
    if (row.transcript === null) {
      return row.transcribing ? '正在识别这一段…' : '排队等待识别…';
    }
    return row.transcript.text.length > 0 ? row.transcript.text : '（未识别到内容）';
  }

  private formatSeconds(positionMs: number, durationMs: number): string {
    const position: string = (Math.max(0, positionMs) / 1000).toFixed(1);
    if (durationMs <= 0) {
      return `${position}s`;
    }
    return `${position}/${(durationMs / 1000).toFixed(1)}s`;
  }

  @Builder
  private StatTile(label: string, value: string) {
    Column({ space: 4 }) {
      Text(value)
        .fontSize(20)
        .fontWeight(FontWeight.Bold)
        .fontColor('#333333')
      Text(label)
        .fontSize(12)
        .fontColor('#999999')
    }
    .layoutWeight(1)
    .alignItems(HorizontalAlign.Center)
  }

  build() {
    Column() {
      Row({ space: 10 }) {
        Text('‹')
          .fontSize(28)
          .fontColor('#333333')
          .width(40)
          .height(40)
          .textAlign(TextAlign.Center)
          .onClick((): void => {
            router.back();
          })
        Column({ space: 2 }) {
          Text('持续录音')
            .fontSize(20)
            .fontWeight(FontWeight.Bold)
            .fontColor('#222222')
          Text(this.statusMessage)
            .fontSize(12)
            .fontColor('#888888')
            .maxLines(1)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)

        // Kept away from the record button: this one ends the session and opens a system
        // dialog, so a mis-tap while recording would be costly.
        Button('导出')
          .fontSize(13)
          .height(32)
          .backgroundColor('#EAF2FF')
          .fontColor('#0066CC')
          .enabled(this.rows.length > 0 && !this.busy)
          .onClick((): void => {
            this.exportRecordings();
          })

        Button('说话人')
          .fontSize(13)
          .height(32)
          .backgroundColor('#F2EEFF')
          .fontColor('#5B3FB5')
          .enabled(this.hasRecordingSpeakerResult() && !this.status.recording && !this.busy)
          .onClick((): void => {
            this.editAllSpeakers();
          })
      }
      .width('100%')
      .padding({ left: 12, right: 16, top: 10, bottom: 10 })

      Row() {
        this.StatTile('已存片段', `${this.status.segments}`)
        this.StatTile('已转录', `${this.status.transcribed}`)
        this.StatTile('待转录', `${this.status.pending}`)
        this.StatTile('占用', this.formatBytes(this.status.storedBytes))
      }
      .width('100%')
      .padding({ top: 14, bottom: 14 })
      .backgroundColor('#F7F8FA')
      .borderRadius(12)
      .margin({ left: 16, right: 16 })

      // Level against the bar it has to clear. Shown only while recording, because at
      // rest it says nothing — and shown at all because without it a threshold that
      // rejects everything looks exactly like a microphone that delivers nothing.
      if (this.status.recording) {
        Row({ space: 8 }) {
          Text('输入电平')
            .fontSize(11)
            .fontColor('#999999')
          Progress({
            value: Math.min(this.status.inputLevel, this.status.inputThreshold * 2),
            total: Math.max(this.status.inputThreshold * 2, 1),
            type: ProgressType.Linear,
          })
            .color(this.status.inputLevel >= this.status.inputThreshold
              ? '#22AA55' : '#CCCCCC')
            .layoutWeight(1)
            .height(6)
          Text(`${this.status.inputLevel} / 门限 ${this.status.inputThreshold}`)
            .fontSize(11)
            .fontColor(this.status.inputLevel >= this.status.inputThreshold
              ? '#22AA55' : '#999999')
        }
        .width('100%')
        .padding({ left: 18, right: 18, top: 10 })
      }

      Column() {
        if (this.rows.length === 0) {
          Text(this.status.recording
            ? '正在监听，说话后会自动存段并转录'
            : '还没有录到内容')
            .fontSize(15)
            .fontColor('#AAAAAA')
            .margin({ top: 48 })
        } else {
          List({ space: 8 }) {
            ForEach(this.rows, (row: TimelineRow) => {
              ListItem() {
                this.SegmentRow(row)
              }
            }, (row: TimelineRow) => row.segment.fingerprint)
          }
          .width('100%')
          .layoutWeight(1)
        }
      }
      .layoutWeight(1)
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 12 })

      Button(this.status.recording ? '停止录音' : '开始持续录音')
        .width('80%')
        .height(52)
        .fontSize(17)
        .backgroundColor(this.status.recording ? '#FF4444' : '#0066CC')
        .enabled(this.engineReady && !this.busy)
        .onClick((): void => {
          this.toggleRecording();
        })
        .margin({ bottom: 24 })
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#FFFFFF')
  }
}
