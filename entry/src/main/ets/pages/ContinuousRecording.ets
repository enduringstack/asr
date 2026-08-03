import { common } from '@kit.AbilityKit';
import { router } from '@kit.ArkUI';
import { abilityAccessCtrl, Permissions } from '@kit.AbilityKit';
import { ErrorEvent, MessageEvents, worker } from '@kit.ArkTS';
import { hilog } from '@kit.PerformanceAnalysisKit';
import {
  ContinuousRecordingService,
  RecordingStatus,
  TimelineRow,
} from '../services/recording/ContinuousRecordingService';
import { SegmentRecord } from '../services/recording/SegmentStore';
import { PlaybackState, SegmentPlayer } from '../services/recording/SegmentPlayer';

const DOMAIN = 0x0001;
const TAG = 'ContinuousRecordingPage';

interface SegmentWorkerRequest {
  msgType: string;
  context?: common.UIAbilityContext;
  requestId?: string;
  data?: Uint8Array;
  audioFormat?: string;
  fileAsrMode?: string;
}

interface SegmentWorkerResponse {
  msgType: string;
  requestId?: string;
  text?: string;
  message?: string;
}

/** Resolvers for segments currently inside the worker, keyed by request id. */
interface PendingTranscription {
  resolve: (text: string) => void;
  reject: (error: Error) => void;
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
  private workerInstance: worker.ThreadWorker | undefined = undefined;
  private pending: Map<string, PendingTranscription> =
    new Map<string, PendingTranscription>();
  private requestSeq: number = 0;
  private active: boolean = false;

  aboutToAppear(): void {
    this.active = true;
    this.startEngine();
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
      (record: SegmentRecord, bytes: Uint8Array): Promise<string> =>
        this.transcribeSegment(record, bytes));
    this.rows = this.service.timeline();
    this.status = this.service.status();
  }

  private onWorkerMessage(event: MessageEvents): void {
    const response = event.data as SegmentWorkerResponse;
    const msgType: string = response.msgType;

    if (msgType === 'sherpa-asr-ready') {
      this.engineReady = true;
      this.statusMessage = '识别引擎就绪';
      // Segments recorded during the model load are waiting on exactly this; telling the
      // service lets it clear their failure counts and start working through them.
      this.service.setEngineReady(true);
      return;
    }
    if (msgType === 'sherpa-asr-segment-done') {
      const entry = this.pending.get(response.requestId ?? '');
      if (entry !== undefined) {
        this.pending.delete(response.requestId ?? '');
        entry.resolve(response.text ?? '');
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
  private transcribeSegment(record: SegmentRecord, bytes: Uint8Array): Promise<string> {
    const instance = this.workerInstance;
    if (instance === undefined || !this.engineReady) {
      return Promise.reject(new Error('识别引擎尚未就绪'));
    }
    // Re-bound as a non-nullable const: the executor below closes over it, and a
    // narrowing from the guard above does not survive into a closure.
    const active: worker.ThreadWorker = instance;
    this.requestSeq += 1;
    const requestId: string = `seg-${this.requestSeq}`;
    return new Promise<string>((resolve: (text: string) => void,
      reject: (error: Error) => void): void => {
      this.pending.set(requestId, { resolve, reject });
      try {
        const request: SegmentWorkerRequest = {
          msgType: 'sherpa-asr-transcribe-segment',
          requestId: requestId,
          data: bytes,
          audioFormat: 'wav',
          fileAsrMode: 'offline',
        };
        active.postMessage(request, [bytes.buffer]);
      } catch (error) {
        this.pending.delete(requestId);
        reject(new Error(`${error}`));
      }
    });
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
          Text('待转录')
            .fontSize(11)
            .fontColor('#B36B00')
            .backgroundColor('#FFF5E6')
            .padding({ left: 6, right: 6, top: 2, bottom: 2 })
            .borderRadius(4)
        }
        Blank()

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
      return '正在排队转录…';
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
