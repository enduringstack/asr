import { common } from '@kit.AbilityKit';
import { router } from '@kit.ArkUI';
import { ErrorEvent, MessageEvents, util, worker } from '@kit.ArkTS';
import { hilog } from '@kit.PerformanceAnalysisKit';
import {
  SpeakerCandidate,
  SpeakerSearchProfile,
  SpeakerTurn,
} from '../common/SpeakerTypes';
import { SpeakerProfileStore } from '../services/speaker/SpeakerProfileStore';

const LOG_DOMAIN: number = 0x0001;
const LOG_TAG: string = 'SpeakerTestCenter';

interface FixedSpeakerTest {
  id: string;
  label: string;
  path: string;
  format: string;
  durationSec: number;
  purpose: string;
}

interface FixedSpeakerTestResult {
  id: string;
  state: string;
  elapsedMs: number;
  rtf: number;
  clusters: number;
  known: number;
  unknown: number;
  turns: number;
  error: string;
}

interface TestWorkerRequest {
  msgType: string;
  context?: common.UIAbilityContext;
  data?: Uint8Array;
  audioFormat?: string;
  fileAsrMode?: string;
  enableSpeakerDiarization?: boolean;
  enableSegmentRescore?: boolean;
  meetingId?: string;
  voiceprints?: SpeakerSearchProfile[];
}

interface TestWorkerResponse {
  msgType: string;
  text?: string;
  message?: string;
  speakerTurns?: SpeakerTurn[];
  speakerCandidates?: SpeakerCandidate[];
}

@Entry
@Component
struct SpeakerTestCenterPage {
  @State ready: boolean = false;
  @State running: boolean = false;
  @State statusMessage: string = '正在加载端侧模型...';
  @State latestResult: string = '';
  @State results: FixedSpeakerTestResult[] = [];
  @State progressText: string = '0 / 17';

  private workerInstance: worker.ThreadWorker | undefined = undefined;
  private profileStore: SpeakerProfileStore = SpeakerProfileStore.getInstance();
  private profileStoreReady: boolean = false;
  private profileLoadComplete: boolean = false;
  private modelReady: boolean = false;
  private currentIndex: number = -1;
  private currentStartedAt: number = 0;
  private tests: FixedSpeakerTest[] = [
    { id: 'a4-0', label: 'AISHELL-4 会议 0–8s', path: 'test/aishell4-meeting-s-r004s04c01-0s.wav', format: 'wav', durationSec: 8, purpose: '公开会议切片回归' },
    { id: 'a4-12', label: 'AISHELL-4 会议 12–20s', path: 'test/aishell4-meeting-s-r004s04c01-12s.wav', format: 'wav', durationSec: 8, purpose: '公开会议切片回归' },
    { id: 'a4-24', label: 'AISHELL-4 会议 24–32s', path: 'test/aishell4-meeting-s-r004s04c01-24s.wav', format: 'wav', durationSec: 8, purpose: '公开会议切片回归' },
    { id: 'ali-1', label: 'AliMeeting R8002-M8002', path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8002_M8002_MS802_3min.m4a', format: 'm4a', durationSec: 180, purpose: '远场稳定性' },
    { id: 'ali-2', label: 'AliMeeting R8002-M8003', path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8002_M8003_MS803_3min.wav', format: 'wav', durationSec: 180, purpose: '远场稳定性' },
    { id: 'ali-3', label: 'AliMeeting R8004-M8005', path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8004_M8005_MS803_3min.wav', format: 'wav', durationSec: 180, purpose: '远场稳定性' },
    { id: 'ali-4', label: 'AliMeeting R8004-M8006', path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8004_M8006_MS805_3min.wav', format: 'wav', durationSec: 180, purpose: '远场稳定性' },
    { id: 'ali-5', label: 'AliMeeting R8005-M8007', path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8005_M8007_MS806_3min.wav', format: 'wav', durationSec: 180, purpose: '远场稳定性' },
    { id: 'ali-6', label: 'AliMeeting R8005-M8008', path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8005_M8008_MS806_3min.wav', format: 'wav', durationSec: 180, purpose: '远场稳定性' },
    { id: 'ali-7', label: 'AliMeeting R8005-M8009', path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8005_M8009_MS802_3min.wav', format: 'wav', durationSec: 180, purpose: '远场稳定性' },
    { id: 'ali-8', label: 'AliMeeting R8006-M8012', path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8006_M8012_MS803_3min.wav', format: 'wav', durationSec: 180, purpose: '远场稳定性' },
    { id: 'ali-9', label: 'AliMeeting R8008-M8014', path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8008_M8014_MS807_3min.wav', format: 'wav', durationSec: 180, purpose: '远场稳定性' },
    { id: 'ali-10', label: 'AliMeeting R8008-M8015', path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8008_M8015_MS808_3min.wav', format: 'wav', durationSec: 180, purpose: '远场稳定性' },
    { id: 'ali-11', label: 'AliMeeting R8008-M8016', path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8008_M8016_MS808_3min.wav', format: 'wav', durationSec: 180, purpose: '远场稳定性' },
    { id: 'ali-12', label: 'AliMeeting R8008-M8017', path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8008_M8017_MS808_3min.wav', format: 'wav', durationSec: 180, purpose: '远场稳定性' },
    { id: 'ali-13', label: 'AliMeeting R8009-M8021', path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8009_M8021_MS810_3min.wav', format: 'wav', durationSec: 180, purpose: '远场稳定性' },
    { id: 'ali-14', label: 'AliMeeting R8009-M8022', path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8009_M8022_MS810_3min.wav', format: 'wav', durationSec: 180, purpose: '远场稳定性' },
  ];

  aboutToAppear(): void {
    const context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;
    this.results = this.emptyResults();
    this.profileStore.init(context)
      .then((): void => {
        this.profileStoreReady = true;
        this.profileLoadComplete = true;
        this.restoreProfiles();
      })
      .catch((_error: Error): void => {
        this.profileLoadComplete = true;
        this.restoreProfiles();
      });
    const instance: worker.ThreadWorker = new worker.ThreadWorker(
      'entry/ets/workers/StreamingSherpaWorker.ets', { name: 'Speaker fixed test worker' });
    instance.onmessage = (event: MessageEvents): void => {
      this.onWorkerMessage(event);
    };
    instance.onerror = (event: ErrorEvent): void => {
      this.statusMessage = `测试引擎异常: ${event.message}`;
      this.running = false;
    };
    this.workerInstance = instance;
    const request: TestWorkerRequest = { msgType: 'sherpa-asr-init', context: context };
    instance.postMessage(request);
  }

  aboutToDisappear(): void {
    if (this.workerInstance) {
      this.workerInstance.terminate();
      this.workerInstance = undefined;
    }
  }

  private emptyResults(): FixedSpeakerTestResult[] {
    const results: FixedSpeakerTestResult[] = [];
    for (let i: number = 0; i < this.tests.length; i++) {
      results.push({
        id: this.tests[i].id, state: '等待', elapsedMs: 0, rtf: 0,
        clusters: 0, known: 0, unknown: 0, turns: 0, error: '',
      });
    }
    return results;
  }

  private resultFor(index: number): FixedSpeakerTestResult {
    if (index >= 0 && index < this.results.length) {
      return this.results[index];
    }
    return {
      id: index >= 0 && index < this.tests.length ? this.tests[index].id : '',
      state: '等待', elapsedMs: 0, rtf: 0, clusters: 0,
      known: 0, unknown: 0, turns: 0, error: '',
    };
  }

  private onWorkerMessage(event: MessageEvents): void {
    const response: TestWorkerResponse = event.data as TestWorkerResponse;
    if (response.msgType === 'sherpa-asr-ready') {
      this.modelReady = true;
      this.statusMessage = '模型已加载，正在恢复本机人物...';
      this.restoreProfiles();
      return;
    }
    if (response.msgType === 'sherpa-voiceprint-restored') {
      this.ready = true;
      this.statusMessage = 'CAM++、说话人分离和本机人物已就绪';
      return;
    }
    if (response.msgType === 'sherpa-asr-self-test-done') {
      this.finishCurrent(response);
      return;
    }
    if (response.msgType === 'sherpa-asr-error') {
      if (this.running && this.currentIndex >= 0) {
        this.failCurrent(response.message || '未知错误');
      } else {
        this.ready = false;
        this.statusMessage = `测试引擎错误: ${response.message || '未知错误'}`;
      }
    }
  }

  private restoreProfiles(): void {
    if (!this.modelReady || !this.profileLoadComplete || !this.workerInstance) {
      return;
    }
    const restore: TestWorkerRequest = {
      msgType: 'sherpa-voiceprint-restore',
      voiceprints: this.profileStoreReady ? this.profileStore.searchProfiles() : [],
    };
    this.workerInstance.postMessage(restore);
  }

  private startAll(): void {
    if (!this.ready || this.running) {
      return;
    }
    this.results = this.emptyResults();
    this.latestResult = '';
    this.running = true;
    this.currentIndex = -1;
    this.runNext();
  }

  private runNext(): void {
    this.currentIndex += 1;
    if (this.currentIndex >= this.tests.length) {
      this.running = false;
      this.progressText = `${this.tests.length} / ${this.tests.length}`;
      let completed: number = 0;
      let failed: number = 0;
      let elapsedMs: number = 0;
      let audioSeconds: number = 0;
      for (let i: number = 0; i < this.results.length; i++) {
        if (this.results[i].state === '完成') {
          completed += 1;
          elapsedMs += this.results[i].elapsedMs;
          audioSeconds += this.tests[i].durationSec;
        } else if (this.results[i].state === '失败') {
          failed += 1;
        }
      }
      const overallRtf: number = audioSeconds > 0 ? elapsedMs / 1000 / audioSeconds : 0;
      this.statusMessage = `固定音频完成 ${completed} 条，失败 ${failed} 条，整体 RTF ${overallRtf.toFixed(4)}`;
      hilog.info(LOG_DOMAIN, LOG_TAG,
        `SUMMARY completed=${completed} failed=${failed} elapsedMs=${elapsedMs} audioSec=${audioSeconds} rtf=${overallRtf.toFixed(4)}`);
      return;
    }
    const test: FixedSpeakerTest = this.tests[this.currentIndex];
    this.progressText = `${this.currentIndex + 1} / ${this.tests.length}`;
    this.statusMessage = `正在处理 ${test.label}`;
    const current: FixedSpeakerTestResult[] = this.results.slice();
    current[this.currentIndex].state = '运行中';
    this.results = current;
    const context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;
    context.resourceManager.getRawFileContent(test.path)
      .then((data: Uint8Array): void => {
        if (!this.workerInstance) {
          this.failCurrent('测试引擎已关闭');
          return;
        }
        this.currentStartedAt = Date.now();
        const request: TestWorkerRequest = {
          msgType: 'sherpa-asr-decode-wav',
          data: data,
          audioFormat: test.format,
          fileAsrMode: 'offline_paraformer_large',
          enableSpeakerDiarization: true,
          enableSegmentRescore: false,
          meetingId: util.generateRandomUUID(true),
        };
        this.workerInstance.postMessage(request, [data.buffer]);
      })
      .catch((error: Error): void => {
        this.failCurrent(error.message);
      });
  }

  private finishCurrent(response: TestWorkerResponse): void {
    if (this.currentIndex < 0 || this.currentIndex >= this.tests.length) {
      return;
    }
    const candidates: SpeakerCandidate[] = response.speakerCandidates || [];
    let known: number = 0;
    for (let i: number = 0; i < candidates.length; i++) {
      if (candidates[i].personId) {
        known += 1;
      }
    }
    const elapsedMs: number = Date.now() - this.currentStartedAt;
    const result: FixedSpeakerTestResult = {
      id: this.tests[this.currentIndex].id,
      state: '完成',
      elapsedMs: elapsedMs,
      rtf: elapsedMs / 1000 / this.tests[this.currentIndex].durationSec,
      clusters: candidates.length,
      known: known,
      unknown: Math.max(0, candidates.length - known),
      turns: response.speakerTurns ? response.speakerTurns.length : 0,
      error: '',
    };
    const results: FixedSpeakerTestResult[] = this.results.slice();
    results[this.currentIndex] = result;
    this.results = results;
    this.latestResult = `${this.tests[this.currentIndex].label} · RTF ${result.rtf.toFixed(3)} · ` +
      `${result.clusters} 簇 / ${result.turns} 段`;
    hilog.info(LOG_DOMAIN, LOG_TAG,
      `RESULT id=${result.id} elapsedMs=${result.elapsedMs} rtf=${result.rtf.toFixed(4)} clusters=${result.clusters} known=${result.known} unknown=${result.unknown} turns=${result.turns}`);
    this.runNext();
  }

  private failCurrent(message: string): void {
    if (this.currentIndex >= 0 && this.currentIndex < this.results.length) {
      const results: FixedSpeakerTestResult[] = this.results.slice();
      results[this.currentIndex].state = '失败';
      results[this.currentIndex].error = message;
      this.results = results;
      this.latestResult = `${this.tests[this.currentIndex].label} · 失败：${message}`;
      hilog.error(LOG_DOMAIN, LOG_TAG,
        `RESULT id=${this.tests[this.currentIndex].id} state=failed error=${message}`);
    }
    this.runNext();
  }

  @Builder
  private ResultRow(test: FixedSpeakerTest, result: FixedSpeakerTestResult) {
    Column() {
      Row() {
        Column() {
          Text(test.label)
            .fontSize(14)
            .fontWeight(FontWeight.Medium)
            .fontColor('#1F2937')
          Text(`${test.purpose} · ${test.durationSec} 秒`)
            .fontSize(11)
            .fontColor('#6B7280')
            .margin({ top: 3 })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
        Text(result.state)
          .fontSize(12)
          .fontColor(result.state === '失败' ? '#B91C1C' :
            result.state === '完成' ? '#047857' : '#6B7280')
      }
      .width('100%')
      if (result.state === '完成') {
        Text(`RTF ${result.rtf.toFixed(3)} · ${result.clusters} 簇 · ` +
          `${result.known} 已知 / ${result.unknown} 未知 · ${result.turns} 段`)
          .fontSize(12)
          .fontColor('#4B5563')
          .width('100%')
          .margin({ top: 7 })
      } else if (result.error.length > 0) {
        Text(result.error)
          .fontSize(11)
          .fontColor('#B91C1C')
          .width('100%')
          .margin({ top: 7 })
      }
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
          Text('声纹固定测试')
            .fontSize(20)
            .fontWeight(FontWeight.Bold)
          Text(this.statusMessage)
            .fontSize(11)
            .fontColor('#6B7280')
            .maxLines(1)
          if (this.latestResult.length > 0) {
            Text(this.latestResult)
              .fontSize(10)
              .fontColor('#047857')
              .maxLines(1)
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
        Text('CAM++ 中英双语 · 26.97 MB · 192 维 · 2 线程')
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .fontColor('#4C1D95')
        Text('Mac 整场基线：AISHELL-4 DER 7.93%，RTF 0.0747；本页展示手机切片回归。')
          .fontSize(11)
          .fontColor('#6D28D9')
          .margin({ top: 4 })
        Text('本页不把短切片簇数当 DER；AliMeeting 无真值，只用于稳定性与耗时回归。')
          .fontSize(11)
          .fontColor('#6D28D9')
          .margin({ top: 3 })
      }
      .alignItems(HorizontalAlign.Start)
      .width('100%')
      .padding(13)
      .backgroundColor('#F5F3FF')
      .borderRadius(12)
      .margin({ left: 16, right: 16, bottom: 10 })

      List() {
        ForEach(this.results, (result: FixedSpeakerTestResult, index: number): void => {
          ListItem() {
            this.ResultRow(this.tests[index], result)
          }
        }, (result: FixedSpeakerTestResult): string =>
          `${result.id}-${result.state}-${result.elapsedMs}`)
      }
      .layoutWeight(1)
      .width('100%')
      .padding({ left: 16, right: 16 })

      Button(this.running ? `正在运行 ${this.progressText}` : '运行全部固定音频')
        .width('80%')
        .height(48)
        .backgroundColor('#0066CC')
        .fontColor('#FFFFFF')
        .enabled(this.ready && !this.running)
        .onClick((): void => {
          this.startAll();
        })
        .margin({ top: 8, bottom: 20 })
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#F7F8FA')
  }
}
