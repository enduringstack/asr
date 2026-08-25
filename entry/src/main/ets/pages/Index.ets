import { abilityAccessCtrl, common, Permissions } from '@kit.AbilityKit';
import { fileIo, picker } from '@kit.CoreFileKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { promptAction, router } from '@kit.ArkUI';
import { ErrorEvent, MessageEvents, util, worker } from '@kit.ArkTS';
import {
  EnrollmentQuality,
  MeetingSpeakerAssignment,
  SPEAKER_ACTIVE_ENROLL_ROUNDS,
  SPEAKER_IDENTITY_THRESHOLD,
  SpeakerCandidate,
  SpeakerProfile,
  SpeakerSearchProfile,
  SpeakerTurn,
  VoicePrototype,
} from '../common/SpeakerTypes';
import { AudioRecordingService, AudioChunk } from '../services/audio/AudioRecordingService';
import {
  createVoicePrototype,
  prototypeSimilarityFloor,
} from '../services/speaker/SpeakerMath';
import {
  SpeakerMigrationResult,
  SpeakerProfileStore,
} from '../services/speaker/SpeakerProfileStore';
import { MeetingSpeakerStore } from '../services/speaker/MeetingSpeakerStore';
import { SpeakerEditSessionStore } from '../services/speaker/SpeakerEditSessionStore';

const DOMAIN = 0x0001;
const TAG = 'ASR';
const MIC_PERMISSION: Permissions = 'ohos.permission.MICROPHONE';
const SHOW_ADVANCED_CONTROLS = true;
const SHOW_TUNING_CONTROLS = false;
const SHOW_VOICEPRINT_CONTROLS = true;
const SHOW_DIAGNOSTICS = true;
const FILE_ASR_MODE_STREAMING = 'streaming';
const FILE_ASR_MODE_OFFLINE_PARAFORMER = 'offline_paraformer_large';
const MAX_IMPORTED_AUDIO_BYTES: number = 256 * 1024 * 1024;

type AsrState = 'idle' | 'initializing' | 'ready' | 'starting' | 'recording' | 'voiceprintStarting' |
  'voiceprintRecording' | 'processing' | 'error';

interface SherpaWorkerRequest {
  msgType: string;
  context?: common.UIAbilityContext;
  pcm?: ArrayBuffer;
  data?: Uint8Array;
  audioFormat?: string;
  fileName?: string;
  fileAsrMode?: string;
  enableSpeakerDiarization?: boolean;
  enableSegmentRescore?: boolean;
  compareEnhanced?: boolean;
  speakerName?: string;
  voiceprints?: SpeakerSearchProfile[];
  meetingId?: string;
}

interface AsrTelemetry {
  chunkDecodeMs?: number;
  partialIntervalMs?: number;
  finalSegmentMs?: number;
  finalRedecodeMs?: number;
  finalRedecodeRtf?: number;
  audioDurationSec?: number;
}

interface SherpaWorkerResponse {
  msgType: string;
  message?: string;
  text?: string;
  punctuationReady?: boolean;
  segmentIndex?: number;
  segmentTotal?: number;
  segmentStart?: number;
  segmentEnd?: number;
  audioDuration?: number;
  speakerName?: string;
  speakerNames?: string[];
  embedding?: number[];
  quality?: EnrollmentQuality;
  meetingId?: string;
  speakerTurns?: SpeakerTurn[];
  speakerCandidates?: SpeakerCandidate[];
  voiceprintReady?: boolean;
  telemetry?: AsrTelemetry;
}

interface SelfTestAudio {
  label: string;
  path: string;
}

interface SelfTestOptions {
  enableSpeakerDiarization: boolean;
  enableSegmentRescore: boolean;
  compareEnhanced: boolean;
  fileAsrMode: string;
}

@Entry
@Component
struct Index {
  @State asrState: AsrState = 'initializing';
  @State recognizedText: string = '';
  @State partialText: string = '';
  @State statusMessage: string = '正在初始化本地语音识别...';
  @State isButtonPressed: boolean = false;
  @State punctuationReady: boolean = false;
  @State enableSpeakerDiarization: boolean = true;
  @State enableSegmentRescore: boolean = false;
  @State compareEnhanced: boolean = false;
  @State useOfflineParaformerForFiles: boolean = true;
  @State isVoiceprintPressed: boolean = false;
  @State voiceprintName: string = '我';
  @State voiceprintSummary: string = '未注册';
  @State voiceprintRound: number = 0;
  @State hasEditableSpeakers: boolean = false;
  @State latestSpeakerPreview: string = '';
  @State audioLevelPercent: number = 0;
  @State recordingDurationText: string = '00:00';

  private audioService: AudioRecordingService = AudioRecordingService.getInstance();
  private workerInstance?: worker.ThreadWorker;
  private profileStore: SpeakerProfileStore = SpeakerProfileStore.getInstance();
  private meetingSpeakerStore: MeetingSpeakerStore | undefined = undefined;
  private speakerEditSessions: SpeakerEditSessionStore = SpeakerEditSessionStore.getInstance();
  @State profileStoreReady: boolean = false;
  private voiceprintProfiles: SpeakerProfile[] = [];
  private enrollmentPrototypes: VoicePrototype[] = [];
  private latestMeetingId: string = '';
  private latestMeetingTitle: string = '';
  private latestSpeakerTurns: SpeakerTurn[] = [];
  private latestSpeakerCandidates: SpeakerCandidate[] = [];
  private recordingStartToken: number = 0;
  private voiceprintStartToken: number = 0;
  private recordingStartTime: number = 0;
  private voiceprintStartTime: number = 0;
  private lastAudioStatusUpdateAt: number = 0;
  private activeImportedAudioBaseText: string = '';
  private activeImportedAudioProgressText: string = '';
  private readonly workerScriptUrl: string = 'entry/ets/workers/StreamingSherpaWorker.ets';
  private activeSelfTestOptions: SelfTestOptions = {
    enableSpeakerDiarization: true,
    enableSegmentRescore: false,
    compareEnhanced: false,
    fileAsrMode: FILE_ASR_MODE_OFFLINE_PARAFORMER
  };
  private readonly selfTestAudios: SelfTestAudio[] = [
    {
      label: 'AliMeeting R8002 M8002 MS802',
      path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8002_M8002_MS802_3min.m4a'
    },
    {
      label: 'AliMeeting R8002 M8003 MS803',
      path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8002_M8003_MS803_3min.wav'
    },
    {
      label: 'AliMeeting R8004 M8005 MS803',
      path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8004_M8005_MS803_3min.wav'
    },
    {
      label: 'AliMeeting R8004 M8006 MS805',
      path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8004_M8006_MS805_3min.wav'
    },
    {
      label: 'AliMeeting R8005 M8007 MS806',
      path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8005_M8007_MS806_3min.wav'
    },
    {
      label: 'AliMeeting R8005 M8008 MS806',
      path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8005_M8008_MS806_3min.wav'
    },
    {
      label: 'AliMeeting R8005 M8009 MS802',
      path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8005_M8009_MS802_3min.wav'
    },
    {
      label: 'AliMeeting R8006 M8012 MS803',
      path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8006_M8012_MS803_3min.wav'
    },
    {
      label: 'AliMeeting R8008 M8014 MS807',
      path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8008_M8014_MS807_3min.wav'
    },
    {
      label: 'AliMeeting R8008 M8015 MS808',
      path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8008_M8015_MS808_3min.wav'
    },
    {
      label: 'AliMeeting R8008 M8016 MS808',
      path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8008_M8016_MS808_3min.wav'
    },
    {
      label: 'AliMeeting R8008 M8017 MS808',
      path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8008_M8017_MS808_3min.wav'
    },
    {
      label: 'AliMeeting R8009 M8021 MS810',
      path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8009_M8021_MS810_3min.wav'
    },
    {
      label: 'AliMeeting R8009 M8022 MS810',
      path: 'test/downloads_audios/chinese_meeting_room_discussion_alimeeting_R8009_M8022_MS810_3min.wav'
    }
  ];
  private selfTestIndex: number = 0;
  private activeSelfTestCount: number = 0;
  private isSelfTestRunning: boolean = false;
  private resumeSelfTestAfterWorkerReady: boolean = false;
  private activeImportedAudioLabel: string = '';

  aboutToAppear(): void {
    this.initializeSpeakerProfiles();
    this.initASR();
  }

  aboutToDisappear(): void {
    this.recordingStartToken += 1;
    this.voiceprintStartToken += 1;
    this.audioService.stop();
    this.audioService.clearCallbacks();
    this.shutdownWorker();
  }

  /**
   * Releases the recogniser while another page is in front.
   *
   * router.pushUrl leaves this page mounted, so without this the continuous recording
   * screen would load a second copy of the models alongside this one. Returning costs a
   * reload, which is the right trade against holding hundreds of megabytes for a screen
   * nobody is looking at.
   */
  onPageHide(): void {
    this.recordingStartToken += 1;
    this.voiceprintStartToken += 1;
    this.audioService.stop();
    this.audioService.clearCallbacks();
    this.shutdownWorker();
    this.asrState = 'initializing';
    this.statusMessage = '已释放识别引擎，返回后自动重新加载';
  }

  onPageShow(): void {
    if (this.profileStoreReady) {
      this.voiceprintProfiles = this.profileStore.list();
      this.updateVoiceprintSummary();
      this.refreshLatestSpeakerAssignments();
    }
    if (!this.workerInstance) {
      this.initASR();
    } else {
      this.restoreVoiceprintsToWorker();
    }
  }

  private getAbilityContext(): common.UIAbilityContext {
    return getContext(this) as common.UIAbilityContext;
  }

  private openContinuousRecording(): void {
    router.pushUrl({ url: 'pages/ContinuousRecording' })
      .catch((error: Error): void => {
        this.statusMessage = `打开持续录音失败: ${error.message}`;
        hilog.error(DOMAIN, TAG, `pushUrl failed: ${error.message}`);
      });
  }

  private openSpeakerProfiles(): void {
    router.pushUrl({ url: 'pages/SpeakerProfiles' })
      .catch((error: Error): void => {
        this.statusMessage = `打开人物与声纹失败: ${error.message}`;
      });
  }

  private openSpeakerTestCenter(): void {
    router.pushUrl({ url: 'pages/SpeakerTestCenter' })
      .catch((error: Error): void => {
        this.statusMessage = `打开固定测试失败: ${error.message}`;
      });
  }

  private openMeetingSpeakerEditor(): void {
    if (this.latestMeetingId.length === 0 || this.latestSpeakerTurns.length === 0) {
      promptAction.showToast({ message: '当前没有可编辑的说话人结果' });
      return;
    }
    const sessionId: string = this.speakerEditSessions.put(
      this.latestMeetingId,
      this.latestMeetingTitle,
      this.latestSpeakerTurns,
      this.latestSpeakerCandidates);
    router.pushUrl({
      url: 'pages/MeetingSpeakerEditor',
      params: {
        sessionId: sessionId,
      },
    }).catch((error: Error): void => {
      this.statusMessage = `打开说话人编辑失败: ${error.message}`;
    });
  }

  private initializeSpeakerProfiles(): void {
    const context: common.UIAbilityContext = this.getAbilityContext();
    this.meetingSpeakerStore = new MeetingSpeakerStore(context);
    this.profileStore.init(context)
      .then((migration: SpeakerMigrationResult): void => {
        this.profileStoreReady = true;
        this.voiceprintProfiles = this.profileStore.list();
        this.updateVoiceprintSummary();
        if (migration.needsReRegister > 0) {
          this.statusMessage = `已保留 ${migration.migratedNames} 位旧人物，请重新注册声纹`;
        }
        if (this.workerInstance) {
          this.restoreVoiceprintsToWorker();
        }
        this.refreshLatestSpeakerAssignments();
      })
      .catch((error: Error): void => {
        this.profileStoreReady = false;
        this.statusMessage = `声纹库初始化失败: ${error.message}`;
        hilog.error(DOMAIN, TAG, `Speaker profile store init failed: ${error.message}`);
      });
  }

  private async ensureMicrophonePermission(): Promise<boolean> {
    const context = this.getAbilityContext();
    const atManager = abilityAccessCtrl.createAtManager();
    try {
      let granted = atManager.checkAccessTokenSync(
        context.applicationInfo.accessTokenId,
        MIC_PERMISSION
      ) === abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED;
      if (granted) {
        return true;
      }

      const result = await atManager.requestPermissionsFromUser(context, [MIC_PERMISSION]);
      granted = result.authResults.length > 0 &&
        result.authResults[0] === abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED;
      if (!granted) {
        this.statusMessage = '需要麦克风权限';
        promptAction.showToast({ message: '请允许麦克风权限后再录音' });
      }
      return granted;
    } catch (error) {
      this.statusMessage = `权限请求失败: ${error}`;
      hilog.error(DOMAIN, TAG, `Microphone permission error: ${error}`);
      return false;
    }
  }

  initASR(): void {
    this.shutdownWorker();
    this.asrState = 'initializing';
    this.statusMessage = '正在加载本地语音识别...';

    this.workerInstance = new worker.ThreadWorker(this.workerScriptUrl, {
      name: 'Sherpa streaming ASR worker'
    });
    this.workerInstance.onmessage = (e: MessageEvents) => {
      this.onWorkerMessage(e);
    };
    this.workerInstance.onmessageerror = (_e: MessageEvents) => {
      this.handleWorkerFatal('识别引擎消息反序列化失败，正在重新加载');
    };
    this.workerInstance.onerror = (e: ErrorEvent) => {
      this.handleWorkerFatal(`识别引擎异常: ${e.message}`);
    };

    this.audioService.setOnChunkCallback((chunk: AudioChunk) => {
      this.onAudioChunk(chunk);
    });
    this.audioService.setStateCallback((state, error) => {
      hilog.info(DOMAIN, TAG, `Audio state: ${state}, error: ${error || 'none'}`);
      if (state === 'error') {
        this.handleAudioFailure(error || '录音启动失败');
      }
    });

    const context = this.getAbilityContext();
    this.postToWorker({ msgType: 'sherpa-asr-init', context });
  }

  private postToWorker(request: SherpaWorkerRequest): void {
    try {
      this.workerInstance?.postMessage(request);
    } catch (error) {
      this.asrState = 'error';
      this.statusMessage = `识别引擎消息发送失败: ${error}`;
      hilog.error(DOMAIN, TAG, `Worker postMessage failed: ${error}`);
    }
  }

  private postAudioFileToWorker(request: SherpaWorkerRequest, data: Uint8Array): void {
    try {
      request.data = data;
      this.workerInstance?.postMessage(request, [data.buffer]);
    } catch (error) {
      this.asrState = 'ready';
      this.statusMessage = `音频测试失败: ${error}`;
      this.activeImportedAudioLabel = '';
      this.activeImportedAudioBaseText = '';
      this.activeImportedAudioProgressText = '';
      hilog.error(DOMAIN, TAG, `Worker audio file postMessage failed: ${error}`);
    }
  }

  private postPcmToWorker(msgType: string, pcm: ArrayBuffer): void {
    try {
      const request: SherpaWorkerRequest = { msgType, pcm };
      this.workerInstance?.postMessage(request, [pcm]);
    } catch (error) {
      this.handleWorkerFatal(`识别引擎音频发送失败: ${error}`);
      hilog.error(DOMAIN, TAG, `Worker PCM postMessage failed: ${error}`);
    }
  }

  private shutdownWorker(): void {
    if (!this.workerInstance) {
      return;
    }
    try {
      this.workerInstance.onmessage = undefined;
      this.workerInstance.onmessageerror = undefined;
      this.workerInstance.onerror = undefined;
      this.workerInstance.terminate();
    } catch (error) {
      hilog.warn(DOMAIN, TAG, `Worker shutdown failed: ${error}`);
    }
    this.workerInstance = undefined;
  }

  private handleWorkerFatal(message: string): void {
    this.isButtonPressed = false;
    this.isVoiceprintPressed = false;
    this.recordingStartToken += 1;
    this.voiceprintStartToken += 1;
    this.audioLevelPercent = 0;
    this.recordingDurationText = '00:00';
    this.audioService.stop();
    this.shutdownWorker();
    this.asrState = 'initializing';
    this.statusMessage = message;
    this.activeImportedAudioLabel = '';
    this.activeImportedAudioBaseText = '';
    this.activeImportedAudioProgressText = '';
    if (this.isSelfTestRunning) {
      const testAudio = this.selfTestAudios[this.selfTestIndex];
      const label = testAudio ? testAudio.label : '未知音频';
      this.recognizedText += `[测试:${label}] 识别引擎重载: ${message}\n`;
      this.selfTestIndex += 1;
      this.resumeSelfTestAfterWorkerReady = this.selfTestIndex < this.activeSelfTestCount;
      this.isSelfTestRunning = this.resumeSelfTestAfterWorkerReady;
    }
    this.initASR();
  }

  private handleAudioFailure(message: string): void {
    this.isButtonPressed = false;
    this.isVoiceprintPressed = false;
    this.recordingStartToken += 1;
    this.voiceprintStartToken += 1;
    this.audioLevelPercent = 0;
    this.recordingDurationText = '00:00';
    this.postToWorker({ msgType: 'sherpa-asr-cancel' });
    this.postToWorker({ msgType: 'sherpa-voiceprint-cancel' });
    this.asrState = 'ready';
    this.statusMessage = `录音失败: ${message}`;
  }

  private getReadyStatusMessage(): string {
    return this.punctuationReady ? '本地语音识别已就绪，按住说话' : '语音识别已就绪，按住说话';
  }

  private formatElapsed(ms: number): string {
    const totalSeconds = Math.max(0, Math.floor(ms / 1000));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds - minutes * 60;
    const minuteText = minutes < 10 ? `0${minutes}` : `${minutes}`;
    const secondText = seconds < 10 ? `0${seconds}` : `${seconds}`;
    return `${minuteText}:${secondText}`;
  }

  private formatAudioSeconds(seconds?: number): string {
    const safeSeconds = Math.max(0, seconds || 0);
    const minutes = Math.floor(safeSeconds / 60);
    const sec = Math.floor(safeSeconds - minutes * 60);
    const secText = sec < 10 ? `0${sec}` : `${sec}`;
    return `${minutes}:${secText}`;
  }

  private setImportedAudioProgress(line: string): void {
    this.activeImportedAudioProgressText = line.endsWith('\n') ? line : `${line}\n`;
    if (this.activeImportedAudioBaseText.length > 0) {
      this.recognizedText = this.activeImportedAudioBaseText + this.activeImportedAudioProgressText;
    } else {
      this.recognizedText += this.activeImportedAudioProgressText;
    }
  }

  private formatFileSize(size: number): string {
    if (size < 1024) {
      return `${size}B`;
    }
    if (size < 1024 * 1024) {
      return `${(size / 1024).toFixed(1)}KB`;
    }
    return `${(size / 1024 / 1024).toFixed(1)}MB`;
  }

  private extractFileName(uri: string): string {
    const queryStart = uri.indexOf('?');
    const cleanUri = queryStart >= 0 ? uri.substring(0, queryStart) : uri;
    const slashIndex = cleanUri.lastIndexOf('/');
    const rawName = slashIndex >= 0 ? cleanUri.substring(slashIndex + 1) : cleanUri;
    try {
      const decodedName = decodeURIComponent(rawName);
      return decodedName.length > 0 ? decodedName : '音频文件';
    } catch (_error) {
      return rawName.length > 0 ? rawName : '音频文件';
    }
  }

  private getAudioFormatFromName(fileName: string): string {
    const lowerName = fileName.toLowerCase();
    if (lowerName.endsWith('.wav')) {
      return 'wav';
    }
    if (lowerName.endsWith('.mp3')) {
      return 'mp3';
    }
    if (lowerName.endsWith('.m4a')) {
      return 'm4a';
    }
    return '';
  }

  private isWavBytes(data: Uint8Array): boolean {
    return data.length >= 12 &&
      data[0] === 0x52 && data[1] === 0x49 && data[2] === 0x46 && data[3] === 0x46 &&
      data[8] === 0x57 && data[9] === 0x41 && data[10] === 0x56 && data[11] === 0x45;
  }

  private isMp3Bytes(data: Uint8Array): boolean {
    if (data.length < 3) {
      return false;
    }
    const hasId3 = data[0] === 0x49 && data[1] === 0x44 && data[2] === 0x33;
    const hasFrameSync = data.length >= 2 && data[0] === 0xFF && (data[1] & 0xE0) === 0xE0;
    return hasId3 || hasFrameSync;
  }

  private isM4aBytes(data: Uint8Array): boolean {
    return data.length >= 12 &&
      data[4] === 0x66 && data[5] === 0x74 && data[6] === 0x79 && data[7] === 0x70;
  }

  private detectAudioFormatFromContent(data: Uint8Array): string {
    if (this.isWavBytes(data)) {
      return 'wav';
    }
    if (this.isMp3Bytes(data)) {
      return 'mp3';
    }
    if (this.isM4aBytes(data)) {
      return 'm4a';
    }
    return '';
  }

  private getCurrentFileOptions(): SelfTestOptions {
    return {
      enableSpeakerDiarization: this.enableSpeakerDiarization,
      enableSegmentRescore: this.enableSegmentRescore,
      compareEnhanced: this.compareEnhanced,
      fileAsrMode: this.useOfflineParaformerForFiles ? FILE_ASR_MODE_OFFLINE_PARAFORMER : FILE_ASR_MODE_STREAMING
    };
  }

  private async readUriBytes(uri: string): Promise<Uint8Array> {
    let audioFile: fileIo.File | undefined = undefined;
    try {
      audioFile = await fileIo.open(uri, fileIo.OpenMode.READ_ONLY);
      const stat = await fileIo.stat(audioFile.fd);
      if (stat.size <= 0) {
        throw new Error('音频文件有问题：文件为空');
      }
      if (stat.size > MAX_IMPORTED_AUDIO_BYTES) {
        throw new Error('音频文件有问题：单个文件不能超过 256 MB');
      }

      const buffer = new ArrayBuffer(stat.size);
      const readBytes = await fileIo.read(audioFile.fd, buffer, { length: stat.size });
      if (readBytes <= 0) {
        throw new Error('音频文件有问题：无法读取音频内容');
      }

      const bytes = new Uint8Array(buffer, 0, readBytes);
      const copied = new Uint8Array(readBytes);
      copied.set(bytes);
      return copied;
    } finally {
      if (audioFile) {
        await fileIo.close(audioFile);
      }
    }
  }

  private normalizeAudioProblemMessage(message: string): string {
    const cleanMessage = message.replace('Error: ', '').trim();
    if (cleanMessage.indexOf('音频文件有问题') >= 0) {
      return cleanMessage;
    }
    return `音频文件有问题：${cleanMessage}`;
  }

  private validateImportedAudio(fileName: string, data: Uint8Array): string {
    const suffixFormat = this.getAudioFormatFromName(fileName);
    if (suffixFormat.length === 0) {
      throw new Error('音频文件有问题：仅支持 MP3、WAV 或 M4A');
    }

    const contentFormat = this.detectAudioFormatFromContent(data);
    if (contentFormat.length === 0) {
      throw new Error('音频文件有问题：文件内容不是有效的 MP3/WAV/M4A');
    }
    if (contentFormat !== suffixFormat) {
      throw new Error('音频文件有问题：文件后缀和音频内容不一致');
    }
    return contentFormat;
  }

  private computeAudioLevelPercent(buffer: ArrayBuffer): number {
    if (buffer.byteLength < 2) {
      return 0;
    }

    const samples = new Int16Array(buffer);
    if (samples.length === 0) {
      return 0;
    }

    let sum = 0;
    for (let i = 0; i < samples.length; ++i) {
      const sample = samples[i] / 32768.0;
      sum += sample * sample;
    }
    const rms = Math.sqrt(sum / samples.length);
    return Math.min(100, Math.round(rms * 300));
  }

  private updateAudioFeedback(chunk: AudioChunk, state: AsrState): void {
    const now = Date.now();
    const startTime = state === 'voiceprintRecording' ? this.voiceprintStartTime : this.recordingStartTime;
    const elapsedMs = startTime > 0 ? now - startTime : 0;
    this.recordingDurationText = this.formatElapsed(elapsedMs);
    this.audioLevelPercent = this.computeAudioLevelPercent(chunk.pcmData);
    if (now - this.lastAudioStatusUpdateAt < 250) {
      return;
    }

    this.lastAudioStatusUpdateAt = now;
    const quietHint = elapsedMs >= 1500 && this.audioLevelPercent < 2 ? ' · 未检测到明显语音' : '';
    if (state === 'voiceprintRecording') {
      this.statusMessage = `声纹注册中 ${this.recordingDurationText} · 音量 ${this.audioLevelPercent}%${quietHint}`;
    } else {
      this.statusMessage = `录音中 ${this.recordingDurationText} · 音量 ${this.audioLevelPercent}%${quietHint}`;
    }
  }

  private cleanSpeakerName(name: string): string {
    const cleanName = name.trim();
    if (cleanName.length === 0) {
      return '我';
    }
    return cleanName.length > 24 ? cleanName.slice(0, 24) : cleanName;
  }

  private updateVoiceprintSummary(names?: string[]): void {
    if (names) {
      this.voiceprintSummary = names.length > 0 ? `${names.length} 位已可识别` : '未注册';
      return;
    }
    let searchable: number = 0;
    let needsReRegister: number = 0;
    for (let i: number = 0; i < this.voiceprintProfiles.length; i++) {
      if (this.voiceprintProfiles[i].state === 'needsReRegister') {
        needsReRegister += 1;
      } else {
        searchable += 1;
      }
    }
    if (searchable === 0 && needsReRegister === 0) {
      this.voiceprintSummary = '未注册';
    } else if (needsReRegister > 0) {
      this.voiceprintSummary = `${searchable} 位可识别，${needsReRegister} 位待重录`;
    } else {
      this.voiceprintSummary = `${searchable} 位已可识别`;
    }
  }

  private restoreVoiceprintsToWorker(): void {
    if (!this.profileStoreReady) {
      return;
    }
    this.voiceprintProfiles = this.profileStore.list();
    this.updateVoiceprintSummary();
    this.postToWorker({
      msgType: 'sherpa-voiceprint-restore',
      voiceprints: this.profileStore.searchProfiles(),
    });
  }

  private async handleVoiceprintRegistered(response: SherpaWorkerResponse): Promise<void> {
    const speakerName = this.cleanSpeakerName(response.speakerName || this.voiceprintName);
    const embedding: number[] | undefined = response.embedding;
    const quality: EnrollmentQuality | undefined = response.quality;
    if (!embedding || embedding.length === 0 || !quality) {
      throw new Error('empty speaker embedding');
    }
    const prototype: VoicePrototype = createVoicePrototype(
      embedding, quality.speechMs, quality.qualityScore, 'activeEnrollment');
    const proposed: VoicePrototype[] = this.enrollmentPrototypes.slice();
    proposed.push(prototype);
    if (proposed.length > 1 &&
      prototypeSimilarityFloor(proposed) < SPEAKER_IDENTITY_THRESHOLD) {
      this.asrState = 'ready';
      this.statusMessage = '这次声音与前一轮差异较大，请重新录制本轮';
      return;
    }
    this.enrollmentPrototypes = proposed;
    this.voiceprintRound = proposed.length;
    if (proposed.length < SPEAKER_ACTIVE_ENROLL_ROUNDS) {
      this.asrState = 'ready';
      this.statusMessage = `第 ${proposed.length} 次录音合格，请继续录制第 ${proposed.length + 1} 次`;
      return;
    }
    try {
      await this.profileStore.saveActiveEnrollment(speakerName, proposed);
    } catch (error) {
      // Keep the first two accepted rounds so a storage/conflict failure does not leave
      // the UI at 3/3 with no profile actually saved.
      this.enrollmentPrototypes = proposed.slice(0, SPEAKER_ACTIVE_ENROLL_ROUNDS - 1);
      this.voiceprintRound = this.enrollmentPrototypes.length;
      throw error;
    }
    this.enrollmentPrototypes = [];
    this.voiceprintRound = 0;
    this.voiceprintProfiles = this.profileStore.list();
    this.restoreVoiceprintsToWorker();
    this.asrState = 'ready';
    this.statusMessage = `已完成 ${speakerName} 的三次声纹注册`;
  }

  private resetEnrollment(): void {
    this.enrollmentPrototypes = [];
    this.voiceprintRound = 0;
    this.statusMessage = '已重置本次注册，请从第 1 次开始';
  }

  private captureSpeakerResult(response: SherpaWorkerResponse, title: string): void {
    const turns: SpeakerTurn[] = response.speakerTurns || [];
    if (turns.length === 0) {
      return;
    }
    this.latestMeetingId = response.meetingId || turns[0].meetingId;
    this.latestMeetingTitle = title;
    this.latestSpeakerTurns = turns;
    this.latestSpeakerCandidates = response.speakerCandidates || [];
    this.latestSpeakerPreview = this.formatSpeakerTurns(turns);
    this.hasEditableSpeakers = true;
  }

  /** Re-applies persisted user labels after returning from the speaker editor. */
  private refreshLatestSpeakerAssignments(): void {
    const store: MeetingSpeakerStore | undefined = this.meetingSpeakerStore;
    if (!store || !this.profileStoreReady || this.latestMeetingId.length === 0 ||
      this.latestSpeakerTurns.length === 0) {
      return;
    }
    store.assignmentsFor(this.latestMeetingId)
      .then((assignments: MeetingSpeakerAssignment[]): void => {
        this.voiceprintProfiles = this.profileStore.list();
        this.latestSpeakerTurns = store.apply(
          this.latestSpeakerTurns, assignments, this.voiceprintProfiles);
        this.latestSpeakerPreview = this.formatSpeakerTurns(this.latestSpeakerTurns);
      })
      .catch((error: Error): void => {
        hilog.warn(DOMAIN, TAG, `refresh speaker assignments failed: ${error.message}`);
      });
  }

  private formatSpeakerTurns(turns: SpeakerTurn[]): string {
    const lines: string[] = [];
    for (let i: number = 0; i < turns.length; i++) {
      const turn: SpeakerTurn = turns[i];
      const name: string = turn.displayName || `未知说话人 ${turn.clusterId}`;
      const start: string = (turn.startMs / 1000).toFixed(1);
      const end: string = (turn.endMs / 1000).toFixed(1);
      lines.push(`[${name} ${start}s–${end}s] ${turn.text}`);
    }
    return lines.join('\n');
  }

  private formatSelfTestOptions(options: SelfTestOptions): string {
    const speaker = options.enableSpeakerDiarization ? '说话人分离:开' : '说话人分离:关';
    const fileModel = options.fileAsrMode === FILE_ASR_MODE_OFFLINE_PARAFORMER ?
      '文件模型:离线模型' : '文件模型:实时模型';
    const rescore = options.enableSegmentRescore ? '分段校正:开' : '分段校正:关';
    const compare = options.compareEnhanced ? '对比:开' : '对比:关';
    return `${speaker}，${fileModel}，${rescore}，${compare}`;
  }

  private onWorkerMessage(e: MessageEvents): void {
    const response = e.data as SherpaWorkerResponse;
    const msgType = response.msgType;
    hilog.info(DOMAIN, TAG, `Worker message: ${msgType}`);

    if (msgType === 'sherpa-asr-ready') {
      this.asrState = this.isSelfTestRunning ? 'processing' : 'ready';
      this.punctuationReady = response.punctuationReady === true;
      this.statusMessage = this.isSelfTestRunning ? '测试引擎已重新加载...' : this.getReadyStatusMessage();
      this.restoreVoiceprintsToWorker();
      if (this.resumeSelfTestAfterWorkerReady) {
        this.resumeSelfTestAfterWorkerReady = false;
        this.runNextSelfTest();
      }
      return;
    }

    if (msgType === 'sherpa-asr-started') {
      this.statusMessage = '录音中...';
      return;
    }

    if (msgType === 'sherpa-asr-partial') {
      this.partialText = response.text as string;
      return;
    }

    if (msgType === 'sherpa-asr-final') {
      const text = response.text as string;
      if (text.trim().length > 0) {
        this.recognizedText += text.trim() + '\n';
      }
      this.captureSpeakerResult(response, '本次录音');
      this.partialText = '';
      return;
    }

    if (msgType === 'sherpa-asr-stopped') {
      this.partialText = '';
      this.asrState = 'ready';
      this.statusMessage = this.getReadyStatusMessage();
      return;
    }

    if (msgType === 'sherpa-asr-telemetry') {
      const t = response.telemetry;
      if (t) {
        const parts: string[] = [];
        if (t.chunkDecodeMs !== undefined) {
          parts.push(`chunk=${t.chunkDecodeMs}ms`);
        }
        if (t.partialIntervalMs !== undefined) {
          parts.push(`partialInterval=${t.partialIntervalMs}ms`);
        }
        if (t.finalSegmentMs !== undefined) {
          parts.push(`finalSegment=${t.finalSegmentMs}ms`);
        }
        if (t.finalRedecodeMs !== undefined) {
          parts.push(`finalRedecode=${t.finalRedecodeMs}ms`);
        }
        if (t.finalRedecodeRtf !== undefined) {
          parts.push(`rtf=${t.finalRedecodeRtf.toFixed(3)}`);
        }
        if (t.audioDurationSec !== undefined) {
          parts.push(`audio=${t.audioDurationSec.toFixed(2)}s`);
        }
        hilog.info(DOMAIN, TAG, `ASR telemetry ${parts.join(' ')}`);
      }
      return;
    }

    if (msgType === 'sherpa-voiceprint-restored') {
      this.updateVoiceprintSummary(response.speakerNames);
      return;
    }

    if (msgType === 'sherpa-voiceprint-started') {
      this.statusMessage = '声纹注册中...';
      return;
    }

    if (msgType === 'sherpa-voiceprint-registered') {
      this.handleVoiceprintRegistered(response)
        .catch((error: Error) => {
          this.asrState = 'ready';
          this.statusMessage = `声纹保存失败: ${error}`;
          hilog.error(DOMAIN, TAG, `Voiceprint save failed: ${error}`);
        });
      return;
    }

    if (msgType === 'sherpa-voiceprint-rejected') {
      this.asrState = 'ready';
      this.isVoiceprintPressed = false;
      this.audioLevelPercent = 0;
      this.statusMessage = response.quality && response.quality.reason.length > 0
        ? response.quality.reason
        : '本次录音质量不足，请重录';
      return;
    }

    if (msgType === 'sherpa-voiceprint-error') {
      const message = response.message as string;
      this.asrState = 'ready';
      this.isVoiceprintPressed = false;
      this.audioLevelPercent = 0;
      this.statusMessage = `声纹注册失败: ${message}`;
      hilog.error(DOMAIN, TAG, `Voiceprint error: ${message}`);
      return;
    }

    if (msgType === 'sherpa-asr-self-test-done') {
      const text = response.text as string;
      const testAudio = this.selfTestAudios[this.selfTestIndex];
      const label = testAudio ? testAudio.label : '未知音频';
      this.recognizedText += `[测试:${label}] ${text.trim().length > 0 ? text.trim() : '(空)'}\n`;
      this.partialText = '';
      this.selfTestIndex += 1;
      this.runNextSelfTest();
      return;
    }

    if (msgType === 'sherpa-asr-audio-file-done') {
      const text = response.text as string;
      const label = this.activeImportedAudioLabel.length > 0 ? this.activeImportedAudioLabel : '音频文件';
      this.captureSpeakerResult(response, label);
      const finalLine = `[导入:${label} 最终] ${text.trim().length > 0 ? text.trim() : '(空)'}\n`;
      if (this.activeImportedAudioBaseText.length > 0) {
        this.recognizedText = this.activeImportedAudioBaseText + finalLine;
      } else {
        this.recognizedText += finalLine;
      }
      this.partialText = '';
      this.asrState = 'ready';
      this.statusMessage = this.getReadyStatusMessage();
      this.activeImportedAudioLabel = '';
      this.activeImportedAudioBaseText = '';
      this.activeImportedAudioProgressText = '';
      return;
    }

    if (msgType === 'sherpa-asr-audio-file-decoding') {
      const label = this.activeImportedAudioLabel.length > 0 ? this.activeImportedAudioLabel : '音频文件';
      this.setImportedAudioProgress(`[导入:${label}] 正在解码音频...`);
      this.partialText = '';
      this.asrState = 'processing';
      this.statusMessage = `正在解码音频 ${label}`;
      return;
    }

    if (msgType === 'sherpa-asr-audio-file-prepared') {
      const label = this.activeImportedAudioLabel.length > 0 ? this.activeImportedAudioLabel : '音频文件';
      const segmentTotal = response.segmentTotal || 0;
      const durationText = this.formatAudioSeconds(response.audioDuration);
      const segmentText = segmentTotal > 0 ? `按约 10 秒切成 ${segmentTotal} 段` : '按约 10 秒切段';
      this.setImportedAudioProgress(`[导入:${label}] 解码完成，${segmentText}，音频时长 ${durationText}`);
      this.partialText = '';
      this.asrState = 'processing';
      this.statusMessage = segmentTotal > 0 ? `已切成 ${segmentTotal} 段，正在识别 1/${segmentTotal}` : '正在识别导入音频';
      return;
    }

    if (msgType === 'sherpa-asr-audio-file-segment') {
      const text = response.text as string;
      const label = this.activeImportedAudioLabel.length > 0 ? this.activeImportedAudioLabel : '音频文件';
      const segmentIndex = response.segmentIndex || 0;
      const segmentTotal = response.segmentTotal || 0;
      const startText = this.formatAudioSeconds(response.segmentStart);
      const endText = this.formatAudioSeconds(response.segmentEnd);
      const progressLabel = segmentTotal > 0 ? `${segmentIndex}/${segmentTotal}` : `${segmentIndex}`;
      this.setImportedAudioProgress(
        `[导入:${label} ${progressLabel} ${startText}-${endText}] ${text.trim().length > 0 ? text.trim() : '(空)'}`
      );
      this.partialText = '';
      this.asrState = 'processing';
      this.statusMessage = `正在识别 ${label} ${progressLabel}`;
      return;
    }

    if (msgType === 'sherpa-asr-audio-file-error') {
      const message = this.normalizeAudioProblemMessage(response.message as string);
      const label = this.activeImportedAudioLabel.length > 0 ? this.activeImportedAudioLabel : '音频文件';
      this.recognizedText += `[导入:${label}] ${message}\n`;
      this.partialText = '';
      this.asrState = 'ready';
      this.statusMessage = message;
      this.activeImportedAudioLabel = '';
      this.activeImportedAudioBaseText = '';
      this.activeImportedAudioProgressText = '';
      promptAction.showToast({ message });
      hilog.error(DOMAIN, TAG, `Imported audio error: ${message}`);
      return;
    }

    if (msgType === 'sherpa-asr-error') {
      const message = response.message as string;
      this.asrState = 'ready';
      this.isButtonPressed = false;
      this.isVoiceprintPressed = false;
      this.audioLevelPercent = 0;
      this.statusMessage = `识别引擎异常: ${message}`;
      hilog.error(DOMAIN, TAG, `Sherpa ASR error: ${message}`);
    }
  }

  onAudioChunk(chunk: AudioChunk): void {
    if (this.asrState === 'recording') {
      this.updateAudioFeedback(chunk, this.asrState);
      this.postPcmToWorker('sherpa-asr-pcm', chunk.pcmData);
      return;
    }

    if (this.asrState === 'voiceprintRecording') {
      this.updateAudioFeedback(chunk, this.asrState);
      this.postPcmToWorker('sherpa-voiceprint-pcm', chunk.pcmData);
    }
  }

  async startRecording(): Promise<void> {
    if (this.asrState !== 'ready') {
      return;
    }

    const token = this.recordingStartToken + 1;
    this.recordingStartToken = token;
    this.asrState = 'starting';
    this.statusMessage = '准备录音...';
    this.partialText = '';
    this.audioLevelPercent = 0;
    this.recordingDurationText = '00:00';

    const hasPermission = await this.ensureMicrophonePermission();
    if (!hasPermission) {
      this.asrState = 'ready';
      return;
    }
    if (token !== this.recordingStartToken || !this.isButtonPressed) {
      this.asrState = 'ready';
      this.statusMessage = this.getReadyStatusMessage();
      return;
    }

    try {
      this.postToWorker({ msgType: 'sherpa-asr-start' });
      await this.audioService.start();
      if (token !== this.recordingStartToken || !this.isButtonPressed) {
        await this.audioService.stop();
        this.postToWorker({ msgType: 'sherpa-asr-cancel' });
        this.asrState = 'ready';
        this.statusMessage = this.getReadyStatusMessage();
        return;
      }
      this.recordingStartTime = Date.now();
      this.lastAudioStatusUpdateAt = 0;
      this.asrState = 'recording';
      this.statusMessage = '录音中 00:00 · 音量 0%';
    } catch (error) {
      this.postToWorker({ msgType: 'sherpa-asr-cancel' });
      this.isButtonPressed = false;
      this.asrState = 'ready';
      this.statusMessage = `录音启动失败: ${error}`;
      hilog.error(DOMAIN, TAG, `Start recording failed: ${error}`);
    }
  }

  async stopRecording(): Promise<void> {
    this.recordingStartToken += 1;
    this.isButtonPressed = false;
    if (this.asrState === 'starting') {
      this.postToWorker({ msgType: 'sherpa-asr-cancel' });
      this.asrState = 'ready';
      this.statusMessage = this.getReadyStatusMessage();
      return;
    }
    if (this.asrState !== 'recording') {
      return;
    }

    await this.audioService.stop();

    this.asrState = 'processing';
    this.statusMessage = '识别收尾中...';
    this.postToWorker({
      msgType: 'sherpa-asr-stop',
      meetingId: util.generateRandomUUID(true),
      enableSpeakerDiarization: this.enableSpeakerDiarization,
      enableSegmentRescore: this.enableSegmentRescore,
      compareEnhanced: this.compareEnhanced
    });
  }

  async startVoiceprintRegistration(): Promise<void> {
    if (this.asrState !== 'ready' || !this.profileStoreReady) {
      if (!this.profileStoreReady) {
        this.statusMessage = '本机加密声纹库尚未就绪，请稍后重试';
      }
      return;
    }

    const token = this.voiceprintStartToken + 1;
    this.voiceprintStartToken = token;
    this.asrState = 'voiceprintStarting';
    this.statusMessage = '准备声纹注册...';
    this.partialText = '';
    this.audioLevelPercent = 0;
    this.recordingDurationText = '00:00';

    const hasPermission = await this.ensureMicrophonePermission();
    if (!hasPermission) {
      this.asrState = 'ready';
      return;
    }
    if (token !== this.voiceprintStartToken || !this.isVoiceprintPressed) {
      this.asrState = 'ready';
      this.statusMessage = this.getReadyStatusMessage();
      return;
    }

    try {
      this.postToWorker({ msgType: 'sherpa-voiceprint-start' });
      await this.audioService.start();
      if (token !== this.voiceprintStartToken || !this.isVoiceprintPressed) {
        await this.audioService.stop();
        this.postToWorker({ msgType: 'sherpa-voiceprint-cancel' });
        this.asrState = 'ready';
        this.statusMessage = this.getReadyStatusMessage();
        return;
      }
      this.voiceprintStartTime = Date.now();
      this.lastAudioStatusUpdateAt = 0;
      this.asrState = 'voiceprintRecording';
      this.statusMessage = '声纹注册中 00:00 · 音量 0%';
    } catch (error) {
      this.postToWorker({ msgType: 'sherpa-voiceprint-cancel' });
      this.isVoiceprintPressed = false;
      this.asrState = 'ready';
      this.statusMessage = `声纹注册启动失败: ${error}`;
      hilog.error(DOMAIN, TAG, `Start voiceprint registration failed: ${error}`);
    }
  }

  async stopVoiceprintRegistration(): Promise<void> {
    this.voiceprintStartToken += 1;
    this.isVoiceprintPressed = false;
    if (this.asrState === 'voiceprintStarting') {
      this.postToWorker({ msgType: 'sherpa-voiceprint-cancel' });
      this.asrState = 'ready';
      this.statusMessage = this.getReadyStatusMessage();
      return;
    }
    if (this.asrState !== 'voiceprintRecording') {
      return;
    }

    const elapsedMs = this.voiceprintStartTime > 0 ? Date.now() - this.voiceprintStartTime : 0;
    await this.audioService.stop();
    if (elapsedMs < 4000) {
      this.postToWorker({ msgType: 'sherpa-voiceprint-cancel' });
      this.asrState = 'ready';
      this.audioLevelPercent = 0;
      this.statusMessage = `本轮太短，请至少再说 ${Math.ceil((4000 - elapsedMs) / 1000)} 秒`;
      return;
    }

    this.asrState = 'processing';
    this.statusMessage = '声纹注册收尾中...';
    this.postToWorker({
      msgType: 'sherpa-voiceprint-stop',
      speakerName: this.cleanSpeakerName(this.voiceprintName)
    });
  }

  clearText(): void {
    this.recognizedText = '';
    this.partialText = '';
  }

  async importAudioForTest(): Promise<void> {
    if (this.asrState !== 'ready') {
      return;
    }

    try {
      const context = this.getAbilityContext();
      const documentPicker = new picker.DocumentViewPicker(context);
      const options = new picker.DocumentSelectOptions();
      options.maxSelectNumber = 1;
      options.fileSuffixFilters = ['音频文件|.mp3,.wav,.m4a'];

      const uris = await documentPicker.select(options);
      if (!uris || uris.length === 0) {
        return;
      }

      const uri = uris[0];
      const fileName = this.extractFileName(uri);
      const data = await this.readUriBytes(uri);
      const audioFormat = this.validateImportedAudio(fileName, data);
      const fileOptions = this.getCurrentFileOptions();

      this.activeImportedAudioLabel = fileName;
      this.asrState = 'processing';
      this.statusMessage = `正在解码音频 ${fileName} (${this.formatFileSize(data.length)})`;
      this.partialText = '';
      this.recognizedText += `[导入] ${fileName}（${this.formatFileSize(data.length)}，${this.formatSelfTestOptions(fileOptions)}）\n`;
      this.activeImportedAudioBaseText = this.recognizedText;
      this.activeImportedAudioProgressText = '';
      this.setImportedAudioProgress(`[导入:${fileName}] 正在解码音频...`);
      this.postAudioFileToWorker({
        msgType: 'sherpa-asr-decode-audio-file',
        meetingId: util.generateRandomUUID(true),
        audioFormat,
        fileName,
        fileAsrMode: fileOptions.fileAsrMode,
        enableSpeakerDiarization: fileOptions.enableSpeakerDiarization,
        enableSegmentRescore: fileOptions.enableSegmentRescore,
        compareEnhanced: fileOptions.compareEnhanced
      }, data);
    } catch (error) {
      const message = this.normalizeAudioProblemMessage(`${error}`);
      this.asrState = 'ready';
      this.partialText = '';
      this.statusMessage = message;
      this.activeImportedAudioLabel = '';
      this.activeImportedAudioBaseText = '';
      this.activeImportedAudioProgressText = '';
      promptAction.showToast({ message });
      hilog.error(DOMAIN, TAG, `Import audio failed: ${error}`);
    }
  }

  private async runSelfTestWithLimit(limit: number): Promise<void> {
    if (this.asrState !== 'ready') {
      return;
    }

    this.isSelfTestRunning = true;
    this.resumeSelfTestAfterWorkerReady = false;
    this.asrState = 'processing';
    this.statusMessage = '测试中...';
    this.partialText = '';
    this.selfTestIndex = 0;
    this.activeSelfTestOptions = {
      enableSpeakerDiarization: this.enableSpeakerDiarization,
      enableSegmentRescore: this.enableSegmentRescore,
      compareEnhanced: this.compareEnhanced,
      fileAsrMode: this.useOfflineParaformerForFiles ? FILE_ASR_MODE_OFFLINE_PARAFORMER : FILE_ASR_MODE_STREAMING
    };
    this.activeSelfTestCount = Math.min(Math.max(limit, 1), this.selfTestAudios.length);
    this.recognizedText += `[测试] 开始 ${this.activeSelfTestCount} 条音频（${this.formatSelfTestOptions(this.activeSelfTestOptions)}）\n`;
    this.runNextSelfTest();
  }

  async runSelfTest(): Promise<void> {
    await this.runSelfTestWithLimit(this.selfTestAudios.length);
  }

  async runSingleSelfTest(): Promise<void> {
    await this.runSelfTestWithLimit(1);
  }

  private async runNextSelfTest(): Promise<void> {
    if (this.selfTestIndex >= this.activeSelfTestCount) {
      this.isSelfTestRunning = false;
      this.resumeSelfTestAfterWorkerReady = false;
      this.asrState = 'ready';
      this.statusMessage = this.getReadyStatusMessage();
      this.recognizedText += `[测试] 完成 ${this.activeSelfTestCount} 条音频\n`;
      return;
    }

    const testAudio = this.selfTestAudios[this.selfTestIndex];
    this.statusMessage = `测试中 ${this.selfTestIndex + 1}/${this.activeSelfTestCount}: ${testAudio.label}`;
    try {
      const context = this.getAbilityContext();
      const audioContent = await context.resourceManager.getRawFileContent(testAudio.path);
      this.postToWorker({
        msgType: 'sherpa-asr-decode-wav',
        meetingId: util.generateRandomUUID(true),
        data: audioContent,
        audioFormat: this.getAudioFormatFromName(testAudio.path),
        fileName: testAudio.label,
        fileAsrMode: this.activeSelfTestOptions.fileAsrMode,
        enableSpeakerDiarization: this.activeSelfTestOptions.enableSpeakerDiarization,
        enableSegmentRescore: this.activeSelfTestOptions.enableSegmentRescore,
        compareEnhanced: this.activeSelfTestOptions.compareEnhanced
      });
    } catch (error) {
      this.recognizedText += `[测试:${testAudio.label}] 读取失败: ${error}\n`;
      hilog.error(DOMAIN, TAG, `Self test failed: ${error}`);
      this.selfTestIndex += 1;
      this.runNextSelfTest();
    }
  }

  build() {
    Column() {
      Row() {
        Circle()
          .width(12)
          .height(12)
          .fill(this.asrState === 'recording' ? '#FF4444' :
                this.asrState === 'ready' ? '#44CC44' :
                this.asrState === 'error' ? '#FF8800' : '#888888')
          .margin({ right: 8 })

        Text(this.statusMessage)
          .fontSize(14)
          .fontColor('#666666')
          .layoutWeight(1)
          .maxLines(2)
          .textOverflow({ overflow: TextOverflow.Ellipsis })

        // Sits in the status row rather than beside the record button: this is a
        // different way of working, not another option for the current recording.
        Button('持续录音')
          .fontSize(12)
          .height(30)
          .backgroundColor('#EAF2FF')
          .fontColor('#0066CC')
          .margin({ left: 8 })
          .onClick((): void => {
            this.openContinuousRecording();
          })

        Button('人物与声纹')
          .fontSize(12)
          .height(30)
          .backgroundColor('#F2EEFF')
          .fontColor('#5B3FB5')
          .margin({ left: 8 })
          .onClick((): void => {
            this.openSpeakerProfiles();
          })
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 8 })

      Column() {
        if (this.recognizedText.length > 0) {
          Scroll() {
            Text(this.recognizedText)
              .fontSize(18)
              .fontColor('#333333')
              .width('100%')
              .wordBreak(WordBreak.BREAK_ALL)
          }
          .layoutWeight(1)
        }

        if (this.partialText.length > 0) {
          Text(this.partialText)
            .fontSize(18)
            .fontColor('#0066CC')
            .width('100%')
            .padding({ top: 8 })
            .backgroundColor('#F0F7FF')
            .borderRadius(8)
            .padding(12)
        }

        if (this.recognizedText.length === 0 && this.partialText.length === 0) {
          Text('识别结果将显示在这里')
            .fontSize(16)
            .fontColor('#AAAAAA')
            .margin({ top: 40 })
        }
      }
      .layoutWeight(1)
      .width('100%')
      .padding({ left: 16, right: 16, top: 8, bottom: 8 })
      .backgroundColor('#FAFAFA')
      .borderRadius(12)
      .margin({ left: 16, right: 16 })

      if (this.hasEditableSpeakers) {
        Column({ space: 6 }) {
          Row() {
            Text('已生成可编辑的说话人分段')
              .fontSize(13)
              .fontColor('#4B5563')
              .layoutWeight(1)
            Button('编辑说话人')
              .fontSize(13)
              .height(34)
              .backgroundColor('#EAF2FF')
              .fontColor('#0066CC')
              .onClick((): void => {
                this.openMeetingSpeakerEditor();
              })
          }
          .width('100%')

          if (this.latestSpeakerPreview.length > 0) {
            Text(this.latestSpeakerPreview)
              .fontSize(12)
              .fontColor('#374151')
              .maxLines(4)
              .textOverflow({ overflow: TextOverflow.Ellipsis })
              .width('100%')
          }
        }
        .width('100%')
        .padding({ left: 20, right: 20, top: 8, bottom: 2 })
      }

      if (SHOW_ADVANCED_CONTROLS || SHOW_DIAGNOSTICS) {
        Column() {
          Row() {
            Text('高级设置')
              .fontSize(13)
              .fontColor('#666666')
              .fontWeight(FontWeight.Medium)
            Blank()
            if (SHOW_DIAGNOSTICS) {
              Text(`音频测试 ${this.selfTestAudios.length} 条`)
                .fontSize(12)
                .fontColor('#999999')
                .margin({ right: 10 })

              Button('导入音频')
                .fontSize(12)
                .height(30)
                .backgroundColor('#EAF2FF')
                .fontColor('#0066CC')
                .enabled(this.asrState === 'ready')
                .onClick(() => {
                  this.importAudioForTest();
                })
                .margin({ right: 8 })

              Button('单条测试')
                .fontSize(12)
                .height(30)
                .backgroundColor('#EAF2FF')
                .fontColor('#0066CC')
                .enabled(this.asrState === 'ready')
                .onClick(() => {
                  this.runSingleSelfTest();
                })
            }
          }
          .width('100%')
          .margin({ bottom: 6 })

          if (SHOW_TUNING_CONTROLS) {
            Row() {
              Text('说话人分离')
                .fontSize(14)
                .fontColor('#333333')
              Blank()
              Toggle({ type: ToggleType.Switch, isOn: this.enableSpeakerDiarization })
                .selectedColor('#0066CC')
                .enabled(this.asrState === 'ready')
                .onChange((isOn: boolean) => {
                  this.enableSpeakerDiarization = isOn;
                })
            }
            .width('100%')
            .height(32)

            Row() {
              Text('分段校正')
                .fontSize(14)
                .fontColor('#333333')
              Blank()
              Toggle({ type: ToggleType.Switch, isOn: this.enableSegmentRescore })
                .selectedColor('#0066CC')
                .enabled(this.asrState === 'ready')
                .onChange((isOn: boolean) => {
                  this.enableSegmentRescore = isOn;
                })
            }
            .width('100%')
            .height(32)

            Row() {
              Text('原始/增强对比')
                .fontSize(14)
                .fontColor('#333333')
              Blank()
              Toggle({ type: ToggleType.Switch, isOn: this.compareEnhanced })
                .selectedColor('#0066CC')
                .enabled(this.asrState === 'ready')
                .onChange((isOn: boolean) => {
                  this.compareEnhanced = isOn;
                })
            }
            .width('100%')
            .height(32)

            Row() {
              Text('音频文件模型')
                .fontSize(14)
                .fontColor('#333333')
              Blank()
              Toggle({ type: ToggleType.Switch, isOn: this.useOfflineParaformerForFiles })
                .selectedColor('#0066CC')
                .enabled(this.asrState === 'ready')
                .onChange((isOn: boolean) => {
                  this.useOfflineParaformerForFiles = isOn;
                })
            }
            .width('100%')
            .height(32)

            Row() {
              Text('声纹名')
                .fontSize(14)
                .fontColor('#333333')
              TextInput({ text: this.voiceprintName, placeholder: '输入姓名' })
                .height(36)
                .fontSize(14)
                .margin({ left: 12 })
                .layoutWeight(1)
                .enabled(this.asrState === 'ready')
                .onChange((value: string) => {
                  this.voiceprintName = value;
                })
            }
            .width('100%')
            .height(40)
          }

          if (SHOW_VOICEPRINT_CONTROLS) {
            Column() {
              Row() {
                Text('我的声纹')
                  .fontSize(14)
                  .fontWeight(FontWeight.Medium)
                  .fontColor('#333333')
                Blank()
                Text(this.voiceprintSummary)
                  .fontSize(12)
                  .fontColor('#666666')
                  .maxLines(1)
                  .textOverflow({ overflow: TextOverflow.Ellipsis })
              }
              .width('100%')

              TextInput({ text: this.voiceprintName, placeholder: '本人显示名称' })
                .height(38)
                .fontSize(14)
                .backgroundColor('#FFFFFF')
                .enabled(this.asrState === 'ready' && this.profileStoreReady)
                .onChange((value: string): void => {
                  this.voiceprintName = value;
                })
                .margin({ top: 6 })

              Row() {
                Text(`注册进度 ${this.voiceprintRound}/${SPEAKER_ACTIVE_ENROLL_ROUNDS}`)
                  .fontSize(12)
                  .fontColor('#666666')
                  .layoutWeight(1)

                Button(this.isVoiceprintPressed ? '松开完成本轮' :
                  `录制第 ${Math.min(this.voiceprintRound + 1, SPEAKER_ACTIVE_ENROLL_ROUNDS)} 次`)
                  .fontSize(12)
                  .height(34)
                  .backgroundColor(this.isVoiceprintPressed ? '#CC6600' : '#EAF2FF')
                  .fontColor(this.isVoiceprintPressed ? '#FFFFFF' : '#0066CC')
                  .enabled(this.profileStoreReady && (this.asrState === 'ready' ||
                    this.asrState === 'voiceprintStarting' || this.asrState === 'voiceprintRecording'))
                  .onTouch((event: TouchEvent): void => {
                    if (event.type === TouchType.Down) {
                      this.isVoiceprintPressed = true;
                      this.startVoiceprintRegistration();
                    } else if (event.type === TouchType.Up || event.type === TouchType.Cancel) {
                      this.isVoiceprintPressed = false;
                      if (this.asrState === 'voiceprintStarting' || this.asrState === 'voiceprintRecording') {
                        this.stopVoiceprintRegistration();
                      }
                    }
                  })
                  .margin({ left: 8 })

                if (this.voiceprintRound > 0) {
                  Button('重来')
                    .fontSize(12)
                    .height(34)
                    .backgroundColor('#F5F5F5')
                    .fontColor('#666666')
                    .enabled(this.asrState === 'ready')
                    .onClick((): void => {
                      this.resetEnrollment();
                    })
                    .margin({ left: 8 })
                }
              }
              .width('100%')
              .margin({ top: 6 })

              Text('每次自然说话 4–6 秒，共 3 次；原始注册音频不会保存')
                .fontSize(12)
                .fontColor('#888888')
                .margin({ top: 4 })
            }
            .width('100%')
          }
        }
        .width('100%')
        .padding({ left: 16, right: 16, top: 8, bottom: 8 })
        .margin({ left: 16, right: 16, top: 8 })
        .backgroundColor('#F7FAFC')
        .borderRadius(8)
      }

      Row() {
        Button('清除')
          .fontSize(14)
          .height(40)
          .backgroundColor('#F5F5F5')
          .fontColor('#666666')
          .onClick(() => {
            this.clearText();
          })
          .enabled(this.recognizedText.length > 0)

        Blank()

        Button(this.isButtonPressed ? '松开结束' : '按住说话')
          .fontSize(16)
          .fontWeight(FontWeight.Medium)
          .height(56)
          .width(160)
          .backgroundColor(this.isButtonPressed ? '#CC0000' : '#0066CC')
          .fontColor('#FFFFFF')
          .borderRadius(28)
          .enabled(this.asrState === 'ready' || this.asrState === 'starting' || this.asrState === 'recording')
          .onTouch((event: TouchEvent) => {
            if (event.type === TouchType.Down) {
              this.isButtonPressed = true;
              this.startRecording();
            } else if (event.type === TouchType.Up || event.type === TouchType.Cancel) {
              this.isButtonPressed = false;
              if (this.asrState === 'starting' || this.asrState === 'recording') {
                this.stopRecording();
              }
            }
          })

        Blank()

        if (SHOW_DIAGNOSTICS) {
          Button('固定测试')
            .fontSize(14)
            .height(40)
            .backgroundColor('#F5F5F5')
            .fontColor('#666666')
            .onClick(() => {
              this.openSpeakerTestCenter();
            })
            .enabled(this.asrState === 'ready')
        } else {
          Text('')
            .width(48)
            .height(40)
        }
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 24 })
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#FFFFFF')
  }
}
