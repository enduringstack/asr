import { abilityAccessCtrl, common, Permissions } from '@kit.AbilityKit';
import { preferences } from '@kit.ArkData';
import { fileIo, picker } from '@kit.CoreFileKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { promptAction } from '@kit.ArkUI';
import { ErrorEvent, MessageEvents, worker } from '@kit.ArkTS';
import { AudioRecordingService, AudioChunk } from '../services/audio/AudioRecordingService';

const DOMAIN = 0x0001;
const TAG = 'ASR';
const MIC_PERMISSION: Permissions = 'ohos.permission.MICROPHONE';
const VOICEPRINT_PREFERENCES = 'asr_voiceprints';
const VOICEPRINT_PROFILES_KEY = 'profiles';
const VOICEPRINT_MODEL_ID = 'sherpa-onnx-3dspeaker-sv-zh-cn';
const VOICEPRINT_EMBEDDING_DIM = 512;
const SHOW_ADVANCED_CONTROLS = true;
const SHOW_TUNING_CONTROLS = false;
const SHOW_VOICEPRINT_CONTROLS = false;
const SHOW_DIAGNOSTICS = true;
const FILE_ASR_MODE_STREAMING = 'streaming';
const FILE_ASR_MODE_OFFLINE_PARAFORMER = 'offline_paraformer_large';

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
  voiceprints?: VoiceprintProfile[];
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

interface VoiceprintProfile {
  name: string;
  embedding: number[];
  modelId?: string;
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
  @State enableSpeakerDiarization: boolean = false;
  @State enableSegmentRescore: boolean = false;
  @State compareEnhanced: boolean = false;
  @State useOfflineParaformerForFiles: boolean = true;
  @State isVoiceprintPressed: boolean = false;
  @State voiceprintName: string = '我';
  @State voiceprintSummary: string = '未注册';
  @State audioLevelPercent: number = 0;
  @State recordingDurationText: string = '00:00';

  private audioService: AudioRecordingService = AudioRecordingService.getInstance();
  private workerInstance?: worker.ThreadWorker;
  private voiceprintProfiles: VoiceprintProfile[] = [];
  private recordingStartToken: number = 0;
  private voiceprintStartToken: number = 0;
  private recordingStartTime: number = 0;
  private voiceprintStartTime: number = 0;
  private lastAudioStatusUpdateAt: number = 0;
  private activeImportedAudioBaseText: string = '';
  private activeImportedAudioProgressText: string = '';
  private readonly workerScriptUrl: string = 'entry/ets/workers/StreamingSherpaWorker.ets';
  private activeSelfTestOptions: SelfTestOptions = {
    enableSpeakerDiarization: false,
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
    this.initASR();
  }

  aboutToDisappear(): void {
    this.recordingStartToken += 1;
    this.voiceprintStartToken += 1;
    this.audioService.stop();
    this.audioService.clearCallbacks();
    this.shutdownWorker();
  }

  private getAbilityContext(): common.UIAbilityContext {
    return getContext(this) as common.UIAbilityContext;
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
    return cleanName.length > 16 ? cleanName.slice(0, 16) : cleanName;
  }

  private updateVoiceprintSummary(names?: string[]): void {
    const speakerNames = names ? names : this.voiceprintProfiles.map((profile: VoiceprintProfile) => profile.name);
    this.voiceprintSummary = speakerNames.length > 0 ? speakerNames.join('、') : '未注册';
  }

  private async loadVoiceprintProfiles(): Promise<VoiceprintProfile[]> {
    const context = this.getAbilityContext();
    const store = await preferences.getPreferences(context, VOICEPRINT_PREFERENCES);
    const raw = await store.get(VOICEPRINT_PROFILES_KEY, '[]') as string;
    const parsed = JSON.parse(raw) as VoiceprintProfile[];
    const profiles: VoiceprintProfile[] = [];
    if (!Array.isArray(parsed)) {
      return profiles;
    }
    for (let i = 0; i < parsed.length; ++i) {
      const profile = parsed[i];
      if (!profile || !profile.name || !Array.isArray(profile.embedding)) {
        continue;
      }
      if (profile.embedding.length !== VOICEPRINT_EMBEDDING_DIM) {
        continue;
      }
      if (profile.modelId && profile.modelId !== VOICEPRINT_MODEL_ID) {
        continue;
      }
      profiles.push({
        name: this.cleanSpeakerName(profile.name),
        embedding: profile.embedding,
        modelId: VOICEPRINT_MODEL_ID
      });
    }
    return profiles;
  }

  private async saveVoiceprintProfiles(): Promise<void> {
    const context = this.getAbilityContext();
    const store = await preferences.getPreferences(context, VOICEPRINT_PREFERENCES);
    await store.put(VOICEPRINT_PROFILES_KEY, JSON.stringify(this.voiceprintProfiles));
    await store.flush();
  }

  private restoreVoiceprintsToWorker(): void {
    this.loadVoiceprintProfiles()
      .then((profiles: VoiceprintProfile[]) => {
        this.voiceprintProfiles = profiles;
        this.updateVoiceprintSummary();
        this.postToWorker({
          msgType: 'sherpa-voiceprint-restore',
          voiceprints: this.voiceprintProfiles
        });
      })
      .catch((error: Error) => {
        hilog.error(DOMAIN, TAG, `Load voiceprints failed: ${error}`);
        this.voiceprintProfiles = [];
        this.updateVoiceprintSummary();
      });
  }

  private async handleVoiceprintRegistered(response: SherpaWorkerResponse): Promise<void> {
    const speakerName = this.cleanSpeakerName(response.speakerName || this.voiceprintName);
    const embedding = response.embedding;
    if (!embedding || embedding.length === 0) {
      throw new Error('empty speaker embedding');
    }

    const profiles: VoiceprintProfile[] = [];
    for (let i = 0; i < this.voiceprintProfiles.length; ++i) {
      if (this.voiceprintProfiles[i].name !== speakerName) {
        profiles.push(this.voiceprintProfiles[i]);
      }
    }
    profiles.push({ name: speakerName, embedding, modelId: VOICEPRINT_MODEL_ID });
    this.voiceprintProfiles = profiles;
    await this.saveVoiceprintProfiles();
    this.updateVoiceprintSummary(response.speakerNames);
    this.asrState = 'ready';
    this.statusMessage = `声纹已注册: ${speakerName}`;
  }

  private async clearVoiceprints(): Promise<void> {
    this.voiceprintProfiles = [];
    this.updateVoiceprintSummary();
    await this.saveVoiceprintProfiles();
    this.postToWorker({
      msgType: 'sherpa-voiceprint-restore',
      voiceprints: []
    });
    this.statusMessage = '声纹库已清空';
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
      if (SHOW_ADVANCED_CONTROLS) {
        this.restoreVoiceprintsToWorker();
      }
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
      enableSpeakerDiarization: this.enableSpeakerDiarization,
      enableSegmentRescore: this.enableSegmentRescore,
      compareEnhanced: this.compareEnhanced
    });
  }

  async startVoiceprintRegistration(): Promise<void> {
    if (this.asrState !== 'ready') {
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
    if (elapsedMs < 2000) {
      this.postToWorker({ msgType: 'sherpa-voiceprint-cancel' });
      this.asrState = 'ready';
      this.audioLevelPercent = 0;
      this.statusMessage = `声纹注册太短，请至少再说 ${Math.ceil((2000 - elapsedMs) / 1000)} 秒`;
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
            Row() {
              Text(`声纹库: ${this.voiceprintSummary}`)
                .fontSize(12)
                .fontColor('#666666')
                .layoutWeight(1)
                .maxLines(1)
                .textOverflow({ overflow: TextOverflow.Ellipsis })

              Button(this.isVoiceprintPressed ? '松开完成' : '按住注册')
                .fontSize(12)
                .height(32)
                .backgroundColor(this.isVoiceprintPressed ? '#CC6600' : '#EAF2FF')
                .fontColor(this.isVoiceprintPressed ? '#FFFFFF' : '#0066CC')
                .enabled(this.asrState === 'ready' || this.asrState === 'voiceprintStarting' ||
                  this.asrState === 'voiceprintRecording')
                .onTouch((event: TouchEvent) => {
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

              Button('清空')
                .fontSize(12)
                .height(32)
                .backgroundColor('#F5F5F5')
                .fontColor('#666666')
                .enabled(this.asrState === 'ready' && this.voiceprintSummary !== '未注册')
                .onClick(() => {
                  this.clearVoiceprints();
                })
                .margin({ left: 8 })
            }
            .width('100%')
            .height(40)
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
          Button('14条测试')
            .fontSize(14)
            .height(40)
            .backgroundColor('#F5F5F5')
            .fontColor('#666666')
            .onClick(() => {
              this.runSelfTest();
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
