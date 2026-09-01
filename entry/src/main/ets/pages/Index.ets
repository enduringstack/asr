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
import { RawPlaybackState, RawResourcePlayer } from '../services/audio/RawResourcePlayer';
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
import {
  buildDatasetSuites,
  buildSpeechModelDescriptors,
  DatasetAction,
  formatTelemetrySummary,
  SpeechModelDescriptor,
  SpeechModelId,
  WorkbenchDatasetSuite,
  WorkbenchMode,
  WorkbenchTelemetry,
} from '../common/VoiceWorkbenchTypes';
import {
  audioUnderstandingLabelText,
  audioUnderstandingLanguageText,
} from '../common/AudioUnderstandingTypes';
import {
  AcousticScenePrediction,
  buildAcousticScenePrediction,
} from '../common/AcousticSceneTypes';
import {
  AcousticSceneFusionResult,
  fuseAcousticScene,
} from '../common/AcousticSceneFusion';
import {
  SceneContextSummary,
  SceneContextTracker,
} from '../services/scene/SceneContextTracker';

const DOMAIN = 0x0001;
const TAG = 'ASR';
const MIC_PERMISSION: Permissions = 'ohos.permission.MICROPHONE';
const LOCATION_PERMISSION: Permissions = 'ohos.permission.APPROXIMATELY_LOCATION';
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

interface SherpaWorkerResponse {
  msgType: string;
  message?: string;
  text?: string;
  rawText?: string;
  punctuationReady?: boolean;
  sceneReady?: boolean;
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
  telemetry?: WorkbenchTelemetry;
  lang?: string;
  emotion?: string;
  event?: string;
  sceneClassIndex?: number;
  sceneConfidence?: number;
  sceneProbabilities?: number[];
  sceneWindowCount?: number;
  elapsedMs?: number;
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
  @State sceneReady: boolean = false;
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
  @State workbenchMode: WorkbenchMode = 'recording';
  @State selectedModel: SpeechModelId = 'asr';
  @State latestRawText: string = '';
  @State latestFinalText: string = '';
  @State latestTelemetryText: string = '等待首次运行';
  @State latestUnderstandingLanguage: string = '';
  @State latestUnderstandingEmotion: string = '';
  @State latestUnderstandingEvent: string = '';
  @State latestSceneAudioName: string = '';
  @State latestSceneAudioConfidence: number = 0;
  @State latestSceneName: string = '';
  @State latestSceneConfidence: number = 0;
  @State latestSceneWindowCount: number = 0;
  @State latestSceneElapsedMs: number = 0;
  @State enableSceneContextFusion: boolean = true;
  @State latestSceneInferenceMode: string = '纯音频';
  @State latestSceneEvidence: string = '';
  @State latestSceneSpeedText: string = '';
  @State datasetPlayingId: string = '';
  @State datasetPlaybackPositionMs: number = 0;
  @State datasetPlaybackDurationMs: number = 0;
  @State datasetPlaybackBusy: boolean = false;
  @State datasetPlaybackError: string = '';

  private audioService: AudioRecordingService = AudioRecordingService.getInstance();
  private datasetAudioPlayer: RawResourcePlayer = new RawResourcePlayer();
  private sceneContextTracker: SceneContextTracker = new SceneContextTracker();
  private workerInstance?: worker.ThreadWorker;
  private profileStore: SpeakerProfileStore = SpeakerProfileStore.getInstance();
  private meetingSpeakerStore: MeetingSpeakerStore | undefined = undefined;
  private speakerEditSessions: SpeakerEditSessionStore = SpeakerEditSessionStore.getInstance();
  private modelDescriptors: SpeechModelDescriptor[] = buildSpeechModelDescriptors();
  private datasetSuites: WorkbenchDatasetSuite[] = buildDatasetSuites();
  private waveBars: number[] = [12, 20, 30, 17, 38, 24, 14, 28, 42, 25, 16, 32, 20, 12];
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
    this.datasetAudioPlayer.setCallback((state: RawPlaybackState): void => {
      this.datasetPlayingId = state.audioId;
      this.datasetPlaybackPositionMs = state.positionMs;
      this.datasetPlaybackDurationMs = state.durationMs;
      if (state.errorMessage.length > 0) {
        this.datasetPlaybackError = `播放失败：${state.errorMessage}`;
      }
    });
    this.initializeSpeakerProfiles();
    this.initASR();
  }

  aboutToDisappear(): void {
    this.recordingStartToken += 1;
    this.voiceprintStartToken += 1;
    this.audioService.stop();
    this.audioService.clearCallbacks();
    this.sceneContextTracker.stop();
    this.datasetAudioPlayer.setCallback(null);
    this.datasetAudioPlayer.stop().catch((error: Error): void => {
      hilog.warn(DOMAIN, TAG, `stop dataset playback failed: ${error.message}`);
    });
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
    this.sceneContextTracker.stop();
    this.datasetAudioPlayer.stop().catch((error: Error): void => {
      hilog.warn(DOMAIN, TAG, `stop hidden dataset playback failed: ${error.message}`);
    });
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

  private openSpeakerIdentityTestCenter(): void {
    router.pushUrl({ url: 'pages/SpeakerIdentityTestCenter' })
      .catch((error: Error): void => {
        this.statusMessage = `打开声纹身份评测失败: ${error.message}`;
      });
  }

  private openAudioUnderstandingTestCenter(): void {
    router.pushUrl({ url: 'pages/AudioUnderstandingTestCenter' })
      .catch((error: Error): void => {
        this.statusMessage = `打开声音理解评测失败: ${error.message}`;
      });
  }

  private openAcousticSceneTestCenter(): void {
    router.pushUrl({ url: 'pages/AcousticSceneTestCenter' })
      .catch((error: Error): void => {
        this.statusMessage = `打开场景感知评测失败: ${error.message}`;
      });
  }

  private selectWorkbenchMode(mode: WorkbenchMode): void {
    if (this.asrState === 'recording' || this.asrState === 'starting' ||
      this.asrState === 'voiceprintRecording' || this.asrState === 'voiceprintStarting' ||
      this.asrState === 'processing') {
      return;
    }
    if (mode === 'recording' && this.datasetPlayingId.length > 0) {
      this.datasetAudioPlayer.stop().catch((error: Error): void => {
        hilog.warn(DOMAIN, TAG, `stop playback on mode switch failed: ${error.message}`);
      });
    }
    this.workbenchMode = mode;
  }

  private modelStatus(modelId: SpeechModelId): string {
    if (modelId === 'asr') {
      if (this.asrState === 'initializing') {
        return '加载中';
      }
      if (this.asrState === 'error') {
        return '需重试';
      }
      return this.asrState === 'processing' ? '处理中' : '已就绪';
    }
    if (modelId === 'punctuation') {
      return this.punctuationReady ? '已就绪' : '基础断句';
    }
    if (modelId === 'voiceprint') {
      return this.profileStoreReady ? this.voiceprintSummary : '加载中';
    }
    if (modelId === 'diarization') {
      return this.enableSpeakerDiarization ? '已启用' : '已关闭';
    }
    if (modelId === 'scene') {
      return this.sceneReady ? '已就绪' : '未加载';
    }
    return this.asrState === 'initializing' ? '加载中' :
      (this.asrState === 'error' ? '需重试' : '已就绪');
  }

  private modelStatusColor(modelId: SpeechModelId): string {
    if (modelId === 'asr') {
      if (this.asrState === 'error') {
        return '#D92D20';
      }
      if (this.asrState === 'initializing' || this.asrState === 'processing') {
        return '#F79009';
      }
      return '#21A366';
    }
    if (modelId === 'punctuation') {
      return this.punctuationReady ? '#21A366' : '#F79009';
    }
    if (modelId === 'voiceprint') {
      return this.voiceprintSummary === '未注册' ? '#98A2B3' : '#21A366';
    }
    if (modelId === 'diarization') {
      return this.enableSpeakerDiarization ? '#21A366' : '#98A2B3';
    }
    if (modelId === 'scene') {
      return this.sceneReady ? '#21A366' : '#F79009';
    }
    return this.asrState === 'error' ? '#D92D20' :
      (this.asrState === 'initializing' ? '#F79009' : '#21A366');
  }

  private modelOutputSummary(modelId: SpeechModelId): string {
    if (modelId === 'asr') {
      return this.latestTelemetryText;
    }
    if (modelId === 'punctuation') {
      if (this.latestRawText.length === 0) {
        return '等待识别文本';
      }
      return `${this.latestRawText.length} 字已处理`;
    }
    if (modelId === 'voiceprint') {
      if (this.latestSpeakerCandidates.length === 0) {
        return '等待足够语音';
      }
      return `${this.latestSpeakerCandidates.length} 个候选`;
    }
    if (modelId === 'diarization') {
      return this.latestSpeakerTurns.length > 0 ?
        `${this.latestSpeakerTurns.length} 个分段` : '等待多人语音';
    }
    if (modelId === 'scene') {
      return this.latestSceneName.length > 0 ?
        `${this.latestSceneName} · ${(this.latestSceneConfidence * 100).toFixed(1)}%` : '等待声音片段';
    }
    if (this.latestUnderstandingEmotion.length === 0 && this.latestUnderstandingEvent.length === 0) {
      return '等待声音片段';
    }
    return `情感 ${audioUnderstandingLabelText(this.latestUnderstandingEmotion)} · ` +
      `事件 ${audioUnderstandingLabelText(this.latestUnderstandingEvent)}`;
  }

  private selectedDatasetSuite(): WorkbenchDatasetSuite {
    for (let i: number = 0; i < this.datasetSuites.length; i++) {
      if (this.datasetSuites[i].modelId === this.selectedModel) {
        return this.datasetSuites[i];
      }
    }
    return this.datasetSuites[0];
  }

  private descriptorFor(modelId: SpeechModelId): SpeechModelDescriptor {
    for (let i: number = 0; i < this.modelDescriptors.length; i++) {
      if (this.modelDescriptors[i].id === modelId) {
        return this.modelDescriptors[i];
      }
    }
    return this.modelDescriptors[0];
  }

  private datasetActionLabel(action: DatasetAction): string {
    if (action === 'open-voiceprint') {
      return '打开 AISHELL-3 身份评测';
    }
    if (action === 'open-diarization') {
      return '打开真实会议分离评测';
    }
    if (action === 'open-understanding') {
      return '打开声音理解真值评测';
    }
    if (action === 'open-scene') {
      return '打开场景感知真值评测';
    }
    return action === 'run-punctuation' ? '运行标点对比' : '运行单条识别';
  }

  private runSelectedDataset(): void {
    const suite: WorkbenchDatasetSuite = this.selectedDatasetSuite();
    if (suite.action === 'open-voiceprint') {
      this.openSpeakerIdentityTestCenter();
      return;
    }
    if (suite.action === 'open-diarization') {
      this.openSpeakerTestCenter();
      return;
    }
    if (suite.action === 'open-understanding') {
      this.openAudioUnderstandingTestCenter();
      return;
    }
    if (suite.action === 'open-scene') {
      this.openAcousticSceneTestCenter();
      return;
    }
    this.runSelfTestWithLimit(1, false).catch((error: Error): void => {
      this.statusMessage = `数据集评测启动失败：${error.message}`;
    });
  }

  private toggleRecording(): void {
    if (this.datasetPlayingId.length > 0) {
      return;
    }
    if (this.asrState === 'ready') {
      this.isButtonPressed = true;
      this.startRecording();
      return;
    }
    if (this.asrState === 'starting' || this.asrState === 'recording') {
      this.isButtonPressed = false;
      this.stopRecording();
    }
  }

  private toggleDatasetPlayback(): void {
    if (this.datasetPlaybackBusy || this.asrState !== 'ready') {
      return;
    }
    const sample: SelfTestAudio = this.selfTestAudios[0];
    this.datasetPlaybackBusy = true;
    this.datasetPlaybackError = '';
    this.datasetAudioPlayer.toggle('workbench-dataset-sample', sample.path,
      this.getAbilityContext().resourceManager)
      .then((): void => {
        this.datasetPlaybackBusy = false;
      })
      .catch((error: Error): void => {
        this.datasetPlaybackBusy = false;
        this.datasetPlaybackError = `播放失败：${error.message}`;
      });
  }

  private formatPlaybackTime(valueMs: number): string {
    const totalSeconds: number = Math.max(0, Math.floor(valueMs / 1000));
    const minutes: number = Math.floor(totalSeconds / 60);
    const seconds: number = totalSeconds % 60;
    return `${minutes}:${seconds < 10 ? `0${seconds}` : `${seconds}`}`;
  }

  private voiceprintCandidateSummary(): string {
    if (this.latestSpeakerCandidates.length === 0) {
      return '累计足够的同一说话人语音后，这里会显示人物候选和相似度。';
    }
    const lines: string[] = [];
    for (let i: number = 0; i < this.latestSpeakerCandidates.length; i++) {
      const candidate: SpeakerCandidate = this.latestSpeakerCandidates[i];
      const name: string = candidate.displayName || `未知说话人 ${candidate.clusterId}`;
      const score: string = candidate.matchScore !== undefined ?
        ` · 相似度 ${candidate.matchScore.toFixed(4)}` : '';
      lines.push(`${name} · ${candidate.segmentCount} 段${score}`);
    }
    return lines.join('\n');
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

  private async ensureOptionalLocationPermission(): Promise<boolean> {
    if (!this.enableSceneContextFusion) {
      return false;
    }
    const context = this.getAbilityContext();
    const atManager = abilityAccessCtrl.createAtManager();
    try {
      let granted = atManager.checkAccessTokenSync(
        context.applicationInfo.accessTokenId,
        LOCATION_PERMISSION,
      ) === abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED;
      if (granted) {
        return true;
      }
      const result = await atManager.requestPermissionsFromUser(context, [LOCATION_PERMISSION]);
      granted = result.authResults.length > 0 &&
        result.authResults[0] === abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED;
      return granted;
    } catch (error) {
      hilog.warn(DOMAIN, TAG, `Optional location permission unavailable: ${error}`);
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
    this.sceneContextTracker.stop();
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
      throw new Error(`${error}`);
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

  private captureAudioUnderstanding(response: SherpaWorkerResponse): void {
    if (response.lang !== undefined) {
      this.latestUnderstandingLanguage = response.lang.trim();
    }
    if (response.emotion !== undefined) {
      this.latestUnderstandingEmotion = response.emotion.trim();
    }
    if (response.event !== undefined) {
      this.latestUnderstandingEvent = response.event.trim();
    }
  }

  private captureAcousticScene(response: SherpaWorkerResponse): void {
    const context: SceneContextSummary = this.sceneContextTracker.stop();
    const audioPrediction: AcousticScenePrediction = buildAcousticScenePrediction(
      response.sceneClassIndex !== undefined ? response.sceneClassIndex : 5,
      response.sceneConfidence !== undefined ? response.sceneConfidence : 0,
      response.sceneProbabilities || [],
      response.sceneWindowCount || 0,
      response.audioDuration || 0,
      response.elapsedMs || 0,
    );
    const fusion: AcousticSceneFusionResult = fuseAcousticScene(
      response.sceneProbabilities || [],
      this.enableSceneContextFusion ? context : {},
      this.enableSceneContextFusion ? this.latestFinalText : '',
      this.enableSceneContextFusion ? this.latestUnderstandingEvent : '',
    );
    const prediction: AcousticScenePrediction = buildAcousticScenePrediction(
      fusion.classIndex,
      fusion.confidence,
      fusion.probabilities,
      response.sceneWindowCount || 0,
      response.audioDuration || 0,
      response.elapsedMs || 0,
    );
    this.latestSceneAudioName = audioPrediction.displayName;
    this.latestSceneAudioConfidence = audioPrediction.confidence;
    this.latestSceneName = prediction.displayName;
    this.latestSceneConfidence = prediction.confidence;
    this.latestSceneWindowCount = prediction.windowCount;
    this.latestSceneElapsedMs = prediction.elapsedMs;
    this.latestSceneInferenceMode = fusion.usedContext ? '音频 + 上下文融合' : '纯音频';
    this.latestSceneEvidence = fusion.evidence.join(' · ');
    this.latestSceneSpeedText = this.enableSceneContextFusion &&
      context.available && context.speedKmh !== undefined ?
      `录音期速度中位数 ${context.speedKmh.toFixed(0)} km/h · ${context.sampleCount} 个定位点` : '';
  }

  private onWorkerMessage(e: MessageEvents): void {
    const response = e.data as SherpaWorkerResponse;
    const msgType = response.msgType;
    hilog.info(DOMAIN, TAG, `Worker message: ${msgType}`);

    if (msgType === 'sherpa-asr-ready') {
      this.asrState = this.isSelfTestRunning ? 'processing' : 'ready';
      this.punctuationReady = response.punctuationReady === true;
      this.sceneReady = response.sceneReady === true;
      this.statusMessage = this.isSelfTestRunning ? '测试引擎已重新加载...' : this.getReadyStatusMessage();
      this.restoreVoiceprintsToWorker();
      if (this.resumeSelfTestAfterWorkerReady) {
        this.resumeSelfTestAfterWorkerReady = false;
        this.runNextSelfTest();
      }
      return;
    }

    if (msgType === 'sherpa-scene-result') {
      this.sceneReady = true;
      this.captureAcousticScene(response);
      return;
    }

    if (msgType === 'sherpa-scene-failed') {
      this.sceneContextTracker.stop();
      this.sceneReady = response.sceneReady === true;
      this.latestSceneName = '场景识别失败';
      hilog.warn(DOMAIN, TAG, `Acoustic scene failed: ${response.message || ''}`);
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
      this.captureAudioUnderstanding(response);
      if (response.rawText !== undefined && response.rawText.trim().length > 0) {
        this.latestRawText = response.rawText.trim();
      }
      if (text.trim().length > 0) {
        this.latestFinalText = text.trim();
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
        this.latestTelemetryText = formatTelemetrySummary(t);
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
      if (response.rawText !== undefined && response.rawText.trim().length > 0) {
        this.latestRawText = response.rawText.trim();
      }
      this.latestFinalText = text.trim();
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
      if (response.rawText !== undefined && response.rawText.trim().length > 0) {
        this.latestRawText = response.rawText.trim();
      }
      this.latestFinalText = text.trim();
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
    this.latestRawText = '';
    this.latestFinalText = '';
    this.latestUnderstandingLanguage = '';
    this.latestUnderstandingEmotion = '';
    this.latestUnderstandingEvent = '';
    this.latestSceneAudioName = '';
    this.latestSceneAudioConfidence = 0;
    this.latestSceneName = '';
    this.latestSceneConfidence = 0;
    this.latestSceneWindowCount = 0;
    this.latestSceneElapsedMs = 0;
    this.latestSceneInferenceMode = '纯音频';
    this.latestSceneEvidence = '';
    this.latestSceneSpeedText = '';
    this.audioLevelPercent = 0;
    this.recordingDurationText = '00:00';

    const hasPermission = await this.ensureMicrophonePermission();
    if (!hasPermission) {
      this.asrState = 'ready';
      return;
    }
    const hasLocationPermission: boolean = await this.ensureOptionalLocationPermission();
    if (token !== this.recordingStartToken || !this.isButtonPressed) {
      this.asrState = 'ready';
      this.statusMessage = this.getReadyStatusMessage();
      return;
    }

    try {
      if (hasLocationPermission) {
        try {
          this.sceneContextTracker.start();
        } catch (error) {
          hilog.warn(DOMAIN, TAG, `Scene context tracking unavailable: ${error}`);
        }
      }
      this.postToWorker({ msgType: 'sherpa-asr-start' });
      await this.audioService.start();
      if (token !== this.recordingStartToken || !this.isButtonPressed) {
        this.sceneContextTracker.stop();
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
      this.sceneContextTracker.stop();
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
    this.latestRawText = '';
    this.latestFinalText = '';
    this.latestTelemetryText = '等待首次运行';
    this.latestUnderstandingLanguage = '';
    this.latestUnderstandingEmotion = '';
    this.latestUnderstandingEvent = '';
    this.latestSceneAudioName = '';
    this.latestSceneAudioConfidence = 0;
    this.latestSceneName = '';
    this.latestSceneConfidence = 0;
    this.latestSceneWindowCount = 0;
    this.latestSceneElapsedMs = 0;
    this.latestSceneInferenceMode = '纯音频';
    this.latestSceneEvidence = '';
    this.latestSceneSpeedText = '';
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
      this.latestRawText = '';
      this.latestFinalText = '';
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

  private async runSelfTestWithLimit(limit: number, diarizationOverride?: boolean): Promise<void> {
    if (this.asrState !== 'ready') {
      return;
    }

    this.isSelfTestRunning = true;
    this.resumeSelfTestAfterWorkerReady = false;
    this.asrState = 'processing';
    this.statusMessage = '测试中...';
    this.partialText = '';
    this.latestRawText = '';
    this.latestFinalText = '';
    this.selfTestIndex = 0;
    this.activeSelfTestOptions = {
      enableSpeakerDiarization: diarizationOverride === undefined ?
        this.enableSpeakerDiarization : diarizationOverride,
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

  @Builder
  private ModelCard(model: SpeechModelDescriptor) {
    Column() {
      Row() {
        Text(model.iconText)
          .fontSize(15)
          .fontWeight(FontWeight.Bold)
          .fontColor(model.accentColor)
          .width(34)
          .height(34)
          .textAlign(TextAlign.Center)
          .backgroundColor(model.softColor)
          .borderRadius(17)
        Column() {
          Text(model.name)
            .fontSize(15)
            .fontWeight(FontWeight.Bold)
            .fontColor('#172033')
          Text(model.subtitle)
            .fontSize(11)
            .fontColor('#6B7485')
            .margin({ top: 2 })
        }
        .alignItems(HorizontalAlign.Start)
        .margin({ left: 9 })
        .layoutWeight(1)
      }
      .width('100%')

      Row() {
        Circle()
          .width(7)
          .height(7)
          .fill(this.modelStatusColor(model.id))
        Text(this.modelStatus(model.id))
          .fontSize(11)
          .fontColor('#465065')
          .margin({ left: 5 })
      }
      .width('100%')
      .margin({ top: 10 })

      Text(this.modelOutputSummary(model.id))
        .fontSize(11)
        .fontColor(model.accentColor)
        .maxLines(1)
        .textOverflow({ overflow: TextOverflow.Ellipsis })
        .width('100%')
        .margin({ top: 6 })
    }
    .width('48.5%')
    .padding(12)
    .backgroundColor(this.selectedModel === model.id ? model.softColor : '#FFFFFF')
    .border({
      width: this.selectedModel === model.id ? 1.5 : 0.5,
      color: this.selectedModel === model.id ? model.accentColor : '#E4E9F0',
    })
    .borderRadius(14)
    .onClick((): void => {
      this.selectedModel = model.id;
    })
  }

  @Builder
  private ModelGrid() {
    Column({ space: 10 }) {
      Row({ space: 10 }) {
        this.ModelCard(this.modelDescriptors[0])
        this.ModelCard(this.modelDescriptors[1])
      }
      .width('100%')
      Row({ space: 10 }) {
        this.ModelCard(this.modelDescriptors[2])
        this.ModelCard(this.modelDescriptors[3])
      }
      .width('100%')
      Row({ space: 10 }) {
        this.ModelCard(this.modelDescriptors[4])
        this.ModelCard(this.modelDescriptors[5])
      }
      .width('100%')
    }
    .width('100%')
  }

  @Builder
  private VoiceprintEnrollmentPanel() {
    Column() {
      Row() {
        Text('注册本人声纹')
          .fontSize(13)
          .fontWeight(FontWeight.Medium)
          .fontColor('#3B2A86')
        Blank()
        Text(`进度 ${this.voiceprintRound}/${SPEAKER_ACTIVE_ENROLL_ROUNDS}`)
          .fontSize(11)
          .fontColor('#7758E8')
      }
      .width('100%')

      TextInput({ text: this.voiceprintName, placeholder: '本人显示名称' })
        .height(40)
        .fontSize(13)
        .backgroundColor('#FFFFFF')
        .enabled(this.asrState === 'ready' && this.profileStoreReady)
        .onChange((value: string): void => {
          this.voiceprintName = value;
        })
        .margin({ top: 8 })

      Row() {
        Button(this.isVoiceprintPressed ? '松开完成本轮' :
          `按住录制第 ${Math.min(this.voiceprintRound + 1, SPEAKER_ACTIVE_ENROLL_ROUNDS)} 次`)
          .fontSize(12)
          .height(38)
          .layoutWeight(1)
          .backgroundColor(this.isVoiceprintPressed ? '#B54708' : '#7758E8')
          .fontColor('#FFFFFF')
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
        Button('管理人物')
          .fontSize(12)
          .height(38)
          .backgroundColor('#FFFFFF')
          .fontColor('#5B3FB5')
          .margin({ left: 8 })
          .onClick((): void => {
            this.openSpeakerProfiles();
          })
      }
      .width('100%')
      .margin({ top: 8 })

      Text('每次自然说话 4–6 秒，共 3 次；原始注册音频不会保存。')
        .fontSize(11)
        .fontColor('#6D6590')
        .margin({ top: 6 })
    }
    .width('100%')
    .padding(12)
    .backgroundColor('#F7F4FF')
    .borderRadius(10)
    .margin({ top: 10 })
  }

  @Builder
  private ModelInspector() {
    Column() {
      Row() {
        Text(`${this.descriptorFor(this.selectedModel).name} 模型效果`)
          .fontSize(16)
          .fontWeight(FontWeight.Bold)
          .fontColor('#172033')
        Blank()
        Text('真实输出')
          .fontSize(11)
          .fontColor('#246BFD')
          .padding({ left: 9, right: 9, top: 4, bottom: 4 })
          .backgroundColor('#EAF1FF')
          .borderRadius(10)
      }
      .width('100%')

      if (this.selectedModel === 'asr') {
        Text('ASR 原始文本')
          .fontSize(11)
          .fontColor('#6B7485')
          .margin({ top: 10 })
        Text(this.latestRawText.length > 0 ? this.latestRawText :
          this.partialText.length > 0 ? this.partialText : '完成一次录音或数据集评测后显示。')
          .fontSize(14)
          .fontColor(this.latestRawText.length > 0 || this.partialText.length > 0 ? '#172033' : '#929BAD')
          .width('100%')
          .margin({ top: 5 })
        Text(this.latestTelemetryText)
          .fontSize(11)
          .fontColor('#246BFD')
          .margin({ top: 8 })
      } else if (this.selectedModel === 'punctuation') {
        Text('输入 · ASR 原始文本')
          .fontSize(11)
          .fontColor('#6B7485')
          .margin({ top: 10 })
        Text(this.latestRawText.length > 0 ? this.latestRawText : '等待 ASR 原始文本')
          .fontSize(13)
          .fontColor(this.latestRawText.length > 0 ? '#465065' : '#929BAD')
          .width('100%')
          .padding(10)
          .backgroundColor('#F5F7FA')
          .borderRadius(8)
          .margin({ top: 5 })
        Text('输出 · 标点与断句')
          .fontSize(11)
          .fontColor('#0FA9BC')
          .margin({ top: 9 })
        Text(this.latestFinalText.length > 0 ? this.latestFinalText : '完成识别后显示标点结果')
          .fontSize(14)
          .fontColor(this.latestFinalText.length > 0 ? '#172033' : '#929BAD')
          .width('100%')
          .padding(10)
          .backgroundColor('#EAF9FB')
          .borderRadius(8)
          .margin({ top: 5 })
      } else if (this.selectedModel === 'voiceprint') {
        Text(this.voiceprintCandidateSummary())
          .fontSize(13)
          .fontColor('#443878')
          .width('100%')
          .margin({ top: 10 })
        this.VoiceprintEnrollmentPanel()
      } else if (this.selectedModel === 'diarization') {
        Row() {
          Column() {
            Text(this.enableSpeakerDiarization ? '说话人分离已启用' : '说话人分离已关闭')
              .fontSize(13)
              .fontWeight(FontWeight.Medium)
              .fontColor('#176B49')
            Text(this.latestSpeakerTurns.length > 0 ?
              `${this.latestSpeakerTurns.length} 个分段，可继续编辑人物` : '多人录音完成后显示时间段与人物聚类')
              .fontSize(11)
              .fontColor('#6B7485')
              .margin({ top: 3 })
          }
          .alignItems(HorizontalAlign.Start)
          .layoutWeight(1)
          Toggle({ type: ToggleType.Switch, isOn: this.enableSpeakerDiarization })
            .selectedColor('#159A61')
            .enabled(this.asrState === 'ready')
            .onChange((isOn: boolean): void => {
              this.enableSpeakerDiarization = isOn;
            })
        }
        .width('100%')
        .margin({ top: 10 })
        if (this.latestSpeakerPreview.length > 0) {
          Text(this.latestSpeakerPreview)
            .fontSize(12)
            .fontColor('#344A40')
            .maxLines(6)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
            .width('100%')
            .padding(10)
            .backgroundColor('#ECF8F2')
            .borderRadius(8)
            .margin({ top: 9 })
          Button('编辑说话人')
            .height(36)
            .fontSize(12)
            .backgroundColor('#159A61')
            .fontColor('#FFFFFF')
            .margin({ top: 8 })
            .onClick((): void => {
              this.openMeetingSpeakerEditor();
            })
        }
      } else if (this.selectedModel === 'understanding') {
        Row({ space: 8 }) {
          Column() {
            Text('语种')
              .fontSize(10)
              .fontColor('#6B7485')
            Text(audioUnderstandingLanguageText(this.latestUnderstandingLanguage))
              .fontSize(13)
              .fontWeight(FontWeight.Bold)
              .fontColor('#4E55E6')
              .margin({ top: 4 })
          }
          .layoutWeight(1)
          .alignItems(HorizontalAlign.Start)
          .padding(10)
          .backgroundColor('#EEF0FF')
          .borderRadius(9)
          Column() {
            Text('情感')
              .fontSize(10)
              .fontColor('#6B7485')
            Text(audioUnderstandingLabelText(this.latestUnderstandingEmotion))
              .fontSize(13)
              .fontWeight(FontWeight.Bold)
              .fontColor('#C45F16')
              .margin({ top: 4 })
          }
          .layoutWeight(1)
          .alignItems(HorizontalAlign.Start)
          .padding(10)
          .backgroundColor('#FFF4E8')
          .borderRadius(9)
          Column() {
            Text('声音事件')
              .fontSize(10)
              .fontColor('#6B7485')
            Text(audioUnderstandingLabelText(this.latestUnderstandingEvent))
              .fontSize(13)
              .fontWeight(FontWeight.Bold)
              .fontColor('#137A4C')
              .margin({ top: 4 })
          }
          .layoutWeight(1)
          .alignItems(HorizontalAlign.Start)
          .padding(10)
          .backgroundColor('#ECF8F2')
          .borderRadius(9)
        }
        .width('100%')
        .margin({ top: 10 })
        Text('声音理解来自最近最多 30 秒录音的 SenseVoice 原生输出，不通过识别文字推断。')
          .fontSize(11)
          .fontColor('#6B7485')
          .width('100%')
          .margin({ top: 8 })
      } else {
        Row() {
          Column() {
            Text('上下文增强')
              .fontSize(13)
              .fontWeight(FontWeight.Medium)
              .fontColor('#80512F')
            Text('可选读取录音期速度；不保存经纬度')
              .fontSize(10)
              .fontColor('#6B7485')
              .margin({ top: 2 })
          }
          .alignItems(HorizontalAlign.Start)
          .layoutWeight(1)
          Toggle({ type: ToggleType.Switch, isOn: this.enableSceneContextFusion })
            .selectedColor('#E87928')
            .enabled(this.asrState === 'ready')
            .onChange((isOn: boolean): void => {
              this.enableSceneContextFusion = isOn;
            })
        }
        .width('100%')
        .margin({ top: 10 })
        Column() {
          Text('端侧环境场景')
            .fontSize(10)
            .fontColor('#6B7485')
          Text(this.latestSceneName.length > 0 ? this.latestSceneName : '等待完成一次录音')
            .fontSize(22)
            .fontWeight(FontWeight.Bold)
            .fontColor(this.latestSceneName.length > 0 ? '#C05A13' : '#929BAD')
            .margin({ top: 7 })
          if (this.latestSceneName.length > 0) {
            Text(`最终置信度 ${(this.latestSceneConfidence * 100).toFixed(1)}% · ` +
              `${this.latestSceneWindowCount} 个窗口 · ${this.latestSceneElapsedMs} ms`)
              .fontSize(11)
              .fontColor('#80512F')
              .margin({ top: 6 })
            Text(`纯音频：${this.latestSceneAudioName} · ` +
              `${(this.latestSceneAudioConfidence * 100).toFixed(1)}%`)
              .fontSize(11)
              .fontColor('#6B7485')
              .margin({ top: 5 })
            if (this.latestSceneInferenceMode !== '纯音频') {
              Text(`融合输出：${this.latestSceneName} · ` +
                `${(this.latestSceneConfidence * 100).toFixed(1)}%`)
                .fontSize(11)
                .fontWeight(FontWeight.Medium)
                .fontColor('#C05A13')
                .margin({ top: 4 })
            }
            Text(this.latestSceneInferenceMode)
              .fontSize(11)
              .fontWeight(FontWeight.Medium)
              .fontColor(this.latestSceneInferenceMode === '纯音频' ? '#6B7485' : '#C05A13')
              .margin({ top: 5 })
            if (this.latestSceneEvidence.length > 0) {
              Text(this.latestSceneEvidence)
                .fontSize(11)
                .fontColor('#80512F')
                .margin({ top: 4 })
            }
            if (this.latestSceneSpeedText.length > 0) {
              Text(this.latestSceneSpeedText)
                .fontSize(10)
                .fontColor('#6B7485')
                .margin({ top: 3 })
            }
          }
        }
        .alignItems(HorizontalAlign.Start)
        .width('100%')
        .padding(13)
        .backgroundColor('#FFF3E8')
        .borderRadius(10)
        .margin({ top: 10 })
        Text('音频模型先给候选场景；开启增强后，仅用可靠速度、明确 ASR 关键词和声音事件修正易混类别。数据集音频没有传感器真值，评测页仍只报告纯音频指标。')
          .fontSize(11)
          .fontColor('#6B7485')
          .width('100%')
          .margin({ top: 8 })
      }
    }
    .width('100%')
    .padding(14)
    .backgroundColor('#FFFFFF')
    .borderRadius(14)
    .border({ width: 0.5, color: '#E4E9F0' })
  }

  @Builder
  private RecordingController() {
    Column() {
      Row() {
        Column() {
          Text(this.recordingDurationText)
            .fontSize(30)
            .fontWeight(FontWeight.Bold)
            .fontColor('#172033')
          Row() {
            Circle()
              .width(8)
              .height(8)
              .fill(this.asrState === 'recording' ? '#20B8CD' : '#A7B0BF')
            Text(this.asrState === 'recording' ? '正在聆听' :
              this.asrState === 'processing' ? '本地模型处理中' : '点击开始录音')
              .fontSize(12)
              .fontColor('#6B7485')
              .margin({ left: 6 })
          }
          .margin({ top: 4 })
        }
        .alignItems(HorizontalAlign.Start)

        Row({ space: 4 }) {
          ForEach(this.waveBars, (height: number): void => {
            Rect()
              .width(3)
              .height(this.asrState === 'recording' ? height : Math.max(8, height / 2))
              .radius(2)
              .fill(this.asrState === 'recording' ? '#246BFD' : '#C8D7F5')
          }, (height: number, index: number): string => `${index}-${height}`)
        }
        .layoutWeight(1)
        .justifyContent(FlexAlign.Center)
        .margin({ left: 14, right: 14 })

        Button(this.asrState === 'recording' || this.asrState === 'starting' ? '结束' : '录音')
          .width(66)
          .height(48)
          .fontSize(13)
          .backgroundColor(this.asrState === 'recording' || this.asrState === 'starting' ?
            '#D92D20' : '#246BFD')
          .fontColor('#FFFFFF')
          .enabled(this.asrState === 'ready' || this.asrState === 'starting' || this.asrState === 'recording')
          .onClick((): void => {
            this.toggleRecording();
          })
      }
      .width('100%')
      if (this.asrState === 'recording') {
        Row() {
          Text(`输入音量 ${this.audioLevelPercent}%`)
            .fontSize(11)
            .fontColor('#246BFD')
          Blank()
          Text('音频仅在本机处理')
            .fontSize(11)
            .fontColor('#6B7485')
        }
        .width('100%')
        .margin({ top: 10 })
      }
    }
    .width('100%')
    .padding(16)
    .backgroundColor('#FFFFFF')
    .borderRadius(16)
    .border({ width: 0.5, color: '#E4E9F0' })
  }

  @Builder
  private ResultPanel() {
    Column() {
      Row() {
        Text('完整结果')
          .fontSize(16)
          .fontWeight(FontWeight.Bold)
          .fontColor('#172033')
        Blank()
        if (this.recognizedText.length > 0) {
          Button('清除')
            .fontSize(11)
            .height(30)
            .backgroundColor('#F1F3F6')
            .fontColor('#5D6677')
            .onClick((): void => {
              this.clearText();
            })
        }
      }
      .width('100%')

      if (this.partialText.length > 0) {
        Text(this.partialText)
          .fontSize(14)
          .fontColor('#246BFD')
          .width('100%')
          .padding(10)
          .backgroundColor('#EEF4FF')
          .borderRadius(8)
          .margin({ top: 9 })
      }
      if (this.latestSpeakerPreview.length > 0) {
        Text(this.latestSpeakerPreview)
          .fontSize(13)
          .fontColor('#344054')
          .maxLines(10)
          .textOverflow({ overflow: TextOverflow.Ellipsis })
          .width('100%')
          .margin({ top: 10 })
      } else if (this.recognizedText.length > 0) {
        Text(this.recognizedText)
          .fontSize(14)
          .fontColor('#344054')
          .maxLines(10)
          .textOverflow({ overflow: TextOverflow.Ellipsis })
          .width('100%')
          .margin({ top: 10 })
      } else if (this.partialText.length === 0) {
        Text('录音或运行数据集后，这里会显示完整流水线结果。')
          .fontSize(13)
          .fontColor('#929BAD')
          .width('100%')
          .margin({ top: 10 })
      }
      if (this.hasEditableSpeakers) {
        Button('编辑说话人')
          .height(34)
          .fontSize(12)
          .backgroundColor('#ECF8F2')
          .fontColor('#137A4C')
          .margin({ top: 9 })
          .onClick((): void => {
            this.openMeetingSpeakerEditor();
          })
      }
    }
    .width('100%')
    .padding(14)
    .backgroundColor('#FFFFFF')
    .borderRadius(14)
    .border({ width: 0.5, color: '#E4E9F0' })
  }

  @Builder
  private DatasetSuiteCard(suite: WorkbenchDatasetSuite) {
    Column() {
      Row() {
        Text(this.descriptorFor(suite.modelId).iconText)
          .fontSize(14)
          .fontWeight(FontWeight.Bold)
          .fontColor(this.descriptorFor(suite.modelId).accentColor)
          .width(32)
          .height(32)
          .textAlign(TextAlign.Center)
          .backgroundColor(this.descriptorFor(suite.modelId).softColor)
          .borderRadius(9)
        Text(this.descriptorFor(suite.modelId).name)
          .fontSize(14)
          .fontWeight(FontWeight.Bold)
          .fontColor('#172033')
          .margin({ left: 8 })
        Blank()
        if (this.selectedModel === suite.modelId) {
          Text('已选择')
            .fontSize(10)
            .fontColor('#246BFD')
        }
      }
      .width('100%')
      Text(suite.title)
        .fontSize(13)
        .fontWeight(FontWeight.Medium)
        .fontColor('#344054')
        .width('100%')
        .margin({ top: 9 })
      Text(suite.sampleSummary)
        .fontSize(11)
        .fontColor('#6B7485')
        .width('100%')
        .margin({ top: 4 })
      Text(suite.description)
        .fontSize(11)
        .fontColor('#929BAD')
        .maxLines(2)
        .textOverflow({ overflow: TextOverflow.Ellipsis })
        .width('100%')
        .margin({ top: 5 })
    }
    .width('48.5%')
    .padding(12)
    .backgroundColor('#FFFFFF')
    .border({
      width: this.selectedModel === suite.modelId ? 1.5 : 0.5,
      color: this.selectedModel === suite.modelId ? '#246BFD' : '#E4E9F0',
    })
    .borderRadius(14)
    .onClick((): void => {
      this.selectedModel = suite.modelId;
    })
  }

  @Builder
  private DatasetContent() {
    Column({ space: 12 }) {
      Column() {
        Row() {
          Column() {
            Text(this.selectedDatasetSuite().title)
              .fontSize(16)
              .fontWeight(FontWeight.Bold)
              .fontColor('#172033')
            Text(this.selectedDatasetSuite().sampleSummary)
              .fontSize(11)
              .fontColor('#6B7485')
              .margin({ top: 4 })
          }
          .alignItems(HorizontalAlign.Start)
          .layoutWeight(1)
          if (this.selectedModel === 'asr' || this.selectedModel === 'punctuation') {
            Button(this.datasetPlayingId.length > 0 ? '停止试听' : '试听原始音频')
              .fontSize(11)
              .height(34)
              .backgroundColor(this.datasetPlayingId.length > 0 ? '#246BFD' : '#EEF4FF')
              .fontColor(this.datasetPlayingId.length > 0 ? '#FFFFFF' : '#246BFD')
              .enabled(!this.datasetPlaybackBusy && this.asrState === 'ready')
              .onClick((): void => {
                this.toggleDatasetPlayback();
              })
          }
        }
        .width('100%')
        if (this.datasetPlayingId.length > 0) {
          Text(`${this.formatPlaybackTime(this.datasetPlaybackPositionMs)} / ` +
            `${this.formatPlaybackTime(this.datasetPlaybackDurationMs)}`)
            .fontSize(11)
            .fontColor('#246BFD')
            .margin({ top: 7 })
        }
        if (this.datasetPlaybackError.length > 0) {
          Text(this.datasetPlaybackError)
            .fontSize(11)
            .fontColor('#B42318')
            .margin({ top: 7 })
        }
      }
      .width('100%')
      .padding(14)
      .backgroundColor('#FFFFFF')
      .borderRadius(14)
      .border({ width: 0.5, color: '#E4E9F0' })

      Column({ space: 10 }) {
        Row({ space: 10 }) {
          this.DatasetSuiteCard(this.datasetSuites[0])
          this.DatasetSuiteCard(this.datasetSuites[1])
        }
        .width('100%')
        Row({ space: 10 }) {
          this.DatasetSuiteCard(this.datasetSuites[2])
          this.DatasetSuiteCard(this.datasetSuites[3])
        }
        .width('100%')
        Row({ space: 10 }) {
          this.DatasetSuiteCard(this.datasetSuites[4])
          this.DatasetSuiteCard(this.datasetSuites[5])
        }
        .width('100%')
      }
      .width('100%')

      Row({ space: 10 }) {
        Button(this.datasetActionLabel(this.selectedDatasetSuite().action))
          .height(46)
          .fontSize(13)
          .layoutWeight(1)
          .backgroundColor('#246BFD')
          .fontColor('#FFFFFF')
          .enabled(this.asrState === 'ready' && this.datasetPlayingId.length === 0)
          .onClick((): void => {
            this.runSelectedDataset();
          })
        if (this.selectedModel === 'asr' || this.selectedModel === 'punctuation') {
          Button('导入音频')
            .height(46)
            .fontSize(13)
            .backgroundColor('#FFFFFF')
            .fontColor('#246BFD')
            .border({ width: 1, color: '#BFD2FF' })
            .enabled(this.asrState === 'ready' && this.datasetPlayingId.length === 0)
            .onClick((): void => {
              this.importAudioForTest();
            })
        }
      }
      .width('100%')

      this.ModelInspector()
      this.ResultPanel()
    }
    .width('100%')
  }

  build() {
    Column() {
      Row() {
        Column() {
          Text('语音智能实验台')
            .fontSize(23)
            .fontWeight(FontWeight.Bold)
            .fontColor('#172033')
          Row() {
            Circle()
              .width(8)
              .height(8)
              .fill(this.asrState === 'error' ? '#D92D20' :
                this.asrState === 'initializing' ? '#F79009' : '#21A366')
            Text(this.statusMessage)
              .fontSize(11)
              .fontColor('#6B7485')
              .maxLines(1)
              .textOverflow({ overflow: TextOverflow.Ellipsis })
              .margin({ left: 6 })
          }
          .margin({ top: 5 })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)
        Button('持续录音')
          .fontSize(11)
          .height(34)
          .backgroundColor('#EEF4FF')
          .fontColor('#246BFD')
          .enabled(this.asrState === 'ready')
          .onClick((): void => {
            this.openContinuousRecording();
          })
        Button('人物')
          .fontSize(11)
          .height(34)
          .backgroundColor('#F2EEFF')
          .fontColor('#6941C6')
          .margin({ left: 7 })
          .enabled(this.asrState === 'ready')
          .onClick((): void => {
            this.openSpeakerProfiles();
          })
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 15, bottom: 10 })

      Row() {
        Button('录音识别')
          .fontSize(13)
          .fontWeight(this.workbenchMode === 'recording' ? FontWeight.Bold : FontWeight.Normal)
          .height(40)
          .layoutWeight(1)
          .backgroundColor(this.workbenchMode === 'recording' ? '#FFFFFF' : '#EEF1F5')
          .fontColor(this.workbenchMode === 'recording' ? '#246BFD' : '#667085')
          .onClick((): void => {
            this.selectWorkbenchMode('recording');
          })
        Button('数据集评测')
          .fontSize(13)
          .fontWeight(this.workbenchMode === 'dataset' ? FontWeight.Bold : FontWeight.Normal)
          .height(40)
          .layoutWeight(1)
          .backgroundColor(this.workbenchMode === 'dataset' ? '#FFFFFF' : '#EEF1F5')
          .fontColor(this.workbenchMode === 'dataset' ? '#246BFD' : '#667085')
          .onClick((): void => {
            this.selectWorkbenchMode('dataset');
          })
      }
      .width('100%')
      .padding(3)
      .backgroundColor('#EEF1F5')
      .borderRadius(12)
      .margin({ left: 16, right: 16, bottom: 10 })

      Scroll() {
        Column({ space: 12 }) {
          if (this.workbenchMode === 'recording') {
            this.RecordingController()
            this.ModelGrid()
            this.ModelInspector()
            this.ResultPanel()
          } else {
            this.DatasetContent()
          }
        }
        .width('100%')
        .padding({ left: 16, right: 16, bottom: 18 })
      }
      .layoutWeight(1)
      .width('100%')
      .scrollBar(BarState.Off)
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#F6F8FC')
  }
}
