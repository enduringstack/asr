import { describe, expect, it } from '@ohos/hypium';
import {
  buildDatasetSuites,
  buildSpeechModelDescriptors,
  formatTelemetrySummary,
  SpeechModelDescriptor,
  WorkbenchDatasetSuite,
} from '../main/ets/common/VoiceWorkbenchTypes';

export default function voiceWorkbenchTypesTest() {
  describe('voiceWorkbenchTypesTest', () => {
    it('keeps the six production models in pipeline order', 0, () => {
      const models: SpeechModelDescriptor[] = buildSpeechModelDescriptors();
      expect(models.length).assertEqual(6);
      expect(models[0].id).assertEqual('asr');
      expect(models[1].id).assertEqual('punctuation');
      expect(models[2].id).assertEqual('voiceprint');
      expect(models[3].id).assertEqual('diarization');
      expect(models[4].id).assertEqual('understanding');
      expect(models[5].id).assertEqual('scene');
    });

    it('offers one dataset path for every model', 0, () => {
      const suites: WorkbenchDatasetSuite[] = buildDatasetSuites();
      expect(suites.length).assertEqual(6);
      const modelIds: string[] = [];
      for (let i: number = 0; i < suites.length; i++) {
        modelIds.push(suites[i].modelId);
      }
      expect(modelIds.indexOf('asr') >= 0).assertTrue();
      expect(modelIds.indexOf('punctuation') >= 0).assertTrue();
      expect(modelIds.indexOf('voiceprint') >= 0).assertTrue();
      expect(modelIds.indexOf('diarization') >= 0).assertTrue();
      expect(modelIds.indexOf('understanding') >= 0).assertTrue();
      expect(modelIds.indexOf('scene') >= 0).assertTrue();
      expect(suites[0].action).assertEqual('open-asr');
      expect(suites[0].sampleSummary).assertEqual('5 套数据 · 每套 20 条');
    });

    it('formats only telemetry values that actually exist', 0, () => {
      expect(formatTelemetrySummary({})).assertEqual('等待首次运行');
      expect(formatTelemetrySummary({ finalRedecodeMs: 125, finalRedecodeRtf: 0.053,
        audioDurationSec: 4.2 })).assertEqual('音频 4.2 秒 · 处理 125 ms · RTF 0.053');
    });
  });
}
