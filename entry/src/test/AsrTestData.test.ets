import { describe, expect, it } from '@ohos/hypium';
import { AsrTestCase, buildAsrTestCases } from '../main/ets/common/AsrTestData';

export default function asrTestDataTest() {
  describe('asrTestDataTest', () => {
    it('contains exactly twenty labeled cases from every AISHELL dataset', 0, () => {
      const cases: AsrTestCase[] = buildAsrTestCases();
      expect(cases.length).assertEqual(100);
      const datasetIds: string[] = ['aishell1', 'aishell2', 'aishell3', 'aishell4', 'aishell5'];
      for (let datasetIndex: number = 0; datasetIndex < datasetIds.length; datasetIndex++) {
        let count: number = 0;
        for (let index: number = 0; index < cases.length; index++) {
          if (cases[index].datasetId === datasetIds[datasetIndex]) {
            count += 1;
          }
        }
        expect(count).assertEqual(20);
      }
    });

    it('keeps identifiers paths references and speaker labels complete and unique', 0, () => {
      const cases: AsrTestCase[] = buildAsrTestCases();
      const ids: string[] = [];
      const paths: string[] = [];
      for (let index: number = 0; index < cases.length; index++) {
        const spec: AsrTestCase = cases[index];
        expect(spec.id.length > 0).assertTrue();
        expect(spec.audioPath.length > 0).assertTrue();
        expect(spec.reference.trim().length > 0).assertTrue();
        expect(spec.speakerIds.length > 0).assertTrue();
        expect(ids.indexOf(spec.id)).assertEqual(-1);
        expect(paths.indexOf(spec.audioPath)).assertEqual(-1);
        ids.push(spec.id);
        paths.push(spec.audioPath);
      }
    });

    it('uses meeting turn truth only for AISHELL-4 and AISHELL-5', 0, () => {
      const cases: AsrTestCase[] = buildAsrTestCases();
      for (let index: number = 0; index < cases.length; index++) {
        const spec: AsrTestCase = cases[index];
        const meetingDataset: boolean = spec.datasetId === 'aishell4' || spec.datasetId === 'aishell5';
        expect(spec.supportsAsr).assertTrue();
        expect(spec.supportsVoiceprint).assertTrue();
        expect(spec.supportsDiarization).assertEqual(meetingDataset);
        expect(spec.speakerTurns.length > 0).assertEqual(meetingDataset);
      }
    });
  });
}
