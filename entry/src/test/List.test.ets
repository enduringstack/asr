import localUnitTest from './LocalUnit.test';
import audioSegmentationTest from './AudioSegmentation.test';
import resamplerTest from './Resampler.test';
import transcriptBuilderTest from './TranscriptBuilder.test';
import speakerMathTest from './SpeakerMath.test';
import speakerIdentityEvaluationTest from './SpeakerIdentityEvaluation.test';
import voiceWorkbenchTypesTest from './VoiceWorkbenchTypes.test';

export default function testsuite() {
  localUnitTest();
  audioSegmentationTest();
  resamplerTest();
  transcriptBuilderTest();
  speakerMathTest();
  speakerIdentityEvaluationTest();
  voiceWorkbenchTypesTest();
}
