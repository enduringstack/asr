import localUnitTest from './LocalUnit.test';
import audioSegmentationTest from './AudioSegmentation.test';
import resamplerTest from './Resampler.test';
import transcriptBuilderTest from './TranscriptBuilder.test';

export default function testsuite() {
  localUnitTest();
  audioSegmentationTest();
  resamplerTest();
  transcriptBuilderTest();
}
