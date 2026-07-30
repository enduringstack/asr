import { describe, it, expect } from '@ohos/hypium';
import { TranscriptBuilder } from '../main/ets/common/TranscriptBuilder';

function plainAppend(base: string, next: string): string {
  return base + next;
}

export default function transcriptBuilderTest() {
  describe('transcriptBuilderTest', () => {
    it('punctuates each segment exactly once instead of the whole transcript', 0, () => {
      // The regression this guards: the progress loop used to call the punctuation model
      // on the full accumulated text after every segment, i.e. O(N^2) model work.
      const punctuationInputs: string[] = [];
      const builder = new TranscriptBuilder(plainAppend, (text: string) => {
        punctuationInputs.push(text);
        return text + '。';
      });

      builder.addSegment('第一段');
      builder.addSegment('第二段');
      builder.addSegment('第三段');

      expect(punctuationInputs.length).assertEqual(3);
      expect(punctuationInputs[0]).assertEqual('第一段');
      expect(punctuationInputs[1]).assertEqual('第二段');
      expect(punctuationInputs[2]).assertEqual('第三段');
      // Total characters punctuated grow linearly, not quadratically.
      let punctuatedChars = 0;
      for (let i = 0; i < punctuationInputs.length; ++i) {
        punctuatedChars += punctuationInputs[i].length;
      }
      expect(punctuatedChars).assertEqual(9);
    });

    it('returns the progress transcript accumulated so far', 0, () => {
      const builder = new TranscriptBuilder(plainAppend, (text: string) => text + '。');
      expect(builder.addSegment('甲')).assertEqual('甲。');
      expect(builder.addSegment('乙')).assertEqual('甲。乙。');
      expect(builder.getProgressText()).assertEqual('甲。乙。');
    });

    it('keeps the raw transcript for the single final full-document pass', 0, () => {
      const builder = new TranscriptBuilder(plainAppend, (text: string) => text + '。');
      builder.addSegment('甲');
      builder.addSegment('乙');
      expect(builder.getRawText()).assertEqual('甲乙');
    });

    it('lets the appender de-duplicate overlapping segment text', 0, () => {
      // Mimics the overlap merge used by the fixed-length fallback planner.
      const dedupe = (base: string, next: string): string => {
        if (base.length > 0 && next.length > 0 && base.endsWith(next.charAt(0))) {
          return base + next.substring(1);
        }
        return base + next;
      };
      const builder = new TranscriptBuilder(dedupe, (text: string) => text);
      builder.addSegment('今天开会');
      builder.addSegment('会讨论预算');
      expect(builder.getRawText()).assertEqual('今天开会讨论预算');
      expect(builder.getProgressText()).assertEqual('今天开会讨论预算');
    });

    it('does not re-punctuate everything when the appender rewrites the tail', 0, () => {
      // An appender that replaces part of the existing tail breaks the "new text is a
      // suffix" assumption. The builder must degrade to punctuating the raw segment
      // rather than falling back to a full-transcript pass.
      const punctuationInputs: string[] = [];
      const rewriteTail = (base: string, next: string): string => {
        if (base.length === 0) {
          return next;
        }
        return base.substring(0, base.length - 1) + next;
      };
      const builder = new TranscriptBuilder(rewriteTail, (text: string) => {
        punctuationInputs.push(text);
        return text;
      });

      builder.addSegment('第一段');
      builder.addSegment('第二段');

      expect(punctuationInputs.length).assertEqual(2);
      expect(punctuationInputs[1]).assertEqual('第二段');
      expect(builder.getRawText()).assertEqual('第一第二段');
    });

    it('ignores segments that contribute nothing', 0, () => {
      const punctuationInputs: string[] = [];
      const builder = new TranscriptBuilder(
        (base: string, _next: string) => base,
        (text: string) => {
          punctuationInputs.push(text);
          return text;
        }
      );
      builder.addSegment('');
      builder.addSegment('被丢弃');
      expect(punctuationInputs.length).assertEqual(0);
      expect(builder.getProgressText()).assertEqual('');
      expect(builder.getRawText()).assertEqual('');
    });
  });
}
