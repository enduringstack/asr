/**
 * Incremental transcript assembly for the segmented offline path.
 *
 * The progress loop used to call the punctuation model on the whole accumulated
 * transcript after every decoded segment. For an N-segment file that is N punctuation
 * passes over a monotonically growing text, i.e. O(N^2) work in a 72 MiB CT-Transformer
 * for output the user only glances at. A three-minute recording paid for 18 such passes.
 *
 * The builder punctuates only the text a segment newly contributed, which makes progress
 * rendering O(N) in total. The caller still runs one full-document punctuation pass at
 * the end, so the delivered transcript is unchanged.
 *
 * This file must stay free of @kit imports so it can run in local (non-device) tests.
 */

export type TextAppender = (base: string, next: string) => string;
export type TextTransform = (text: string) => string;

export class TranscriptBuilder {
  private appender: TextAppender;
  private punctuate: TextTransform;
  private rawText: string = '';
  private progressText: string = '';

  constructor(appender: TextAppender, punctuate: TextTransform) {
    this.appender = appender;
    this.punctuate = punctuate;
  }

  /**
   * Add one decoded segment and return the transcript to display right now.
   */
  addSegment(segmentText: string): string {
    const before = this.rawText;
    this.rawText = this.appender(before, segmentText);

    // Normally the appender only appends, so the newly contributed text is the suffix.
    // An appender that rewrites the tail (overlap de-duplication can) breaks that
    // assumption; fall back to the raw segment text rather than re-punctuating
    // everything, because progress text is transient and the final pass is authoritative.
    let added = segmentText;
    if (this.rawText.length >= before.length && this.rawText.startsWith(before)) {
      added = this.rawText.substring(before.length);
    }
    if (added.length === 0) {
      return this.progressText;
    }

    this.progressText += this.punctuate(added);
    return this.progressText;
  }

  /** Concatenated segment texts, unpunctuated — the input for the final pass. */
  getRawText(): string {
    return this.rawText;
  }

  /** Cheaply punctuated text as shown during progress. */
  getProgressText(): string {
    return this.progressText;
  }
}
