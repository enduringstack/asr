export type AsrEvaluationState = 'waiting' | 'running' | 'completed' | 'failed';
export type AsrEvaluationFilter = 'all' | 'errors';
export type AsrEvaluationProtocol = 'full_window' | 'oracle_turns' | 'speaker_pipeline';

export interface AsrTimedTurn {
  speakerId: string;
  startSec: number;
  endSec: number;
  text: string;
}

export interface AsrAnalyzedTurn extends AsrTimedTurn {
  overlapped: boolean;
}

export interface AsrEditScore {
  referenceNormalized: string;
  hypothesisNormalized: string;
  referenceChars: number;
  substitutions: number;
  deletions: number;
  insertions: number;
  errors: number;
  cer: number;
  exact: boolean;
}

export interface AsrCaseResult {
  state: AsrEvaluationState;
  score?: AsrEditScore;
  audioDurationSec: number;
  elapsedMs: number;
}

export interface AsrTurnCaseResult extends AsrCaseResult {
  overlapped: boolean;
}

export interface AsrTurnEvaluationSummary {
  all: AsrEvaluationSummary;
  nonOverlap: AsrEvaluationSummary;
  overlap: AsrEvaluationSummary;
}

export interface AsrSpeakerTranscript {
  speakerId: string;
  text: string;
}

export interface AsrSpeakerAssignment {
  referenceSpeakerId: string;
  hypothesisSpeakerId: string;
  score: AsrEditScore;
}

export interface AsrPermutationScore {
  score: AsrEditScore;
  assignments: AsrSpeakerAssignment[];
}

export interface AsrDiarizationTurn {
  speakerId: string;
  startSec: number;
  endSec: number;
}

export interface AsrDiarizationSpeakerAssignment {
  referenceSpeakerId: string;
  hypothesisSpeakerId: string;
  overlapSec: number;
}

export interface AsrDiarizationScore {
  der: number;
  referenceSpeakerSec: number;
  missedSpeakerSec: number;
  falseAlarmSpeakerSec: number;
  confusionSpeakerSec: number;
  referenceOverlapSec: number;
  predictedOverlapSec: number;
  overlapTruePositiveSec: number;
  overlapRecall: number;
  assignments: AsrDiarizationSpeakerAssignment[];
}

export interface AsrTimeRange {
  startSec: number;
  endSec: number;
}

export interface AsrOverlapDetectionScore {
  referenceOverlapSec: number;
  predictedOverlapSec: number;
  truePositiveSec: number;
  recall: number;
}

export interface AsrEvaluationSummary {
  evaluated: number;
  completed: number;
  failed: number;
  exact: number;
  referenceChars: number;
  substitutions: number;
  deletions: number;
  insertions: number;
  errors: number;
  microCer: number;
  macroCer: number;
  audioDurationSec: number;
  elapsedMs: number;
  rtf: number;
}

interface EditCell {
  distance: number;
  substitutions: number;
  deletions: number;
  insertions: number;
}

interface PermutationCell extends EditCell {
  assignment: number[];
}

interface OverlapPermutationCell {
  overlapSec: number;
  assignment: number[];
}

const REMOVED_PUNCTUATION: string =
  '，。！？；：、“”‘’（）【】《》〈〉「」『』〔〕—…·,.!?;:\"\'()[]{}<>-_\\/|~`@#$%^&*+=';

function normalizeCharacter(character: string): string {
  const code: number = character.charCodeAt(0);
  if (code === 12288) {
    return ' ';
  }
  if (code >= 65281 && code <= 65374) {
    return String.fromCharCode(code - 65248);
  }
  return character;
}

/** Normalization used for every reference and prediction before Mandarin CER scoring. */
export function normalizeAsrText(text: string): string {
  const withoutTags: string = text.replace(/<[^>]*>/g, '');
  let result: string = '';
  for (let i: number = 0; i < withoutTags.length; i++) {
    const character: string = normalizeCharacter(withoutTags.charAt(i)).toLowerCase();
    if (/\s/.test(character) || REMOVED_PUNCTUATION.indexOf(character) >= 0) {
      continue;
    }
    result += character;
  }
  return result;
}

function chooseBetter(current: EditCell, candidate: EditCell): EditCell {
  if (candidate.distance < current.distance) {
    return candidate;
  }
  if (candidate.distance > current.distance) {
    return current;
  }
  // Stable tie-breaking makes the displayed S/D/I counts reproducible.
  if (candidate.substitutions > current.substitutions) {
    return candidate;
  }
  if (candidate.substitutions < current.substitutions) {
    return current;
  }
  if (candidate.deletions < current.deletions) {
    return candidate;
  }
  return current;
}

export function scoreAsrText(reference: string, hypothesis: string): AsrEditScore {
  const ref: string = normalizeAsrText(reference);
  const hyp: string = normalizeAsrText(hypothesis);
  const rows: EditCell[][] = [];
  for (let i: number = 0; i <= ref.length; i++) {
    const row: EditCell[] = [];
    for (let j: number = 0; j <= hyp.length; j++) {
      row.push({ distance: 0, substitutions: 0, deletions: 0, insertions: 0 });
    }
    rows.push(row);
  }
  for (let i: number = 1; i <= ref.length; i++) {
    rows[i][0] = { distance: i, substitutions: 0, deletions: i, insertions: 0 };
  }
  for (let j: number = 1; j <= hyp.length; j++) {
    rows[0][j] = { distance: j, substitutions: 0, deletions: 0, insertions: j };
  }

  for (let i: number = 1; i <= ref.length; i++) {
    for (let j: number = 1; j <= hyp.length; j++) {
      const diagonal: EditCell = rows[i - 1][j - 1];
      if (ref.charAt(i - 1) === hyp.charAt(j - 1)) {
        rows[i][j] = {
          distance: diagonal.distance,
          substitutions: diagonal.substitutions,
          deletions: diagonal.deletions,
          insertions: diagonal.insertions,
        };
        continue;
      }
      let best: EditCell = {
        distance: diagonal.distance + 1,
        substitutions: diagonal.substitutions + 1,
        deletions: diagonal.deletions,
        insertions: diagonal.insertions,
      };
      const deleted: EditCell = rows[i - 1][j];
      best = chooseBetter(best, {
        distance: deleted.distance + 1,
        substitutions: deleted.substitutions,
        deletions: deleted.deletions + 1,
        insertions: deleted.insertions,
      });
      const inserted: EditCell = rows[i][j - 1];
      best = chooseBetter(best, {
        distance: inserted.distance + 1,
        substitutions: inserted.substitutions,
        deletions: inserted.deletions,
        insertions: inserted.insertions + 1,
      });
      rows[i][j] = best;
    }
  }

  const result: EditCell = rows[ref.length][hyp.length];
  const denominator: number = ref.length > 0 ? ref.length : 1;
  return {
    referenceNormalized: ref,
    hypothesisNormalized: hyp,
    referenceChars: ref.length,
    substitutions: result.substitutions,
    deletions: result.deletions,
    insertions: result.insertions,
    errors: result.distance,
    cer: result.distance / denominator,
    exact: result.distance === 0,
  };
}

export function buildAsrEvaluationSummary(rows: AsrCaseResult[]): AsrEvaluationSummary {
  const summary: AsrEvaluationSummary = {
    evaluated: 0,
    completed: 0,
    failed: 0,
    exact: 0,
    referenceChars: 0,
    substitutions: 0,
    deletions: 0,
    insertions: 0,
    errors: 0,
    microCer: 0,
    macroCer: 0,
    audioDurationSec: 0,
    elapsedMs: 0,
    rtf: 0,
  };
  let macroCerTotal: number = 0;
  for (let i: number = 0; i < rows.length; i++) {
    const row: AsrCaseResult = rows[i];
    if (row.state !== 'completed' && row.state !== 'failed') {
      continue;
    }
    summary.evaluated += 1;
    if (row.state === 'completed') {
      summary.completed += 1;
    } else {
      summary.failed += 1;
    }
    if (row.score !== undefined) {
      summary.referenceChars += row.score.referenceChars;
      summary.substitutions += row.score.substitutions;
      summary.deletions += row.score.deletions;
      summary.insertions += row.score.insertions;
      summary.errors += row.score.errors;
      macroCerTotal += row.score.cer;
      if (row.state === 'completed' && row.score.exact) {
        summary.exact += 1;
      }
    }
    summary.audioDurationSec += row.audioDurationSec;
    summary.elapsedMs += row.elapsedMs;
  }
  summary.microCer = summary.referenceChars > 0 ? summary.errors / summary.referenceChars : 0;
  summary.macroCer = summary.evaluated > 0 ? macroCerTotal / summary.evaluated : 0;
  summary.rtf = summary.audioDurationSec > 0 ?
    (summary.elapsedMs / 1000) / summary.audioDurationSec : 0;
  return summary;
}

/** Marks only cross-speaker intersections as overlap; touching boundaries are not overlap. */
export function analyzeAsrTurns(turns: AsrTimedTurn[]): AsrAnalyzedTurn[] {
  const result: AsrAnalyzedTurn[] = [];
  for (let i: number = 0; i < turns.length; i++) {
    const current: AsrTimedTurn = turns[i];
    let overlapped: boolean = false;
    for (let j: number = 0; j < turns.length; j++) {
      if (i === j || current.speakerId === turns[j].speakerId) {
        continue;
      }
      if (current.startSec < turns[j].endSec && turns[j].startSec < current.endSec) {
        overlapped = true;
        break;
      }
    }
    result.push({
      speakerId: current.speakerId,
      startSec: current.startSec,
      endSec: current.endSec,
      text: current.text,
      overlapped,
    });
  }
  return result;
}

export function buildAsrTurnEvaluationSummary(rows: AsrTurnCaseResult[]): AsrTurnEvaluationSummary {
  const allRows: AsrCaseResult[] = [];
  const nonOverlapRows: AsrCaseResult[] = [];
  const overlapRows: AsrCaseResult[] = [];
  for (let i: number = 0; i < rows.length; i++) {
    const row: AsrTurnCaseResult = rows[i];
    const base: AsrCaseResult = {
      state: row.state,
      score: row.score,
      audioDurationSec: row.audioDurationSec,
      elapsedMs: row.elapsedMs,
    };
    allRows.push(base);
    if (row.overlapped) {
      overlapRows.push(base);
    } else {
      nonOverlapRows.push(base);
    }
  }
  return {
    all: buildAsrEvaluationSummary(allRows),
    nonOverlap: buildAsrEvaluationSummary(nonOverlapRows),
    overlap: buildAsrEvaluationSummary(overlapRows),
  };
}

function addEditCells(left: EditCell, right: EditCell): EditCell {
  return {
    distance: left.distance + right.distance,
    substitutions: left.substitutions + right.substitutions,
    deletions: left.deletions + right.deletions,
    insertions: left.insertions + right.insertions,
  };
}

function betterPermutation(current: PermutationCell | undefined,
  candidate: PermutationCell): PermutationCell {
  if (current === undefined) {
    return candidate;
  }
  const preferred: EditCell = chooseBetter(current, candidate);
  return preferred === candidate ? candidate : current;
}

function countBits(value: number): number {
  let count: number = 0;
  let remaining: number = value;
  while (remaining > 0) {
    count += remaining & 1;
    remaining = remaining >>> 1;
  }
  return count;
}

/**
 * Concatenated minimum-permutation CER for speaker-attributed transcripts.
 * Speaker labels themselves are deliberately ignored: the assignment with the fewest
 * character edits is selected, with empty speakers accounting for missed/extra streams.
 */
export function scoreSpeakerPermutation(reference: AsrSpeakerTranscript[],
  hypothesis: AsrSpeakerTranscript[]): AsrPermutationScore {
  const size: number = Math.max(reference.length, hypothesis.length);
  if (size === 0) {
    return {
      score: scoreAsrText('', ''),
      assignments: [],
    };
  }
  if (size > 12) {
    throw new Error('说话人数量超过端侧评测上限');
  }

  const pairScores: AsrEditScore[][] = [];
  for (let i: number = 0; i < size; i++) {
    const row: AsrEditScore[] = [];
    const referenceText: string = i < reference.length ? reference[i].text : '';
    for (let j: number = 0; j < size; j++) {
      const hypothesisText: string = j < hypothesis.length ? hypothesis[j].text : '';
      row.push(scoreAsrText(referenceText, hypothesisText));
    }
    pairScores.push(row);
  }

  const stateCount: number = 1 << size;
  const states: Array<PermutationCell | undefined> = new Array(stateCount);
  states[0] = {
    distance: 0,
    substitutions: 0,
    deletions: 0,
    insertions: 0,
    assignment: [],
  };
  for (let mask: number = 0; mask < stateCount; mask++) {
    const current: PermutationCell | undefined = states[mask];
    if (current === undefined) {
      continue;
    }
    const referenceIndex: number = countBits(mask);
    if (referenceIndex >= size) {
      continue;
    }
    for (let hypothesisIndex: number = 0; hypothesisIndex < size; hypothesisIndex++) {
      const bit: number = 1 << hypothesisIndex;
      if ((mask & bit) !== 0) {
        continue;
      }
      const pair: AsrEditScore = pairScores[referenceIndex][hypothesisIndex];
      const total: EditCell = addEditCells(current, {
        distance: pair.errors,
        substitutions: pair.substitutions,
        deletions: pair.deletions,
        insertions: pair.insertions,
      });
      const assignment: number[] = current.assignment.slice();
      assignment.push(hypothesisIndex);
      const candidate: PermutationCell = {
        distance: total.distance,
        substitutions: total.substitutions,
        deletions: total.deletions,
        insertions: total.insertions,
        assignment,
      };
      const nextMask: number = mask | bit;
      states[nextMask] = betterPermutation(states[nextMask], candidate);
    }
  }

  const finalCell: PermutationCell = states[stateCount - 1] as PermutationCell;
  const assignments: AsrSpeakerAssignment[] = [];
  let referenceChars: number = 0;
  let referenceNormalized: string = '';
  let hypothesisNormalized: string = '';
  for (let i: number = 0; i < size; i++) {
    const hypothesisIndex: number = finalCell.assignment[i];
    const pair: AsrEditScore = pairScores[i][hypothesisIndex];
    referenceChars += pair.referenceChars;
    if (referenceNormalized.length > 0) {
      referenceNormalized += '|';
      hypothesisNormalized += '|';
    }
    referenceNormalized += pair.referenceNormalized;
    hypothesisNormalized += pair.hypothesisNormalized;
    assignments.push({
      referenceSpeakerId: i < reference.length ? reference[i].speakerId : '',
      hypothesisSpeakerId: hypothesisIndex < hypothesis.length ? hypothesis[hypothesisIndex].speakerId : '',
      score: pair,
    });
  }
  const denominator: number = referenceChars > 0 ? referenceChars : 1;
  return {
    score: {
      referenceNormalized,
      hypothesisNormalized,
      referenceChars,
      substitutions: finalCell.substitutions,
      deletions: finalCell.deletions,
      insertions: finalCell.insertions,
      errors: finalCell.distance,
      cer: finalCell.distance / denominator,
      exact: finalCell.distance === 0,
    },
    assignments,
  };
}

/** Concatenates chronological turns belonging to the same anonymous speaker stream. */
export function groupSpeakerTranscripts(turns: AsrSpeakerTranscript[]): AsrSpeakerTranscript[] {
  const result: AsrSpeakerTranscript[] = [];
  const indexBySpeaker: Map<string, number> = new Map<string, number>();
  for (let i: number = 0; i < turns.length; i++) {
    const existing: number | undefined = indexBySpeaker.get(turns[i].speakerId);
    if (existing === undefined) {
      indexBySpeaker.set(turns[i].speakerId, result.length);
      result.push({ speakerId: turns[i].speakerId, text: turns[i].text });
    } else {
      result[existing].text += turns[i].text;
    }
  }
  return result;
}

function diarizationSpeakerIds(turns: AsrDiarizationTurn[]): string[] {
  const result: string[] = [];
  const seen: Set<string> = new Set<string>();
  for (let i: number = 0; i < turns.length; i++) {
    if (!seen.has(turns[i].speakerId)) {
      seen.add(turns[i].speakerId);
      result.push(turns[i].speakerId);
    }
  }
  return result;
}

function intervalOverlapSeconds(left: AsrDiarizationTurn, right: AsrDiarizationTurn): number {
  return Math.max(0, Math.min(left.endSec, right.endSec) - Math.max(left.startSec, right.startSec));
}

function speakerPairOverlap(reference: AsrDiarizationTurn[], hypothesis: AsrDiarizationTurn[],
  referenceSpeakerId: string, hypothesisSpeakerId: string): number {
  let total: number = 0;
  for (let i: number = 0; i < reference.length; i++) {
    if (reference[i].speakerId !== referenceSpeakerId) {
      continue;
    }
    for (let j: number = 0; j < hypothesis.length; j++) {
      if (hypothesis[j].speakerId === hypothesisSpeakerId) {
        total += intervalOverlapSeconds(reference[i], hypothesis[j]);
      }
    }
  }
  return total;
}

function bestDiarizationAssignment(reference: AsrDiarizationTurn[], hypothesis: AsrDiarizationTurn[]):
  AsrDiarizationSpeakerAssignment[] {
  const referenceIds: string[] = diarizationSpeakerIds(reference);
  const hypothesisIds: string[] = diarizationSpeakerIds(hypothesis);
  const size: number = Math.max(referenceIds.length, hypothesisIds.length);
  if (size === 0) {
    return [];
  }
  if (size > 12) {
    throw new Error('说话人数量超过端侧评测上限');
  }
  const overlap: number[][] = [];
  for (let i: number = 0; i < size; i++) {
    const row: number[] = [];
    for (let j: number = 0; j < size; j++) {
      row.push(i < referenceIds.length && j < hypothesisIds.length ?
        speakerPairOverlap(reference, hypothesis, referenceIds[i], hypothesisIds[j]) : 0);
    }
    overlap.push(row);
  }
  const stateCount: number = 1 << size;
  const states: Array<OverlapPermutationCell | undefined> = new Array(stateCount);
  states[0] = { overlapSec: 0, assignment: [] };
  for (let mask: number = 0; mask < stateCount; mask++) {
    const current: OverlapPermutationCell | undefined = states[mask];
    if (current === undefined) {
      continue;
    }
    const referenceIndex: number = countBits(mask);
    if (referenceIndex >= size) {
      continue;
    }
    for (let hypothesisIndex: number = 0; hypothesisIndex < size; hypothesisIndex++) {
      const bit: number = 1 << hypothesisIndex;
      if ((mask & bit) !== 0) {
        continue;
      }
      const assignment: number[] = current.assignment.slice();
      assignment.push(hypothesisIndex);
      const candidate: OverlapPermutationCell = {
        overlapSec: current.overlapSec + overlap[referenceIndex][hypothesisIndex],
        assignment,
      };
      const nextMask: number = mask | bit;
      if (states[nextMask] === undefined ||
        candidate.overlapSec > (states[nextMask] as OverlapPermutationCell).overlapSec) {
        states[nextMask] = candidate;
      }
    }
  }
  const finalCell: OverlapPermutationCell = states[stateCount - 1] as OverlapPermutationCell;
  const result: AsrDiarizationSpeakerAssignment[] = [];
  for (let i: number = 0; i < size; i++) {
    const hypothesisIndex: number = finalCell.assignment[i];
    result.push({
      referenceSpeakerId: i < referenceIds.length ? referenceIds[i] : '',
      hypothesisSpeakerId: hypothesisIndex < hypothesisIds.length ? hypothesisIds[hypothesisIndex] : '',
      overlapSec: overlap[i][hypothesisIndex],
    });
  }
  return result;
}

function activeDiarizationSpeakers(turns: AsrDiarizationTurn[], timeSec: number): Set<string> {
  const result: Set<string> = new Set<string>();
  for (let i: number = 0; i < turns.length; i++) {
    if (turns[i].startSec <= timeSec && timeSec < turns[i].endSec) {
      result.add(turns[i].speakerId);
    }
  }
  return result;
}

/**
 * No-collar DER with overlap included. This intentionally penalizes a single-stream
 * diarizer when the reference contains simultaneous speakers instead of hiding overlap.
 */
export function scoreSpeakerDiarization(reference: AsrDiarizationTurn[],
  hypothesis: AsrDiarizationTurn[]): AsrDiarizationScore {
  const assignments: AsrDiarizationSpeakerAssignment[] =
    bestDiarizationAssignment(reference, hypothesis);
  const referenceByHypothesis: Map<string, string> = new Map<string, string>();
  for (let i: number = 0; i < assignments.length; i++) {
    if (assignments[i].hypothesisSpeakerId.length > 0) {
      referenceByHypothesis.set(assignments[i].hypothesisSpeakerId, assignments[i].referenceSpeakerId);
    }
  }
  const boundaries: number[] = [];
  for (let i: number = 0; i < reference.length; i++) {
    boundaries.push(reference[i].startSec);
    boundaries.push(reference[i].endSec);
  }
  for (let i: number = 0; i < hypothesis.length; i++) {
    boundaries.push(hypothesis[i].startSec);
    boundaries.push(hypothesis[i].endSec);
  }
  boundaries.sort((left: number, right: number): number => left - right);
  const uniqueBoundaries: number[] = [];
  for (let i: number = 0; i < boundaries.length; i++) {
    if (uniqueBoundaries.length === 0 ||
      boundaries[i] > uniqueBoundaries[uniqueBoundaries.length - 1]) {
      uniqueBoundaries.push(boundaries[i]);
    }
  }

  let referenceSpeakerSec: number = 0;
  let missedSpeakerSec: number = 0;
  let falseAlarmSpeakerSec: number = 0;
  let confusionSpeakerSec: number = 0;
  let referenceOverlapSec: number = 0;
  let predictedOverlapSec: number = 0;
  let overlapTruePositiveSec: number = 0;
  for (let i: number = 0; i + 1 < uniqueBoundaries.length; i++) {
    const start: number = uniqueBoundaries[i];
    const end: number = uniqueBoundaries[i + 1];
    const duration: number = end - start;
    if (duration <= 0) {
      continue;
    }
    const middle: number = start + duration / 2;
    const referenceActive: Set<string> = activeDiarizationSpeakers(reference, middle);
    const hypothesisActive: Set<string> = activeDiarizationSpeakers(hypothesis, middle);
    const referenceCount: number = referenceActive.size;
    const hypothesisCount: number = hypothesisActive.size;
    let correctCount: number = 0;
    hypothesisActive.forEach((hypothesisSpeakerId: string): void => {
      const mapped: string | undefined = referenceByHypothesis.get(hypothesisSpeakerId);
      if (mapped !== undefined && mapped.length > 0 && referenceActive.has(mapped)) {
        correctCount += 1;
      }
    });
    referenceSpeakerSec += referenceCount * duration;
    missedSpeakerSec += Math.max(0, referenceCount - hypothesisCount) * duration;
    falseAlarmSpeakerSec += Math.max(0, hypothesisCount - referenceCount) * duration;
    confusionSpeakerSec += Math.max(0, Math.min(referenceCount, hypothesisCount) - correctCount) * duration;
    if (referenceCount >= 2) {
      referenceOverlapSec += duration;
    }
    if (hypothesisCount >= 2) {
      predictedOverlapSec += duration;
    }
    if (referenceCount >= 2 && hypothesisCount >= 2) {
      overlapTruePositiveSec += duration;
    }
  }
  const errors: number = missedSpeakerSec + falseAlarmSpeakerSec + confusionSpeakerSec;
  return {
    der: referenceSpeakerSec > 0 ? errors / referenceSpeakerSec : 0,
    referenceSpeakerSec,
    missedSpeakerSec,
    falseAlarmSpeakerSec,
    confusionSpeakerSec,
    referenceOverlapSec,
    predictedOverlapSec,
    overlapTruePositiveSec,
    overlapRecall: referenceOverlapSec > 0 ? overlapTruePositiveSec / referenceOverlapSec : 0,
    assignments,
  };
}

function timeInRanges(ranges: AsrTimeRange[], timeSec: number): boolean {
  for (let i: number = 0; i < ranges.length; i++) {
    if (ranges[i].startSec <= timeSec && timeSec < ranges[i].endSec) {
      return true;
    }
  }
  return false;
}

/** Scores Pyannote's raw overlap ranges separately from the single-stream final turns. */
export function scoreSpeakerOverlapDetection(reference: AsrDiarizationTurn[],
  predictedRanges: AsrTimeRange[]): AsrOverlapDetectionScore {
  const boundaries: number[] = [];
  for (let i: number = 0; i < reference.length; i++) {
    boundaries.push(reference[i].startSec);
    boundaries.push(reference[i].endSec);
  }
  for (let i: number = 0; i < predictedRanges.length; i++) {
    boundaries.push(predictedRanges[i].startSec);
    boundaries.push(predictedRanges[i].endSec);
  }
  boundaries.sort((left: number, right: number): number => left - right);
  let referenceOverlapSec: number = 0;
  let predictedOverlapSec: number = 0;
  let truePositiveSec: number = 0;
  for (let i: number = 0; i + 1 < boundaries.length; i++) {
    const start: number = boundaries[i];
    const end: number = boundaries[i + 1];
    const duration: number = end - start;
    if (duration <= 0) {
      continue;
    }
    const middle: number = start + duration / 2;
    const referenceOverlap: boolean = activeDiarizationSpeakers(reference, middle).size >= 2;
    const predictedOverlap: boolean = timeInRanges(predictedRanges, middle);
    if (referenceOverlap) {
      referenceOverlapSec += duration;
    }
    if (predictedOverlap) {
      predictedOverlapSec += duration;
    }
    if (referenceOverlap && predictedOverlap) {
      truePositiveSec += duration;
    }
  }
  return {
    referenceOverlapSec,
    predictedOverlapSec,
    truePositiveSec,
    recall: referenceOverlapSec > 0 ? truePositiveSec / referenceOverlapSec : 0,
  };
}

export function asrResultMatchesFilter(filter: AsrEvaluationFilter,
  state: AsrEvaluationState, score?: AsrEditScore): boolean {
  if (filter === 'errors') {
    return state === 'failed' || (state === 'completed' && score !== undefined && !score.exact);
  }
  return true;
}
