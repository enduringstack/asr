export type AsrEvaluationState = 'waiting' | 'running' | 'completed' | 'failed';
export type AsrEvaluationFilter = 'all' | 'errors';

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

export function asrResultMatchesFilter(filter: AsrEvaluationFilter,
  state: AsrEvaluationState, score?: AsrEditScore): boolean {
  if (filter === 'errors') {
    return state === 'failed' || (state === 'completed' && score !== undefined && !score.exact);
  }
  return true;
}
