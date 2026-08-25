/**
 * Packages the recordings and their transcripts into one file the user can take away.
 *
 * The audio and the text are useless apart. A folder of WAVs cannot be reviewed, and a
 * transcript cannot be checked against anything — the whole reason to export is to sit
 * down with both and see whether the recogniser heard what it claimed. So the export is
 * a single archive in which the correspondence is explicit and survives being copied
 * around: each row of the index names the exact file it came from.
 *
 * One archive rather than many files is also the only workable shape for the save
 * dialog. Asking someone to confirm a destination once per segment would be unusable
 * after an afternoon of recording.
 *
 * Three views of the same data go in, because they answer different questions:
 *   transcript.txt   read it
 *   transcript.csv   sort, filter, or diff it in a spreadsheet
 *   index.json       process it
 * The CSV carries a byte-order mark, without which Excel renders Chinese as mojibake.
 */

import { fileIo } from '@kit.CoreFileKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { zlib } from '@kit.BasicServicesKit';
import { SegmentRecord } from './SegmentStore';
import { TranscriptRecord } from './TranscriptionQueue';
import { TimelineRow } from './ContinuousRecordingService';
import { SpeakerTurn } from '../../common/SpeakerTypes';

const DOMAIN = 0x0001;
const TAG = 'RecordingExporter';
const EXPORT_DIR = 'export';

export interface ExportResult {
  /** Sandbox path of the finished archive, ready to be copied to a user location. */
  archivePath: string;
  fileName: string;
  segments: number;
  bytes: number;
}

/*
 * The shape of index.json, spelled out.
 *
 * ArkTS rejects object literals that do not correspond to a declared type, so a
 * Record<string, ...> cannot be built inline. Declaring the shape is better anyway: this
 * file is the machine-readable half of the export, and anyone writing a script against it
 * now has the field names and types in one place.
 */
interface ExportedSegment {
  audio: string;
  startedAt: number;
  startedAtText: string;
  durationMs: number;
  transcribed: boolean;
  text: string;
  /** How long the recogniser held this segment, and which one it was. Zero and empty on
   * segments transcribed before those were recorded. Carried out of the sandbox because
   * whether recognition can keep pace with capture is only observable on the device. */
  decodeMs: number;
  engine: string;
  speakerTurns: ExportedSpeakerTurn[];
}

interface ExportedSpeakerTurn {
  clusterId: string;
  speaker: string;
  startMs: number;
  endMs: number;
  text: string;
  identity: string;
}

interface ExportManifest {
  exportedAt: number;
  segments: ExportedSegment[];
}

export class RecordingExporter {
  /**
   * Builds the archive and returns where it landed.
   *
   * Everything happens inside the sandbox: zlib only operates there, and the caller
   * copies the result out through whatever destination the user picks.
   */
  async build(filesDir: string, rows: TimelineRow[]): Promise<ExportResult> {
    if (rows.length === 0) {
      throw new Error('没有可导出的录音');
    }
    const stamp: string = this.stampNow();
    const name: string = `asr_export_${stamp}`;
    const root: string = `${filesDir}/${EXPORT_DIR}`;
    const staging: string = `${root}/${name}`;
    const audioDir: string = `${staging}/audio`;
    const archivePath: string = `${root}/${name}.zip`;

    // A previous export that failed part-way would otherwise contribute stale files to
    // this one, since the archive is built from whatever the folder happens to contain.
    this.removeTree(staging);
    this.deleteQuietly(archivePath);
    this.ensureDir(root);
    this.ensureDir(staging);
    this.ensureDir(audioDir);

    // Oldest first: an export is read as a record of what happened, in the order it did.
    const ordered: TimelineRow[] = rows.slice().sort(
      (a: TimelineRow, b: TimelineRow): number =>
        a.segment.startedAt - b.segment.startedAt);

    const copied: TimelineRow[] = [];
    let totalBytes: number = 0;
    for (let i: number = 0; i < ordered.length; i++) {
      const row: TimelineRow = ordered[i];
      const target: string = `${audioDir}/${this.audioName(row.segment)}`;
      if (this.copyFile(row.segment.path, target)) {
        copied.push(row);
        totalBytes += row.segment.bytes;
      } else {
        // A segment whose audio has already been pruned still has a transcript worth
        // keeping, but the index must not point at a file that is not in the archive.
        hilog.warn(DOMAIN, TAG, `skipping missing audio for ${row.segment.id}`);
      }
    }
    if (copied.length === 0) {
      this.removeTree(staging);
      throw new Error('录音文件已不存在，无法导出');
    }

    this.writeText(`${staging}/transcript.txt`, this.buildReadable(copied));
    this.writeText(`${staging}/transcript.csv`, this.buildCsv(copied));
    this.writeText(`${staging}/index.json`, this.buildJson(copied));

    try {
      await zlib.compressFile(staging, archivePath, { level: zlib.CompressLevel.COMPRESS_LEVEL_DEFAULT_COMPRESSION });
    } catch (error) {
      this.removeTree(staging);
      throw new Error(`打包失败: ${error}`);
    }
    // The staging tree has served its purpose; leaving it would double the space the
    // export costs on a device that is still recording.
    this.removeTree(staging);

    let archiveBytes: number = 0;
    try {
      archiveBytes = fileIo.statSync(archivePath).size;
    } catch (_error) {
      archiveBytes = totalBytes;
    }
    hilog.info(DOMAIN, TAG,
      `export ready ${archivePath} segments=${copied.length} bytes=${archiveBytes}`);
    const result: ExportResult = {
      archivePath: archivePath,
      fileName: `${name}.zip`,
      segments: copied.length,
      bytes: archiveBytes,
    };
    return result;
  }

  /** Removes archives from earlier exports, which are pure duplication once saved. */
  cleanup(filesDir: string): void {
    const root: string = `${filesDir}/${EXPORT_DIR}`;
    try {
      if (!fileIo.accessSync(root)) {
        return;
      }
      const names: string[] = fileIo.listFileSync(root);
      for (let i: number = 0; i < names.length; i++) {
        this.deleteQuietly(`${root}/${names[i]}`);
      }
    } catch (error) {
      hilog.warn(DOMAIN, TAG, `cleanup failed: ${error}`);
    }
  }

  /**
   * Names the audio after its start time.
   *
   * The internal id is a millisecond epoch, which sorts correctly but tells a reader
   * nothing. A name carrying the wall-clock time means the file is identifiable on its
   * own, even if it gets separated from the index.
   */
  private audioName(segment: SegmentRecord): string {
    const date = new Date(segment.startedAt);
    const pad = (value: number): string => value < 10 ? `0${value}` : `${value}`;
    return `${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}` +
      `_${segment.id}.wav`;
  }

  private buildReadable(rows: TimelineRow[]): string {
    const lines: string[] = [];
    lines.push(`录音转录导出  ${this.readableStamp(new Date())}`);
    lines.push(`共 ${rows.length} 段`);
    lines.push('');
    for (let i: number = 0; i < rows.length; i++) {
      const row: TimelineRow = rows[i];
      const transcript: TranscriptRecord | null = row.transcript;
      const seconds: string = (row.segment.durationMs / 1000).toFixed(1);
      lines.push(`[${this.clock(row.segment.startedAt)}] (${seconds}s) ` +
        `${this.audioName(row.segment)}`);
      lines.push(this.textOf(transcript));
      lines.push('');
    }
    return lines.join('\n');
  }

  private buildCsv(rows: TimelineRow[]): string {
    const lines: string[] = [];
    // Excel decides the encoding from this mark; without it Chinese text arrives broken
    // and the export looks like it lost the data.
    lines.push('﻿音频文件,开始时间,时长(秒),转录文本');
    for (let i: number = 0; i < rows.length; i++) {
      const row: TimelineRow = rows[i];
      lines.push([
        this.csvCell(this.audioName(row.segment)),
        this.csvCell(this.readableStamp(new Date(row.segment.startedAt))),
        this.csvCell((row.segment.durationMs / 1000).toFixed(1)),
        this.csvCell(this.textOf(row.transcript)),
      ].join(','));
    }
    return lines.join('\n');
  }

  private buildJson(rows: TimelineRow[]): string {
    const items: ExportedSegment[] = [];
    for (let i: number = 0; i < rows.length; i++) {
      const row: TimelineRow = rows[i];
      // Re-bound so the field reads below do not each have to re-prove it is not null.
      const transcript: TranscriptRecord | null = row.transcript;
      const item: ExportedSegment = {
        audio: `audio/${this.audioName(row.segment)}`,
        startedAt: row.segment.startedAt,
        startedAtText: this.readableStamp(new Date(row.segment.startedAt)),
        durationMs: row.segment.durationMs,
        transcribed: transcript !== null,
        text: transcript !== null ? transcript.text : '',
        decodeMs: transcript !== null ? transcript.decodeMs : 0,
        engine: transcript !== null ? transcript.engine : '',
        speakerTurns: transcript !== null ? this.exportTurns(transcript.speakerTurns) : [],
      };
      items.push(item);
    }
    const manifest: ExportManifest = { exportedAt: Date.now(), segments: items };
    return JSON.stringify(manifest, null, 2);
  }

  /** Export display information only; personId, scores, candidates and embeddings stay local. */
  private exportTurns(turns?: SpeakerTurn[]): ExportedSpeakerTurn[] {
    const result: ExportedSpeakerTurn[] = [];
    if (!turns) {
      return result;
    }
    for (let i: number = 0; i < turns.length; i++) {
      const turn: SpeakerTurn = turns[i];
      result.push({
        clusterId: turn.clusterId,
        speaker: turn.displayName || `未知说话人 ${turn.clusterId}`,
        startMs: turn.startMs,
        endMs: turn.endMs,
        text: turn.text,
        identity: turn.identity,
      });
    }
    return result;
  }

  private textOf(transcript: TranscriptRecord | null): string {
    if (transcript === null) {
      return '（尚未转录）';
    }
    return transcript.text.length > 0 ? transcript.text : '（未识别到内容）';
  }

  /** RFC 4180: quote always, and double any quote inside. Transcripts contain commas. */
  private csvCell(value: string): string {
    return `"${value.split('"').join('""')}"`;
  }

  private clock(timestamp: number): string {
    const date = new Date(timestamp);
    const pad = (value: number): string => value < 10 ? `0${value}` : `${value}`;
    return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  }

  private readableStamp(date: Date): string {
    const pad = (value: number): string => value < 10 ? `0${value}` : `${value}`;
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ` +
      `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  }

  private stampNow(): string {
    const date = new Date();
    const pad = (value: number): string => value < 10 ? `0${value}` : `${value}`;
    return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}` +
      `_${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
  }

  private copyFile(from: string, to: string): boolean {
    try {
      if (!fileIo.accessSync(from)) {
        return false;
      }
      fileIo.copyFileSync(from, to);
      return true;
    } catch (error) {
      hilog.warn(DOMAIN, TAG, `copy ${from} failed: ${error}`);
      return false;
    }
  }

  private writeText(path: string, content: string): void {
    try {
      const file: fileIo.File = fileIo.openSync(
        path, fileIo.OpenMode.CREATE | fileIo.OpenMode.TRUNC | fileIo.OpenMode.WRITE_ONLY);
      fileIo.writeSync(file.fd, content);
      fileIo.fsyncSync(file.fd);
      fileIo.closeSync(file);
    } catch (error) {
      hilog.error(DOMAIN, TAG, `write ${path} failed: ${error}`);
    }
  }

  private ensureDir(path: string): void {
    try {
      if (!fileIo.accessSync(path)) {
        fileIo.mkdirSync(path, true);
      }
    } catch (error) {
      hilog.error(DOMAIN, TAG, `mkdir ${path} failed: ${error}`);
    }
  }

  private removeTree(path: string): void {
    try {
      if (fileIo.accessSync(path)) {
        fileIo.rmdirSync(path);
      }
    } catch (_error) {
      // Never existed, or already gone; either way there is nothing to clean up.
    }
  }

  private deleteQuietly(path: string): void {
    try {
      if (fileIo.accessSync(path)) {
        fileIo.unlinkSync(path);
      }
    } catch (_error) {
      // Same as above — absence is the desired end state.
    }
  }
}
