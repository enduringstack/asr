#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  createReadStream,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const TOOLS_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(TOOLS_DIRECTORY, '..');
const MODEL_NAME = 'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17';
const SOURCE_URL =
  'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/' +
  `${MODEL_NAME}.tar.bz2`;
const TARGET_MODEL = join(
  PROJECT_ROOT,
  'entry',
  'src',
  'main',
  'resources',
  'rawfile',
  MODEL_NAME,
  'model.int8.onnx',
);
const PATCH_PATH = join(TOOLS_DIRECTORY, 'sensevoice-model.patch.json');
const CACHE_DIRECTORY = join(PROJECT_ROOT, '.hvigor', 'sensevoice-model');
const CACHED_ARCHIVE = join(CACHE_DIRECTORY, `${MODEL_NAME}.tar.bz2`);
const EXTRACT_DIRECTORY = join(CACHE_DIRECTORY, 'source');
const SOURCE_MODEL = join(EXTRACT_DIRECTORY, MODEL_NAME, 'model.int8.onnx');
const SOURCE_MEMBER = `${MODEL_NAME}/model.int8.onnx`;

async function sha256File(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) {
    hash.update(chunk);
  }
  return hash.digest('hex');
}

function run(command, args) {
  const result = spawnSync(command, args, {
    cwd: PROJECT_ROOT,
    stdio: 'inherit',
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`${command} exited with status ${result.status}`);
  }
}

function patchModel(source, patch) {
  if (patch.format !== 'sensevoice-copy-add-v1') {
    throw new Error(`Unsupported patch format: ${patch.format}`);
  }
  const output = Buffer.allocUnsafe(patch.newSize);
  const extra = Buffer.alloc(patch.extraSize);
  for (const run of patch.extraRuns) {
    Buffer.from(run.data, 'base64').copy(extra, run.offset);
  }

  let oldPosition = 0;
  let newPosition = 0;
  let diffPosition = 0;
  let extraPosition = 0;
  let mutationIndex = 0;

  for (const [diffLength, extraLength, oldSeek] of patch.controls) {
    if (oldPosition < 0 || oldPosition + diffLength > source.length) {
      throw new Error('Patch attempted to read outside the source model');
    }
    source.copy(
      output,
      newPosition,
      oldPosition,
      oldPosition + diffLength,
    );
    while (
      mutationIndex < patch.diffMutations.length &&
      patch.diffMutations[mutationIndex][0] < diffPosition + diffLength
    ) {
      const [position, value] = patch.diffMutations[mutationIndex];
      if (position < diffPosition) {
        throw new Error('Patch mutations are not ordered');
      }
      const outputIndex = newPosition + position - diffPosition;
      output[outputIndex] = (output[outputIndex] + value) & 0xff;
      mutationIndex += 1;
    }
    newPosition += diffLength;
    oldPosition += diffLength;
    diffPosition += diffLength;

    extra.copy(
      output,
      newPosition,
      extraPosition,
      extraPosition + extraLength,
    );
    newPosition += extraLength;
    extraPosition += extraLength;
    oldPosition += oldSeek;
  }

  if (
    newPosition !== patch.newSize ||
    extraPosition !== patch.extraSize ||
    mutationIndex !== patch.diffMutations.length
  ) {
    throw new Error('Patch did not consume the expected output data');
  }
  return output;
}

export async function prepareSenseVoiceModel(options = {}) {
  const patch = JSON.parse(readFileSync(PATCH_PATH, 'utf8'));
  const force = options.force === true;
  if (
    !force &&
    existsSync(TARGET_MODEL) &&
    (await sha256File(TARGET_MODEL)) === patch.outputSha256
  ) {
    console.log('[SenseVoice] Verified packaged model.');
    return TARGET_MODEL;
  }

  mkdirSync(CACHE_DIRECTORY, { recursive: true });
  mkdirSync(EXTRACT_DIRECTORY, { recursive: true });
  const archiveOverride = process.env.SENSEVOICE_ARCHIVE_PATH;
  const archive = archiveOverride ? resolve(archiveOverride) : CACHED_ARCHIVE;
  if (!existsSync(archive)) {
    const temporaryArchive = `${archive}.download-${process.pid}`;
    console.log('[SenseVoice] Downloading the official INT8 model...');
    run('curl', [
      '--fail',
      '--location',
      '--retry',
      '5',
      '--retry-all-errors',
      '--output',
      temporaryArchive,
      SOURCE_URL,
    ]);
    renameSync(temporaryArchive, archive);
  }

  run('tar', ['-xjf', archive, '-C', EXTRACT_DIRECTORY, SOURCE_MEMBER]);
  const sourceHash = await sha256File(SOURCE_MODEL);
  if (sourceHash !== patch.sourceSha256) {
    throw new Error(
      `Official SenseVoice source hash mismatch: ${sourceHash}`,
    );
  }

  console.log('[SenseVoice] Applying verified emotion-token patch...');
  const output = patchModel(readFileSync(SOURCE_MODEL), patch);
  const outputHash = createHash('sha256').update(output).digest('hex');
  if (outputHash !== patch.outputSha256) {
    throw new Error(`Packaged SenseVoice hash mismatch: ${outputHash}`);
  }

  mkdirSync(dirname(TARGET_MODEL), { recursive: true });
  const temporaryModel = `${TARGET_MODEL}.write-${process.pid}`;
  try {
    writeFileSync(temporaryModel, output);
    renameSync(temporaryModel, TARGET_MODEL);
  } finally {
    if (existsSync(temporaryModel)) {
      unlinkSync(temporaryModel);
    }
  }
  console.log(`[SenseVoice] Prepared ${TARGET_MODEL}`);
  return TARGET_MODEL;
}

if (resolve(process.argv[1] ?? '') === fileURLToPath(import.meta.url)) {
  await prepareSenseVoiceModel({ force: process.argv.includes('--force') });
}
