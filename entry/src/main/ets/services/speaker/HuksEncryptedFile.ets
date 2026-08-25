import { cryptoFramework } from '@kit.CryptoArchitectureKit';
import { fileIo } from '@kit.CoreFileKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { util } from '@kit.ArkTS';
import { huks } from '@kit.UniversalKeystoreKit';

const DOMAIN: number = 0x0001;
const TAG: string = 'HuksEncryptedFile';
const KEY_ALIAS: string = 'asr.voiceprints.aes-gcm.v2';
const AAD_TEXT: string = 'ASR voiceprint profile schema v2';
const NONCE_BYTES: number = 12;
const AE_TAG_BYTES: number = 16;
const HUKS_CHUNK_BYTES: number = 48 * 1024;

interface EncryptedEnvelope {
  version: number;
  algorithm: string;
  nonce: string;
  ciphertext: string;
  authenticationTag: string;
}

/**
 * Device-bound encrypted file backed by a non-exportable HUKS AES-256-GCM key.
 *
 * Only the random nonce and authenticated ciphertext are written to disk. HUKS keeps
 * the key in the device keystore; copying the file to another device is insufficient to
 * recover biometric templates.
 */
export class HuksEncryptedFile {
  async writeText(path: string, plaintext: string): Promise<void> {
    await this.ensureKey();
    const nonce: Uint8Array = cryptoFramework.createRandom().generateRandomSync(NONCE_BYTES).data;
    const input: Uint8Array = new util.TextEncoder('utf-8').encodeInto(plaintext);
    const sealed: Uint8Array = await this.crypt(input, nonce, true);
    if (sealed.length < AE_TAG_BYTES) {
      throw new Error('HUKS did not return an AES-GCM authentication tag');
    }
    const ciphertext: Uint8Array = sealed.slice(0, sealed.length - AE_TAG_BYTES);
    const authenticationTag: Uint8Array = sealed.slice(sealed.length - AE_TAG_BYTES);
    const base64: util.Base64Helper = new util.Base64Helper();
    const envelope: EncryptedEnvelope = {
      version: 2,
      algorithm: 'HUKS-AES-256-GCM',
      nonce: base64.encodeToStringSync(nonce),
      ciphertext: base64.encodeToStringSync(ciphertext),
      authenticationTag: base64.encodeToStringSync(authenticationTag),
    };
    this.atomicWrite(path, JSON.stringify(envelope));
  }

  async readText(path: string): Promise<string> {
    await this.ensureKey();
    const raw: string = fileIo.readTextSync(path);
    const envelope: EncryptedEnvelope = JSON.parse(raw) as EncryptedEnvelope;
    if (envelope.version !== 2 || envelope.algorithm !== 'HUKS-AES-256-GCM') {
      throw new Error('unsupported encrypted voiceprint format');
    }
    const base64: util.Base64Helper = new util.Base64Helper();
    const nonce: Uint8Array = base64.decodeSync(envelope.nonce);
    const ciphertext: Uint8Array = base64.decodeSync(envelope.ciphertext);
    const authenticationTag: Uint8Array = base64.decodeSync(envelope.authenticationTag);
    if (nonce.length !== NONCE_BYTES || ciphertext.length === 0 ||
      authenticationTag.length !== AE_TAG_BYTES) {
      throw new Error('invalid encrypted voiceprint envelope');
    }
    const plaintext: Uint8Array = await this.crypt(
      ciphertext, nonce, false, authenticationTag);
    return new util.TextDecoder('utf-8').decode(plaintext);
  }

  exists(path: string): boolean {
    try {
      return fileIo.accessSync(path);
    } catch (_error) {
      return false;
    }
  }

  private async ensureKey(): Promise<void> {
    const empty: huks.HuksOptions = { properties: [] };
    if (await huks.isKeyItemExist(KEY_ALIAS, empty)) {
      return;
    }
    const options: huks.HuksOptions = {
      properties: [
        { tag: huks.HuksTag.HUKS_TAG_ALGORITHM, value: huks.HuksKeyAlg.HUKS_ALG_AES },
        {
          tag: huks.HuksTag.HUKS_TAG_PURPOSE,
          value: huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_ENCRYPT |
            huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_DECRYPT,
        },
        { tag: huks.HuksTag.HUKS_TAG_KEY_SIZE, value: huks.HuksKeySize.HUKS_AES_KEY_SIZE_256 },
        { tag: huks.HuksTag.HUKS_TAG_BLOCK_MODE, value: huks.HuksCipherMode.HUKS_MODE_GCM },
        { tag: huks.HuksTag.HUKS_TAG_PADDING, value: huks.HuksKeyPadding.HUKS_PADDING_NONE },
      ],
    };
    try {
      await huks.generateKeyItem(KEY_ALIAS, options);
    } catch (error) {
      // A concurrent page may have created the same key between the existence check and
      // generate call. Re-check before treating it as a real failure.
      if (!(await huks.isKeyItemExist(KEY_ALIAS, empty))) {
        throw error;
      }
    }
  }

  private operationOptions(nonce: Uint8Array, encrypt: boolean,
    authenticationTag?: Uint8Array): huks.HuksOptions {
    const purpose: huks.HuksKeyPurpose = encrypt
      ? huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_ENCRYPT
      : huks.HuksKeyPurpose.HUKS_KEY_PURPOSE_DECRYPT;
    const properties: huks.HuksParam[] = [
        { tag: huks.HuksTag.HUKS_TAG_ALGORITHM, value: huks.HuksKeyAlg.HUKS_ALG_AES },
        { tag: huks.HuksTag.HUKS_TAG_PURPOSE, value: purpose },
        { tag: huks.HuksTag.HUKS_TAG_BLOCK_MODE, value: huks.HuksCipherMode.HUKS_MODE_GCM },
        { tag: huks.HuksTag.HUKS_TAG_PADDING, value: huks.HuksKeyPadding.HUKS_PADDING_NONE },
        { tag: huks.HuksTag.HUKS_TAG_NONCE, value: nonce },
        {
          tag: huks.HuksTag.HUKS_TAG_ASSOCIATED_DATA,
          value: new util.TextEncoder('utf-8').encodeInto(AAD_TEXT),
        },
      ];
    if (!encrypt && authenticationTag) {
      properties.push({
        tag: huks.HuksTag.HUKS_TAG_AE_TAG,
        value: authenticationTag,
      });
    }
    return { properties: properties };
  }

  private async crypt(input: Uint8Array, nonce: Uint8Array, encrypt: boolean,
    authenticationTag?: Uint8Array): Promise<Uint8Array> {
    const initOptions: huks.HuksOptions = this.operationOptions(
      nonce, encrypt, authenticationTag);
    const session: huks.HuksSessionHandle = await huks.initSession(KEY_ALIAS, initOptions);
    const outputs: Uint8Array[] = [];
    try {
      let offset: number = 0;
      while (input.length - offset > HUKS_CHUNK_BYTES) {
        const chunk: Uint8Array = input.slice(offset, offset + HUKS_CHUNK_BYTES);
        const updated: huks.HuksReturnResult = await huks.updateSession(
          session.handle, { inData: chunk });
        this.collectOutput(outputs, updated.outData);
        offset += chunk.length;
      }
      const tail: Uint8Array = input.slice(offset);
      const finished: huks.HuksReturnResult = await huks.finishSession(
        session.handle, { inData: tail });
      this.collectOutput(outputs, finished.outData);
      return this.concat(outputs);
    } catch (error) {
      try {
        await huks.abortSession(session.handle, {});
      } catch (abortError) {
        hilog.warn(DOMAIN, TAG, `abort failed: ${abortError}`);
      }
      throw error;
    }
  }

  private collectOutput(outputs: Uint8Array[], value?: Uint8Array): void {
    if (!value || value.length === 0) {
      return;
    }
    const copy: Uint8Array = new Uint8Array(value.length);
    copy.set(value);
    outputs.push(copy);
  }

  private concat(chunks: Uint8Array[]): Uint8Array {
    let length: number = 0;
    for (let i: number = 0; i < chunks.length; i++) {
      length += chunks[i].length;
    }
    const result: Uint8Array = new Uint8Array(length);
    let offset: number = 0;
    for (let i: number = 0; i < chunks.length; i++) {
      result.set(chunks[i], offset);
      offset += chunks[i].length;
    }
    return result;
  }

  private atomicWrite(path: string, content: string): void {
    const tempPath: string = `${path}.tmp`;
    let file: fileIo.File | undefined = undefined;
    try {
      file = fileIo.openSync(tempPath,
        fileIo.OpenMode.CREATE | fileIo.OpenMode.TRUNC | fileIo.OpenMode.WRITE_ONLY);
      fileIo.writeSync(file.fd, content);
      fileIo.fsyncSync(file.fd);
      fileIo.closeSync(file);
      file = undefined;
      fileIo.renameSync(tempPath, path);
    } catch (error) {
      if (file) {
        try {
          fileIo.closeSync(file);
        } catch (_closeError) {
        }
      }
      throw error;
    }
  }
}
