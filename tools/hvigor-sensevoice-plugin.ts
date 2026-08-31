import type { HvigorPlugin } from '@ohos/hvigor';
import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';

export const senseVoiceModelPlugin: HvigorPlugin = {
  pluginId: 'asr.sensevoice-model',
  apply(node) {
    const projectRoot = node.getNodePath();
    execFileSync(
      process.execPath,
      [resolve(projectRoot, 'tools', 'prepare_sensevoice_model.mjs')],
      {
        cwd: projectRoot,
        stdio: 'inherit',
      },
    );
  },
};
