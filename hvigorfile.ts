import { appTasks } from '@ohos/hvigor-ohos-plugin';
import { senseVoiceModelPlugin } from './tools/hvigor-sensevoice-plugin';

export default {
  system: appTasks, /* Built-in plugin of Hvigor. It cannot be modified. */
  plugins: [senseVoiceModelPlugin]
}
