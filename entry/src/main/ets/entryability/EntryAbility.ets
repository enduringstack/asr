import { AbilityConstant, ConfigurationConstant, UIAbility, Want } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { window } from '@kit.ArkUI';

const DOMAIN = 0x0000;
const TAG = 'ASREntry';

export default class EntryAbility extends UIAbility {
  private mainWindow?: window.Window;

  onCreate(want: Want, launchParam: AbilityConstant.LaunchParam): void {
    try {
      this.context.getApplicationContext().setColorMode(ConfigurationConstant.ColorMode.COLOR_MODE_NOT_SET);
    } catch (err) {
      hilog.error(DOMAIN, TAG, 'Failed to set colorMode. Cause: %{public}s', JSON.stringify(err));
    }
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onCreate');
  }

  onDestroy(): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onDestroy');
  }

  onWindowStageCreate(windowStage: window.WindowStage): void {
    // Main window is created, set main page for this ability
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onWindowStageCreate');

    windowStage.loadContent('pages/Index', (err) => {
      if (err.code) {
        hilog.error(DOMAIN, TAG, 'Failed to load the content. Cause: %{public}s', JSON.stringify(err));
        return;
      }
      hilog.info(DOMAIN, TAG, 'Succeeded in loading the content.');
      this.enableKeepScreenOn(windowStage);
    });
  }

  onWindowStageDestroy(): void {
    // Main window is destroyed, release UI related resources
    this.disableKeepScreenOn();
    this.mainWindow = undefined;
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onWindowStageDestroy');
  }

  onForeground(): void {
    // Ability has brought to foreground
    this.setKeepScreenOn(true);
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onForeground');
  }

  onBackground(): void {
    // Ability has back to background
    this.setKeepScreenOn(false);
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onBackground');
  }

  private async enableKeepScreenOn(windowStage: window.WindowStage): Promise<void> {
    try {
      this.mainWindow = await windowStage.getMainWindow();
      await this.mainWindow.setWindowKeepScreenOn(true);
      hilog.info(DOMAIN, TAG, 'Window keep screen on enabled.');
    } catch (err) {
      hilog.error(DOMAIN, TAG, 'Failed to enable keep screen on. Cause: %{public}s', JSON.stringify(err));
    }
  }

  private async disableKeepScreenOn(): Promise<void> {
    await this.setKeepScreenOn(false);
  }

  private async setKeepScreenOn(enabled: boolean): Promise<void> {
    if (!this.mainWindow) {
      return;
    }

    try {
      await this.mainWindow.setWindowKeepScreenOn(enabled);
      hilog.info(DOMAIN, TAG, 'Window keep screen on: %{public}s', enabled ? 'enabled' : 'disabled');
    } catch (err) {
      hilog.error(DOMAIN, TAG, 'Failed to set keep screen on. Cause: %{public}s', JSON.stringify(err));
    }
  }
}
