import { common } from '@kit.AbilityKit';
import { promptAction, router } from '@kit.ArkUI';
import { SpeakerProfile } from '../common/SpeakerTypes';
import { MeetingSpeakerStore } from '../services/speaker/MeetingSpeakerStore';
import { SpeakerProfileStore } from '../services/speaker/SpeakerProfileStore';

@Entry
@Component
struct SpeakerProfilesPage {
  @State profiles: SpeakerProfile[] = [];
  @State statusMessage: string = '正在读取本机声纹库...';
  @State editingPersonId: string = '';
  @State editingName: string = '';
  @State pendingDeleteId: string = '';

  private store: SpeakerProfileStore = SpeakerProfileStore.getInstance();
  private meetingStore: MeetingSpeakerStore | undefined = undefined;

  aboutToAppear(): void {
    const context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;
    this.meetingStore = new MeetingSpeakerStore(context);
    this.store.init(context)
      .then((): void => {
        this.refresh();
        this.statusMessage = this.profiles.length > 0
          ? `本机已保存 ${this.profiles.length} 位人物`
          : '还没有人物声纹';
      })
      .catch((error: Error): void => {
        this.statusMessage = `声纹库读取失败: ${error.message}`;
      });
  }

  private refresh(): void {
    this.profiles = this.store.list();
  }

  private stateText(profile: SpeakerProfile): string {
    if (profile.state === 'needsReRegister') {
      return '需要重新录制';
    }
    if (profile.state === 'stable') {
      return '稳定';
    }
    return '候选，建议跨会议再确认一次';
  }

  private roleText(profile: SpeakerProfile): string {
    return profile.role === 'self' ? '本人' : '参会者';
  }

  private durationText(ms: number): string {
    return `${(ms / 1000).toFixed(1)} 秒`;
  }

  private beginRename(profile: SpeakerProfile): void {
    this.editingPersonId = profile.personId;
    this.editingName = profile.displayName;
    this.pendingDeleteId = '';
  }

  private saveRename(): void {
    if (this.editingPersonId.length === 0 || this.editingName.trim().length === 0) {
      return;
    }
    this.store.rename(this.editingPersonId, this.editingName)
      .then((): void => {
        this.editingPersonId = '';
        this.editingName = '';
        this.refresh();
        this.statusMessage = '姓名已更新，历史和后续会议都会显示新名称';
      })
      .catch((error: Error): void => {
        this.statusMessage = `改名失败: ${error.message}`;
      });
  }

  private deleteProfile(profile: SpeakerProfile): void {
    if (this.pendingDeleteId !== profile.personId) {
      this.pendingDeleteId = profile.personId;
      this.editingPersonId = '';
      this.statusMessage = `再次点击“确认删除”将删除 ${profile.displayName} 的本机声纹`;
      return;
    }
    this.store.remove(profile.personId)
      .then((): Promise<void> => {
        return this.meetingStore ? this.meetingStore.clearPerson(profile.personId) :
          Promise.resolve<void>();
      })
      .then((): void => {
        this.pendingDeleteId = '';
        this.refresh();
        this.statusMessage = '人物声纹已从本机删除';
      })
      .catch((error: Error): void => {
        this.statusMessage = `删除失败: ${error.message}`;
      });
  }

  @Builder
  private ProfileCard(profile: SpeakerProfile) {
    Column() {
      Row() {
        Column() {
          Text(profile.displayName)
            .fontSize(17)
            .fontWeight(FontWeight.Medium)
            .fontColor('#1F2937')
          Text(`${this.roleText(profile)} · ${this.stateText(profile)}`)
            .fontSize(12)
            .fontColor(profile.state === 'needsReRegister' ? '#B45309' : '#6B7280')
            .margin({ top: 3 })
        }
        .alignItems(HorizontalAlign.Start)
        .layoutWeight(1)

        Button('改名')
          .fontSize(12)
          .height(32)
          .backgroundColor('#EAF2FF')
          .fontColor('#0066CC')
          .onClick((): void => {
            this.beginRename(profile);
          })

        Button(this.pendingDeleteId === profile.personId ? '确认删除' : '删除')
          .fontSize(12)
          .height(32)
          .backgroundColor(this.pendingDeleteId === profile.personId ? '#FEE2E2' : '#F5F5F5')
          .fontColor('#B91C1C')
          .margin({ left: 8 })
          .onClick((): void => {
            this.deleteProfile(profile);
          })
      }
      .width('100%')

      Row() {
        Text(`${profile.sampleCount} 个模板`)
          .fontSize(12)
          .fontColor('#6B7280')
        Text(`累计 ${this.durationText(profile.totalSpeechMs)}`)
          .fontSize(12)
          .fontColor('#6B7280')
          .margin({ left: 16 })
        Text(`${profile.sourceSessionCount} 个来源`)
          .fontSize(12)
          .fontColor('#6B7280')
          .margin({ left: 16 })
      }
      .width('100%')
      .margin({ top: 10 })

      if (this.editingPersonId === profile.personId) {
        Row() {
          TextInput({ text: this.editingName, placeholder: '输入新姓名' })
            .height(38)
            .layoutWeight(1)
            .onChange((value: string): void => {
              this.editingName = value;
            })
          Button('保存')
            .height(38)
            .fontSize(13)
            .margin({ left: 8 })
            .backgroundColor('#0066CC')
            .enabled(this.editingName.trim().length > 0)
            .onClick((): void => {
              this.saveRename();
            })
        }
        .width('100%')
        .margin({ top: 10 })
      }

      if (profile.state === 'needsReRegister') {
        Text(profile.role === 'self'
          ? '返回首页完成三次录音后即可重新启用识别。'
          : '请在会议说话人编辑页重新选择“记住此人”。')
          .fontSize(12)
          .fontColor('#92400E')
          .margin({ top: 10 })
      }
    }
    .width('100%')
    .padding(14)
    .backgroundColor('#FFFFFF')
    .borderRadius(12)
    .margin({ bottom: 10 })
  }

  build() {
    Column() {
      Row() {
        Button('返回')
          .fontSize(13)
          .height(34)
          .backgroundColor('#F3F4F6')
          .fontColor('#374151')
          .onClick((): void => {
            router.back();
          })
        Text('人物与声纹')
          .fontSize(20)
          .fontWeight(FontWeight.Bold)
          .fontColor('#111827')
          .margin({ left: 12 })
        Blank()
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 16, bottom: 12 })

      Column() {
        Text('声纹只用于本机会议标记')
          .fontSize(15)
          .fontWeight(FontWeight.Medium)
          .fontColor('#4C1D95')
        Text('模板由设备密钥加密，不上传，也不会随录音和转录导出。')
          .fontSize(12)
          .fontColor('#6D28D9')
          .margin({ top: 4 })
      }
      .alignItems(HorizontalAlign.Start)
      .width('100%')
      .padding(14)
      .backgroundColor('#F5F3FF')
      .borderRadius(12)
      .margin({ left: 16, right: 16, bottom: 10 })

      Text(this.statusMessage)
        .fontSize(13)
        .fontColor('#6B7280')
        .width('100%')
        .padding({ left: 18, right: 18, bottom: 8 })

      if (this.profiles.length === 0) {
        Column() {
          Text('还没有已保存的人物')
            .fontSize(17)
            .fontColor('#374151')
          Text('本人可在首页录制三次完成注册；其他人可在会议转录后标记并记住。')
            .fontSize(13)
            .fontColor('#6B7280')
            .textAlign(TextAlign.Center)
            .margin({ top: 8 })
        }
        .layoutWeight(1)
        .justifyContent(FlexAlign.Center)
        .padding(30)
      } else {
        List() {
          ForEach(this.profiles, (profile: SpeakerProfile): void => {
            ListItem() {
              this.ProfileCard(profile)
            }
          }, (profile: SpeakerProfile): string => profile.personId)
        }
        .layoutWeight(1)
        .width('100%')
        .padding({ left: 16, right: 16 })
      }

      Button('返回首页录制本人声纹')
        .width('80%')
        .height(46)
        .backgroundColor('#0066CC')
        .fontColor('#FFFFFF')
        .margin({ left: 16, right: 16, top: 8, bottom: 18 })
        .onClick((): void => {
          promptAction.showToast({ message: '请在首页“我的声纹”区域完成三次录音' });
          router.back();
        })
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#F7F8FA')
  }
}
