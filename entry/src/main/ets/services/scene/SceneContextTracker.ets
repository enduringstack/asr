import { geoLocationManager } from '@kit.LocationKit';
import { SceneMotionEvidence } from '../../common/AcousticSceneFusion';

export interface SceneContextSummary extends SceneMotionEvidence {
  available: boolean;
  sampleCount: number;
}

interface SpeedSample {
  speedKmh: number;
  accuracyKmh: number;
  timeStamp: number;
}

/**
 * Collects only short-lived speed evidence while the user is recording.
 * Coordinates are intentionally neither retained nor exposed.
 */
export class SceneContextTracker {
  private active: boolean = false;
  private samples: SpeedSample[] = [];

  private readonly locationCallback = (location: geoLocationManager.Location): void => {
    if (!Number.isFinite(location.speed) || location.speed < 0 ||
      location.speedAccuracy === undefined || !Number.isFinite(location.speedAccuracy)) {
      return;
    }
    this.samples.push({
      speedKmh: location.speed * 3.6,
      accuracyKmh: Math.max(0, location.speedAccuracy * 3.6),
      timeStamp: location.timeStamp,
    });
    if (this.samples.length > 30) {
      this.samples.shift();
    }
  };

  start(): boolean {
    this.stop();
    this.samples = [];
    try {
      if (!geoLocationManager.isLocationEnabled()) {
        return false;
      }
      const request: geoLocationManager.ContinuousLocationRequest = {
        interval: 2,
        locationScenario: geoLocationManager.UserActivityScenario.TRANSPORT,
      };
      geoLocationManager.on('locationChange', request, this.locationCallback);
      this.active = true;
      return true;
    } catch (_error) {
      this.active = false;
      return false;
    }
  }

  stop(): SceneContextSummary {
    if (this.active) {
      try {
        geoLocationManager.off('locationChange', this.locationCallback);
      } catch (_error) {
        // A missing service or a user-disabled location switch must not block ASR.
      }
    }
    this.active = false;
    return this.summary();
  }

  summary(): SceneContextSummary {
    if (this.samples.length === 0) {
      return { available: false, sampleCount: 0 };
    }
    const orderedSpeed: number[] = this.samples.map((value: SpeedSample): number =>
      value.speedKmh).sort((left: number, right: number): number => left - right);
    const orderedAccuracy: number[] = this.samples.map((value: SpeedSample): number =>
      value.accuracyKmh).sort((left: number, right: number): number => left - right);
    const middle: number = Math.floor(orderedSpeed.length / 2);
    const latest: SpeedSample = this.samples[this.samples.length - 1];
    return {
      available: true,
      sampleCount: this.samples.length,
      speedKmh: orderedSpeed[middle],
      speedAccuracyKmh: orderedAccuracy[middle],
      locationAgeMs: Math.max(0, Date.now() - latest.timeStamp),
    };
  }
}
