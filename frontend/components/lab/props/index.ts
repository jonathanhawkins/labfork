// Lab Props - 3D equipment for activity visualization
// All props are game-like visual elements that appear during activities

export * from './GPU3D';
export * from './Microphone3D';
export * from './Speaker3D';
export * from './ServerRack3D';
export * from './WaveformDisplay3D';
export * from './Supercomputer3D';
export * from './EmotionVerify3D';
export * from './ContributorDevices3D';

import * as THREE from 'three';
import { PropType } from '../activities';
import { createGPU3D, GPU3DRefs, GPU3DOptions, animateGPU3D, disposeGPU3D } from './GPU3D';
import { createMicrophone3D, Microphone3DRefs, Microphone3DOptions, animateMicrophone3D, disposeMicrophone3D } from './Microphone3D';
import { createSpeaker3D, Speaker3DRefs, Speaker3DOptions, animateSpeaker3D, disposeSpeaker3D } from './Speaker3D';
import { createServerRack3D, ServerRack3DRefs, ServerRack3DOptions, animateServerRack3D, disposeServerRack3D } from './ServerRack3D';
import { createWaveformDisplay3D, WaveformDisplay3DRefs, WaveformDisplay3DOptions, animateWaveformDisplay3D, disposeWaveformDisplay3D } from './WaveformDisplay3D';
import { createSupercomputer3D, Supercomputer3DRefs, Supercomputer3DOptions, animateSupercomputer3D, disposeSupercomputer3D } from './Supercomputer3D';
import { createEmotionVerify3D, EmotionVerify3DRefs, EmotionVerify3DOptions, animateEmotionVerify3D, disposeEmotionVerify3D } from './EmotionVerify3D';

// Union type for all prop refs
export type Prop3DRefs =
  | GPU3DRefs
  | Microphone3DRefs
  | Speaker3DRefs
  | ServerRack3DRefs
  | WaveformDisplay3DRefs
  | Supercomputer3DRefs
  | EmotionVerify3DRefs;

// Union type for all prop options
export type Prop3DOptions =
  | GPU3DOptions
  | Microphone3DOptions
  | Speaker3DOptions
  | ServerRack3DOptions
  | WaveformDisplay3DOptions
  | Supercomputer3DOptions
  | EmotionVerify3DOptions;

/**
 * Factory function to create any prop type
 */
export function createProp3D(
  type: PropType,
  options: {
    position: [number, number, number];
    scale?: number;
    accentColor?: number;
  }
): Prop3DRefs | null {
  switch (type) {
    case 'gpu':
      return createGPU3D(options as GPU3DOptions);
    case 'microphone':
      return createMicrophone3D(options as Microphone3DOptions);
    case 'speaker':
      return createSpeaker3D(options as Speaker3DOptions);
    case 'server':
      return createServerRack3D(options as ServerRack3DOptions);
    case 'waveform':
      return createWaveformDisplay3D(options as WaveformDisplay3DOptions);
    case 'supercomputer':
      return createSupercomputer3D(options as Supercomputer3DOptions);
    case 'emotion-verify':
      return createEmotionVerify3D(options as EmotionVerify3DOptions);
    case 'none':
    default:
      return null;
  }
}

/**
 * Factory function to animate any prop
 */
export function animateProp3D(
  type: PropType,
  refs: Prop3DRefs,
  time: number,
  options?: {
    progress?: number;
    isActive?: boolean;
    audioLevel?: number;
    loadLevel?: number;
  }
): void {
  switch (type) {
    case 'gpu':
      animateGPU3D(refs as GPU3DRefs, time, {
        gpuUtilization: options?.loadLevel ? options.loadLevel * 100 : undefined,
        progress: options?.progress,
      });
      break;
    case 'microphone':
      animateMicrophone3D(refs as Microphone3DRefs, time, {
        isRecording: options?.isActive,
        audioLevel: options?.audioLevel,
      });
      break;
    case 'speaker':
      animateSpeaker3D(refs as Speaker3DRefs, time, {
        isPlaying: options?.isActive,
        audioLevel: options?.audioLevel,
      });
      break;
    case 'server':
      animateServerRack3D(refs as ServerRack3DRefs, time, {
        isProcessing: options?.isActive,
        loadLevel: options?.loadLevel,
      });
      break;
    case 'waveform':
      animateWaveformDisplay3D(refs as WaveformDisplay3DRefs, time, {
        isActive: options?.isActive,
        intensity: options?.audioLevel,
      });
      break;
    case 'supercomputer':
      animateSupercomputer3D(refs as Supercomputer3DRefs, time, {
        isProcessing: options?.isActive,
        loadLevel: options?.loadLevel,
        progress: options?.progress,
      });
      break;
    case 'emotion-verify':
      animateEmotionVerify3D(refs as EmotionVerify3DRefs, time, {
        isVerifying: options?.isActive,
        verified: options?.progress === 100,
      });
      break;
  }
}

/**
 * Factory function to dispose any prop
 */
export function disposeProp3D(type: PropType, refs: Prop3DRefs): void {
  switch (type) {
    case 'gpu':
      disposeGPU3D(refs as GPU3DRefs);
      break;
    case 'microphone':
      disposeMicrophone3D(refs as Microphone3DRefs);
      break;
    case 'speaker':
      disposeSpeaker3D(refs as Speaker3DRefs);
      break;
    case 'server':
      disposeServerRack3D(refs as ServerRack3DRefs);
      break;
    case 'waveform':
      disposeWaveformDisplay3D(refs as WaveformDisplay3DRefs);
      break;
    case 'supercomputer':
      disposeSupercomputer3D(refs as Supercomputer3DRefs);
      break;
    case 'emotion-verify':
      disposeEmotionVerify3D(refs as EmotionVerify3DRefs);
      break;
  }
}

/**
 * Get the THREE.Group from any prop refs
 */
export function getProp3DGroup(refs: Prop3DRefs): THREE.Group {
  return refs.group;
}
