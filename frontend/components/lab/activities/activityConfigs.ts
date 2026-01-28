// Activity Configurations
// Maps activity types to their 3D visualization properties

import { ActivityType } from './index';

/**
 * Demo prop positions in the Lab3D scene (corners)
 * These match the positions in Lab3D.tsx createDemoProps
 */
const DEMO_PROP_POSITIONS = {
  supercomputer: [-6, 0, -5] as [number, number, number],  // Back-left corner
  microphone: [-6, 0, 5] as [number, number, number],      // Front-left corner
  speaker: [6, 0, 5] as [number, number, number],          // Front-right corner
  server: [6, 0, -5] as [number, number, number],          // Back-right corner
};

/**
 * Registry of all activity types with their visual configurations
 * propPosition directs agents to the corresponding demo prop in the scene
 */
export const ACTIVITY_REGISTRY: ActivityType[] = [
  {
    id: 'training',
    name: 'Model Training',
    description: 'Fine-tuning or training a model',
    prop: 'none',  // Don't create new prop - use demo supercomputer
    propPosition: DEMO_PROP_POSITIONS.supercomputer,  // Go to supercomputer
    propScale: 3.0,
    agentAnimation: 'focused',
    particleEffect: 'sparks',
    color: 0xff6b6b,  // Warm red for heat
    priority: 100,
  },
  {
    id: 'recording',
    name: 'Voice Recording',
    description: 'Recording voice samples',
    prop: 'none',  // Don't create new prop - use demo microphone
    propPosition: DEMO_PROP_POSITIONS.microphone,  // Go to microphone
    propScale: 2.5,
    agentAnimation: 'working',
    particleEffect: 'audio-wave',
    color: 0x4ecdc4,  // Teal for audio
    priority: 90,
  },
  {
    id: 'generation',
    name: 'Speech Generation',
    description: 'Generating speech from text',
    prop: 'none',  // Don't create new prop - use demo speaker
    propPosition: DEMO_PROP_POSITIONS.speaker,  // Go to speaker
    propScale: 2.5,
    agentAnimation: 'excited',
    particleEffect: 'audio-wave',
    color: 0xffe66d,  // Yellow for output
    priority: 80,
  },
  {
    id: 'live-transform',
    name: 'Live Voice Transform',
    description: 'Real-time voice transformation',
    prop: 'none',
    propPosition: DEMO_PROP_POSITIONS.supercomputer,  // Go to supercomputer
    propScale: 2.5,
    agentAnimation: 'focused',
    particleEffect: 'data-flow',
    color: 0xa855f7,  // Purple for magic
    priority: 85,
  },
  {
    id: 'inference',
    name: 'Model Inference',
    description: 'Running model inference',
    prop: 'none',  // Don't create new prop - use demo server
    propPosition: DEMO_PROP_POSITIONS.server,  // Go to server
    propScale: 2.5,
    agentAnimation: 'working',
    particleEffect: 'data-flow',
    color: 0x3b82f6,  // Blue for computing
    priority: 70,
  },
  {
    id: 'processing',
    name: 'Audio Processing',
    description: 'Processing audio files',
    prop: 'none',
    propPosition: DEMO_PROP_POSITIONS.supercomputer,  // Go to supercomputer
    propScale: 2.0,
    agentAnimation: 'thinking',
    particleEffect: 'glow',
    color: 0x22d3ee,  // Cyan for processing
    priority: 60,
  },
  {
    id: 'task',
    name: 'Agent Task',
    description: 'Claude agent working on a task',
    prop: 'none',
    propPosition: DEMO_PROP_POSITIONS.supercomputer,  // Tasks go to supercomputer
    agentAnimation: 'working',
    particleEffect: 'glow',
    color: 0x818cf8,  // Indigo for tasks
    priority: 50,
  },
];

/**
 * Get activity config by ID
 */
export function getActivityConfig(id: string): ActivityType | undefined {
  return ACTIVITY_REGISTRY.find(a => a.id === id);
}

/**
 * Get default activity config for unknown activity types
 */
export function getDefaultActivityConfig(id: string): ActivityType {
  return {
    id,
    name: 'Unknown Activity',
    description: 'Unknown activity type',
    prop: 'none',
    propPosition: [0, 0, 0],
    agentAnimation: 'idle',
    particleEffect: 'none',
    color: 0x64748b,
    priority: 0,
  };
}

/**
 * Prop position offsets for multiple simultaneous activities
 */
export const PROP_OFFSETS: Record<number, [number, number, number][]> = {
  1: [[0, 0, 0]],
  2: [[-1, 0, 0], [1, 0, 0]],
  3: [[-1.5, 0, 0], [0, 0, 0], [1.5, 0, 0]],
  4: [[-1.5, 0, -0.5], [1.5, 0, -0.5], [-1.5, 0, 0.5], [1.5, 0, 0.5]],
};

/**
 * Get offset for prop based on how many activities are active
 */
export function getPropOffset(index: number, total: number): [number, number, number] {
  const offsets = PROP_OFFSETS[Math.min(total, 4)] || PROP_OFFSETS[4];
  return offsets[index % offsets.length] || [0, 0, 0];
}
