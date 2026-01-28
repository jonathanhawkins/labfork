// Activity System - Core types and registry
// Connects real backend activities to 3D visualizations

export type PropType =
  | 'gpu'
  | 'microphone'
  | 'speaker'
  | 'server'
  | 'waveform'
  | 'supercomputer'
  | 'emotion-verify'
  | 'none';

export type AgentAnimation =
  | 'idle'
  | 'working'
  | 'thinking'
  | 'excited'
  | 'focused';

export type ParticleEffect =
  | 'data-flow'
  | 'audio-wave'
  | 'sparks'
  | 'glow'
  | 'none';

/**
 * Activity state from the backend
 */
export interface ActivityState {
  id: string;
  type: ActivityType['id'];
  active: boolean;
  progress?: number;        // 0-100
  metrics?: Record<string, number | string>;
  message?: string;
  startedAt?: string;
  assignedAgent?: string;   // Which 3D agent should interact
}

/**
 * Activity type definition
 * Defines how an activity maps to 3D visualization
 */
export interface ActivityType {
  id: string;
  name: string;
  description: string;
  prop: PropType;
  propPosition: [number, number, number];
  propScale?: number;
  agentAnimation: AgentAnimation;
  particleEffect: ParticleEffect;
  color: number;            // Accent color for the prop
  priority: number;         // Higher = more important (determines agent assignment)
}

/**
 * Combined activity data for rendering
 */
export interface ActivityWithConfig extends ActivityState {
  config: ActivityType;
}

// Re-export everything
export * from './activityConfigs';
export * from './useLabActivities';
