/**
 * Activity Visualization Utilities
 *
 * Maps activity configurations to 3D scene properties for Lab3D rendering.
 */

import {
  ActivityConfig,
  ActivityVisualization,
  AgentAnimation,
  ParticleEffect,
  WorkLocation,
} from './types';

/**
 * Work location positions in the 3D scene
 * These correspond to prop positions in the Lab3D scene
 */
export const WORK_LOCATION_POSITIONS: Record<WorkLocation, [number, number, number]> = {
  supercomputer: [-6, 0, -5],
  microphone: [-6, 0, 5],
  speaker: [6, 0, 5],
  server: [6, 0, -5],
  desk: [0, 0, 0],
  center: [0, 0, 0],
  custom: [0, 0, 0],
};

/**
 * Animation speed multipliers for different animations
 */
export const ANIMATION_SPEEDS: Record<AgentAnimation, number> = {
  idle: 0.5,
  focused: 1.0,
  typing: 2.0,
  walking: 1.5,
  celebrating: 1.2,
  thinking: 0.7,
};

/**
 * Particle effect configurations for Three.js
 */
export interface ParticleConfig {
  count: number;
  size: number;
  speed: number;
  spread: number;
  color: number;
  opacity: number;
  lifetime: number;
}

/**
 * Default particle configurations by effect type
 */
export const PARTICLE_CONFIGS: Record<ParticleEffect, ParticleConfig> = {
  none: {
    count: 0,
    size: 0,
    speed: 0,
    spread: 0,
    color: 0xffffff,
    opacity: 0,
    lifetime: 0,
  },
  sparks: {
    count: 50,
    size: 0.05,
    speed: 2.0,
    spread: 1.0,
    color: 0xffaa00,
    opacity: 0.8,
    lifetime: 1.0,
  },
  'data-flow': {
    count: 100,
    size: 0.03,
    speed: 1.5,
    spread: 0.5,
    color: 0x00ffaa,
    opacity: 0.6,
    lifetime: 2.0,
  },
  'audio-waves': {
    count: 30,
    size: 0.1,
    speed: 0.5,
    spread: 2.0,
    color: 0x4ecdc4,
    opacity: 0.4,
    lifetime: 1.5,
  },
  'code-rain': {
    count: 80,
    size: 0.04,
    speed: 3.0,
    spread: 1.5,
    color: 0x22c55e,
    opacity: 0.7,
    lifetime: 0.8,
  },
  stars: {
    count: 40,
    size: 0.08,
    speed: 0.3,
    spread: 3.0,
    color: 0xffffff,
    opacity: 0.9,
    lifetime: 3.0,
  },
  smoke: {
    count: 20,
    size: 0.2,
    speed: 0.2,
    spread: 0.8,
    color: 0x888888,
    opacity: 0.3,
    lifetime: 4.0,
  },
};

/**
 * 3D visualization settings derived from activity config
 */
export interface ActivityVisualization3D {
  /** Position where agent should move */
  targetPosition: [number, number, number];
  /** Animation state name */
  animation: AgentAnimation;
  /** Animation speed multiplier */
  animationSpeed: number;
  /** Whether to show typing effect on workstation */
  showTyping: boolean;
  /** Particle effect configuration */
  particles: ParticleConfig;
  /** Accent color for glow effects */
  accentColor: number;
  /** Prop ID to highlight (if any) */
  highlightProp?: string;
  /** Whether to show progress bar */
  showProgress: boolean;
}

/**
 * Convert activity config to 3D visualization settings
 */
export function activityTo3DConfig(
  activity: ActivityConfig,
  customPosition?: [number, number, number]
): ActivityVisualization3D {
  const viz = activity.visualization;

  // Determine target position
  let targetPosition: [number, number, number];
  if (customPosition) {
    targetPosition = customPosition;
  } else if (viz.customPosition && viz.workLocation === 'custom') {
    targetPosition = viz.customPosition;
  } else if (viz.workLocation) {
    targetPosition = WORK_LOCATION_POSITIONS[viz.workLocation];
  } else {
    targetPosition = WORK_LOCATION_POSITIONS.center;
  }

  // Get animation settings
  const animation = viz.animation;
  const animationSpeed = ANIMATION_SPEEDS[animation];

  // Determine if typing should be shown
  const showTyping = animation === 'typing' || animation === 'focused';

  // Get particle config with color override
  const baseParticles = PARTICLE_CONFIGS[viz.particles];
  const particles: ParticleConfig = {
    ...baseParticles,
    color: viz.color || baseParticles.color,
  };

  return {
    targetPosition,
    animation,
    animationSpeed,
    showTyping,
    particles,
    accentColor: viz.color,
    highlightProp: viz.highlightProp ? viz.prop : undefined,
    showProgress: viz.showProgress ?? false,
  };
}

/**
 * Get glow material properties for an activity color
 */
export function getGlowMaterialProps(color: number, intensity: number = 1.0) {
  return {
    color,
    emissive: color,
    emissiveIntensity: 0.3 * intensity,
    transparent: true,
    opacity: 0.8,
  };
}

/**
 * Interpolate between two colors
 */
export function lerpColor(color1: number, color2: number, t: number): number {
  const r1 = (color1 >> 16) & 0xff;
  const g1 = (color1 >> 8) & 0xff;
  const b1 = color1 & 0xff;

  const r2 = (color2 >> 16) & 0xff;
  const g2 = (color2 >> 8) & 0xff;
  const b2 = color2 & 0xff;

  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const b = Math.round(b1 + (b2 - b1) * t);

  return (r << 16) | (g << 8) | b;
}

/**
 * Get animation state from activity status
 */
export function getAnimationFromStatus(
  status: 'idle' | 'working' | 'thinking'
): AgentAnimation {
  switch (status) {
    case 'working':
      return 'focused';
    case 'thinking':
      return 'thinking';
    default:
      return 'idle';
  }
}

/**
 * Calculate progress ring geometry
 */
export function getProgressRingGeometry(
  progress: number,
  radius: number = 0.5,
  thickness: number = 0.05
): { startAngle: number; endAngle: number; innerRadius: number; outerRadius: number } {
  const normalizedProgress = Math.max(0, Math.min(100, progress)) / 100;
  return {
    startAngle: -Math.PI / 2,
    endAngle: -Math.PI / 2 + normalizedProgress * Math.PI * 2,
    innerRadius: radius - thickness / 2,
    outerRadius: radius + thickness / 2,
  };
}

export default activityTo3DConfig;
