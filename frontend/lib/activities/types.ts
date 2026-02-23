/**
 * Activity Type Definitions
 *
 * Activities represent work being done in the lab by agents.
 * Each activity has visualization settings for the 3D scene.
 */

/**
 * Agent animation states
 */
export type AgentAnimation =
  | "idle"        // Standing still, small bob
  | "focused"     // Focused work, minimal movement
  | "typing"      // Rapid arm movement at workstation
  | "walking"     // Moving to target location
  | "celebrating" // Success animation
  | "thinking";   // Chin-scratch pose

/**
 * Particle effect types for activities
 */
export type ParticleEffect =
  | "none"        // No particles
  | "sparks"      // Small spark particles
  | "data-flow"   // Flowing data particles
  | "audio-waves" // Sound wave visualization
  | "code-rain"   // Matrix-style code rain
  | "stars"       // Sparkle stars
  | "smoke";      // Processing smoke

/**
 * Work location where agent should move for this activity
 */
export type WorkLocation =
  | "supercomputer"
  | "microphone"
  | "speaker"
  | "server"
  | "desk"
  | "center"
  | "custom";

/**
 * Visualization configuration for an activity
 */
export interface ActivityVisualization {
  /** 3D prop to spawn or highlight (optional) */
  prop?: string;
  /** Agent animation to use */
  animation: AgentAnimation;
  /** Particle effect to display */
  particles: ParticleEffect;
  /** Accent color for glow effects (hex number) */
  color: number;
  /** Where the agent should move to work */
  workLocation?: WorkLocation;
  /** Custom work position if workLocation is "custom" */
  customPosition?: [number, number, number];
  /** Whether to show progress bar */
  showProgress?: boolean;
  /** Whether to highlight the associated prop */
  highlightProp?: boolean;
}

/**
 * Agent behavior configuration for an activity
 */
export interface ActivityAgentBehavior {
  /** Default agent status during this activity */
  defaultStatus: "idle" | "working" | "thinking";
  /** Typing speed multiplier (1.0 = normal) */
  typingSpeed?: number;
  /** Whether agent should face the associated prop */
  faceProp?: boolean;
  /** Custom facing direction (radians) */
  facingDirection?: number;
}

/**
 * Progress detection patterns
 */
export interface ActivityProgressDetection {
  /** Regex patterns to detect progress in agent output */
  patterns: string[];
  /** Pattern to detect completion */
  completionPattern?: string;
  /** Message to show when completed */
  completionMessage?: string;
  /** Extract progress percentage from match groups */
  progressExtractor?: string;
}

/**
 * Complete activity configuration
 */
export interface ActivityConfig {
  /** Unique identifier for this activity type */
  id: string;
  /** Display name */
  name: string;
  /** Short description */
  description: string;
  /** Lucide icon name or custom SVG path */
  icon?: string;
  /** Visualization settings */
  visualization: ActivityVisualization;
  /** Agent behavior settings */
  agentBehavior: ActivityAgentBehavior;
  /** Progress detection settings */
  progressDetection?: ActivityProgressDetection;
  /** Tags for filtering/grouping */
  tags?: string[];
  /** Sort priority (lower = earlier) */
  priority?: number;
}

/**
 * Activity instance (a running activity)
 */
export interface ActivityInstance {
  /** Unique instance ID */
  id: string;
  /** Activity type ID */
  type: string;
  /** Activity configuration */
  config: ActivityConfig;
  /** Whether currently active */
  active: boolean;
  /** Progress percentage (0-100) */
  progress?: number;
  /** Status message */
  message?: string;
  /** Assigned agent ID */
  assignedAgent?: string;
  /** Start timestamp */
  startedAt?: string;
  /** Estimated completion time */
  estimatedCompletion?: string;
}

/**
 * Built-in activity definitions
 */
export const BUILTIN_ACTIVITIES: ActivityConfig[] = [
  {
    id: "training",
    name: "Model Training",
    description: "Fine-tuning the voice model",
    icon: "Brain",
    visualization: {
      prop: "supercomputer",
      animation: "focused",
      particles: "data-flow",
      color: 0x4ade80, // Green
      workLocation: "supercomputer",
      showProgress: true,
      highlightProp: true,
    },
    agentBehavior: {
      defaultStatus: "working",
      typingSpeed: 1.0,
      faceProp: true,
    },
    progressDetection: {
      patterns: [
        "Epoch\\s+(\\d+)/(\\d+)",
        "loss:\\s+([\\d.]+)",
        "step\\s+(\\d+)/(\\d+)",
      ],
      completionPattern: "Training complete|Saved checkpoint",
      completionMessage: "Training completed",
      progressExtractor: "($1 / $2) * 100",
    },
    tags: ["training", "ml"],
    priority: 1,
  },
  {
    id: "recording",
    name: "Voice Recording",
    description: "Recording voice samples",
    icon: "Mic",
    visualization: {
      prop: "microphone",
      animation: "focused",
      particles: "audio-waves",
      color: 0x4ecdc4, // Teal
      workLocation: "microphone",
      highlightProp: true,
    },
    agentBehavior: {
      defaultStatus: "working",
      faceProp: true,
    },
    tags: ["audio", "input"],
    priority: 2,
  },
  {
    id: "generation",
    name: "Speech Generation",
    description: "Generating synthesized speech",
    icon: "Sparkles",
    visualization: {
      prop: "speaker",
      animation: "focused",
      particles: "audio-waves",
      color: 0xffe66d, // Yellow
      workLocation: "speaker",
      highlightProp: true,
    },
    agentBehavior: {
      defaultStatus: "working",
      faceProp: true,
    },
    progressDetection: {
      patterns: ["Generating\\s+(\\d+)/(\\d+)"],
      completionPattern: "Generation complete|Saved audio",
    },
    tags: ["audio", "output"],
    priority: 3,
  },
  {
    id: "evaluation",
    name: "Model Evaluation",
    description: "Evaluating model performance",
    icon: "ChartBar",
    visualization: {
      animation: "thinking",
      particles: "sparks",
      color: 0x3b82f6, // Blue
      workLocation: "desk",
      showProgress: true,
    },
    agentBehavior: {
      defaultStatus: "thinking",
    },
    progressDetection: {
      patterns: ["Evaluating\\s+(\\d+)/(\\d+)", "Metric:\\s+([\\w]+)"],
      completionPattern: "Evaluation complete|Results saved",
    },
    tags: ["evaluation"],
    priority: 4,
  },
  {
    id: "research",
    name: "Web Research",
    description: "Searching for papers and techniques",
    icon: "Search",
    visualization: {
      animation: "typing",
      particles: "code-rain",
      color: 0xa855f7, // Purple
      workLocation: "desk",
    },
    agentBehavior: {
      defaultStatus: "thinking",
      typingSpeed: 0.8,
    },
    tags: ["research"],
    priority: 5,
  },
  {
    id: "implementation",
    name: "Code Implementation",
    description: "Writing and implementing code",
    icon: "Code",
    visualization: {
      animation: "typing",
      particles: "code-rain",
      color: 0x22c55e, // Green
      workLocation: "desk",
      showProgress: true,
    },
    agentBehavior: {
      defaultStatus: "working",
      typingSpeed: 1.2,
    },
    tags: ["coding"],
    priority: 6,
  },
  {
    id: "idle",
    name: "Idle",
    description: "Agent is idle",
    icon: "Pause",
    visualization: {
      animation: "idle",
      particles: "none",
      color: 0x94a3b8, // Gray
      workLocation: "center",
    },
    agentBehavior: {
      defaultStatus: "idle",
    },
    tags: ["idle"],
    priority: 99,
  },
];

/**
 * Get a built-in activity config by ID
 */
export function getBuiltinActivity(id: string): ActivityConfig | undefined {
  return BUILTIN_ACTIVITIES.find((a) => a.id === id);
}

/**
 * Merge activity config with defaults
 */
export function mergeActivityDefaults(
  partial: Partial<ActivityConfig>
): ActivityConfig {
  const idle = getBuiltinActivity("idle")!;

  return {
    id: partial.id || "unknown",
    name: partial.name || "Unknown Activity",
    description: partial.description || "",
    icon: partial.icon,
    visualization: {
      ...idle.visualization,
      ...partial.visualization,
    },
    agentBehavior: {
      ...idle.agentBehavior,
      ...partial.agentBehavior,
    },
    progressDetection: partial.progressDetection,
    tags: partial.tags,
    priority: partial.priority ?? 50,
  };
}

/**
 * Default export
 */
export default BUILTIN_ACTIVITIES;
