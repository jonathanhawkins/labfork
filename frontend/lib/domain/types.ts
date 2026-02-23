/**
 * Domain Plugin Type Definitions
 *
 * This module defines the schema for domain plugins that allow the AI Research Lab
 * Platform to support multiple research domains (voice cloning, trading, robotics, etc.)
 */

/**
 * Background style options for 3D scene
 */
export type BackgroundStyle =
  | 'sky'      // Soft sky gradient (default for voice-clone)
  | 'space'    // Dark space with stars
  | 'grid'     // Cyberpunk grid floor
  | 'gradient' // Custom gradient
  | 'minimal'; // Plain white

/**
 * Domain branding configuration
 */
export interface DomainBranding {
  /** Primary brand color (hex, e.g., "#3b82f6") */
  primaryColor: string;
  /** Accent color for highlights (hex) */
  accentColor: string;
  /** Background style for 3D scene */
  backgroundStyle: BackgroundStyle;
  /** Optional logo URL or path */
  logo?: string;
  /** Optional custom background color (hex) */
  backgroundColor?: string;
  /** Optional gradient colors for 'gradient' backgroundStyle */
  gradientColors?: [string, string];
}

/**
 * 3D prop definition for domain scenes
 */
export interface DomainProp {
  /** Unique identifier for this prop instance */
  id: string;
  /** Prop type from the built-in prop library */
  type: 'supercomputer' | 'microphone' | 'speaker' | 'server' | 'waveform' |
        'emotion-verify' | 'chart-wall' | 'robot-arm' | 'camera' | 'terminal' |
        'molecule' | 'custom';
  /** Position in 3D space [x, y, z] */
  position: [number, number, number];
  /** Scale multiplier (default: 1) */
  scale?: number;
  /** Y-axis rotation in radians */
  rotation?: number;
  /** Optional accent color override (hex number) */
  accentColor?: number;
  /** For custom props: path to prop definition */
  customPath?: string;
}

/**
 * 3D scene configuration
 */
export interface DomainScene {
  /** Props to render in the scene */
  props: DomainProp[];
  /** Decorations like plants, floating cubes */
  decorations?: {
    plants?: boolean;
    floatingCubes?: boolean;
    particles?: boolean;
  };
  /** Lighting configuration */
  lighting?: {
    ambientIntensity?: number;
    mainLightIntensity?: number;
    mainLightColor?: string;
  };
  /** Fog configuration */
  fog?: {
    enabled?: boolean;
    near?: number;
    far?: number;
  };
}

/**
 * Evaluation metric definition
 */
export interface EvaluationMetric {
  /** Unique identifier for the metric */
  id: string;
  /** Display name */
  name: string;
  /** Description of what the metric measures */
  description?: string;
  /** Expected value range [min, max] */
  range?: [number, number];
  /** Whether higher values are better */
  higherIsBetter: boolean;
  /** Unit of measurement (e.g., "Hz", "%", "ms") */
  unit?: string;
  /** How to compute this metric (for reference) */
  computation?: string;
}

/**
 * Research configuration
 */
export interface DomainResearch {
  /** arXiv categories to search (e.g., ["cs.SD", "cs.CL"]) */
  arxivCategories: string[];
  /** Keywords for web searches */
  keywords: string[];
  /** Additional search sources */
  additionalSources?: ('semantic-scholar' | 'github' | 'papers-with-code')[];
  /** Max papers to auto-ingest per research session */
  maxPapersPerSession?: number;
}

/**
 * Evaluation configuration
 */
export interface DomainEvaluation {
  /** Available metrics for this domain */
  metrics: EvaluationMetric[];
  /** Primary metric ID for sorting/comparison */
  primaryMetric: string;
  /** Whether to run baseline comparisons */
  baselineComparison?: boolean;
  /** Path to baseline results file */
  baselinePath?: string;
}

/**
 * Hardware requirements
 */
export interface DomainHardware {
  /** Minimum GPU VRAM in GB */
  minGpuVram?: number;
  /** Recommended GPU VRAM in GB */
  recommendedGpuVram?: number;
  /** Minimum system RAM in GB */
  minRam?: number;
  /** Whether GPU is required */
  gpuRequired?: boolean;
  /** Supported platforms */
  platforms?: ('darwin' | 'linux' | 'win32')[];
}

/**
 * Prompt templates configuration
 */
export interface DomainPrompts {
  /** Path to research agent prompt template */
  research?: string;
  /** Path to implementation agent prompt template */
  implementation?: string;
  /** Path to evaluation agent prompt template */
  evaluation?: string;
  /** Preamble to add to all prompts */
  preamble?: string;
}

/**
 * Complete domain configuration
 */
export interface DomainConfig {
  /** Human-readable domain name */
  name: string;
  /** URL-safe slug (e.g., "voice-clone") */
  slug: string;
  /** Short description */
  description: string;
  /** Longer description for marketing/onboarding */
  longDescription?: string;
  /** Difficulty level for users */
  difficulty?: 'beginner' | 'intermediate' | 'advanced';
  /** Visual branding */
  branding: DomainBranding;
  /** 3D scene configuration */
  scene: DomainScene;
  /** Research configuration */
  research: DomainResearch;
  /** Evaluation configuration */
  evaluation?: DomainEvaluation;
  /** Hardware requirements */
  hardware?: DomainHardware;
  /** Prompt templates */
  prompts?: DomainPrompts;
  /** Optional tags for discovery */
  tags?: string[];
  /** Version of this config (for migrations) */
  version?: string;
}

/**
 * Default values for domain configuration
 */
export const DEFAULT_DOMAIN_CONFIG: Partial<DomainConfig> = {
  difficulty: 'intermediate',
  branding: {
    primaryColor: '#3b82f6',
    accentColor: '#22c55e',
    backgroundStyle: 'sky',
  },
  scene: {
    props: [],
    decorations: {
      plants: true,
      floatingCubes: true,
      particles: true,
    },
    lighting: {
      ambientIntensity: 0.6,
      mainLightIntensity: 0.8,
      mainLightColor: '#fff5ee',
    },
    fog: {
      enabled: true,
      near: 10,
      far: 30,
    },
  },
  research: {
    arxivCategories: ['cs.LG'],
    keywords: ['machine learning'],
    maxPapersPerSession: 5,
  },
  hardware: {
    gpuRequired: false,
    minRam: 8,
  },
};

/**
 * Type guard to check if an object is a valid DomainConfig
 */
export function isDomainConfig(obj: unknown): obj is DomainConfig {
  if (!obj || typeof obj !== 'object') return false;
  const config = obj as Record<string, unknown>;

  // Check required fields
  if (typeof config.name !== 'string') return false;
  if (typeof config.slug !== 'string') return false;
  if (typeof config.description !== 'string') return false;

  // Check branding
  if (!config.branding || typeof config.branding !== 'object') return false;
  const branding = config.branding as Record<string, unknown>;
  if (typeof branding.primaryColor !== 'string') return false;
  if (typeof branding.accentColor !== 'string') return false;

  // Check scene
  if (!config.scene || typeof config.scene !== 'object') return false;
  const scene = config.scene as Record<string, unknown>;
  if (!Array.isArray(scene.props)) return false;

  // Check research
  if (!config.research || typeof config.research !== 'object') return false;
  const research = config.research as Record<string, unknown>;
  if (!Array.isArray(research.arxivCategories)) return false;
  if (!Array.isArray(research.keywords)) return false;

  return true;
}

/**
 * Merge partial config with defaults
 */
export function mergeWithDefaults(partial: Partial<DomainConfig>): DomainConfig {
  return {
    name: partial.name || 'Unnamed Lab',
    slug: partial.slug || 'unnamed',
    description: partial.description || 'A research lab',
    difficulty: partial.difficulty || DEFAULT_DOMAIN_CONFIG.difficulty!,
    branding: {
      ...DEFAULT_DOMAIN_CONFIG.branding!,
      ...partial.branding,
    },
    scene: {
      ...DEFAULT_DOMAIN_CONFIG.scene!,
      ...partial.scene,
      decorations: {
        ...DEFAULT_DOMAIN_CONFIG.scene!.decorations,
        ...partial.scene?.decorations,
      },
      lighting: {
        ...DEFAULT_DOMAIN_CONFIG.scene!.lighting,
        ...partial.scene?.lighting,
      },
      fog: {
        ...DEFAULT_DOMAIN_CONFIG.scene!.fog,
        ...partial.scene?.fog,
      },
    },
    research: {
      ...DEFAULT_DOMAIN_CONFIG.research!,
      ...partial.research,
    },
    evaluation: partial.evaluation,
    hardware: {
      ...DEFAULT_DOMAIN_CONFIG.hardware,
      ...partial.hardware,
    },
    prompts: partial.prompts,
    tags: partial.tags,
    version: partial.version || '1.0',
  };
}

/**
 * Convert hex string color to number for Three.js
 */
export function hexToNumber(hex: string): number {
  return parseInt(hex.replace('#', ''), 16);
}

/**
 * Convert number to hex string
 */
export function numberToHex(num: number): string {
  return '#' + num.toString(16).padStart(6, '0');
}
