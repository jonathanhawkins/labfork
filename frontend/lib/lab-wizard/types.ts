/**
 * Lab Wizard Type Definitions
 *
 * Types for the multi-step lab creation wizard.
 */

import type { DomainConfig } from "@/lib/domain/types";

/**
 * Wizard step identifiers
 */
export type LabWizardStep =
  | "welcome"
  | "domain"
  | "hardware"
  | "research"
  | "review";

/**
 * Hardware type options
 */
export type HardwareType = "local" | "remote-ssh" | "cloud";

/**
 * Cloud provider options
 */
export type CloudProvider = "runpod" | "aws" | "gcp" | "lambda-labs";

/**
 * GPU information
 */
export interface GpuInfo {
  /** GPU name (e.g., "NVIDIA RTX 4090") */
  name: string;
  /** VRAM in GB */
  vram: number;
  /** CUDA version (e.g., "12.1") */
  cudaVersion?: string;
  /** Driver version */
  driverVersion?: string;
  /** Whether GPU is available */
  available: boolean;
  /** Compute capability */
  computeCapability?: string;
}

/**
 * SSH connection configuration
 */
export interface SSHConfig {
  /** Remote host address (IP or hostname) */
  host: string;
  /** SSH port (default: 22) */
  port?: number;
  /** Username for SSH connection */
  user: string;
  /** Path to SSH private key */
  keyPath?: string;
  /** Remote working directory */
  remoteDir?: string;
  /** Whether connection has been tested */
  tested?: boolean;
  /** Whether connection has been verified */
  verified?: boolean;
  /** Test result status */
  testStatus?: "success" | "failed" | "pending";
  /** Error message if failed */
  testError?: string;
  /** Detected GPU on remote machine */
  remoteGpu?: GpuInfo;
}

/**
 * Cloud provider configuration
 */
export interface CloudConfig {
  /** Selected cloud provider */
  provider: CloudProvider;
  /** API key for the provider */
  apiKey?: string;
  /** Region selection */
  region?: string;
  /** Instance type */
  instanceType?: string;
  /** Estimated hourly cost */
  hourlyRate?: number;
  /** Whether API key is valid */
  validated?: boolean;
}

/**
 * System information
 */
export interface SystemInfo {
  /** OS name */
  os?: string;
  /** Platform (darwin, linux, win32) */
  platform?: string;
  /** CPU architecture */
  arch?: string;
  /** Total RAM in GB */
  totalMemory?: number;
  /** Free RAM in GB */
  freeMemory?: number;
}

/**
 * Local hardware configuration
 */
export interface LocalConfig {
  /** Detected GPU info */
  gpu?: GpuInfo;
  /** System information */
  systemInfo?: SystemInfo;
  /** Whether Ollama is installed */
  ollamaInstalled?: boolean;
  /** Ollama version */
  ollamaVersion?: string;
  /** Available Ollama models */
  ollamaModels?: string[];
  /** System RAM in GB */
  systemRam?: number;
}

/**
 * Complete hardware configuration
 */
export interface HardwareConfig {
  /** Selected hardware type */
  type: HardwareType;
  /** Local hardware config (if type is 'local') */
  local?: LocalConfig;
  /** SSH config (if type is 'remote-ssh') */
  ssh?: SSHConfig;
  /** Cloud config (if type is 'cloud') */
  cloud?: CloudConfig;
  /** Claude API key (required for analysis) */
  claudeApiKey?: string;
  /** Whether Claude API key is validated */
  claudeApiKeyValid?: boolean;
}

/**
 * Research goal input
 */
export interface ResearchGoal {
  /** Goal description from user */
  description: string;
  /** Keywords for the research goal */
  keywords?: string[];
  /** Raw goal text from user (alias for description) */
  goalText?: string;
  /** AI-suggested domain slug */
  suggestedDomain?: string;
  /** AI-suggested arxiv categories */
  suggestedCategories?: string[];
  /** AI-suggested keywords */
  suggestedKeywords?: string[];
  /** AI-recommended papers to start with */
  recommendedPapers?: RecommendedPaper[];
  /** AI-generated initial tasks */
  initialTasks?: InitialTask[];
  /** Whether goal has been analyzed */
  analyzed?: boolean;
}

/**
 * Recommended paper from goal analysis
 */
export interface RecommendedPaper {
  /** Paper title */
  title: string;
  /** Paper authors */
  authors?: string;
  /** arXiv ID if available */
  arxivId?: string;
  /** Why this paper is relevant */
  reason?: string;
  /** Relevance score 0-100 */
  relevanceScore?: number;
}

/**
 * Initial task generated from goal
 */
export interface InitialTask {
  /** Task subject */
  subject: string;
  /** Task description */
  description: string;
  /** Task type */
  type: "research" | "implementation" | "evaluation" | "setup";
  /** Estimated hours */
  estimatedHours?: number;
  /** Priority */
  priority?: "high" | "medium" | "low";
}

/**
 * Paper references for initial research
 */
export interface ResearchPaperRef {
  /** Paper ID */
  id: string;
  /** Paper title */
  title: string;
  /** Source (arxiv, doi, etc.) */
  source: string;
  /** Source identifier */
  sourceId: string;
}

/**
 * Research configuration from wizard
 */
export interface ResearchConfig {
  /** Research setup path */
  path: "papers" | "goal";
  /** Papers added if path is 'papers' */
  papers?: ResearchPaperRef[];
  /** Research goal if path is 'goal' */
  goal?: ResearchGoal;
}

/**
 * Complete lab configuration
 */
export interface LabConfig {
  /** Domain configuration */
  domain?: Partial<DomainConfig>;
  /** Selected existing domain slug (if not creating new) */
  existingDomainSlug?: string;
  /** Whether creating new domain or using existing */
  createNewDomain: boolean;
  /** Hardware configuration */
  hardware: HardwareConfig;
  /** Research configuration */
  research: ResearchConfig;
  /** Agent budget limits */
  budget?: {
    /** Max tokens per day */
    maxTokensPerDay?: number;
    /** Max API spend per day in USD */
    maxSpendPerDay?: number;
    /** Preferred model for agents */
    preferredModel?: string;
  };
}

/**
 * Wizard validation state
 */
export interface WizardValidation {
  /** Whether current step is valid */
  isValid: boolean;
  /** Validation errors by field */
  errors: Record<string, string>;
  /** Warning messages */
  warnings?: string[];
}

/**
 * Complete wizard state
 */
export interface LabWizardState {
  /** Current wizard step */
  currentStep: LabWizardStep;
  /** Lab configuration being built */
  config: LabConfig;
  /** Validation state for current step */
  validation: WizardValidation;
  /** Whether wizard is in loading state */
  isLoading: boolean;
  /** Whether wizard has been saved (for continue later) */
  savedAt?: string;
  /** Session ID for save/restore */
  sessionId?: string;
}

/**
 * Wizard step definition
 */
export interface WizardStepDef {
  /** Step ID */
  id: LabWizardStep;
  /** Display title */
  title: string;
  /** Display label (shorter than title) */
  label: string;
  /** Short description */
  description: string;
  /** Estimated time to complete */
  estimatedMinutes?: number;
  /** Whether step is optional */
  optional?: boolean;
}

/**
 * Wizard steps configuration
 */
export const WIZARD_STEPS: WizardStepDef[] = [
  {
    id: "welcome",
    title: "Welcome",
    label: "Welcome",
    description: "Get started with your research lab",
    estimatedMinutes: 1,
  },
  {
    id: "domain",
    title: "Domain",
    label: "Domain",
    description: "Choose your research focus",
    estimatedMinutes: 2,
  },
  {
    id: "hardware",
    title: "Hardware",
    label: "Hardware",
    description: "Configure compute resources",
    estimatedMinutes: 3,
  },
  {
    id: "research",
    title: "Research",
    label: "Research",
    description: "Define your initial goals",
    estimatedMinutes: 3,
  },
  {
    id: "review",
    title: "Review",
    label: "Review",
    description: "Confirm and launch",
    estimatedMinutes: 1,
  },
];

/**
 * Cloud provider display info (as record)
 */
export const CLOUD_PROVIDERS_MAP: Record<CloudProvider, {
  name: string;
  description: string;
  gpuOptions: string[];
  baseUrl: string;
}> = {
  runpod: {
    name: "RunPod",
    description: "GPU cloud for AI/ML workloads",
    gpuOptions: ["RTX 4090", "A100", "H100"],
    baseUrl: "https://runpod.io",
  },
  aws: {
    name: "AWS",
    description: "Amazon Web Services EC2 GPU instances",
    gpuOptions: ["g4dn.xlarge", "p3.2xlarge", "p4d.24xlarge"],
    baseUrl: "https://aws.amazon.com",
  },
  gcp: {
    name: "Google Cloud",
    description: "Google Cloud Platform GPU instances",
    gpuOptions: ["n1-standard-4 + T4", "a2-highgpu-1g"],
    baseUrl: "https://cloud.google.com",
  },
  "lambda-labs": {
    name: "Lambda Labs",
    description: "GPU cloud built for deep learning",
    gpuOptions: ["1x A10", "1x A100", "8x A100"],
    baseUrl: "https://lambdalabs.com",
  },
};

/**
 * Cloud provider info for UI (as array)
 */
export interface CloudProviderInfo {
  id: CloudProvider;
  name: string;
  icon: string;
  description: string;
}

export const CLOUD_PROVIDERS: CloudProviderInfo[] = [
  { id: "runpod", name: "RunPod", icon: "🚀", description: "GPU cloud for AI/ML" },
  { id: "aws", name: "AWS", icon: "☁️", description: "Amazon EC2 GPU instances" },
  { id: "gcp", name: "GCP", icon: "🌐", description: "Google Cloud Platform" },
  { id: "lambda-labs", name: "Lambda", icon: "λ", description: "Lambda Labs GPU cloud" },
];

/**
 * Default lab configuration
 */
export const DEFAULT_LAB_CONFIG: LabConfig = {
  createNewDomain: false,
  hardware: {
    type: "local",
  },
  research: {
    path: "goal",
  },
};

/**
 * Default wizard state
 */
export const DEFAULT_WIZARD_STATE: LabWizardState = {
  currentStep: "welcome",
  config: DEFAULT_LAB_CONFIG,
  validation: {
    isValid: true,
    errors: {},
  },
  isLoading: false,
};

/**
 * Get step index from step ID
 */
export function getStepIndex(step: LabWizardStep): number {
  return WIZARD_STEPS.findIndex((s) => s.id === step);
}

/**
 * Get next step
 */
export function getNextStep(current: LabWizardStep): LabWizardStep | null {
  const index = getStepIndex(current);
  if (index < WIZARD_STEPS.length - 1) {
    return WIZARD_STEPS[index + 1].id;
  }
  return null;
}

/**
 * Get previous step
 */
export function getPrevStep(current: LabWizardStep): LabWizardStep | null {
  const index = getStepIndex(current);
  if (index > 0) {
    return WIZARD_STEPS[index - 1].id;
  }
  return null;
}

/**
 * Check if step is completed based on config
 */
export function isStepCompleted(
  step: LabWizardStep,
  config: LabConfig
): boolean {
  switch (step) {
    case "welcome":
      return true; // Welcome is always complete after viewing

    case "domain":
      return !!(config.existingDomainSlug || config.domain?.slug);

    case "hardware":
      if (config.hardware.type === "local") {
        return config.hardware.local?.gpu?.available === true;
      }
      if (config.hardware.type === "remote-ssh") {
        return config.hardware.ssh?.testStatus === "success";
      }
      if (config.hardware.type === "cloud") {
        return config.hardware.cloud?.validated === true;
      }
      return false;

    case "research":
      if (config.research.path === "papers") {
        return (config.research.papers?.length ?? 0) > 0;
      }
      return config.research.goal?.analyzed === true;

    case "review":
      return false; // Review is never "complete" until submitted

    default:
      return false;
  }
}

/**
 * Get total estimated time for wizard
 */
export function getEstimatedTime(): number {
  return WIZARD_STEPS.reduce(
    (total, step) => total + (step.estimatedMinutes || 0),
    0
  );
}

/**
 * Validate hardware configuration
 */
export function validateHardware(config: HardwareConfig): WizardValidation {
  const errors: Record<string, string> = {};
  const warnings: string[] = [];

  if (config.type === "local") {
    if (!config.local?.gpu?.available) {
      warnings.push("No GPU detected. Some features may be limited.");
    }
    if (!config.local?.ollamaInstalled) {
      warnings.push("Ollama not detected. Install for free local inference.");
    }
  }

  if (config.type === "remote-ssh") {
    if (!config.ssh?.host) {
      errors.host = "SSH host is required";
    }
    if (!config.ssh?.user) {
      errors.user = "SSH username is required";
    }
    if (config.ssh?.testStatus === "failed") {
      errors.connection = config.ssh.testError || "Connection failed";
    }
  }

  if (config.type === "cloud") {
    if (!config.cloud?.provider) {
      errors.provider = "Select a cloud provider";
    }
    if (!config.cloud?.apiKey) {
      errors.apiKey = "API key is required";
    }
  }

  return {
    isValid: Object.keys(errors).length === 0,
    errors,
    warnings: warnings.length > 0 ? warnings : undefined,
  };
}

/**
 * Generate lab session ID
 */
export function generateSessionId(): string {
  return `lab_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}
