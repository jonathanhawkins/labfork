/**
 * GPU Detection Utility
 *
 * Detects local GPU hardware and capabilities.
 */

import type { GpuInfo, LocalConfig } from "./types";

/**
 * GPU detection result from API
 */
export interface GpuDetectionResult {
  success: boolean;
  gpu?: GpuInfo;
  error?: string;
}

/**
 * System info result from API
 */
export interface SystemInfoResult {
  success: boolean;
  info?: {
    gpu?: GpuInfo;
    ram?: number;
    platform?: string;
    ollamaInstalled?: boolean;
    ollamaVersion?: string;
    ollamaModels?: string[];
  };
  error?: string;
}

/**
 * Detect local GPU by calling the hardware API
 * Returns GpuInfo or null
 */
export async function detectLocalGpu(): Promise<GpuInfo | null> {
  try {
    const response = await fetch("/api/lab/hardware");
    const data = await response.json();

    if (data.success && data.gpu) {
      return data.gpu;
    }

    return null;
  } catch (error) {
    return null;
  }
}

/**
 * Get complete local system information
 * Returns system info or null
 */
export async function detectLocalSystem(): Promise<SystemInfoResult["info"] | null> {
  try {
    const response = await fetch("/api/lab/hardware?full=true");
    const data = await response.json();

    if (data.success) {
      return {
        gpu: data.gpu,
        ram: data.ram,
        platform: data.platform,
        ollamaInstalled: data.ollamaInstalled,
        ollamaVersion: data.ollamaVersion,
        ollamaModels: data.ollamaModels,
      };
    }

    return null;
  } catch (error) {
    return null;
  }
}

/**
 * Convert system info to LocalConfig
 */
export function systemInfoToLocalConfig(
  info: SystemInfoResult["info"] | null,
  gpu: GpuInfo | null
): LocalConfig {
  return {
    gpu: gpu ?? info?.gpu,
    systemInfo: info ? {
      platform: info.platform,
      totalMemory: info.ram,
    } : undefined,
    systemRam: info?.ram,
    ollamaInstalled: info?.ollamaInstalled,
    ollamaVersion: info?.ollamaVersion,
    ollamaModels: info?.ollamaModels,
  };
}

/**
 * GPU requirement levels for domains
 */
export interface GpuRequirement {
  minVram: number;
  recommendedVram: number;
  description: string;
}

/**
 * Domain GPU requirements
 */
export const DOMAIN_GPU_REQUIREMENTS: Record<string, GpuRequirement> = {
  "voice-clone": {
    minVram: 8,
    recommendedVram: 24,
    description: "Voice synthesis requires significant VRAM for real-time processing",
  },
  "quant-trading": {
    minVram: 4,
    recommendedVram: 8,
    description: "Trading models can run on smaller GPUs",
  },
  robotics: {
    minVram: 12,
    recommendedVram: 24,
    description: "Robot learning benefits from larger GPU memory",
  },
  biotech: {
    minVram: 16,
    recommendedVram: 48,
    description: "Molecular simulations require substantial GPU resources",
  },
  default: {
    minVram: 8,
    recommendedVram: 16,
    description: "General ML research",
  },
};

/**
 * Check if GPU meets domain requirements
 */
export function checkGpuMeetsDomain(
  gpu: GpuInfo | undefined,
  domainSlug: string
): {
  meets: boolean;
  level: "none" | "minimum" | "recommended" | "exceeded";
  message: string;
  warning?: string;
} {
  const req = DOMAIN_GPU_REQUIREMENTS[domainSlug] || DOMAIN_GPU_REQUIREMENTS.default;

  if (!gpu || !gpu.available) {
    return {
      meets: false,
      level: "none",
      message: "No GPU detected. Consider using cloud or remote GPU.",
      warning: "No GPU available",
    };
  }

  if (gpu.vram >= req.recommendedVram) {
    return {
      meets: true,
      level: "exceeded",
      message: `Excellent! ${gpu.name} exceeds recommended ${req.recommendedVram}GB VRAM.`,
    };
  }

  if (gpu.vram >= req.minVram) {
    return {
      meets: true,
      level: "minimum",
      message: `${gpu.name} meets minimum ${req.minVram}GB, but ${req.recommendedVram}GB recommended.`,
    };
  }

  return {
    meets: false,
    level: "none",
    message: `${gpu.name} has ${gpu.vram}GB but ${req.minVram}GB minimum required.`,
    warning: `Insufficient VRAM: ${gpu.vram}GB < ${req.minVram}GB minimum`,
  };
}

/**
 * Format GPU info for display
 */
export function formatGpuInfo(gpu: GpuInfo): string {
  const parts = [gpu.name];

  if (gpu.vram) {
    parts.push(`${gpu.vram}GB`);
  }

  if (gpu.cudaVersion) {
    parts.push(`CUDA ${gpu.cudaVersion}`);
  }

  return parts.join(" - ");
}

/**
 * Get GPU recommendation based on domain
 */
export function getGpuRecommendation(domainSlug: string): string {
  const req = DOMAIN_GPU_REQUIREMENTS[domainSlug] || DOMAIN_GPU_REQUIREMENTS.default;

  if (req.recommendedVram >= 48) {
    return `${req.recommendedVram}GB+ recommended: A100, H100, or multiple consumer GPUs`;
  }
  if (req.recommendedVram >= 24) {
    return `${req.recommendedVram}GB+ recommended: RTX 4090, RTX 3090, or A10`;
  }
  if (req.recommendedVram >= 16) {
    return `${req.recommendedVram}GB+ recommended: RTX 4080, RTX 3080 Ti, or T4`;
  }
  if (req.recommendedVram >= 8) {
    return `${req.recommendedVram}GB+ recommended: RTX 4070 or better`;
  }
  return "Any modern GPU with 4GB+ VRAM";
}

/**
 * Mock GPU for testing (used in development)
 */
export const MOCK_GPU: GpuInfo = {
  name: "NVIDIA GeForce RTX 4090",
  vram: 24,
  cudaVersion: "12.1",
  driverVersion: "535.104.05",
  available: true,
  computeCapability: "8.9",
};

// Alias for tests
export const MOCK_GPU_INFO = MOCK_GPU;

/**
 * Mock system info for testing
 */
export const MOCK_SYSTEM_INFO: SystemInfoResult["info"] = {
  gpu: MOCK_GPU,
  ram: 64,
  platform: "darwin",
  ollamaInstalled: true,
  ollamaVersion: "0.1.17",
  ollamaModels: ["qwen3-coder:30b", "llama3:8b"],
};
