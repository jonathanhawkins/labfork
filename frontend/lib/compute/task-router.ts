/**
 * Task Router
 *
 * Intelligent task-to-device matching system for the distributed compute network.
 * Handles device capability matching, load balancing, and speculative decoding coordination.
 */

import type {
  ComputeTask,
  ComputeDevice,
  DeviceTier,
  TaskType,
  DeviceCapabilities,
} from "./types";

/**
 * Device matching score
 */
export interface DeviceScore {
  /** The device being scored */
  device: ComputeDevice;
  /** Total score (higher is better) */
  score: number;
  /** Score breakdown for debugging */
  breakdown: {
    computeScore: number;
    memoryScore: number;
    cacheScore: number;
    tierScore: number;
    latencyScore: number;
    reliabilityScore: number;
    loadBalanceScore: number;
  };
}

/**
 * Task routing requirements
 */
export interface TaskRequirements {
  /** Minimum compute power (TFLOPS) */
  minCompute?: number;
  /** Minimum memory (GB) */
  minMemory?: number;
  /** Required model ID */
  modelId: string;
  /** Minimum device tier */
  minTier?: DeviceTier;
  /** Prefer geographic region */
  preferredRegion?: string;
  /** Maximum acceptable latency (ms) */
  maxLatency?: number;
}

/**
 * Speculative decoding task pair
 */
export interface SpeculativeTaskPair {
  /** Draft token generation task (crowd/standard tier) */
  draftTask: ComputeTask;
  /** Verification task (power tier) */
  verifyTask: ComputeTask;
  /** Expected latency (ms) */
  estimatedLatency: number;
}

/**
 * Router configuration
 */
export interface RouterConfig {
  /** Weight for compute score (0-1) */
  computeWeight: number;
  /** Weight for memory score (0-1) */
  memoryWeight: number;
  /** Weight for cache hit (0-1) */
  cacheWeight: number;
  /** Weight for tier matching (0-1) */
  tierWeight: number;
  /** Weight for latency (0-1) */
  latencyWeight: number;
  /** Weight for reliability (0-1) */
  reliabilityWeight: number;
  /** Weight for load balancing (0-1) */
  loadBalanceWeight: number;
  /** Enable geographic routing */
  enableGeoRouting: boolean;
  /** Enable speculative decoding coordination */
  enableSpeculativeDecoding: boolean;
}

/**
 * Default router configuration
 */
const DEFAULT_CONFIG: RouterConfig = {
  computeWeight: 0.25,
  memoryWeight: 0.15,
  cacheWeight: 0.20,
  tierWeight: 0.15,
  latencyWeight: 0.10,
  reliabilityWeight: 0.10,
  loadBalanceWeight: 0.05,
  enableGeoRouting: false,
  enableSpeculativeDecoding: true,
};

/**
 * Task Router
 *
 * Matches tasks to optimal devices based on:
 * - Device capabilities (compute, memory, platform)
 * - Model caching (prefer devices with model cached)
 * - Device tier matching
 * - Geographic latency
 * - Device reliability history
 * - Load balancing
 */
export class TaskRouter {
  private config: RouterConfig;

  constructor(config: Partial<RouterConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Find the best device for a task
   */
  findBestDevice(
    task: ComputeTask,
    availableDevices: ComputeDevice[]
  ): ComputeDevice | null {
    if (availableDevices.length === 0) {
      return null;
    }

    // Extract task requirements
    const requirements = this.extractRequirements(task);

    // Filter devices by requirements
    const candidates = this.filterByRequirements(availableDevices, requirements);

    if (candidates.length === 0) {
      return null;
    }

    // Score all candidate devices
    const scores = candidates.map((device) =>
      this.scoreDevice(device, task, requirements, candidates.length)
    );

    // Sort by score (highest first)
    scores.sort((a, b) => b.score - a.score);

    return scores[0].device;
  }

  /**
   * Find multiple devices for parallel task execution
   */
  findDevicesForParallel(
    task: ComputeTask,
    availableDevices: ComputeDevice[],
    count: number
  ): ComputeDevice[] {
    const requirements = this.extractRequirements(task);
    const candidates = this.filterByRequirements(availableDevices, requirements);

    if (candidates.length === 0) {
      return [];
    }

    // Score all candidates
    const scores = candidates.map((device) =>
      this.scoreDevice(device, task, requirements, candidates.length)
    );

    // Sort by score and take top N
    scores.sort((a, b) => b.score - a.score);

    return scores.slice(0, count).map((s) => s.device);
  }

  /**
   * Create speculative decoding task pair
   *
   * Pairs a draft task (for crowd/standard devices) with a verification task (for power devices)
   */
  createSpeculativeTaskPair(
    originalTask: ComputeTask,
    draftDevices: ComputeDevice[],
    verifyDevices: ComputeDevice[]
  ): SpeculativeTaskPair | null {
    if (!this.config.enableSpeculativeDecoding) {
      return null;
    }

    // Find best draft device (crowd/standard tier)
    const draftDevice = this.findBestDevice(originalTask, draftDevices);
    if (!draftDevice) {
      return null;
    }

    // Find best verification device (power tier)
    const verifyDevice = this.findBestDevice(originalTask, verifyDevices);
    if (!verifyDevice) {
      return null;
    }

    // Create draft task (generates speculative tokens)
    const draftTask: ComputeTask = {
      ...originalTask,
      id: `${originalTask.id}_draft`,
      type: "draft_tokens",
      config: {
        ...originalTask.config,
        minTier: "crowd",
        // Use smaller/faster model for draft
        modelId: this.selectDraftModel(originalTask.config.modelId),
      },
      priority: originalTask.priority + 0.5, // Slightly higher priority
      reward: originalTask.reward * 0.3, // 30% of reward
    };

    // Create verification task (validates and continues)
    const verifyTask: ComputeTask = {
      ...originalTask,
      id: `${originalTask.id}_verify`,
      type: "validation",
      config: {
        ...originalTask.config,
        minTier: "power",
      },
      priority: originalTask.priority + 1, // Highest priority
      reward: originalTask.reward * 0.7, // 70% of reward
    };

    // Estimate latency
    const estimatedLatency = this.estimateTaskLatency(draftTask, draftDevice) +
      this.estimateTaskLatency(verifyTask, verifyDevice);

    return {
      draftTask,
      verifyTask,
      estimatedLatency,
    };
  }

  /**
   * Score a device for a given task
   */
  private scoreDevice(
    device: ComputeDevice,
    task: ComputeTask,
    requirements: TaskRequirements,
    totalCandidates: number
  ): DeviceScore {
    const breakdown = {
      computeScore: this.scoreCompute(device.capabilities, requirements),
      memoryScore: this.scoreMemory(device.capabilities, requirements),
      cacheScore: this.scoreCacheHit(device.capabilities, requirements.modelId),
      tierScore: this.scoreTier(device.tier, task),
      latencyScore: this.scoreLatency(device, requirements),
      reliabilityScore: this.scoreReliability(device),
      loadBalanceScore: this.scoreLoadBalance(device, totalCandidates),
    };

    // Calculate weighted total score
    const score =
      breakdown.computeScore * this.config.computeWeight +
      breakdown.memoryScore * this.config.memoryWeight +
      breakdown.cacheScore * this.config.cacheWeight +
      breakdown.tierScore * this.config.tierWeight +
      breakdown.latencyScore * this.config.latencyWeight +
      breakdown.reliabilityScore * this.config.reliabilityWeight +
      breakdown.loadBalanceScore * this.config.loadBalanceWeight;

    return {
      device,
      score,
      breakdown,
    };
  }

  /**
   * Extract task requirements from task configuration
   */
  private extractRequirements(task: ComputeTask): TaskRequirements {
    const requirements: TaskRequirements = {
      modelId: task.config.modelId,
      minTier: task.config.minTier,
    };

    // Set compute/memory requirements based on task type
    switch (task.type) {
      case "full_inference":
        requirements.minCompute = 5.0; // 5 TFLOPS
        requirements.minMemory = 4.0; // 4 GB
        break;
      case "shard_inference":
        requirements.minCompute = 2.0;
        requirements.minMemory = 2.0;
        break;
      case "draft_tokens":
        requirements.minCompute = 0.5;
        requirements.minMemory = 1.0;
        break;
      case "embedding":
        requirements.minCompute = 1.0;
        requirements.minMemory = 2.0;
        break;
      case "validation":
        requirements.minCompute = 10.0; // Power tier required
        requirements.minMemory = 8.0;
        break;
      case "simulation":
        // Full CFD simulations require power tier (4090 GPU)
        requirements.minCompute = 40.0; // 40+ TFLOPS (RTX 4090 class)
        requirements.minMemory = 16.0; // 16GB+ VRAM for OpenFOAM
        requirements.minTier = "power";
        break;
    }

    return requirements;
  }

  /**
   * Filter devices by minimum requirements
   */
  private filterByRequirements(
    devices: ComputeDevice[],
    requirements: TaskRequirements
  ): ComputeDevice[] {
    const tierOrder: DeviceTier[] = ["crowd", "standard", "power"];

    return devices.filter((device) => {
      // Check tier requirement
      if (requirements.minTier) {
        const deviceTierIndex = tierOrder.indexOf(device.tier);
        const minTierIndex = tierOrder.indexOf(requirements.minTier);
        if (deviceTierIndex < minTierIndex) {
          return false;
        }
      }

      // Check compute requirement
      if (
        requirements.minCompute &&
        device.capabilities.compute < requirements.minCompute
      ) {
        return false;
      }

      // Check memory requirement
      if (
        requirements.minMemory &&
        device.capabilities.memory < requirements.minMemory
      ) {
        return false;
      }

      return true;
    });
  }

  /**
   * Score device compute capability
   */
  private scoreCompute(
    capabilities: DeviceCapabilities,
    requirements: TaskRequirements
  ): number {
    const minCompute = requirements.minCompute || 1.0;
    const computeRatio = capabilities.compute / minCompute;

    // Normalize to 0-100 scale with diminishing returns
    return Math.min(100, Math.log2(computeRatio + 1) * 50);
  }

  /**
   * Score device memory capability
   */
  private scoreMemory(
    capabilities: DeviceCapabilities,
    requirements: TaskRequirements
  ): number {
    const minMemory = requirements.minMemory || 1.0;
    const memoryRatio = capabilities.memory / minMemory;

    // Normalize to 0-100 scale
    return Math.min(100, memoryRatio * 50);
  }

  /**
   * Score model cache hit
   */
  private scoreCacheHit(capabilities: DeviceCapabilities, modelId: string): number {
    return capabilities.cachedModels.includes(modelId) ? 100 : 0;
  }

  /**
   * Score tier matching
   */
  private scoreTier(deviceTier: DeviceTier, task: ComputeTask): number {
    const taskType = task.type;

    // Simulations REQUIRE power tier - CFD/physics are GPU-intensive
    if (taskType === "simulation") {
      if (deviceTier === "power") return 100;
      return 0; // Other tiers cannot run simulations
    }

    // Prefer higher tiers for complex tasks
    if (taskType === "full_inference" || taskType === "validation") {
      if (deviceTier === "power") return 100;
      if (deviceTier === "standard") return 50;
      return 25;
    }

    // Prefer crowd tier for draft tokens (more efficient distribution)
    if (taskType === "draft_tokens") {
      if (deviceTier === "crowd") return 100;
      if (deviceTier === "standard") return 75;
      return 25; // Don't waste power tier on drafts
    }

    // Standard tier good for most tasks
    if (deviceTier === "standard") return 100;
    if (deviceTier === "power") return 90;
    return 70;
  }

  /**
   * Score device latency (geographic/network)
   */
  private scoreLatency(
    device: ComputeDevice,
    requirements: TaskRequirements
  ): number {
    if (!this.config.enableGeoRouting) {
      return 50; // Neutral score
    }

    // Mock latency scoring based on bandwidth
    // In production, would use actual latency measurements
    const bandwidth = device.capabilities.bandwidth;

    if (bandwidth >= 100) return 100; // Excellent
    if (bandwidth >= 50) return 75; // Good
    if (bandwidth >= 10) return 50; // Adequate
    return 25; // Slow
  }

  /**
   * Score device reliability based on completion history
   */
  private scoreReliability(device: ComputeDevice): number {
    const stats = device.stats;

    // No history = neutral score
    if (stats.tasksCompleted === 0) {
      return 50;
    }

    // Calculate success rate (assuming we track failures)
    // For now, use tasks completed as proxy for reliability
    const completionScore = Math.min(100, stats.tasksCompleted * 2);

    // Factor in compute time (more time = more reliable)
    const uptimeScore = Math.min(100, stats.totalComputeTime / 36); // 1 hour = 100

    return (completionScore + uptimeScore) / 2;
  }

  /**
   * Score for load balancing (prefer devices with fewer tasks)
   */
  private scoreLoadBalance(device: ComputeDevice, totalCandidates: number): number {
    // Add small random component to distribute across similar devices
    return Math.random() * 100;
  }

  /**
   * Estimate task execution latency on device
   */
  private estimateTaskLatency(task: ComputeTask, device: ComputeDevice): number {
    // Base latency on device compute power
    const baseLatency = 5000 / device.capabilities.compute; // ms

    // Scale by token count
    const tokenMultiplier = task.config.maxTokens / 100;

    return baseLatency * tokenMultiplier;
  }

  /**
   * Select draft model for speculative decoding
   */
  private selectDraftModel(originalModel: string): string {
    // Map to smaller/faster model variants
    const draftModels: Record<string, string> = {
      "Llama-3.1-8B-Instruct-q4f32_1-MLC": "Llama-3.2-1B-Instruct-q4f16_1-MLC",
      "Phi-3.5-mini-instruct-q4f16_1-MLC": "SmolLM2-135M-Instruct-q0f16-MLC",
    };

    return draftModels[originalModel] || originalModel;
  }

  /**
   * Update router configuration
   */
  updateConfig(config: Partial<RouterConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   */
  getConfig(): RouterConfig {
    return { ...this.config };
  }
}
