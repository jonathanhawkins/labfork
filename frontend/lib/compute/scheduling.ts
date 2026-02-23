/**
 * Fair Scheduling
 *
 * Fair scheduling algorithms for distributed compute network.
 * Prevents device starvation and ensures all contributors get work.
 */

import type { ComputeDevice, ComputeTask, DeviceTier } from "./types";

/**
 * Device contribution metrics
 */
export interface DeviceContribution {
  /** Device ID */
  deviceId: string;
  /** Total tasks completed */
  tasksCompleted: number;
  /** Total compute time (seconds) */
  totalComputeTime: number;
  /** Credits earned */
  creditsEarned: number;
  /** Last task assigned timestamp */
  lastAssignedAt?: number;
  /** Time since last task (ms) */
  timeSinceLastTask: number;
  /** Current wait time for next task (ms) */
  waitTime: number;
}

/**
 * Scheduling fairness metrics
 */
export interface FairnessMetrics {
  /** Gini coefficient (0 = perfect equality, 1 = perfect inequality) */
  giniCoefficient: number;
  /** Standard deviation of task distribution */
  taskDistributionStdDev: number;
  /** Devices currently starving (long wait) */
  starvingDevices: number;
  /** Average wait time (ms) */
  avgWaitTime: number;
  /** Max wait time (ms) */
  maxWaitTime: number;
}

/**
 * Scheduling policy
 */
export type SchedulingPolicy = "fifo" | "fair_share" | "weighted_fair" | "priority_boost";

/**
 * Scheduler configuration
 */
export interface SchedulerConfig {
  /** Scheduling policy */
  policy: SchedulingPolicy;
  /** Starvation threshold (ms) - devices waiting this long get priority boost */
  starvationThreshold: number;
  /** Priority boost amount for starving devices */
  starvationBoost: number;
  /** Enable reliability-based weighting */
  enableReliabilityWeighting: boolean;
  /** Reliability weight factor (0-1) */
  reliabilityWeight: number;
  /** Enable tier-based fair share */
  enableTierFairShare: boolean;
}

/**
 * Default scheduler configuration
 */
const DEFAULT_CONFIG: SchedulerConfig = {
  policy: "weighted_fair",
  starvationThreshold: 60000, // 1 minute
  starvationBoost: 5.0,
  enableReliabilityWeighting: true,
  reliabilityWeight: 0.3,
  enableTierFairShare: true,
};

/**
 * Fair Scheduler
 *
 * Implements fair scheduling algorithms to ensure:
 * - No device starvation
 * - Equitable task distribution
 * - Priority boost for reliable contributors
 * - Tier-based fair share
 */
export class FairScheduler {
  private config: SchedulerConfig;
  private deviceMetrics: Map<string, DeviceContribution> = new Map();
  private assignmentHistory: Array<{ deviceId: string; timestamp: number }> = [];
  private readonly historyLimit = 1000; // Keep last 1000 assignments

  constructor(config: Partial<SchedulerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Calculate priority score for device to receive next task
   */
  calculateDevicePriority(
    device: ComputeDevice,
    task: ComputeTask,
    allDevices: ComputeDevice[]
  ): number {
    // Update device metrics
    this.updateDeviceMetrics(device);

    const metrics = this.deviceMetrics.get(device.id);
    if (!metrics) {
      return 0;
    }

    switch (this.config.policy) {
      case "fifo":
        return this.calculateFIFOPriority(metrics);
      case "fair_share":
        return this.calculateFairSharePriority(metrics, allDevices);
      case "weighted_fair":
        return this.calculateWeightedFairPriority(device, metrics, allDevices);
      case "priority_boost":
        return this.calculatePriorityBoostPriority(device, metrics, task);
      default:
        return 0;
    }
  }

  /**
   * Select best device for task using fair scheduling
   */
  selectDevice(
    task: ComputeTask,
    eligibleDevices: ComputeDevice[]
  ): ComputeDevice | null {
    if (eligibleDevices.length === 0) {
      return null;
    }

    // Calculate priority for each device
    const devicePriorities = eligibleDevices.map((device) => ({
      device,
      priority: this.calculateDevicePriority(device, task, eligibleDevices),
    }));

    // Sort by priority (highest first)
    devicePriorities.sort((a, b) => b.priority - a.priority);

    return devicePriorities[0].device;
  }

  /**
   * Record task assignment to device
   */
  recordAssignment(deviceId: string, taskId: string): void {
    const now = Date.now();

    // Update assignment history
    this.assignmentHistory.push({ deviceId, timestamp: now });

    // Trim history if needed
    if (this.assignmentHistory.length > this.historyLimit) {
      this.assignmentHistory.shift();
    }

    // Update device metrics
    const metrics = this.deviceMetrics.get(deviceId);
    if (metrics) {
      metrics.lastAssignedAt = now;
      metrics.timeSinceLastTask = 0;
      metrics.waitTime = 0;
    }
  }

  /**
   * Get fairness metrics across all devices
   */
  getFairnessMetrics(devices: ComputeDevice[]): FairnessMetrics {
    // Update all device metrics
    devices.forEach((device) => this.updateDeviceMetrics(device));

    const metrics = Array.from(this.deviceMetrics.values());

    if (metrics.length === 0) {
      return {
        giniCoefficient: 0,
        taskDistributionStdDev: 0,
        starvingDevices: 0,
        avgWaitTime: 0,
        maxWaitTime: 0,
      };
    }

    // Calculate Gini coefficient for task distribution
    const giniCoefficient = this.calculateGiniCoefficient(
      metrics.map((m) => m.tasksCompleted)
    );

    // Calculate standard deviation of task distribution
    const taskDistributionStdDev = this.calculateStdDev(
      metrics.map((m) => m.tasksCompleted)
    );

    // Count starving devices
    const starvingDevices = metrics.filter(
      (m) => m.waitTime >= this.config.starvationThreshold
    ).length;

    // Calculate wait times
    const waitTimes = metrics.map((m) => m.waitTime);
    const avgWaitTime = waitTimes.reduce((sum, t) => sum + t, 0) / waitTimes.length;
    const maxWaitTime = Math.max(...waitTimes);

    return {
      giniCoefficient,
      taskDistributionStdDev,
      starvingDevices,
      avgWaitTime,
      maxWaitTime,
    };
  }

  /**
   * Get device contribution metrics
   */
  getDeviceMetrics(deviceId: string): DeviceContribution | null {
    return this.deviceMetrics.get(deviceId) || null;
  }

  /**
   * Reset scheduler state
   */
  reset(): void {
    this.deviceMetrics.clear();
    this.assignmentHistory = [];
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<SchedulerConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   */
  getConfig(): SchedulerConfig {
    return { ...this.config };
  }

  /**
   * Update device metrics based on current state
   */
  private updateDeviceMetrics(device: ComputeDevice): void {
    let metrics = this.deviceMetrics.get(device.id);

    if (!metrics) {
      metrics = {
        deviceId: device.id,
        tasksCompleted: device.stats.tasksCompleted,
        totalComputeTime: device.stats.totalComputeTime,
        creditsEarned: device.stats.creditsEarned,
        timeSinceLastTask: 0,
        waitTime: 0,
      };
      this.deviceMetrics.set(device.id, metrics);
    }

    // Update from device stats
    metrics.tasksCompleted = device.stats.tasksCompleted;
    metrics.totalComputeTime = device.stats.totalComputeTime;
    metrics.creditsEarned = device.stats.creditsEarned;

    // Calculate wait time
    const now = Date.now();
    if (metrics.lastAssignedAt) {
      metrics.timeSinceLastTask = now - metrics.lastAssignedAt;
      metrics.waitTime = metrics.timeSinceLastTask;
    } else {
      // New device - use registration time
      const registeredAt = new Date(device.registeredAt).getTime();
      metrics.waitTime = now - registeredAt;
    }
  }

  /**
   * Calculate FIFO priority (first in, first out)
   */
  private calculateFIFOPriority(metrics: DeviceContribution): number {
    // Longer wait time = higher priority
    return metrics.waitTime;
  }

  /**
   * Calculate fair share priority
   */
  private calculateFairSharePriority(
    metrics: DeviceContribution,
    allDevices: ComputeDevice[]
  ): number {
    // Calculate average tasks completed
    const avgTasksCompleted =
      allDevices.reduce((sum, d) => sum + d.stats.tasksCompleted, 0) /
      allDevices.length;

    // Devices below average get higher priority
    const fairnessGap = avgTasksCompleted - metrics.tasksCompleted;

    // Combine with wait time
    return fairnessGap * 100 + metrics.waitTime;
  }

  /**
   * Calculate weighted fair priority (considers reliability)
   */
  private calculateWeightedFairPriority(
    device: ComputeDevice,
    metrics: DeviceContribution,
    allDevices: ComputeDevice[]
  ): number {
    // Base fair share priority
    let priority = this.calculateFairSharePriority(metrics, allDevices);

    // Apply reliability weighting if enabled
    if (this.config.enableReliabilityWeighting) {
      const reliabilityScore = this.calculateReliabilityScore(device);
      priority *= 1 + reliabilityScore * this.config.reliabilityWeight;
    }

    // Apply tier-based fair share if enabled
    if (this.config.enableTierFairShare) {
      const tierBoost = this.calculateTierFairShareBoost(device, allDevices);
      priority += tierBoost;
    }

    // Boost for starving devices
    if (metrics.waitTime >= this.config.starvationThreshold) {
      priority += this.config.starvationBoost * 1000;
    }

    return priority;
  }

  /**
   * Calculate priority boost priority
   */
  private calculatePriorityBoostPriority(
    device: ComputeDevice,
    metrics: DeviceContribution,
    task: ComputeTask
  ): number {
    let priority = metrics.waitTime;

    // Boost for task priority
    priority += task.priority * 500;

    // Boost for tier matching
    if (task.config.minTier === device.tier) {
      priority += 1000;
    }

    // Boost for reliable devices
    const reliabilityScore = this.calculateReliabilityScore(device);
    priority += reliabilityScore * 500;

    return priority;
  }

  /**
   * Calculate device reliability score (0-1)
   */
  private calculateReliabilityScore(device: ComputeDevice): number {
    const stats = device.stats;

    // No history = neutral score
    if (stats.tasksCompleted === 0) {
      return 0.5;
    }

    // Reliability based on completion count and compute time
    const completionScore = Math.min(1, stats.tasksCompleted / 100);
    const uptimeScore = Math.min(1, stats.totalComputeTime / 3600); // 1 hour = max

    return (completionScore + uptimeScore) / 2;
  }

  /**
   * Calculate tier-based fair share boost
   */
  private calculateTierFairShareBoost(
    device: ComputeDevice,
    allDevices: ComputeDevice[]
  ): number {
    // Group devices by tier
    const devicesByTier = allDevices.filter((d) => d.tier === device.tier);

    if (devicesByTier.length === 0) {
      return 0;
    }

    // Calculate average tasks for this tier
    const avgTasksForTier =
      devicesByTier.reduce((sum, d) => sum + d.stats.tasksCompleted, 0) /
      devicesByTier.length;

    // Boost if below tier average
    const tierGap = avgTasksForTier - device.stats.tasksCompleted;

    return tierGap > 0 ? tierGap * 50 : 0;
  }

  /**
   * Calculate Gini coefficient (inequality measure)
   */
  private calculateGiniCoefficient(values: number[]): number {
    if (values.length === 0) {
      return 0;
    }

    // Sort values
    const sorted = [...values].sort((a, b) => a - b);
    const n = sorted.length;
    const sum = sorted.reduce((a, b) => a + b, 0);

    if (sum === 0) {
      return 0; // Perfect equality when all values are 0
    }

    let numerator = 0;
    for (let i = 0; i < n; i++) {
      numerator += (i + 1) * sorted[i];
    }

    const gini = (2 * numerator) / (n * sum) - (n + 1) / n;

    return Math.max(0, Math.min(1, gini)); // Clamp to [0, 1]
  }

  /**
   * Calculate standard deviation
   */
  private calculateStdDev(values: number[]): number {
    if (values.length === 0) {
      return 0;
    }

    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    const variance =
      values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;

    return Math.sqrt(variance);
  }
}

/**
 * Export singleton scheduler instance - use globalThis for Next.js hot reload persistence
 */
declare global {
  var __fairScheduler: FairScheduler | undefined;
}

export function getScheduler(): FairScheduler {
  if (!globalThis.__fairScheduler) {
    globalThis.__fairScheduler = new FairScheduler();
  }
  return globalThis.__fairScheduler;
}
