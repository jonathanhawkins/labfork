/**
 * Network Analytics
 *
 * Aggregate analytics and health monitoring for the compute network.
 * Tracks historical data, calculates health scores, and provides
 * comprehensive network insights.
 */

import type { NetworkStats, DeviceTier, ComputeTask } from "./types";

/**
 * Time period for analytics
 */
export type TimePeriod = "1h" | "24h" | "7d" | "30d";

/**
 * Latency percentile data
 */
export interface LatencyPercentiles {
  p50: number; // median
  p95: number;
  p99: number;
}

/**
 * Contributor tier stats
 */
export interface TierStats {
  tier: DeviceTier;
  count: number;
  totalCompute: number; // TFLOPS
  tasksCompleted: number;
  creditsEarned: number;
}

/**
 * Task throughput stats
 */
export interface ThroughputStats {
  current: number; // tasks per hour
  average: number;
  peak: number;
  timestamp: string;
}

/**
 * Network analytics data
 */
export interface NetworkAnalytics {
  /** Total network compute power in TFLOPS */
  totalTFLOPS: number;
  /** Contributors by tier */
  contributorsByTier: TierStats[];
  /** Task completion rate (0-100) */
  completionRate: number;
  /** Average task latency in ms */
  averageLatency: number;
  /** Latency percentiles */
  latencyPercentiles: LatencyPercentiles;
  /** Network health score (0-100) */
  healthScore: number;
  /** Tasks per hour stats */
  throughput: ThroughputStats;
  /** Total tasks completed */
  totalTasksCompleted: number;
  /** Total credits distributed */
  totalCreditsDistributed: number;
  /** Active contributors count */
  activeContributors: number;
  /** Time period covered */
  period: TimePeriod;
  /** Timestamp of analytics */
  timestamp: string;
}

/**
 * Historical data point
 */
export interface HistoricalDataPoint {
  timestamp: string;
  value: number;
}

/**
 * Historical analytics data
 */
export interface HistoricalAnalytics {
  tflops: HistoricalDataPoint[];
  contributors: HistoricalDataPoint[];
  throughput: HistoricalDataPoint[];
  latency: HistoricalDataPoint[];
}

/**
 * Calculate network health score (0-100)
 *
 * Health factors:
 * - Active devices (30%)
 * - Task completion rate (25%)
 * - Average latency (20%)
 * - Network compute power (15%)
 * - Queue depth (10%)
 */
export function calculateHealthScore(
  stats: NetworkStats,
  averageLatency: number = 0,
  completionRate: number = 100
): number {
  let score = 0;

  // Active devices score (0-30 points)
  // Full points if >= 10 devices online
  const deviceScore = Math.min((stats.onlineDevices / 10) * 30, 30);
  score += deviceScore;

  // Completion rate score (0-25 points)
  score += (completionRate / 100) * 25;

  // Latency score (0-20 points)
  // Lower is better. Full points if < 1000ms
  const latencyScore = Math.max(0, 20 - (averageLatency / 1000) * 20);
  score += latencyScore;

  // Compute power score (0-15 points)
  // Full points if >= 50 TFLOPS
  const computeScore = Math.min((stats.totalCompute / 50) * 15, 15);
  score += computeScore;

  // Queue depth score (0-10 points)
  // Full points if pending tasks < processing tasks
  const queueRatio = stats.processingTasks > 0
    ? stats.pendingTasks / stats.processingTasks
    : 0;
  const queueScore = Math.max(0, 10 - queueRatio * 5);
  score += queueScore;

  return Math.min(Math.round(score), 100);
}

/**
 * Calculate tier-specific statistics
 */
export function calculateTierStats(
  stats: NetworkStats,
  tasks: ComputeTask[] = []
): TierStats[] {
  const tiers: DeviceTier[] = ["power", "standard", "crowd"];

  return tiers.map((tier) => {
    // Count devices in this tier
    const count = stats.devicesByTier[tier] || 0;

    // Estimate compute based on tier (rough averages)
    const avgComputeByTier: Record<DeviceTier, number> = {
      power: 50, // RTX 4090, A100
      standard: 10, // Apple Silicon, mid-range GPUs
      crowd: 2, // Browsers, phones
    };
    const totalCompute = count * avgComputeByTier[tier];

    // Calculate tasks completed by this tier (if we have task data)
    const tasksCompleted = tasks.filter(
      (t) => t.status === "completed"
      // Would need to track tier on task for accurate count
    ).length;

    // Estimate credits earned
    const creditsEarned = tasksCompleted * 5; // Rough average

    return {
      tier,
      count,
      totalCompute,
      tasksCompleted,
      creditsEarned,
    };
  });
}

/**
 * Calculate latency percentiles from task data
 */
export function calculateLatencyPercentiles(
  tasks: ComputeTask[]
): LatencyPercentiles {
  // Extract compute times from completed tasks
  const latencies = tasks
    .filter((t) => t.status === "completed" && t.result?.metrics.computeTime)
    .map((t) => t.result!.metrics.computeTime)
    .sort((a, b) => a - b);

  if (latencies.length === 0) {
    return { p50: 0, p95: 0, p99: 0 };
  }

  const getPercentile = (p: number) => {
    const index = Math.ceil((p / 100) * latencies.length) - 1;
    return latencies[Math.max(0, index)];
  };

  return {
    p50: getPercentile(50),
    p95: getPercentile(95),
    p99: getPercentile(99),
  };
}

/**
 * Calculate task throughput (tasks per hour)
 */
export function calculateThroughput(
  completedToday: number,
  hoursElapsed: number = 1
): ThroughputStats {
  const current = Math.round(completedToday / Math.max(hoursElapsed, 0.1));

  return {
    current,
    average: current, // Would track over time in production
    peak: current * 1.5, // Rough estimate
    timestamp: new Date().toISOString(),
  };
}

/**
 * Generate comprehensive network analytics
 */
export function generateNetworkAnalytics(
  stats: NetworkStats,
  tasks: ComputeTask[] = [],
  period: TimePeriod = "24h"
): NetworkAnalytics {
  // Calculate tier stats
  const contributorsByTier = calculateTierStats(stats, tasks);

  // Calculate latency metrics
  const latencyPercentiles = calculateLatencyPercentiles(tasks);
  const averageLatency = latencyPercentiles.p50;

  // Calculate completion rate
  const totalTasks = tasks.length;
  const completedTasks = tasks.filter((t) => t.status === "completed").length;
  const completionRate = totalTasks > 0
    ? Math.round((completedTasks / totalTasks) * 100)
    : 100;

  // Calculate health score
  const healthScore = calculateHealthScore(stats, averageLatency, completionRate);

  // Calculate throughput
  const hoursInPeriod = period === "1h" ? 1 : period === "24h" ? 24 : period === "7d" ? 168 : 720;
  const throughput = calculateThroughput(stats.completedToday, hoursInPeriod);

  return {
    totalTFLOPS: stats.totalCompute,
    contributorsByTier,
    completionRate,
    averageLatency,
    latencyPercentiles,
    healthScore,
    throughput,
    totalTasksCompleted: stats.completedToday,
    totalCreditsDistributed: stats.creditsToday,
    activeContributors: stats.onlineDevices,
    period,
    timestamp: new Date().toISOString(),
  };
}

/**
 * Format TFLOPS for display
 */
export function formatTFLOPS(tflops: number): string {
  if (tflops >= 1000) {
    return `${(tflops / 1000).toFixed(1)}P`;
  }
  if (tflops >= 1) {
    return `${tflops.toFixed(1)}T`;
  }
  return `${(tflops * 1000).toFixed(0)}G`;
}

/**
 * Format latency for display
 */
export function formatLatency(ms: number): string {
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(1)}s`;
  }
  return `${Math.round(ms)}ms`;
}

/**
 * Get health status label
 */
export function getHealthStatus(score: number): {
  label: string;
  color: string;
} {
  if (score >= 80) {
    return { label: "Excellent", color: "text-green-400" };
  }
  if (score >= 60) {
    return { label: "Good", color: "text-yellow-400" };
  }
  if (score >= 40) {
    return { label: "Fair", color: "text-orange-400" };
  }
  return { label: "Poor", color: "text-red-400" };
}

/**
 * Get tier color class
 */
export function getTierColor(tier: DeviceTier): string {
  const colors: Record<DeviceTier, string> = {
    power: "bg-purple-500",
    standard: "bg-blue-500",
    crowd: "bg-green-500",
  };
  return colors[tier];
}

/**
 * Get tier label
 */
export function getTierLabel(tier: DeviceTier): string {
  const labels: Record<DeviceTier, string> = {
    power: "Power",
    standard: "Standard",
    crowd: "Crowd",
  };
  return labels[tier];
}
