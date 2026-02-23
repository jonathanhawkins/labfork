/**
 * Research Metrics Aggregator
 *
 * Platform-wide statistics aggregation including labs, papers, tasks,
 * collaborations, technique adoption rates, success rates, cost tracking,
 * and community growth metrics.
 */

import {
  PlatformMetrics,
  MetricsPeriod,
  LabMetrics,
  PaperMetrics,
  TechniqueMetrics,
  TaskMetrics,
  CollaborationMetrics,
  AdoptionRates,
  SuccessRates,
  GrowthMetrics,
  CostMetrics,
  DomainCount,
  SourceCount,
  CategoryCount,
  CollaboratorCount,
} from "./types";

// ============================================================================
// Metrics Aggregator Interface
// ============================================================================

export interface MetricsAggregator {
  history: Map<string, PlatformMetrics>;
  current: PlatformMetrics;
  snapshots: MetricsSnapshot[];
}

export interface MetricsSnapshot {
  id: string;
  timestamp: string;
  period: MetricsPeriod;
  metrics: PlatformMetrics;
}

export interface MetricsConfig {
  snapshotRetentionDays: number;
  aggregationIntervalMs: number;
  costPerApiCall: number;
  costPerComputeHour: number;
  costPerStorageGB: number;
}

const DEFAULT_CONFIG: MetricsConfig = {
  snapshotRetentionDays: 365,
  aggregationIntervalMs: 3600000, // 1 hour
  costPerApiCall: 0.001,
  costPerComputeHour: 0.5,
  costPerStorageGB: 0.1,
};

// ============================================================================
// Factory Functions
// ============================================================================

export function createMetricsAggregator(): MetricsAggregator {
  const now = new Date().toISOString();

  return {
    history: new Map(),
    current: createEmptyMetrics(now, "daily"),
    snapshots: [],
  };
}

function createEmptyMetrics(
  timestamp: string,
  period: MetricsPeriod
): PlatformMetrics {
  return {
    id: `metrics-${Date.now()}`,
    timestamp,
    period,
    labs: createEmptyLabMetrics(),
    papers: createEmptyPaperMetrics(),
    techniques: createEmptyTechniqueMetrics(),
    tasks: createEmptyTaskMetrics(),
    collaborations: createEmptyCollaborationMetrics(),
    adoptionRates: createEmptyAdoptionRates(),
    successRates: createEmptySuccessRates(),
    growth: createEmptyGrowthMetrics(),
    costs: createEmptyCostMetrics(),
  };
}

function createEmptyLabMetrics(): LabMetrics {
  return {
    total: 0,
    active: 0,
    new: 0,
    inactive: 0,
    avgTechniquesPerLab: 0,
    avgCollaborationsPerLab: 0,
    topDomains: [],
  };
}

function createEmptyPaperMetrics(): PaperMetrics {
  return {
    total: 0,
    processed: 0,
    pending: 0,
    failed: 0,
    avgProcessingTime: 0,
    topSources: [],
  };
}

function createEmptyTechniqueMetrics(): TechniqueMetrics {
  return {
    total: 0,
    active: 0,
    deprecated: 0,
    experimental: 0,
    avgUsageCount: 0,
    topCategories: [],
    evolutionGenerations: 0,
  };
}

function createEmptyTaskMetrics(): TaskMetrics {
  return {
    total: 0,
    completed: 0,
    inProgress: 0,
    pending: 0,
    failed: 0,
    avgCompletionTime: 0,
    completionRate: 0,
  };
}

function createEmptyCollaborationMetrics(): CollaborationMetrics {
  return {
    total: 0,
    active: 0,
    completed: 0,
    avgParticipants: 0,
    avgDuration: 0,
    successRate: 0,
    topCollaborators: [],
  };
}

function createEmptyAdoptionRates(): AdoptionRates {
  return {
    newTechniques: 0,
    synergyDiscoveries: 0,
    evolutionExperiments: 0,
    collaborationJoins: 0,
  };
}

function createEmptySuccessRates(): SuccessRates {
  return {
    taskCompletion: 0,
    collaborationSuccess: 0,
    gapResolution: 0,
    evolutionImprovement: 0,
  };
}

function createEmptyGrowthMetrics(): GrowthMetrics {
  return {
    labsGrowth: 0,
    papersGrowth: 0,
    techniquesGrowth: 0,
    collaborationsGrowth: 0,
    weekOverWeek: 0,
    monthOverMonth: 0,
  };
}

function createEmptyCostMetrics(): CostMetrics {
  return {
    totalApiCalls: 0,
    apiCostEstimate: 0,
    computeHours: 0,
    storageGB: 0,
    avgCostPerPaper: 0,
    avgCostPerDiscovery: 0,
  };
}

// ============================================================================
// Data Aggregation
// ============================================================================

export interface AggregationSource {
  labs: LabData[];
  papers: PaperData[];
  techniques: TechniqueData[];
  tasks: TaskData[];
  collaborations: CollaborationData[];
  discoveries: DiscoveryData[];
  apiCalls: number;
  computeHours: number;
  storageGB: number;
}

interface LabData {
  id: string;
  isActive: boolean;
  createdAt: string;
  techniquesCount: number;
  collaborationsCount: number;
  domains: string[];
}

interface PaperData {
  id: string;
  status: "pending" | "processed" | "failed";
  source: string;
  processingTimeMs?: number;
  createdAt: string;
}

interface TechniqueData {
  id: string;
  status: "active" | "deprecated" | "experimental";
  category: string;
  usageCount: number;
  generation: number;
  createdAt: string;
}

interface TaskData {
  id: string;
  status: "pending" | "in_progress" | "completed" | "failed";
  completionTimeMs?: number;
  createdAt: string;
}

interface CollaborationData {
  id: string;
  status: "active" | "completed" | "failed";
  participantCount: number;
  durationDays?: number;
  participants: { labId: string; labName: string }[];
  createdAt: string;
}

interface DiscoveryData {
  id: string;
  type: string;
  createdAt: string;
}

export function aggregateMetrics(
  aggregator: MetricsAggregator,
  source: AggregationSource,
  period: MetricsPeriod = "daily",
  config: MetricsConfig = DEFAULT_CONFIG
): PlatformMetrics {
  const now = new Date();
  const timestamp = now.toISOString();

  // Calculate period boundaries
  const periodStart = getPeriodStart(now, period);

  // Aggregate each section
  const labs = aggregateLabMetrics(source.labs, periodStart);
  const papers = aggregatePaperMetrics(source.papers, periodStart);
  const techniques = aggregateTechniqueMetrics(source.techniques, periodStart);
  const tasks = aggregateTaskMetrics(source.tasks, periodStart);
  const collaborations = aggregateCollaborationMetrics(
    source.collaborations,
    periodStart
  );
  const adoptionRates = calculateAdoptionRates(source, periodStart);
  const successRates = calculateSuccessRates(tasks, collaborations, source);
  const growth = calculateGrowth(aggregator, source, period);
  const costs = calculateCosts(source, config);

  const metrics: PlatformMetrics = {
    id: `metrics-${period}-${Date.now()}`,
    timestamp,
    period,
    labs,
    papers,
    techniques,
    tasks,
    collaborations,
    adoptionRates,
    successRates,
    growth,
    costs,
  };

  // Store in aggregator
  aggregator.current = metrics;
  aggregator.history.set(metrics.id, metrics);

  return metrics;
}

function getPeriodStart(now: Date, period: MetricsPeriod): Date {
  const start = new Date(now);

  switch (period) {
    case "daily":
      start.setHours(0, 0, 0, 0);
      break;
    case "weekly":
      start.setDate(start.getDate() - start.getDay());
      start.setHours(0, 0, 0, 0);
      break;
    case "monthly":
      start.setDate(1);
      start.setHours(0, 0, 0, 0);
      break;
    case "quarterly":
      start.setMonth(Math.floor(start.getMonth() / 3) * 3, 1);
      start.setHours(0, 0, 0, 0);
      break;
    case "yearly":
      start.setMonth(0, 1);
      start.setHours(0, 0, 0, 0);
      break;
    case "all-time":
      start.setFullYear(2000);
      break;
  }

  return start;
}

function aggregateLabMetrics(labs: LabData[], periodStart: Date): LabMetrics {
  const activeLabs = labs.filter((l) => l.isActive);
  const newLabs = labs.filter((l) => new Date(l.createdAt) >= periodStart);
  const inactiveLabs = labs.filter((l) => !l.isActive);

  // Calculate averages
  const avgTechniques =
    activeLabs.length > 0
      ? activeLabs.reduce((sum, l) => sum + l.techniquesCount, 0) /
        activeLabs.length
      : 0;

  const avgCollaborations =
    activeLabs.length > 0
      ? activeLabs.reduce((sum, l) => sum + l.collaborationsCount, 0) /
        activeLabs.length
      : 0;

  // Count domains
  const domainCounts: Record<string, number> = {};
  for (const lab of labs) {
    for (const domain of lab.domains) {
      domainCounts[domain] = (domainCounts[domain] || 0) + 1;
    }
  }

  const topDomains: DomainCount[] = Object.entries(domainCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([domain, count]) => ({
      domain,
      count,
      percentage: labs.length > 0 ? (count / labs.length) * 100 : 0,
    }));

  return {
    total: labs.length,
    active: activeLabs.length,
    new: newLabs.length,
    inactive: inactiveLabs.length,
    avgTechniquesPerLab: avgTechniques,
    avgCollaborationsPerLab: avgCollaborations,
    topDomains,
  };
}

function aggregatePaperMetrics(
  papers: PaperData[],
  periodStart: Date
): PaperMetrics {
  const processed = papers.filter((p) => p.status === "processed");
  const pending = papers.filter((p) => p.status === "pending");
  const failed = papers.filter((p) => p.status === "failed");

  // Calculate average processing time
  const processedWithTime = processed.filter((p) => p.processingTimeMs);
  const avgTime =
    processedWithTime.length > 0
      ? processedWithTime.reduce((sum, p) => sum + (p.processingTimeMs || 0), 0) /
        processedWithTime.length
      : 0;

  // Count sources
  const sourceCounts: Record<string, number> = {};
  for (const paper of papers) {
    sourceCounts[paper.source] = (sourceCounts[paper.source] || 0) + 1;
  }

  const topSources: SourceCount[] = Object.entries(sourceCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([source, count]) => ({
      source,
      count,
      percentage: papers.length > 0 ? (count / papers.length) * 100 : 0,
    }));

  return {
    total: papers.length,
    processed: processed.length,
    pending: pending.length,
    failed: failed.length,
    avgProcessingTime: avgTime,
    topSources,
  };
}

function aggregateTechniqueMetrics(
  techniques: TechniqueData[],
  periodStart: Date
): TechniqueMetrics {
  const active = techniques.filter((t) => t.status === "active");
  const deprecated = techniques.filter((t) => t.status === "deprecated");
  const experimental = techniques.filter((t) => t.status === "experimental");

  // Calculate average usage
  const avgUsage =
    techniques.length > 0
      ? techniques.reduce((sum, t) => sum + t.usageCount, 0) / techniques.length
      : 0;

  // Count categories
  const categoryCounts: Record<string, number> = {};
  for (const tech of techniques) {
    categoryCounts[tech.category] = (categoryCounts[tech.category] || 0) + 1;
  }

  const topCategories: CategoryCount[] = Object.entries(categoryCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([category, count]) => ({
      category,
      count,
      percentage:
        techniques.length > 0 ? (count / techniques.length) * 100 : 0,
    }));

  // Max generation
  const maxGeneration = Math.max(...techniques.map((t) => t.generation), 0);

  return {
    total: techniques.length,
    active: active.length,
    deprecated: deprecated.length,
    experimental: experimental.length,
    avgUsageCount: avgUsage,
    topCategories,
    evolutionGenerations: maxGeneration,
  };
}

function aggregateTaskMetrics(
  tasks: TaskData[],
  periodStart: Date
): TaskMetrics {
  const completed = tasks.filter((t) => t.status === "completed");
  const inProgress = tasks.filter((t) => t.status === "in_progress");
  const pending = tasks.filter((t) => t.status === "pending");
  const failed = tasks.filter((t) => t.status === "failed");

  // Calculate average completion time
  const completedWithTime = completed.filter((t) => t.completionTimeMs);
  const avgTime =
    completedWithTime.length > 0
      ? completedWithTime.reduce((sum, t) => sum + (t.completionTimeMs || 0), 0) /
        completedWithTime.length
      : 0;

  // Completion rate
  const finishedTasks = completed.length + failed.length;
  const completionRate =
    finishedTasks > 0 ? (completed.length / finishedTasks) * 100 : 0;

  return {
    total: tasks.length,
    completed: completed.length,
    inProgress: inProgress.length,
    pending: pending.length,
    failed: failed.length,
    avgCompletionTime: avgTime,
    completionRate,
  };
}

function aggregateCollaborationMetrics(
  collaborations: CollaborationData[],
  periodStart: Date
): CollaborationMetrics {
  const active = collaborations.filter((c) => c.status === "active");
  const completed = collaborations.filter((c) => c.status === "completed");
  const failed = collaborations.filter((c) => c.status === "failed");

  // Calculate averages
  const avgParticipants =
    collaborations.length > 0
      ? collaborations.reduce((sum, c) => sum + c.participantCount, 0) /
        collaborations.length
      : 0;

  const completedWithDuration = completed.filter((c) => c.durationDays);
  const avgDuration =
    completedWithDuration.length > 0
      ? completedWithDuration.reduce((sum, c) => sum + (c.durationDays || 0), 0) /
        completedWithDuration.length
      : 0;

  // Success rate
  const finishedCollabs = completed.length + failed.length;
  const successRate =
    finishedCollabs > 0 ? (completed.length / finishedCollabs) * 100 : 0;

  // Top collaborators
  const labCounts: Record<string, { count: number; name: string }> = {};
  for (const collab of collaborations) {
    for (const participant of collab.participants) {
      if (!labCounts[participant.labId]) {
        labCounts[participant.labId] = { count: 0, name: participant.labName };
      }
      labCounts[participant.labId].count++;
    }
  }

  const topCollaborators: CollaboratorCount[] = Object.entries(labCounts)
    .sort((a, b) => b[1].count - a[1].count)
    .slice(0, 10)
    .map(([labId, data]) => ({
      labId,
      labName: data.name,
      count: data.count,
    }));

  return {
    total: collaborations.length,
    active: active.length,
    completed: completed.length,
    avgParticipants,
    avgDuration,
    successRate,
    topCollaborators,
  };
}

function calculateAdoptionRates(
  source: AggregationSource,
  periodStart: Date
): AdoptionRates {
  const newTechniques = source.techniques.filter(
    (t) => new Date(t.createdAt) >= periodStart
  ).length;

  const recentDiscoveries = source.discoveries.filter(
    (d) => new Date(d.createdAt) >= periodStart
  );

  const synergyDiscoveries = recentDiscoveries.filter(
    (d) => d.type === "synergy"
  ).length;

  const evolutionExperiments = recentDiscoveries.filter(
    (d) => d.type === "evolution"
  ).length;

  const recentCollabs = source.collaborations.filter(
    (c) => new Date(c.createdAt) >= periodStart
  );

  const collaborationJoins = recentCollabs.reduce(
    (sum, c) => sum + c.participantCount,
    0
  );

  return {
    newTechniques,
    synergyDiscoveries,
    evolutionExperiments,
    collaborationJoins,
  };
}

function calculateSuccessRates(
  tasks: TaskMetrics,
  collaborations: CollaborationMetrics,
  source: AggregationSource
): SuccessRates {
  // Evolution improvement (techniques with generation > 1 that improved)
  const evolvedTechniques = source.techniques.filter((t) => t.generation > 1);
  const improvedTechniques = evolvedTechniques.filter((t) => t.usageCount > 0);
  const evolutionImprovement =
    evolvedTechniques.length > 0
      ? (improvedTechniques.length / evolvedTechniques.length) * 100
      : 0;

  // Gap resolution (simulated - would need actual gap data)
  const gapResolution = 75; // Placeholder

  return {
    taskCompletion: tasks.completionRate,
    collaborationSuccess: collaborations.successRate,
    gapResolution,
    evolutionImprovement,
  };
}

function calculateGrowth(
  aggregator: MetricsAggregator,
  source: AggregationSource,
  period: MetricsPeriod
): GrowthMetrics {
  // Get previous metrics for comparison
  const previousMetrics = getPreviousPeriodMetrics(aggregator, period);

  if (!previousMetrics) {
    return createEmptyGrowthMetrics();
  }

  const calcGrowth = (current: number, previous: number): number =>
    previous > 0 ? ((current - previous) / previous) * 100 : 0;

  return {
    labsGrowth: calcGrowth(source.labs.length, previousMetrics.labs.total),
    papersGrowth: calcGrowth(
      source.papers.length,
      previousMetrics.papers.total
    ),
    techniquesGrowth: calcGrowth(
      source.techniques.length,
      previousMetrics.techniques.total
    ),
    collaborationsGrowth: calcGrowth(
      source.collaborations.length,
      previousMetrics.collaborations.total
    ),
    weekOverWeek: calcGrowth(source.labs.length, previousMetrics.labs.total),
    monthOverMonth: calcGrowth(source.labs.length, previousMetrics.labs.total),
  };
}

function getPreviousPeriodMetrics(
  aggregator: MetricsAggregator,
  period: MetricsPeriod
): PlatformMetrics | null {
  const snapshots = aggregator.snapshots.filter((s) => s.period === period);
  if (snapshots.length < 2) return null;

  // Sort by timestamp and get second-to-last
  snapshots.sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );
  return snapshots[1]?.metrics || null;
}

function calculateCosts(
  source: AggregationSource,
  config: MetricsConfig
): CostMetrics {
  const apiCostEstimate = source.apiCalls * config.costPerApiCall;
  const computeCost = source.computeHours * config.costPerComputeHour;
  const storageCost = source.storageGB * config.costPerStorageGB;
  const totalCost = apiCostEstimate + computeCost + storageCost;

  const processedPapers = source.papers.filter(
    (p) => p.status === "processed"
  ).length;
  const avgCostPerPaper = processedPapers > 0 ? totalCost / processedPapers : 0;

  const discoveries = source.discoveries.length;
  const avgCostPerDiscovery = discoveries > 0 ? totalCost / discoveries : 0;

  return {
    totalApiCalls: source.apiCalls,
    apiCostEstimate,
    computeHours: source.computeHours,
    storageGB: source.storageGB,
    avgCostPerPaper,
    avgCostPerDiscovery,
  };
}

// ============================================================================
// Snapshot Management
// ============================================================================

export function takeSnapshot(
  aggregator: MetricsAggregator,
  period: MetricsPeriod = "daily"
): MetricsSnapshot {
  const snapshot: MetricsSnapshot = {
    id: `snapshot-${Date.now()}`,
    timestamp: new Date().toISOString(),
    period,
    metrics: { ...aggregator.current },
  };

  aggregator.snapshots.push(snapshot);
  return snapshot;
}

export function getSnapshots(
  aggregator: MetricsAggregator,
  period?: MetricsPeriod,
  limit: number = 30
): MetricsSnapshot[] {
  let snapshots = aggregator.snapshots;

  if (period) {
    snapshots = snapshots.filter((s) => s.period === period);
  }

  return snapshots
    .sort(
      (a, b) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    )
    .slice(0, limit);
}

export function cleanupOldSnapshots(
  aggregator: MetricsAggregator,
  config: MetricsConfig = DEFAULT_CONFIG
): number {
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - config.snapshotRetentionDays);

  const originalLength = aggregator.snapshots.length;
  aggregator.snapshots = aggregator.snapshots.filter(
    (s) => new Date(s.timestamp) > cutoff
  );

  return originalLength - aggregator.snapshots.length;
}

// ============================================================================
// Query Functions
// ============================================================================

export function getCurrentMetrics(
  aggregator: MetricsAggregator
): PlatformMetrics {
  return aggregator.current;
}

export function getMetricsByPeriod(
  aggregator: MetricsAggregator,
  period: MetricsPeriod
): PlatformMetrics[] {
  return Array.from(aggregator.history.values())
    .filter((m) => m.period === period)
    .sort(
      (a, b) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
}

export function getMetricsTrend(
  aggregator: MetricsAggregator,
  metricPath: string,
  limit: number = 7
): { timestamp: string; value: number }[] {
  const snapshots = getSnapshots(aggregator, "daily", limit);

  return snapshots.map((s) => ({
    timestamp: s.timestamp,
    value: getNestedValue(s.metrics, metricPath) || 0,
  }));
}

function getNestedValue(obj: unknown, path: string): number | undefined {
  const parts = path.split(".");
  let current: unknown = obj;

  for (const part of parts) {
    if (current && typeof current === "object" && part in current) {
      current = (current as Record<string, unknown>)[part];
    } else {
      return undefined;
    }
  }

  return typeof current === "number" ? current : undefined;
}

// ============================================================================
// Summary Generation
// ============================================================================

export interface MetricsSummary {
  highlights: string[];
  concerns: string[];
  recommendations: string[];
}

export function generateSummary(metrics: PlatformMetrics): MetricsSummary {
  const highlights: string[] = [];
  const concerns: string[] = [];
  const recommendations: string[] = [];

  // Highlights
  if (metrics.growth.labsGrowth > 10) {
    highlights.push(
      `Strong lab growth: ${metrics.growth.labsGrowth.toFixed(1)}% increase`
    );
  }

  if (metrics.successRates.taskCompletion > 80) {
    highlights.push(
      `High task completion rate: ${metrics.successRates.taskCompletion.toFixed(1)}%`
    );
  }

  if (metrics.adoptionRates.synergyDiscoveries > 10) {
    highlights.push(
      `Active synergy discovery: ${metrics.adoptionRates.synergyDiscoveries} new synergies`
    );
  }

  // Concerns
  if (metrics.papers.failed / metrics.papers.total > 0.1) {
    concerns.push(
      `High paper failure rate: ${((metrics.papers.failed / metrics.papers.total) * 100).toFixed(1)}%`
    );
  }

  if (metrics.labs.inactive > metrics.labs.active) {
    concerns.push(`More inactive than active labs`);
  }

  if (metrics.successRates.collaborationSuccess < 50) {
    concerns.push(
      `Low collaboration success rate: ${metrics.successRates.collaborationSuccess.toFixed(1)}%`
    );
  }

  // Recommendations
  if (metrics.techniques.experimental > metrics.techniques.active * 0.3) {
    recommendations.push(`Consider validating experimental techniques`);
  }

  if (metrics.costs.avgCostPerDiscovery > 10) {
    recommendations.push(`Optimize discovery costs (currently $${metrics.costs.avgCostPerDiscovery.toFixed(2)}/discovery)`);
  }

  if (metrics.collaborations.avgParticipants < 3) {
    recommendations.push(`Encourage larger collaboration teams`);
  }

  return { highlights, concerns, recommendations };
}

// ============================================================================
// Export
// ============================================================================

export const metricsConfig = DEFAULT_CONFIG;
