/**
 * Community Intelligence Types
 *
 * Core types for weekly digests, trending alerts, opportunity board,
 * meta-agent dashboard, and platform metrics.
 */

// ============================================================================
// Weekly Digest Types
// ============================================================================

export interface WeeklyDigest {
  id: string;
  weekNumber: number;
  year: number;
  startDate: string;
  endDate: string;
  generatedAt: string;
  publishedAt?: string;
  status: DigestStatus;

  // Content sections
  summary: DigestSummary;
  breakthroughs: Breakthrough[];
  trendingTechniques: TrendingTechnique[];
  newSynergies: SynergyHighlight[];
  researchOpportunities: OpportunityHighlight[];
  evolutionHighlights: EvolutionHighlight[];
  collaborationUpdates: CollaborationUpdate[];
  topContributors: ContributorHighlight[];

  // Stats
  stats: WeeklyStats;
}

export type DigestStatus = "draft" | "scheduled" | "published" | "archived";

export interface DigestSummary {
  headline: string;
  keyHighlights: string[];
  wordCount: number;
  readTimeMinutes: number;
}

export interface Breakthrough {
  id: string;
  title: string;
  description: string;
  significance: SignificanceLevel;
  discoveredBy: string;
  labId: string;
  labName: string;
  techniqueIds: string[];
  metrics: BreakthroughMetrics;
  timestamp: string;
}

export interface BreakthroughMetrics {
  improvementPercent: number;
  affectedDomains: string[];
  potentialApplications: number;
  communityInterest: number;
}

export interface TrendingTechnique {
  id: string;
  name: string;
  category: string;
  trendScore: number;
  adoptionRate: number;
  weeklyGrowth: number;
  usageCount: number;
  topLabs: string[];
  domains: string[];
}

export interface SynergyHighlight {
  id: string;
  techniques: string[];
  synergyScore: number;
  discoveredBy: string;
  labId: string;
  description: string;
  potentialImpact: SignificanceLevel;
  timestamp: string;
}

export interface OpportunityHighlight {
  id: string;
  title: string;
  gapType: GapType;
  difficulty: DifficultyLevel;
  impact: SignificanceLevel;
  domain: string;
  claimedBy?: string;
  bountyAmount?: number;
}

export interface EvolutionHighlight {
  id: string;
  techniqueId: string;
  techniqueName: string;
  generation: number;
  fitnessImprovement: number;
  parentTechniques: string[];
  newCapabilities: string[];
  timestamp: string;
}

export interface CollaborationUpdate {
  id: string;
  metaTaskId: string;
  title: string;
  status: string;
  participantCount: number;
  recentProgress: string;
  completedObjectives: number;
  totalObjectives: number;
}

export interface ContributorHighlight {
  labId: string;
  labName: string;
  contributions: number;
  discoveries: number;
  collaborations: number;
  weeklyScore: number;
  rank: number;
}

export interface WeeklyStats {
  totalPapers: number;
  newPapers: number;
  totalTechniques: number;
  newTechniques: number;
  totalLabs: number;
  activeLabs: number;
  synergiesDiscovered: number;
  gapsFilled: number;
  collaborationsStarted: number;
  collaborationsCompleted: number;
  evolutionGenerations: number;
  topDomain: string;
}

// ============================================================================
// Trending Alerts Types
// ============================================================================

export interface TrendingAlert {
  id: string;
  type: AlertType;
  significance: SignificanceLevel;
  title: string;
  description: string;
  relatedIds: string[];
  metadata: AlertMetadata;
  createdAt: string;
  expiresAt?: string;
  read: boolean;
  dismissed: boolean;
}

export type AlertType =
  | "synergy"
  | "breakthrough"
  | "trend"
  | "gap-filled"
  | "collaboration"
  | "evolution"
  | "opportunity";

export type SignificanceLevel = "critical" | "high" | "medium" | "low";

export interface AlertMetadata {
  sourceLabId?: string;
  sourceLabName?: string;
  techniqueIds?: string[];
  domain?: string;
  impactScore?: number;
  actionUrl?: string;
  [key: string]: unknown;
}

export interface AlertPreferences {
  userId: string;
  enabled: boolean;
  types: AlertType[];
  minSignificance: SignificanceLevel;
  emailEnabled: boolean;
  emailDigest: "immediate" | "daily" | "weekly" | "none";
  quietHoursStart?: string;
  quietHoursEnd?: string;
  mutedLabs: string[];
  mutedDomains: string[];
}

export interface AlertSubscription {
  id: string;
  userId: string;
  preferences: AlertPreferences;
  createdAt: string;
  updatedAt: string;
}

// ============================================================================
// Opportunity Board Types
// ============================================================================

export interface ResearchOpportunity {
  id: string;
  title: string;
  description: string;
  gapType: GapType;
  difficulty: DifficultyLevel;
  estimatedEffort: EffortEstimate;
  impact: SignificanceLevel;
  domain: string;
  tags: string[];

  // Status
  status: OpportunityStatus;
  claimedBy?: ClaimInfo;
  progress: number;

  // Bounty
  hasBounty: boolean;
  bounty?: BountyInfo;

  // Metadata
  sourceGapId?: string;
  relatedTechniques: string[];
  requiredExpertise: string[];
  suggestedApproaches: string[];

  createdAt: string;
  updatedAt: string;
  deadline?: string;
}

export type GapType =
  | "missing-technique"
  | "unexplored-combination"
  | "domain-adaptation"
  | "performance-optimization"
  | "scalability"
  | "theoretical-gap"
  | "tooling"
  | "documentation";

export type DifficultyLevel = "beginner" | "intermediate" | "advanced" | "expert";

export interface EffortEstimate {
  minHours: number;
  maxHours: number;
  recommendedTeamSize: number;
}

export type OpportunityStatus = "open" | "claimed" | "in-progress" | "review" | "completed" | "expired";

export interface ClaimInfo {
  labId: string;
  labName: string;
  claimedAt: string;
  expectedCompletion?: string;
}

export interface BountyInfo {
  amount: number;
  currency: string;
  sponsor: string;
  conditions: string[];
  expiresAt?: string;
}

export interface OpportunityLeaderboard {
  entries: LeaderboardEntry[];
  period: "weekly" | "monthly" | "all-time";
  updatedAt: string;
}

export interface LeaderboardEntry {
  rank: number;
  labId: string;
  labName: string;
  opportunitiesCompleted: number;
  totalBountyEarned: number;
  impactScore: number;
  streak: number;
  badges: string[];
}

// ============================================================================
// Meta-Agent Dashboard Types
// ============================================================================

export interface MetaAgentStatus {
  id: string;
  name: MetaAgentName;
  displayName: string;
  status: AgentStatus;
  health: HealthStatus;

  // Activity
  lastActivity: string;
  currentTask?: string;
  queuedTasks: number;

  // Performance
  metrics: AgentMetrics;
  recentDiscoveries: Discovery[];
  activityLog: ActivityLogEntry[];

  // Control
  isEnabled: boolean;
  pausedAt?: string;
  pauseReason?: string;
}

export type MetaAgentName =
  | "synergy-detector"
  | "pattern-recognizer"
  | "gap-analyzer"
  | "evolution-engine"
  | "transfer-agent";

export type AgentStatus = "running" | "idle" | "paused" | "error" | "initializing";

export type HealthStatus = "healthy" | "degraded" | "unhealthy" | "unknown";

export interface AgentMetrics {
  totalRuns: number;
  successfulRuns: number;
  failedRuns: number;
  avgExecutionTimeMs: number;
  discoveriesCount: number;
  lastHourActivity: number;
  last24HourActivity: number;
  uptime: number;
  errorRate: number;
}

export interface Discovery {
  id: string;
  type: string;
  description: string;
  significance: SignificanceLevel;
  timestamp: string;
  metadata: Record<string, unknown>;
}

export interface ActivityLogEntry {
  id: string;
  timestamp: string;
  action: string;
  details: string;
  result: "success" | "failure" | "pending";
  duration?: number;
}

export interface MetaAgentDashboard {
  agents: MetaAgentStatus[];
  systemHealth: SystemHealth;
  recentAlerts: TrendingAlert[];
  summary: DashboardSummary;
  updatedAt: string;
}

export interface SystemHealth {
  overall: HealthStatus;
  cpu: number;
  memory: number;
  activeConnections: number;
  queueDepth: number;
  errorRate: number;
  latency: number;
}

export interface DashboardSummary {
  totalDiscoveries: number;
  discoveriesToday: number;
  activeAgents: number;
  pausedAgents: number;
  pendingTasks: number;
  alertsToday: number;
}

// ============================================================================
// Platform Metrics Types
// ============================================================================

export interface PlatformMetrics {
  id: string;
  timestamp: string;
  period: MetricsPeriod;

  // Counts
  labs: LabMetrics;
  papers: PaperMetrics;
  techniques: TechniqueMetrics;
  tasks: TaskMetrics;
  collaborations: CollaborationMetrics;

  // Rates
  adoptionRates: AdoptionRates;
  successRates: SuccessRates;

  // Growth
  growth: GrowthMetrics;

  // Cost
  costs: CostMetrics;
}

export type MetricsPeriod = "daily" | "weekly" | "monthly" | "quarterly" | "yearly" | "all-time";

export interface LabMetrics {
  total: number;
  active: number;
  new: number;
  inactive: number;
  avgTechniquesPerLab: number;
  avgCollaborationsPerLab: number;
  topDomains: DomainCount[];
}

export interface DomainCount {
  domain: string;
  count: number;
  percentage: number;
}

export interface PaperMetrics {
  total: number;
  processed: number;
  pending: number;
  failed: number;
  avgProcessingTime: number;
  topSources: SourceCount[];
}

export interface SourceCount {
  source: string;
  count: number;
  percentage: number;
}

export interface TechniqueMetrics {
  total: number;
  active: number;
  deprecated: number;
  experimental: number;
  avgUsageCount: number;
  topCategories: CategoryCount[];
  evolutionGenerations: number;
}

export interface CategoryCount {
  category: string;
  count: number;
  percentage: number;
}

export interface TaskMetrics {
  total: number;
  completed: number;
  inProgress: number;
  pending: number;
  failed: number;
  avgCompletionTime: number;
  completionRate: number;
}

export interface CollaborationMetrics {
  total: number;
  active: number;
  completed: number;
  avgParticipants: number;
  avgDuration: number;
  successRate: number;
  topCollaborators: CollaboratorCount[];
}

export interface CollaboratorCount {
  labId: string;
  labName: string;
  count: number;
}

export interface AdoptionRates {
  newTechniques: number;
  synergyDiscoveries: number;
  evolutionExperiments: number;
  collaborationJoins: number;
}

export interface SuccessRates {
  taskCompletion: number;
  collaborationSuccess: number;
  gapResolution: number;
  evolutionImprovement: number;
}

export interface GrowthMetrics {
  labsGrowth: number;
  papersGrowth: number;
  techniquesGrowth: number;
  collaborationsGrowth: number;
  weekOverWeek: number;
  monthOverMonth: number;
}

export interface CostMetrics {
  totalApiCalls: number;
  apiCostEstimate: number;
  computeHours: number;
  storageGB: number;
  avgCostPerPaper: number;
  avgCostPerDiscovery: number;
}

// ============================================================================
// Helper Types
// ============================================================================

export interface DateRange {
  start: string;
  end: string;
}

export interface PaginatedResult<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}

export interface FilterOptions {
  domain?: string;
  difficulty?: DifficultyLevel;
  impact?: SignificanceLevel;
  status?: string;
  dateRange?: DateRange;
  tags?: string[];
}
