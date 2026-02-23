/**
 * Community Intelligence Module
 *
 * Exports all community intelligence features including
 * weekly digests, trending alerts, opportunity board,
 * meta-agent dashboard, and platform metrics.
 */

// Types
export * from "./types";

// Weekly Digest Generator
export {
  createDigestGenerator,
  generateDigest,
  publishDigest,
  scheduleDigest,
  archiveDigest,
  getDigest,
  getLatestDigest,
  getDigestArchive,
  getWeekNumber,
  getWeekDateRange,
  shouldAutoPublish,
  digestGeneratorConfig,
} from "./digest";
export type {
  DigestGenerator,
  DigestConfig,
  DigestSourceData,
  DiscoveryData as DigestDiscoveryData,
  TechniqueData as DigestTechniqueData,
  SynergyData as DigestSynergyData,
  GapData as DigestGapData,
  EvolutionData as DigestEvolutionData,
  CollaborationData as DigestCollaborationData,
  LabData as DigestLabData,
  PaperData as DigestPaperData,
} from "./digest";

// Trending Alerts System
export {
  createAlertsSystem,
  createDefaultPreferences,
  detectAlert,
  getAlert,
  markAlertRead,
  dismissAlert,
  getRecentAlerts,
  getAlertsPaginated,
  cleanupExpiredAlerts,
  subscribe,
  unsubscribe,
  getSubscription,
  updatePreferences,
  muteSource,
  unmuteSource,
  shouldDeliverAlert,
  getAlertsForUser,
  processAlertQueue,
  getAlertStats,
  alertsConfig,
} from "./alerts";
export type {
  AlertsSystem,
  AlertConfig,
  DetectionContext,
  AlertStats,
} from "./alerts";

// Opportunity Board
export {
  createOpportunityBoard,
  createOpportunity,
  addOpportunity,
  getOpportunity,
  updateOpportunity,
  removeOpportunity,
  claimOpportunity,
  unclaimOpportunity,
  startProgress,
  updateProgress,
  submitForReview,
  completeOpportunity,
  expireOpportunity,
  getOpportunities,
  getOpenOpportunities,
  getOpportunitiesByDomain,
  getOpportunitiesWithBounty,
  getClaimedOpportunities,
  searchOpportunities,
  addBounty,
  removeBounty,
  getTotalBountyPool,
  getLeaderboard,
  getLabRank,
  getBoardStats,
  cleanupExpiredClaims,
  boardConfig,
} from "./opportunities";
export type {
  OpportunityBoard,
  BoardConfig,
  BoardStats,
} from "./opportunities";

// Meta-Agent Dashboard
export {
  createAgentDashboard,
  getAgentStatus,
  getAllAgentStatuses,
  updateAgentStatus,
  setAgentHealth,
  enableAgent,
  disableAgent,
  pauseAgent,
  resumeAgent,
  logActivity,
  recordDiscovery,
  getRecentDiscoveries,
  updateSystemHealth,
  setSystemMetrics,
  queueTask,
  dequeueTask,
  generateDashboard,
  cleanupOldActivity,
  agentConfig,
  AGENT_DEFINITIONS,
} from "./agents";
export type {
  AgentDashboardState,
  AgentConfig,
} from "./agents";

// Research Metrics Aggregator
export {
  createMetricsAggregator,
  aggregateMetrics,
  takeSnapshot,
  getSnapshots,
  cleanupOldSnapshots,
  getCurrentMetrics,
  getMetricsByPeriod,
  getMetricsTrend,
  generateSummary,
  metricsConfig,
} from "./metrics";
export type {
  MetricsAggregator,
  MetricsSnapshot,
  MetricsConfig,
  AggregationSource,
  MetricsSummary,
} from "./metrics";
