/**
 * Distributed Compute Network - Public API
 *
 * Main exports for the LabFork distributed compute system.
 */

// Core types
export type {
  DeviceTier,
  DevicePlatform,
  TaskType,
  TaskStatus,
  DeviceStatus,
  DeviceCapabilities,
  DeviceAvailability,
  ComputeDevice,
  TaskInput,
  TaskConfig,
  ComputeTask,
  TaskResult,
  NetworkStats,
  RegisterDeviceRequest,
  SubmitTaskRequest,
  HeartbeatRequest,
  TaskAssignment,
  CompleteTaskRequest,
  CreditTransaction,
  UserCredits,
  SpeculativeDecodingConfig,
} from "./types";

export {
  generateDeviceId,
  generateTaskId,
  generateTransactionId,
  classifyDeviceTier,
  calculateTaskReward,
  DEFAULT_AVAILABILITY,
} from "./types";

// GPU detection
export type { GPUInfo, BenchmarkResult, TierInfo } from "./gpu-detect";
export {
  detectGPU,
  runBenchmark,
  classifyTier,
  getTierInfo,
  formatTFLOPS,
  formatMemory,
} from "./gpu-detect";

// Device agent
export type {
  AgentStatus,
  AgentEvents,
  DeviceStats,
  AgentConfig,
} from "./device-agent";
export { DeviceAgent, getDeviceAgent } from "./device-agent";

// React hooks
export type {
  DeviceAgentState,
  DeviceAgentActions,
  UseDeviceAgentReturn,
  UseDeviceAgentOptions,
} from "./useDeviceAgent";
export {
  useDeviceAgent,
  useDeviceAgentStats,
  useDeviceInfo,
  useCurrentTask,
  formatUptime,
  formatComputeTime,
  formatCredits,
  getStatusColor,
  getStatusLabel,
} from "./useDeviceAgent";

// WebLLM engine (browser-side)
export type {
  ModelId,
  ModelInfo,
  LoadProgress,
  EngineEvents,
} from "./webllm-engine";
export {
  WebLLMEngine,
  getWebLLMEngine,
  AVAILABLE_MODELS,
  MODEL_INFO,
} from "./webllm-engine";

// Orchestrator (server-side only)
export { getOrchestrator } from "./orchestrator";

// Speculative decoding
export type {
  DraftToken,
  DraftSequence,
  TokenVerification,
  VerificationResult,
  SpeculativeTask,
  SpeculativeStats,
} from "./speculative-decoding";

export {
  generateDraftId,
  generateVerificationId,
  shouldAcceptToken,
  calculateAvgConfidence,
  calculateSpeedupFactor,
  estimateTimeSaved,
  batchDrafts,
  validateDraft,
  mergeFinalSequence,
  tokensToText,
  DEFAULT_SPECULATIVE_CONFIG,
} from "./speculative-decoding";

export type {
  DraftGenerationOptions,
  DraftGenerationResult,
} from "./draft-generator";

export {
  DraftGenerator,
  getDraftGenerator,
  resetDraftGenerator,
} from "./draft-generator";

export type {
  VerificationOptions,
  VerificationResultWithMetrics,
  BatchVerificationOptions,
} from "./draft-verifier";

export {
  DraftVerifier,
  getDraftVerifier,
  resetDraftVerifier,
} from "./draft-verifier";

// PWA utilities
export type { BatteryStatus, PWAStatus } from "./pwa-utils";
export {
  isPWAInstalled,
  canInstallPWA,
  getPWAStatus,
  registerServiceWorker,
  unregisterServiceWorker,
  requestNotificationPermission,
  getBatteryStatus,
  monitorBattery,
  shouldContributeCompute,
  subscribeToPushNotifications,
  unsubscribeFromPushNotifications,
  queueTaskForSync,
} from "./pwa-utils";

// PWA React hook
export type { UsePWAOptions, UsePWAResult } from "./usePWA";
export { usePWA } from "./usePWA";

// Task queue
export type { QueuedTask, QueueStats, QueueConfig } from "./task-queue";
export { TaskQueue } from "./task-queue";

// Task router
export type {
  DeviceScore,
  TaskRequirements,
  SpeculativeTaskPair,
  RouterConfig,
} from "./task-router";
export { TaskRouter } from "./task-router";

// Fair scheduling
export type {
  DeviceContribution,
  FairnessMetrics,
  SchedulingPolicy,
  SchedulerConfig,
} from "./scheduling";
export { FairScheduler, getScheduler } from "./scheduling";

// Network analytics
export type {
  TimePeriod,
  LatencyPercentiles,
  TierStats,
  ThroughputStats,
  NetworkAnalytics,
  HistoricalDataPoint,
  HistoricalAnalytics,
} from "./analytics";
export {
  calculateHealthScore,
  calculateTierStats,
  calculateLatencyPercentiles,
  calculateThroughput,
  generateNetworkAnalytics,
  formatLatency,
  getHealthStatus,
  getTierColor,
  getTierLabel,
} from "./analytics";

// Network events hook
export type { CompletedTask, NetworkEventsState } from "./useNetworkEvents";
export {
  useNetworkEvents,
  formatTierBreakdown,
  calculateTasksPerHour,
  calculateNetworkHealth,
} from "./useNetworkEvents";

// Contributor system
export type { ContributorRank, Badge, ContributorProfile } from "./user-types";
export {
  BADGE_DEFINITIONS,
  calculateRank,
  checkBadgeEligibility,
  generateDisplayName,
} from "./user-types";

// Contributor hooks
export { useContributor, useLeaderboard, useContributionStats } from "./useContributor";

// Onboarding
export type { OnboardingState } from "./onboarding";
export {
  hasCompletedOnboarding,
  markOnboardingComplete,
  resetOnboarding,
  getOnboardingState,
} from "./onboarding";
