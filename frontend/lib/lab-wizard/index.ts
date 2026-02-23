/**
 * Lab Wizard Module
 *
 * Utilities for the lab creation wizard.
 */

// Types
export * from "./types";

// GPU Detection
export {
  detectLocalGpu,
  detectLocalSystem,
  systemInfoToLocalConfig,
  checkGpuMeetsDomain,
  formatGpuInfo,
  getGpuRecommendation,
  DOMAIN_GPU_REQUIREMENTS,
  MOCK_GPU,
  MOCK_SYSTEM_INFO,
  type GpuDetectionResult,
  type SystemInfoResult,
} from "./gpu-detection";

// SSH Testing
export {
  testSSHConnection,
  detectRemoteGpu,
  getRemoteSystemInfo,
  getSSHStatusInfo,
  formatSSHError,
  validateSSHConfig,
  formatSSHConnection,
  KNOWN_SSH_HOSTS,
  MOCK_SSH_RESULT,
  MOCK_REMOTE_SYSTEM,
  type SSHTestResult,
  type RemoteGpuResult,
  type RemoteSystemInfo,
  type SSHConnectionStatus,
} from "./ssh-tester";

// Goal Analysis
export {
  analyzeGoal,
  generateInitialTasks,
  parseGoalAnalysisResponse,
  estimateTimeline,
  getTaskTypeInfo,
  getPriorityInfo,
  applyAnalysisToGoal,
  GOAL_ANALYSIS_SYSTEM_PROMPT,
  generateGoalPrompt,
  MOCK_GOAL_ANALYSIS,
  type GoalAnalysisResult,
} from "./goal-analyzer";

// Scaffolding
export {
  generateDomainYaml,
  generateHardwareSection,
  generateInitialTasksFromConfig,
  createLab,
  validateLabConfig,
  getLabDirectoryStructure,
  generatePrompt,
  estimateCreationTime,
  DEFAULT_PROMPTS,
  MOCK_SCAFFOLDING_RESULT,
  type DomainYamlContent,
  type ScaffoldingResult,
} from "./scaffolding";
