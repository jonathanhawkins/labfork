/**
 * Distributed Compute Network Types
 *
 * Core type definitions for the LabFork compute network.
 */

/**
 * Device capability tiers
 */
export type DeviceTier = "power" | "standard" | "crowd";

/**
 * Device platform/backend
 */
export type DevicePlatform = "cuda" | "metal" | "webgpu" | "cpu";

/**
 * Task types that can be executed
 */
export type TaskType =
  | "full_inference"
  | "shard_inference"
  | "draft_tokens"
  | "embedding"
  | "embeddings"
  | "validation"
  | "draft_generation"
  | "draft_verification"
  | "summarization"
  | "classification"
  | "simulation"; // Physics/CFD simulations (water harvester, heat transfer, etc.)

/**
 * Task status
 */
export type TaskStatus =
  | "pending"
  | "assigned"
  | "processing"
  | "completed"
  | "failed"
  | "timeout";

/**
 * Device status
 */
export type DeviceStatus = "online" | "busy" | "offline" | "paused";

/**
 * Device capabilities
 */
export interface DeviceCapabilities {
  /** Estimated compute in TFLOPS */
  compute: number;
  /** Available memory in GB */
  memory: number;
  /** Network bandwidth in Mbps */
  bandwidth: number;
  /** Platform/backend */
  platform: DevicePlatform;
  /** GPU name if available */
  gpuName?: string;
  /** Cached model IDs */
  cachedModels: string[];
}

/**
 * Device availability preferences
 */
export interface DeviceAvailability {
  /** Only contribute on WiFi */
  wifiOnly: boolean;
  /** Only contribute when charging (mobile) */
  chargingOnly: boolean;
  /** Minimum battery level (mobile) */
  minBattery: number;
  /** Maximum GPU utilization % */
  maxUtilization: number;
}

/**
 * Registered device
 */
export interface ComputeDevice {
  /** Unique device ID */
  id: string;
  /** User ID if authenticated */
  userId?: string;
  /** Display name */
  name: string;
  /** Device tier classification */
  tier: DeviceTier;
  /** Device capabilities */
  capabilities: DeviceCapabilities;
  /** Availability preferences */
  availability: DeviceAvailability;
  /** Current status */
  status: DeviceStatus;
  /** Currently assigned task ID */
  currentTaskId?: string;
  /** Last heartbeat timestamp */
  lastHeartbeat: string;
  /** Registration timestamp */
  registeredAt: string;
  /** Lifetime stats */
  stats: {
    tasksCompleted: number;
    creditsEarned: number;
    totalComputeTime: number; // seconds
  };
}

/**
 * Task input data
 */
export interface TaskInput {
  /** Input tokens for inference */
  tokens?: number[];
  /** Text prompt */
  prompt?: string;
  /** Input embedding */
  embedding?: number[];
  /** Model shard index for pipeline parallel */
  shardIndex?: number;
  /** Total shards in pipeline */
  shardCount?: number;
  /** Simulation parameters (for simulation tasks) */
  simulationParams?: {
    type: string; // e.g., 'water_harvester', 'heat_transfer'
    labSlug: string; // e.g., 'water-harvester'
    parameters: Record<string, unknown>;
    mode: 'quick' | 'full';
  };
}

/**
 * Speculative decoding configuration
 */
export interface SpeculativeDecodingConfig {
  /** Number of draft tokens to generate */
  draftCount: number;
  /** Acceptance threshold (0-1) */
  acceptanceThreshold: number;
  /** Draft model ID */
  draftModelId: string;
  /** Verification model ID */
  verifyModelId: string;
  /** Maximum drafts to batch together */
  batchSize: number;
}

/**
 * Task configuration
 */
export interface TaskConfig {
  /** Model ID to use */
  modelId: string;
  /** Maximum tokens to generate */
  maxTokens: number;
  /** Temperature for sampling */
  temperature: number;
  /** Deadline timestamp */
  deadline?: string;
  /** Minimum device tier required */
  minTier?: DeviceTier;
  /** Speculative decoding configuration */
  speculativeDecoding?: SpeculativeDecodingConfig;
}

/**
 * Compute task
 */
export interface ComputeTask {
  /** Unique task ID */
  id: string;
  /** Task type */
  type: TaskType;
  /** Task input */
  input: TaskInput;
  /** Task configuration */
  config: TaskConfig;
  /** Task status */
  status: TaskStatus;
  /** Assigned device ID */
  assignedDeviceId?: string;
  /** Priority (higher = more urgent) */
  priority: number;
  /** Credit reward for completion */
  reward: number;
  /** Creation timestamp */
  createdAt: string;
  /** Assignment timestamp */
  assignedAt?: string;
  /** Completion timestamp */
  completedAt?: string;
  /** Submitter ID (lab or user) */
  submitterId: string;
  /** Result data */
  result?: TaskResult;
}

/**
 * Task result
 */
export interface TaskResult {
  /** Output tokens */
  tokens?: number[];
  /** Output text */
  text?: string;
  /** Output embedding */
  embedding?: number[];
  /** Output logits (for verification) */
  logits?: number[];
  /** Whether result came from real WebLLM or mock execution */
  computeMode?: 'webllm' | 'mock';
  /** Execution metrics */
  metrics: {
    computeTime: number; // ms
    tokensPerSecond?: number;
  };
}

/**
 * Network statistics
 */
export interface NetworkStats {
  /** Total registered devices */
  totalDevices: number;
  /** Currently online devices */
  onlineDevices: number;
  /** Devices by tier */
  devicesByTier: Record<DeviceTier, number>;
  /** Total network compute (TFLOPS) */
  totalCompute: number;
  /** Tasks in queue */
  pendingTasks: number;
  /** Tasks being processed */
  processingTasks: number;
  /** Tasks completed today */
  completedToday: number;
  /** Credits distributed today */
  creditsToday: number;
}

/**
 * Device registration request
 */
export interface RegisterDeviceRequest {
  /** Device name */
  name: string;
  /** Device capabilities */
  capabilities: DeviceCapabilities;
  /** Availability preferences */
  availability?: Partial<DeviceAvailability>;
}

/**
 * Task submission request
 */
export interface SubmitTaskRequest {
  /** Task type */
  type: TaskType;
  /** Task input */
  input: TaskInput;
  /** Task configuration */
  config: TaskConfig;
  /** Priority (default 1) */
  priority?: number;
}

/**
 * Heartbeat request
 */
export interface HeartbeatRequest {
  /** Device ID */
  deviceId: string;
  /** Current status */
  status: DeviceStatus;
  /** Current task progress (0-100) */
  taskProgress?: number;
}

/**
 * Task assignment response
 */
export interface TaskAssignment {
  /** Task to execute */
  task: ComputeTask;
  /** Model download URL if needed */
  modelUrl?: string;
}

/**
 * Task completion request
 */
export interface CompleteTaskRequest {
  /** Task ID */
  taskId: string;
  /** Device ID */
  deviceId: string;
  /** Success or failure */
  success: boolean;
  /** Result if successful */
  result?: TaskResult;
  /** Error message if failed */
  error?: string;
}

/**
 * Credit transaction
 */
export interface CreditTransaction {
  /** Transaction ID */
  id: string;
  /** User ID */
  userId: string;
  /** Amount (positive = earn, negative = spend) */
  amount: number;
  /** Transaction type */
  type: "earn" | "spend" | "bonus" | "refund";
  /** Related task ID */
  taskId?: string;
  /** Description */
  description: string;
  /** Timestamp */
  createdAt: string;
}

/**
 * User credit balance
 */
export interface UserCredits {
  /** User ID */
  userId: string;
  /** Current balance */
  balance: number;
  /** Lifetime earned */
  totalEarned: number;
  /** Lifetime spent */
  totalSpent: number;
  /** Last updated */
  updatedAt: string;
}

/**
 * Generate unique IDs
 */
export function generateDeviceId(): string {
  return `dev_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

export function generateTaskId(): string {
  return `task_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

export function generateTransactionId(): string {
  return `tx_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * Classify device tier based on capabilities
 */
export function classifyDeviceTier(capabilities: DeviceCapabilities): DeviceTier {
  const { compute, memory, platform } = capabilities;

  // Power tier: High-end GPUs (4090, A100, etc.)
  if (compute >= 40 && memory >= 16 && (platform === "cuda" || platform === "metal")) {
    return "power";
  }

  // Standard tier: Mid-range GPUs, Apple Silicon with good memory
  if (compute >= 5 && memory >= 8) {
    return "standard";
  }

  // Crowd tier: Everything else (browsers, phones, low-end)
  return "crowd";
}

/**
 * Calculate credit reward for task
 */
export function calculateTaskReward(type: TaskType, config: TaskConfig): number {
  const baseRewards: Record<string, number> = {
    full_inference: 10,
    shard_inference: 5,
    draft_tokens: 1,
    embedding: 2,
    embeddings: 2,
    validation: 0.5,
    draft_generation: 0.8,
    draft_verification: 3,
    classification: 1.5,
    summarization: 3,
    simulation: 25, // High reward - GPU-intensive CFD/physics simulations
  };

  let reward = baseRewards[type] || 1; // Default to 1 credit if type unknown

  // Scale by token count (for inference tasks)
  if (config.maxTokens > 100) {
    reward *= config.maxTokens / 100;
  }

  return Math.round(reward * 10) / 10; // Round to 1 decimal
}

/**
 * Default availability settings
 */
export const DEFAULT_AVAILABILITY: DeviceAvailability = {
  wifiOnly: true,
  chargingOnly: false,
  minBattery: 20,
  maxUtilization: 80,
};
