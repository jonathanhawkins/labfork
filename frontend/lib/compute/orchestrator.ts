/**
 * Compute Orchestrator
 *
 * In-memory orchestrator for the distributed compute network.
 * Handles device registration, task queue, and assignment.
 */

import type {
  ComputeDevice,
  ComputeTask,
  DeviceTier,
  TaskType,
  TaskStatus,
  DeviceStatus,
  NetworkStats,
  RegisterDeviceRequest,
  SubmitTaskRequest,
  CompleteTaskRequest,
  TaskAssignment,
  HeartbeatRequest,
  UserCredits,
  CreditTransaction,
} from "./types";
import {
  generateDeviceId,
  generateTaskId,
  generateTransactionId,
  classifyDeviceTier,
  calculateTaskReward,
  DEFAULT_AVAILABILITY,
} from "./types";

/**
 * In-memory storage for MVP
 * In production, use Redis + PostgreSQL
 */
class ComputeOrchestrator {
  private devices: Map<string, ComputeDevice> = new Map();
  private tasks: Map<string, ComputeTask> = new Map();
  private taskQueue: string[] = []; // Task IDs in priority order
  private credits: Map<string, UserCredits> = new Map();
  private transactions: CreditTransaction[] = [];

  // Stats tracking
  private completedToday = 0;
  private creditsToday = 0;
  private lastStatsReset = new Date().toDateString();

  constructor() {
    // Reset daily stats at midnight
    this.checkDailyReset();
  }

  private checkDailyReset() {
    const today = new Date().toDateString();
    if (today !== this.lastStatsReset) {
      this.completedToday = 0;
      this.creditsToday = 0;
      this.lastStatsReset = today;
    }
  }

  /**
   * Register a new device
   */
  registerDevice(request: RegisterDeviceRequest, userId?: string): ComputeDevice {
    const id = generateDeviceId();
    const now = new Date().toISOString();

    const device: ComputeDevice = {
      id,
      userId,
      name: request.name,
      tier: classifyDeviceTier(request.capabilities),
      capabilities: request.capabilities,
      availability: { ...DEFAULT_AVAILABILITY, ...request.availability },
      status: "online",
      lastHeartbeat: now,
      registeredAt: now,
      stats: {
        tasksCompleted: 0,
        creditsEarned: 0,
        totalComputeTime: 0,
      },
    };

    this.devices.set(id, device);

    // Initialize credits for user if needed
    if (userId && !this.credits.has(userId)) {
      this.credits.set(userId, {
        userId,
        balance: 100, // Free starter credits
        totalEarned: 100,
        totalSpent: 0,
        updatedAt: now,
      });
    }

    return device;
  }

  /**
   * Update device heartbeat
   */
  heartbeat(request: HeartbeatRequest): ComputeDevice | null {
    const device = this.devices.get(request.deviceId);
    if (!device) return null;

    device.lastHeartbeat = new Date().toISOString();
    device.status = request.status;

    // Update task progress if provided
    if (request.taskProgress !== undefined && device.currentTaskId) {
      const task = this.tasks.get(device.currentTaskId);
      if (task) {
        // Could store progress on task if needed
      }
    }

    return device;
  }

  /**
   * Get device by ID
   */
  getDevice(deviceId: string): ComputeDevice | null {
    return this.devices.get(deviceId) || null;
  }

  /**
   * Get all online devices
   */
  getOnlineDevices(): ComputeDevice[] {
    const now = Date.now();
    const timeoutMs = 60000; // 1 minute timeout

    return Array.from(this.devices.values()).filter((device) => {
      const lastSeen = new Date(device.lastHeartbeat).getTime();
      const isRecent = now - lastSeen < timeoutMs;

      // Mark offline if not seen recently
      if (!isRecent && device.status !== "offline") {
        device.status = "offline";
        // Reassign any task
        if (device.currentTaskId) {
          this.reassignTask(device.currentTaskId);
          device.currentTaskId = undefined;
        }
      }

      return isRecent && device.status !== "offline";
    });
  }

  /**
   * Get all devices including recently offline (for visualization)
   * Returns devices seen within the last 5 minutes
   */
  getAllDevicesForVisualization(): ComputeDevice[] {
    const now = Date.now();
    const visualizationTimeoutMs = 300000; // 5 minutes - keep showing devices for a bit after disconnect

    return Array.from(this.devices.values()).filter((device) => {
      const lastSeen = new Date(device.lastHeartbeat).getTime();
      return now - lastSeen < visualizationTimeoutMs;
    });
  }

  /**
   * Submit a new task
   */
  submitTask(request: SubmitTaskRequest, submitterId: string): ComputeTask {
    const id = generateTaskId();
    const now = new Date().toISOString();
    const reward = calculateTaskReward(request.type, request.config);

    const task: ComputeTask = {
      id,
      type: request.type,
      input: request.input,
      config: request.config,
      status: "pending",
      priority: request.priority || 1,
      reward,
      createdAt: now,
      submitterId,
    };

    this.tasks.set(id, task);
    this.insertTaskInQueue(id, task.priority);

    // Try immediate assignment
    this.assignPendingTasks();

    return task;
  }

  /**
   * Insert task in queue maintaining priority order
   */
  private insertTaskInQueue(taskId: string, priority: number) {
    // Find insertion point (higher priority first)
    let insertIndex = this.taskQueue.length;
    for (let i = 0; i < this.taskQueue.length; i++) {
      const existingTask = this.tasks.get(this.taskQueue[i]);
      if (existingTask && existingTask.priority < priority) {
        insertIndex = i;
        break;
      }
    }
    this.taskQueue.splice(insertIndex, 0, taskId);
  }

  /**
   * Get task by ID
   */
  getTask(taskId: string): ComputeTask | null {
    return this.tasks.get(taskId) || null;
  }

  /**
   * Assign pending tasks to available devices
   */
  private assignPendingTasks() {
    const availableDevices = this.getOnlineDevices().filter(
      (d) => d.status === "online" && !d.currentTaskId
    );

    for (const taskId of [...this.taskQueue]) {
      const task = this.tasks.get(taskId);
      if (!task || task.status !== "pending") continue;

      // Find best device for this task
      const device = this.findBestDevice(task, availableDevices);
      if (!device) continue;

      // Assign task
      task.status = "assigned";
      task.assignedDeviceId = device.id;
      task.assignedAt = new Date().toISOString();

      device.status = "busy";
      device.currentTaskId = taskId;

      // Remove from queue
      const queueIndex = this.taskQueue.indexOf(taskId);
      if (queueIndex > -1) {
        this.taskQueue.splice(queueIndex, 1);
      }

      // Remove device from available pool
      const deviceIndex = availableDevices.indexOf(device);
      if (deviceIndex > -1) {
        availableDevices.splice(deviceIndex, 1);
      }
    }
  }

  /**
   * Find best device for a task
   */
  private findBestDevice(
    task: ComputeTask,
    availableDevices: ComputeDevice[]
  ): ComputeDevice | null {
    // Filter by minimum tier if specified
    let candidates = availableDevices;
    if (task.config.minTier) {
      const tierOrder: DeviceTier[] = ["crowd", "standard", "power"];
      const minTierIndex = tierOrder.indexOf(task.config.minTier);
      candidates = candidates.filter(
        (d) => tierOrder.indexOf(d.tier) >= minTierIndex
      );
    }

    if (candidates.length === 0) return null;

    // Score devices (higher is better)
    const scored = candidates.map((device) => {
      let score = 0;

      // Prefer higher compute
      score += device.capabilities.compute * 10;

      // Prefer devices with model cached
      if (device.capabilities.cachedModels.includes(task.config.modelId)) {
        score += 100;
      }

      // Prefer higher tier for complex tasks
      if (task.type === "full_inference" || task.type === "shard_inference") {
        if (device.tier === "power") score += 50;
        else if (device.tier === "standard") score += 25;
      }

      // Slight randomization to distribute load
      score += Math.random() * 5;

      return { device, score };
    });

    // Sort by score descending
    scored.sort((a, b) => b.score - a.score);

    return scored[0]?.device || null;
  }

  /**
   * Get next task assignment for a device
   */
  getNextTask(deviceId: string): TaskAssignment | null {
    const device = this.devices.get(deviceId);
    if (!device) return null;

    // Check if device already has a task
    if (device.currentTaskId) {
      const task = this.tasks.get(device.currentTaskId);
      if (task && (task.status === "assigned" || task.status === "processing")) {
        return { task };
      }
    }

    // Try to assign a new task
    this.assignPendingTasks();

    // Check again
    if (device.currentTaskId) {
      const task = this.tasks.get(device.currentTaskId);
      if (task) {
        task.status = "processing";
        return { task };
      }
    }

    return null;
  }

  /**
   * Complete a task
   */
  completeTask(request: CompleteTaskRequest): { success: boolean; credits?: number } {
    const task = this.tasks.get(request.taskId);
    const device = this.devices.get(request.deviceId);

    if (!task || !device) {
      return { success: false };
    }

    if (task.assignedDeviceId !== device.id) {
      return { success: false };
    }

    const now = new Date().toISOString();

    if (request.success && request.result) {
      task.status = "completed";
      task.result = request.result;
      task.completedAt = now;

      // Update device stats
      device.stats.tasksCompleted++;
      device.stats.totalComputeTime += request.result.metrics.computeTime / 1000;

      // Award credits
      if (device.userId) {
        this.awardCredits(device.userId, task.reward, task.id);
        device.stats.creditsEarned += task.reward;
      }

      // Update daily stats
      this.checkDailyReset();
      this.completedToday++;
      this.creditsToday += task.reward;
    } else {
      task.status = "failed";
      // Could implement retry logic here
    }

    // Free up device
    device.status = "online";
    device.currentTaskId = undefined;

    // Try to assign more tasks
    this.assignPendingTasks();

    return {
      success: true,
      credits: request.success ? task.reward : 0,
    };
  }

  /**
   * Reassign a task (e.g., when device goes offline)
   */
  private reassignTask(taskId: string) {
    const task = this.tasks.get(taskId);
    if (!task) return;

    task.status = "pending";
    task.assignedDeviceId = undefined;
    task.assignedAt = undefined;

    // Re-add to queue with higher priority
    this.insertTaskInQueue(taskId, task.priority + 1);
  }

  /**
   * Award credits to a user
   */
  private awardCredits(userId: string, amount: number, taskId?: string) {
    let userCredits = this.credits.get(userId);
    if (!userCredits) {
      userCredits = {
        userId,
        balance: 0,
        totalEarned: 0,
        totalSpent: 0,
        updatedAt: new Date().toISOString(),
      };
      this.credits.set(userId, userCredits);
    }

    userCredits.balance += amount;
    userCredits.totalEarned += amount;
    userCredits.updatedAt = new Date().toISOString();

    // Record transaction
    this.transactions.push({
      id: generateTransactionId(),
      userId,
      amount,
      type: "earn",
      taskId,
      description: `Earned ${amount} credits for completing task`,
      createdAt: new Date().toISOString(),
    });
  }

  /**
   * Spend credits (for submitting tasks)
   */
  spendCredits(userId: string, amount: number, description: string): boolean {
    const userCredits = this.credits.get(userId);
    if (!userCredits || userCredits.balance < amount) {
      return false;
    }

    userCredits.balance -= amount;
    userCredits.totalSpent += amount;
    userCredits.updatedAt = new Date().toISOString();

    this.transactions.push({
      id: generateTransactionId(),
      userId,
      amount: -amount,
      type: "spend",
      description,
      createdAt: new Date().toISOString(),
    });

    return true;
  }

  /**
   * Get user credits
   */
  getUserCredits(userId: string): UserCredits | null {
    return this.credits.get(userId) || null;
  }

  /**
   * Get network statistics
   */
  getNetworkStats(): NetworkStats {
    this.checkDailyReset();

    const allDevices = Array.from(this.devices.values());
    const onlineDevices = this.getOnlineDevices();

    const devicesByTier: Record<DeviceTier, number> = {
      power: 0,
      standard: 0,
      crowd: 0,
    };

    let totalCompute = 0;
    for (const device of onlineDevices) {
      devicesByTier[device.tier]++;
      totalCompute += device.capabilities.compute;
    }

    const pendingTasks = this.taskQueue.length;
    const processingTasks = Array.from(this.tasks.values()).filter(
      (t) => t.status === "processing" || t.status === "assigned"
    ).length;

    return {
      totalDevices: allDevices.length,
      onlineDevices: onlineDevices.length,
      devicesByTier,
      totalCompute: Math.round(totalCompute * 10) / 10,
      pendingTasks,
      processingTasks,
      completedToday: this.completedToday,
      creditsToday: Math.round(this.creditsToday * 10) / 10,
    };
  }

  /**
   * Get tasks for a submitter
   */
  getTasksBySubmitter(submitterId: string): ComputeTask[] {
    return Array.from(this.tasks.values()).filter(
      (t) => t.submitterId === submitterId
    );
  }

  /**
   * Get all pending tasks in the queue
   */
  getPendingTasks(limit = 50): ComputeTask[] {
    return this.taskQueue
      .slice(0, limit)
      .map((id) => this.tasks.get(id))
      .filter((t): t is ComputeTask => t !== undefined);
  }

  /**
   * Get all tasks (for admin/debugging)
   */
  getAllTasks(limit = 100): ComputeTask[] {
    return Array.from(this.tasks.values())
      .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
      .slice(0, limit);
  }

  /**
   * Get device stats for leaderboard
   */
  getLeaderboard(limit = 10): Array<{
    deviceId: string;
    name: string;
    tier: DeviceTier;
    tasksCompleted: number;
    creditsEarned: number;
  }> {
    return Array.from(this.devices.values())
      .sort((a, b) => b.stats.creditsEarned - a.stats.creditsEarned)
      .slice(0, limit)
      .map((d) => ({
        deviceId: d.id,
        name: d.name,
        tier: d.tier,
        tasksCompleted: d.stats.tasksCompleted,
        creditsEarned: d.stats.creditsEarned,
      }));
  }

  /**
   * Auto-generate tasks to keep the queue filled
   * Creates meaningful work based on LabFork activities
   */
  private autoTaskInterval: NodeJS.Timeout | null = null;
  private autoTaskEnabled = false;

  startAutoTaskGeneration() {
    if (this.autoTaskInterval) return;

    this.autoTaskEnabled = true;
    console.log('[Orchestrator] Starting auto-task generation');

    // Generate initial batch - always have tasks ready for incoming devices
    this.generateTaskBatch(10);

    // Continue generating every 10 seconds
    this.autoTaskInterval = setInterval(() => {
      if (!this.autoTaskEnabled) return;

      const onlineDevices = this.getOnlineDevices().length;
      const pendingTasks = this.taskQueue.length;

      // Always keep at least 5 tasks in queue, more if devices are online
      const targetQueueSize = Math.max(5, onlineDevices * 3);
      if (pendingTasks < targetQueueSize) {
        const toGenerate = Math.min(10, targetQueueSize - pendingTasks);
        this.generateTaskBatch(toGenerate);
      }
    }, 10000);
  }

  stopAutoTaskGeneration() {
    this.autoTaskEnabled = false;
    if (this.autoTaskInterval) {
      clearInterval(this.autoTaskInterval);
      this.autoTaskInterval = null;
    }
    console.log('[Orchestrator] Stopped auto-task generation');
  }

  /**
   * Generate a batch of realistic tasks
   */
  private generateTaskBatch(count = 5) {
    const taskTypes: Array<{
      type: TaskType;
      generator: () => { input: any; config: any };
    }> = [
      {
        type: 'embeddings',
        generator: () => ({
          input: {
            text: this.getRandomPaperAbstract(),
            taskDescription: 'Generate embeddings for paper similarity search',
          },
          config: {
            modelId: 'all-MiniLM-L6-v2',
            maxTokens: 512,
          },
        }),
      },
      {
        type: 'summarization',
        generator: () => ({
          input: {
            text: this.getRandomPaperAbstract(),
            taskDescription: 'Summarize research paper for lab digest',
          },
          config: {
            modelId: 'Qwen/Qwen2.5-0.5B-Instruct',
            maxTokens: 256,
          },
        }),
      },
      {
        type: 'draft_tokens',
        generator: () => ({
          input: {
            prompt: this.getRandomPrompt(),
            taskDescription: 'Generate draft tokens for speculative decoding',
          },
          config: {
            modelId: 'Qwen/Qwen2.5-0.5B-Instruct',
            maxTokens: 64,
            draftCount: 4,
          },
        }),
      },
      {
        type: 'classification',
        generator: () => ({
          input: {
            text: this.getRandomTechniqueDescription(),
            categories: ['audio', 'vision', 'nlp', 'multimodal', 'optimization'],
            taskDescription: 'Classify research technique domain',
          },
          config: {
            modelId: 'Qwen/Qwen2.5-0.5B-Instruct',
            maxTokens: 32,
          },
        }),
      },
    ];

    for (let i = 0; i < count; i++) {
      const taskDef = taskTypes[Math.floor(Math.random() * taskTypes.length)];
      const { input, config } = taskDef.generator();

      this.submitTask(
        {
          type: taskDef.type,
          input,
          config,
          priority: Math.floor(Math.random() * 3) + 1, // 1-3 priority
        },
        'system' // Submitted by system
      );
    }

    console.log(`[Orchestrator] Generated ${count} tasks, queue size: ${this.taskQueue.length}`);
  }

  /**
   * Sample paper abstracts for realistic tasks
   */
  private getRandomPaperAbstract(): string {
    const abstracts = [
      "We present a novel approach to emotional speech synthesis using disentangled prosody representations. By separating content, speaker, and emotion embeddings, we achieve controllable synthesis with natural expressiveness.",
      "This paper introduces a cross-lingual voice conversion system that preserves prosodic patterns across languages while adapting phonetic content. Our method uses attention mechanisms to align prosodic features.",
      "We propose a zero-shot text-to-speech system capable of cloning voices from a single reference audio. The system uses a neural codec language model to capture speaker characteristics.",
      "Our work presents an efficient method for real-time pitch modification in speech synthesis, enabling dynamic emotional control without artifacts.",
      "We introduce a multi-speaker synthesis framework that learns shared prosody representations across speakers, enabling style transfer and emotion interpolation.",
    ];
    return abstracts[Math.floor(Math.random() * abstracts.length)];
  }

  /**
   * Sample prompts for draft token generation
   */
  private getRandomPrompt(): string {
    const prompts = [
      "Explain how prosody affects emotional perception in synthesized speech",
      "Describe the architecture of a typical voice cloning system",
      "What are the key differences between autoregressive and non-autoregressive TTS",
      "How can attention mechanisms improve speech synthesis quality",
      "Summarize recent advances in emotional speech synthesis",
    ];
    return prompts[Math.floor(Math.random() * prompts.length)];
  }

  /**
   * Sample technique descriptions for classification
   */
  private getRandomTechniqueDescription(): string {
    const techniques = [
      "A neural network approach using mel-spectrogram prediction with WaveNet vocoder for high-quality speech synthesis",
      "Contrastive learning framework for learning speaker-independent emotion representations from unlabeled audio",
      "Transformer-based architecture with cross-attention for aligning text and speech features in TTS",
      "Diffusion model for generating natural prosody patterns conditioned on text and speaker embeddings",
      "VAE-based disentanglement of speaker identity and emotional expression in speech signals",
    ];
    return techniques[Math.floor(Math.random() * techniques.length)];
  }

  /**
   * Get queue status for debugging
   */
  getQueueStatus(): {
    pending: number;
    processing: number;
    autoTaskEnabled: boolean;
    devices: number;
  } {
    return {
      pending: this.taskQueue.length,
      processing: Array.from(this.tasks.values()).filter(
        (t) => t.status === 'processing' || t.status === 'assigned'
      ).length,
      autoTaskEnabled: this.autoTaskEnabled,
      devices: this.getOnlineDevices().length,
    };
  }
}

// Singleton instance - use globalThis to persist across Next.js hot reloads
declare global {
  var __computeOrchestrator: ComputeOrchestrator | undefined;
}

export function getOrchestrator(): ComputeOrchestrator {
  if (!globalThis.__computeOrchestrator) {
    console.log('[Orchestrator] Creating new instance');
    globalThis.__computeOrchestrator = new ComputeOrchestrator();
    // DEPRECATED: Auto-task generation removed. Crowd tasks are now generated
    // by the Cloudflare Workers cron handler (workers/src/index.ts) and stored
    // in D1. The in-memory orchestrator is only used for local API compatibility.
  }
  return globalThis.__computeOrchestrator;
}

export { ComputeOrchestrator };
