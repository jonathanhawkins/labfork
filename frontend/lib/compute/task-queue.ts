/**
 * Task Queue
 *
 * Priority queue implementation with fair scheduling for distributed compute tasks.
 * Handles task routing, timeout, and retry logic.
 */

import type { ComputeTask, TaskStatus, DeviceTier } from "./types";

/**
 * Task queue entry with metadata
 */
export interface QueuedTask {
  /** The task */
  task: ComputeTask;
  /** Number of retry attempts */
  retryCount: number;
  /** Timeout timestamp (when task should be considered timed out) */
  timeoutAt?: number;
  /** Task was reassigned (used for priority boost) */
  wasReassigned: boolean;
}

/**
 * Queue statistics
 */
export interface QueueStats {
  /** Total tasks in queue */
  totalTasks: number;
  /** Tasks by priority level */
  tasksByPriority: Record<number, number>;
  /** Tasks by tier requirement */
  tasksByTier: Record<DeviceTier | "any", number>;
  /** Average wait time (ms) */
  avgWaitTime: number;
  /** Tasks waiting for retry */
  tasksAwaitingRetry: number;
}

/**
 * Queue configuration
 */
export interface QueueConfig {
  /** Maximum retry attempts per task */
  maxRetries: number;
  /** Task timeout in milliseconds */
  taskTimeout: number;
  /** Enable priority decay (lower priority of old tasks) */
  enablePriorityDecay: boolean;
  /** Priority decay rate (per minute) */
  priorityDecayRate: number;
}

/**
 * Default queue configuration
 */
const DEFAULT_CONFIG: QueueConfig = {
  maxRetries: 3,
  taskTimeout: 300000, // 5 minutes
  enablePriorityDecay: false,
  priorityDecayRate: 0.1,
};

/**
 * Priority queue for compute tasks
 *
 * Maintains tasks in priority order with support for:
 * - Fair scheduling across different priority levels
 * - Timeout and retry logic
 * - Priority boost for reassigned tasks
 * - Filtering by device tier requirements
 */
export class TaskQueue {
  private queue: QueuedTask[] = [];
  private config: QueueConfig;
  private taskMap: Map<string, QueuedTask> = new Map();
  private lastDecayTime: number = Date.now();

  constructor(config: Partial<QueueConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Add a task to the queue
   */
  enqueue(task: ComputeTask): void {
    // Check if task already in queue
    if (this.taskMap.has(task.id)) {
      console.warn(`Task ${task.id} already in queue`);
      return;
    }

    const queuedTask: QueuedTask = {
      task,
      retryCount: 0,
      wasReassigned: false,
    };

    this.taskMap.set(task.id, queuedTask);
    this.insertTask(queuedTask);
  }

  /**
   * Re-enqueue a failed/timed-out task for retry
   */
  retry(taskId: string, reason: "timeout" | "failure"): boolean {
    const queuedTask = this.taskMap.get(taskId);
    if (!queuedTask) {
      return false;
    }

    // Check retry limit
    if (queuedTask.retryCount >= this.config.maxRetries) {
      console.warn(`Task ${taskId} exceeded max retries`);
      this.taskMap.delete(taskId);
      return false;
    }

    queuedTask.retryCount++;
    queuedTask.wasReassigned = true;
    queuedTask.task.status = "pending";
    queuedTask.task.assignedDeviceId = undefined;
    queuedTask.task.assignedAt = undefined;
    queuedTask.timeoutAt = undefined;

    // Boost priority for retries
    queuedTask.task.priority += 1;

    // Re-insert into queue
    this.removeFromQueue(taskId);
    this.insertTask(queuedTask);

    return true;
  }

  /**
   * Remove a task from the queue
   */
  dequeue(taskId: string): QueuedTask | null {
    const queuedTask = this.taskMap.get(taskId);
    if (!queuedTask) {
      return null;
    }

    this.removeFromQueue(taskId);
    this.taskMap.delete(taskId);

    return queuedTask;
  }

  /**
   * Peek at the next task without removing it
   * Optionally filter by minimum tier requirement
   */
  peek(minTier?: DeviceTier): QueuedTask | null {
    if (this.queue.length === 0) {
      return null;
    }

    if (!minTier) {
      return this.queue[0];
    }

    // Filter by tier requirement
    const tierOrder: DeviceTier[] = ["crowd", "standard", "power"];
    const minTierIndex = tierOrder.indexOf(minTier);

    for (const queuedTask of this.queue) {
      const taskMinTier = queuedTask.task.config.minTier;
      if (!taskMinTier) {
        return queuedTask; // No tier requirement
      }

      const taskTierIndex = tierOrder.indexOf(taskMinTier);
      if (minTierIndex >= taskTierIndex) {
        return queuedTask; // Device tier meets requirement
      }
    }

    return null;
  }

  /**
   * Get all tasks matching criteria
   */
  filter(predicate: (task: QueuedTask) => boolean): QueuedTask[] {
    return this.queue.filter(predicate);
  }

  /**
   * Get task by ID
   */
  get(taskId: string): QueuedTask | null {
    return this.taskMap.get(taskId) || null;
  }

  /**
   * Get queue size
   */
  size(): number {
    return this.queue.length;
  }

  /**
   * Check if queue is empty
   */
  isEmpty(): boolean {
    return this.queue.length === 0;
  }

  /**
   * Clear the queue
   */
  clear(): void {
    this.queue = [];
    this.taskMap.clear();
  }

  /**
   * Check for timed-out tasks and retry them
   */
  processTimeouts(): string[] {
    const now = Date.now();
    const timedOutTasks: string[] = [];

    Array.from(this.taskMap.values()).forEach((queuedTask) => {
      if (queuedTask.timeoutAt && now >= queuedTask.timeoutAt) {
        timedOutTasks.push(queuedTask.task.id);
      }
    });

    // Retry timed-out tasks
    for (const taskId of timedOutTasks) {
      const retried = this.retry(taskId, "timeout");
      if (!retried) {
        // Task exceeded max retries, mark as failed
        const queuedTask = this.taskMap.get(taskId);
        if (queuedTask) {
          queuedTask.task.status = "timeout";
        }
      }
    }

    return timedOutTasks;
  }

  /**
   * Apply priority decay to old tasks
   */
  applyPriorityDecay(): void {
    if (!this.config.enablePriorityDecay) {
      return;
    }

    const now = Date.now();
    const minutesElapsed = (now - this.lastDecayTime) / 60000;

    if (minutesElapsed < 1) {
      return; // Only decay once per minute
    }

    let needsReorder = false;

    Array.from(this.taskMap.values()).forEach((queuedTask) => {
      const taskAge = (now - new Date(queuedTask.task.createdAt).getTime()) / 60000;
      const decay = taskAge * this.config.priorityDecayRate;

      if (decay > 0) {
        const oldPriority = queuedTask.task.priority;
        queuedTask.task.priority = Math.max(0, oldPriority - decay);
        if (queuedTask.task.priority !== oldPriority) {
          needsReorder = true;
        }
      }
    });

    this.lastDecayTime = now;

    if (needsReorder) {
      this.reorderQueue();
    }
  }

  /**
   * Get queue statistics
   */
  getStats(): QueueStats {
    const tasksByPriority: Record<number, number> = {};
    const tasksByTier: Record<DeviceTier | "any", number> = {
      power: 0,
      standard: 0,
      crowd: 0,
      any: 0,
    };

    let totalWaitTime = 0;
    const now = Date.now();

    for (const queuedTask of this.queue) {
      // Count by priority
      const priority = Math.floor(queuedTask.task.priority);
      tasksByPriority[priority] = (tasksByPriority[priority] || 0) + 1;

      // Count by tier
      const minTier = queuedTask.task.config.minTier || "any";
      tasksByTier[minTier]++;

      // Calculate wait time
      const createdAt = new Date(queuedTask.task.createdAt).getTime();
      totalWaitTime += now - createdAt;
    }

    return {
      totalTasks: this.queue.length,
      tasksByPriority,
      tasksByTier,
      avgWaitTime: this.queue.length > 0 ? totalWaitTime / this.queue.length : 0,
      tasksAwaitingRetry: this.queue.filter((qt) => qt.retryCount > 0).length,
    };
  }

  /**
   * Mark task as assigned and set timeout
   */
  markAssigned(taskId: string): void {
    const queuedTask = this.taskMap.get(taskId);
    if (queuedTask) {
      queuedTask.timeoutAt = Date.now() + this.config.taskTimeout;
    }
  }

  /**
   * Export queue state (for debugging/monitoring)
   */
  toArray(): QueuedTask[] {
    return [...this.queue];
  }

  /**
   * Insert task into queue maintaining priority order
   */
  private insertTask(queuedTask: QueuedTask): void {
    const priority = queuedTask.task.priority;

    // Binary search for insertion point
    let left = 0;
    let right = this.queue.length;

    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (this.queue[mid].task.priority >= priority) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }

    this.queue.splice(left, 0, queuedTask);
  }

  /**
   * Remove task from queue array
   */
  private removeFromQueue(taskId: string): void {
    const index = this.queue.findIndex((qt) => qt.task.id === taskId);
    if (index >= 0) {
      this.queue.splice(index, 1);
    }
  }

  /**
   * Re-order queue after priority changes
   */
  private reorderQueue(): void {
    this.queue.sort((a, b) => b.task.priority - a.task.priority);
  }
}
