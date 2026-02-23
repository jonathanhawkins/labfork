/**
 * Meta-Task Service
 *
 * Manages meta-tasks that span multiple labs, including creation,
 * participant management, objective tracking, and result integration.
 */

import {
  MetaTask,
  MetaTaskId,
  MetaTaskStatus,
  MetaTaskCategory,
  MetaTaskRequirements,
  MetaTaskTimeline,
  Participant,
  ParticipantId,
  ParticipantRole,
  ParticipantStatus,
  ParticipantCommitment,
  ParticipantContribution,
  JointObjective,
  ObjectiveStatus,
  ObjectiveMilestone,
  Deliverable,
  SharedResult,
  ResultType,
  ResultQuality,
  CompletionCriterion,
  WorkflowEvent,
  WorkflowEventType,
  META_TASK_STATUS_FLOW,
  createMetaTaskId,
  createParticipantId,
  createDefaultRequirements,
  createDefaultTimeline,
  createEmptyResultQuality,
  createEmptyParticipantCommitment,
} from "./types";

// ============================================================================
// Meta-Task Manager
// ============================================================================

export interface MetaTaskManager {
  tasks: Map<MetaTaskId, MetaTask>;
  events: WorkflowEvent[];
}

export function createMetaTaskManager(): MetaTaskManager {
  return {
    tasks: new Map(),
    events: [],
  };
}

// ============================================================================
// Task Creation
// ============================================================================

export interface CreateMetaTaskInput {
  title: string;
  description: string;
  category: MetaTaskCategory;
  leadLabId: string;
  leadLabName: string;
  leadExpertise: string[];
  requirements?: Partial<MetaTaskRequirements>;
  timeline?: Partial<MetaTaskTimeline>;
  tags?: string[];
  domains?: string[];
  visibility?: "public" | "private" | "invite_only";
}

export function createMetaTask(
  manager: MetaTaskManager,
  input: CreateMetaTaskInput
): MetaTask {
  const now = new Date().toISOString();
  const id = createMetaTaskId();
  const leadId = createParticipantId();

  const lead: Participant = {
    id: leadId,
    labId: input.leadLabId,
    labName: input.leadLabName,
    role: "lead",
    status: "active",
    expertise: input.leadExpertise,
    commitment: createEmptyParticipantCommitment(),
    joinedAt: now,
    lastActiveAt: now,
    contributions: [],
    assignedObjectives: [],
  };

  const task: MetaTask = {
    id,
    title: input.title,
    description: input.description,
    category: input.category,
    status: "proposed",
    lead,
    participants: [lead],
    objectives: [],
    results: [],
    requirements: {
      ...createDefaultRequirements(),
      ...input.requirements,
    },
    timeline: {
      ...createDefaultTimeline(),
      ...input.timeline,
    },
    completionCriteria: [],
    tags: input.tags || [],
    domains: input.domains || [],
    visibility: input.visibility || "public",
    createdAt: now,
    updatedAt: now,
  };

  manager.tasks.set(id, task);

  recordEvent(manager, {
    metaTaskId: id,
    type: "task_created",
    actor: leadId,
    description: `Meta-task "${input.title}" created`,
    data: { category: input.category },
  });

  return task;
}

// ============================================================================
// Status Management
// ============================================================================

export function canTransitionTo(
  currentStatus: MetaTaskStatus,
  newStatus: MetaTaskStatus
): boolean {
  const allowedTransitions = META_TASK_STATUS_FLOW[currentStatus];
  return allowedTransitions.includes(newStatus);
}

export function transitionStatus(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  newStatus: MetaTaskStatus,
  actorId: ParticipantId
): MetaTask | null {
  const task = manager.tasks.get(taskId);
  if (!task) return null;

  if (!canTransitionTo(task.status, newStatus)) {
    throw new Error(
      `Cannot transition from ${task.status} to ${newStatus}`
    );
  }

  const oldStatus = task.status;
  task.status = newStatus;
  task.updatedAt = new Date().toISOString();

  if (newStatus === "active" && !task.startedAt) {
    task.startedAt = task.updatedAt;
  }
  if (newStatus === "completed") {
    task.completedAt = task.updatedAt;
  }

  recordEvent(manager, {
    metaTaskId: taskId,
    type: "status_changed",
    actor: actorId,
    description: `Status changed from ${oldStatus} to ${newStatus}`,
    data: { oldStatus, newStatus },
  });

  return task;
}

export function startRecruiting(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  actorId: ParticipantId
): MetaTask | null {
  return transitionStatus(manager, taskId, "recruiting", actorId);
}

export function acceptTask(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  actorId: ParticipantId
): MetaTask | null {
  const task = manager.tasks.get(taskId);
  if (!task) return null;

  // Check minimum participants
  const activeParticipants = task.participants.filter(
    (p) => p.status === "active"
  );
  if (activeParticipants.length < task.requirements.minParticipants) {
    throw new Error(
      `Need at least ${task.requirements.minParticipants} participants`
    );
  }

  return transitionStatus(manager, taskId, "accepted", actorId);
}

export function startTask(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  actorId: ParticipantId
): MetaTask | null {
  return transitionStatus(manager, taskId, "active", actorId);
}

export function submitForReview(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  actorId: ParticipantId
): MetaTask | null {
  const task = manager.tasks.get(taskId);
  if (!task) return null;

  // Check if all objectives are completed
  const incompleteObjectives = task.objectives.filter(
    (o) => o.status !== "completed"
  );
  if (incompleteObjectives.length > 0) {
    throw new Error(
      `${incompleteObjectives.length} objectives still incomplete`
    );
  }

  return transitionStatus(manager, taskId, "reviewing", actorId);
}

export function completeTask(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  actorId: ParticipantId
): MetaTask | null {
  const task = manager.tasks.get(taskId);
  if (!task) return null;

  // Check completion criteria
  const unmetCriteria = task.completionCriteria.filter((c) => !c.met);
  if (unmetCriteria.length > 0) {
    throw new Error(`${unmetCriteria.length} completion criteria not met`);
  }

  return transitionStatus(manager, taskId, "completed", actorId);
}

// ============================================================================
// Participant Management
// ============================================================================

export interface JoinTaskInput {
  labId: string;
  labName: string;
  expertise: string[];
  role: ParticipantRole;
  commitment: ParticipantCommitment;
}

export function addParticipant(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  input: JoinTaskInput
): Participant | null {
  const task = manager.tasks.get(taskId);
  if (!task) return null;

  // Check max participants
  if (task.participants.length >= task.requirements.maxParticipants) {
    throw new Error("Maximum participants reached");
  }

  // Check if lab already participating
  if (task.participants.some((p) => p.labId === input.labId)) {
    throw new Error("Lab already participating");
  }

  const now = new Date().toISOString();
  const participant: Participant = {
    id: createParticipantId(),
    labId: input.labId,
    labName: input.labName,
    role: input.role,
    status: "active",
    expertise: input.expertise,
    commitment: input.commitment,
    joinedAt: now,
    lastActiveAt: now,
    contributions: [],
    assignedObjectives: [],
  };

  task.participants.push(participant);
  task.updatedAt = now;

  recordEvent(manager, {
    metaTaskId: taskId,
    type: "participant_joined",
    actor: participant.id,
    description: `${input.labName} joined as ${input.role}`,
    data: { labId: input.labId, role: input.role },
  });

  return participant;
}

export function removeParticipant(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  participantId: ParticipantId,
  reason?: string
): boolean {
  const task = manager.tasks.get(taskId);
  if (!task) return false;

  const participant = task.participants.find((p) => p.id === participantId);
  if (!participant) return false;

  // Cannot remove lead
  if (participant.role === "lead") {
    throw new Error("Cannot remove task lead");
  }

  participant.status = "withdrawn";
  task.updatedAt = new Date().toISOString();

  // Unassign their objectives
  for (const objective of task.objectives) {
    objective.assignedTo = objective.assignedTo.filter(
      (id) => id !== participantId
    );
  }

  recordEvent(manager, {
    metaTaskId: taskId,
    type: "participant_left",
    actor: participantId,
    description: `${participant.labName} left the task${reason ? `: ${reason}` : ""}`,
    data: { labId: participant.labId, reason },
  });

  return true;
}

export function updateParticipantRole(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  participantId: ParticipantId,
  newRole: ParticipantRole,
  actorId: ParticipantId
): Participant | null {
  const task = manager.tasks.get(taskId);
  if (!task) return null;

  const participant = task.participants.find((p) => p.id === participantId);
  if (!participant) return null;

  // Only lead can promote to co-lead
  if (newRole === "co-lead") {
    const actor = task.participants.find((p) => p.id === actorId);
    if (!actor || actor.role !== "lead") {
      throw new Error("Only lead can assign co-lead role");
    }
  }

  const oldRole = participant.role;
  participant.role = newRole;
  task.updatedAt = new Date().toISOString();

  recordEvent(manager, {
    metaTaskId: taskId,
    type: "task_updated",
    actor: actorId,
    description: `${participant.labName} role changed from ${oldRole} to ${newRole}`,
    data: { participantId, oldRole, newRole },
  });

  return participant;
}

export function recordContribution(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  participantId: ParticipantId,
  contribution: Omit<ParticipantContribution, "id" | "timestamp">
): ParticipantContribution | null {
  const task = manager.tasks.get(taskId);
  if (!task) return null;

  const participant = task.participants.find((p) => p.id === participantId);
  if (!participant) return null;

  const now = new Date().toISOString();
  const fullContribution: ParticipantContribution = {
    id: `contrib-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    timestamp: now,
    ...contribution,
  };

  participant.contributions.push(fullContribution);
  participant.lastActiveAt = now;
  task.updatedAt = now;

  return fullContribution;
}

// ============================================================================
// Objective Management
// ============================================================================

export interface CreateObjectiveInput {
  title: string;
  description: string;
  priority: "critical" | "high" | "medium" | "low";
  assignedTo?: ParticipantId[];
  dependencies?: string[];
  deadline?: string;
}

export function addObjective(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  input: CreateObjectiveInput,
  actorId: ParticipantId
): JointObjective | null {
  const task = manager.tasks.get(taskId);
  if (!task) return null;

  const now = new Date().toISOString();
  const objective: JointObjective = {
    id: `obj-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    title: input.title,
    description: input.description,
    priority: input.priority,
    status: "pending",
    assignedTo: input.assignedTo || [],
    dependencies: input.dependencies || [],
    milestones: [],
    deliverables: [],
    deadline: input.deadline,
    progress: 0,
    createdAt: now,
    updatedAt: now,
  };

  task.objectives.push(objective);
  task.updatedAt = now;

  // Assign to participants
  for (const participantId of objective.assignedTo) {
    const participant = task.participants.find((p) => p.id === participantId);
    if (participant) {
      participant.assignedObjectives.push(objective.id);
    }
  }

  recordEvent(manager, {
    metaTaskId: taskId,
    type: "objective_created",
    actor: actorId,
    description: `Objective "${input.title}" created`,
    data: { objectiveId: objective.id, priority: input.priority },
  });

  return objective;
}

export function updateObjectiveProgress(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  objectiveId: string,
  progress: number,
  actorId: ParticipantId
): JointObjective | null {
  const task = manager.tasks.get(taskId);
  if (!task) return null;

  const objective = task.objectives.find((o) => o.id === objectiveId);
  if (!objective) return null;

  objective.progress = Math.max(0, Math.min(100, progress));
  objective.updatedAt = new Date().toISOString();
  task.updatedAt = objective.updatedAt;

  // Auto-update status based on progress
  if (progress === 0 && objective.status === "in_progress") {
    objective.status = "pending";
  } else if (progress > 0 && progress < 100 && objective.status === "pending") {
    objective.status = "in_progress";
  } else if (progress === 100 && objective.status !== "completed") {
    objective.status = "completed";
    recordEvent(manager, {
      metaTaskId: taskId,
      type: "objective_completed",
      actor: actorId,
      description: `Objective "${objective.title}" completed`,
      data: { objectiveId },
    });
  }

  recordEvent(manager, {
    metaTaskId: taskId,
    type: "objective_updated",
    actor: actorId,
    description: `Objective "${objective.title}" progress: ${progress}%`,
    data: { objectiveId, progress },
  });

  return objective;
}

export function completeObjective(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  objectiveId: string,
  actorId: ParticipantId
): JointObjective | null {
  return updateObjectiveProgress(manager, taskId, objectiveId, 100, actorId);
}

export function addMilestone(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  objectiveId: string,
  milestone: Omit<ObjectiveMilestone, "id" | "completed" | "completedAt" | "completedBy">,
  actorId: ParticipantId
): ObjectiveMilestone | null {
  const task = manager.tasks.get(taskId);
  if (!task) return null;

  const objective = task.objectives.find((o) => o.id === objectiveId);
  if (!objective) return null;

  const fullMilestone: ObjectiveMilestone = {
    id: `ms-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    completed: false,
    ...milestone,
  };

  objective.milestones.push(fullMilestone);
  objective.updatedAt = new Date().toISOString();
  task.updatedAt = objective.updatedAt;

  return fullMilestone;
}

export function completeMilestone(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  objectiveId: string,
  milestoneId: string,
  actorId: ParticipantId
): ObjectiveMilestone | null {
  const task = manager.tasks.get(taskId);
  if (!task) return null;

  const objective = task.objectives.find((o) => o.id === objectiveId);
  if (!objective) return null;

  const milestone = objective.milestones.find((m) => m.id === milestoneId);
  if (!milestone) return null;

  const now = new Date().toISOString();
  milestone.completed = true;
  milestone.completedAt = now;
  milestone.completedBy = actorId;
  objective.updatedAt = now;
  task.updatedAt = now;

  recordEvent(manager, {
    metaTaskId: taskId,
    type: "milestone_reached",
    actor: actorId,
    description: `Milestone "${milestone.title}" completed`,
    data: { objectiveId, milestoneId },
  });

  return milestone;
}

// ============================================================================
// Result Management
// ============================================================================

export interface CreateResultInput {
  objectiveId: string;
  title: string;
  description: string;
  type: ResultType;
  contributors: ParticipantId[];
  data: {
    summary: string;
    details?: Record<string, unknown>;
    metrics?: Record<string, number>;
    references?: string[];
  };
}

export function submitResult(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  input: CreateResultInput,
  actorId: ParticipantId
): SharedResult | null {
  const task = manager.tasks.get(taskId);
  if (!task) return null;

  const objective = task.objectives.find((o) => o.id === input.objectiveId);
  if (!objective) {
    throw new Error("Objective not found");
  }

  const now = new Date().toISOString();
  const result: SharedResult = {
    id: `result-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    objectiveId: input.objectiveId,
    title: input.title,
    description: input.description,
    type: input.type,
    contributors: input.contributors,
    data: {
      summary: input.data.summary,
      details: input.data.details || {},
      metrics: input.data.metrics,
      artifacts: [],
      references: input.data.references || [],
    },
    quality: createEmptyResultQuality(),
    status: "draft",
    createdAt: now,
    updatedAt: now,
  };

  task.results.push(result);
  task.updatedAt = now;

  recordEvent(manager, {
    metaTaskId: taskId,
    type: "result_submitted",
    actor: actorId,
    description: `Result "${input.title}" submitted`,
    data: { resultId: result.id, type: input.type },
  });

  return result;
}

export function mergeResults(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  resultIds: string[],
  mergedTitle: string,
  mergedDescription: string,
  actorId: ParticipantId
): SharedResult | null {
  const task = manager.tasks.get(taskId);
  if (!task) return null;

  const results = resultIds
    .map((id) => task.results.find((r) => r.id === id))
    .filter((r): r is SharedResult => r !== undefined);

  if (results.length !== resultIds.length) {
    throw new Error("Some results not found");
  }

  // Merge contributors
  const allContributors = new Set<ParticipantId>();
  for (const result of results) {
    for (const contrib of result.contributors) {
      allContributors.add(contrib);
    }
  }

  // Merge data
  const mergedDetails: Record<string, unknown> = {};
  const mergedMetrics: Record<string, number> = {};
  const mergedReferences: string[] = [];

  for (const result of results) {
    Object.assign(mergedDetails, result.data.details);
    if (result.data.metrics) {
      Object.assign(mergedMetrics, result.data.metrics);
    }
    mergedReferences.push(...result.data.references);
  }

  const now = new Date().toISOString();
  const mergedResult: SharedResult = {
    id: `result-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    objectiveId: results[0].objectiveId,
    title: mergedTitle,
    description: mergedDescription,
    type: results[0].type,
    contributors: Array.from(allContributors),
    data: {
      summary: results.map((r) => r.data.summary).join("\n\n"),
      details: mergedDetails,
      metrics: Object.keys(mergedMetrics).length > 0 ? mergedMetrics : undefined,
      artifacts: results.flatMap((r) => r.data.artifacts),
      references: Array.from(new Set(mergedReferences)),
    },
    quality: createEmptyResultQuality(),
    status: "draft",
    createdAt: now,
    updatedAt: now,
    mergedFrom: resultIds,
  };

  task.results.push(mergedResult);
  task.updatedAt = now;

  recordEvent(manager, {
    metaTaskId: taskId,
    type: "result_merged",
    actor: actorId,
    description: `Merged ${resultIds.length} results into "${mergedTitle}"`,
    data: { resultId: mergedResult.id, sourceIds: resultIds },
  });

  return mergedResult;
}

export function reviewResult(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  resultId: string,
  quality: Partial<ResultQuality>,
  actorId: ParticipantId
): SharedResult | null {
  const task = manager.tasks.get(taskId);
  if (!task) return null;

  const result = task.results.find((r) => r.id === resultId);
  if (!result) return null;

  result.quality = {
    ...result.quality,
    ...quality,
  };
  result.status = "reviewed";
  result.updatedAt = new Date().toISOString();
  task.updatedAt = result.updatedAt;

  return result;
}

export function approveResult(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  resultId: string,
  actorId: ParticipantId
): SharedResult | null {
  const task = manager.tasks.get(taskId);
  if (!task) return null;

  const result = task.results.find((r) => r.id === resultId);
  if (!result) return null;

  result.status = "approved";
  result.updatedAt = new Date().toISOString();
  task.updatedAt = result.updatedAt;

  return result;
}

// ============================================================================
// Completion Criteria
// ============================================================================

export function addCompletionCriterion(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  criterion: Omit<CompletionCriterion, "id" | "met" | "verifiedBy" | "verifiedAt">,
  actorId: ParticipantId
): CompletionCriterion | null {
  const task = manager.tasks.get(taskId);
  if (!task) return null;

  const fullCriterion: CompletionCriterion = {
    id: `crit-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    met: false,
    ...criterion,
  };

  task.completionCriteria.push(fullCriterion);
  task.updatedAt = new Date().toISOString();

  return fullCriterion;
}

export function verifyCriterion(
  manager: MetaTaskManager,
  taskId: MetaTaskId,
  criterionId: string,
  met: boolean,
  actorId: ParticipantId
): CompletionCriterion | null {
  const task = manager.tasks.get(taskId);
  if (!task) return null;

  const criterion = task.completionCriteria.find((c) => c.id === criterionId);
  if (!criterion) return null;

  const now = new Date().toISOString();
  criterion.met = met;
  criterion.verifiedBy = actorId;
  criterion.verifiedAt = now;
  task.updatedAt = now;

  return criterion;
}

export function checkAllCriteriaMet(task: MetaTask): boolean {
  return task.completionCriteria.every((c) => c.met);
}

// ============================================================================
// Query Functions
// ============================================================================

export function getTask(
  manager: MetaTaskManager,
  taskId: MetaTaskId
): MetaTask | undefined {
  return manager.tasks.get(taskId);
}

export function getAllTasks(manager: MetaTaskManager): MetaTask[] {
  return Array.from(manager.tasks.values());
}

export function getTasksByStatus(
  manager: MetaTaskManager,
  status: MetaTaskStatus
): MetaTask[] {
  return getAllTasks(manager).filter((t) => t.status === status);
}

export function getTasksByCategory(
  manager: MetaTaskManager,
  category: MetaTaskCategory
): MetaTask[] {
  return getAllTasks(manager).filter((t) => t.category === category);
}

export function getTasksByParticipant(
  manager: MetaTaskManager,
  labId: string
): MetaTask[] {
  return getAllTasks(manager).filter((t) =>
    t.participants.some((p) => p.labId === labId && p.status === "active")
  );
}

export function getOpenTasks(manager: MetaTaskManager): MetaTask[] {
  return getAllTasks(manager).filter(
    (t) =>
      (t.status === "proposed" || t.status === "recruiting") &&
      t.visibility !== "private"
  );
}

export function getTaskEvents(
  manager: MetaTaskManager,
  taskId: MetaTaskId
): WorkflowEvent[] {
  return manager.events.filter((e) => e.metaTaskId === taskId);
}

// ============================================================================
// Statistics
// ============================================================================

export interface MetaTaskStats {
  totalTasks: number;
  byStatus: Record<MetaTaskStatus, number>;
  byCategory: Record<MetaTaskCategory, number>;
  totalParticipants: number;
  averageParticipants: number;
  completionRate: number;
  averageDuration: number;
}

export function calculateStats(manager: MetaTaskManager): MetaTaskStats {
  const tasks = getAllTasks(manager);

  const byStatus: Record<MetaTaskStatus, number> = {
    proposed: 0,
    recruiting: 0,
    accepted: 0,
    active: 0,
    reviewing: 0,
    completed: 0,
    cancelled: 0,
    failed: 0,
  };

  const byCategory: Record<MetaTaskCategory, number> = {
    exploration: 0,
    integration: 0,
    benchmark: 0,
    dataset: 0,
    replication: 0,
    extension: 0,
    application: 0,
  };

  let totalParticipants = 0;
  let completedCount = 0;
  let totalDuration = 0;
  let durationCount = 0;

  for (const task of tasks) {
    byStatus[task.status]++;
    byCategory[task.category]++;
    totalParticipants += task.participants.filter(
      (p) => p.status === "active"
    ).length;

    if (task.status === "completed") {
      completedCount++;
      if (task.startedAt && task.completedAt) {
        const duration =
          new Date(task.completedAt).getTime() -
          new Date(task.startedAt).getTime();
        totalDuration += duration;
        durationCount++;
      }
    }
  }

  return {
    totalTasks: tasks.length,
    byStatus,
    byCategory,
    totalParticipants,
    averageParticipants: tasks.length > 0 ? totalParticipants / tasks.length : 0,
    completionRate:
      tasks.length > 0
        ? completedCount / tasks.filter((t) => t.status !== "proposed").length
        : 0,
    averageDuration:
      durationCount > 0 ? totalDuration / durationCount / (1000 * 60 * 60 * 24) : 0,
  };
}

// ============================================================================
// Event Recording
// ============================================================================

function recordEvent(
  manager: MetaTaskManager,
  event: Omit<WorkflowEvent, "id" | "timestamp">
): WorkflowEvent {
  const fullEvent: WorkflowEvent = {
    id: `event-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    timestamp: new Date().toISOString(),
    ...event,
  };

  manager.events.push(fullEvent);
  return fullEvent;
}

// ============================================================================
// Global Manager
// ============================================================================

let globalManager: MetaTaskManager | null = null;

export function getGlobalMetaTaskManager(): MetaTaskManager {
  if (!globalManager) {
    globalManager = createMetaTaskManager();
  }
  return globalManager;
}

export function resetGlobalMetaTaskManager(): void {
  globalManager = null;
}
