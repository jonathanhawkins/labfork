/**
 * Workflow Coordinator
 *
 * Orchestrates collaboration workflows across labs including:
 * - Proposal lifecycle management
 * - Lab invitation system
 * - Task assignment coordination
 * - Progress tracking
 * - Result merging
 * - Conflict resolution
 */

import {
  MetaTask,
  MetaTaskId,
  MetaTaskStatus,
  Participant,
  ParticipantId,
  ParticipantRole,
  ParticipantCommitment,
  Invitation,
  InvitationId,
  JointObjective,
  SharedResult,
  Conflict,
  ConflictType,
  Resolution,
  WorkflowEvent,
  WorkflowEventType,
  CollaborationConfig,
  DEFAULT_COLLABORATION_CONFIG,
  LabProfile,
  createInvitationId,
  createParticipantId,
} from "./types";
import {
  MetaTaskManager,
  getGlobalMetaTaskManager,
  getTask,
  addParticipant,
  removeParticipant,
  transitionStatus,
  updateObjectiveProgress,
  mergeResults as mergeTaskResults,
  reviewResult,
  approveResult,
  JoinTaskInput,
} from "./meta-task";

// ============================================================================
// Workflow Coordinator Types
// ============================================================================

export interface WorkflowCoordinator {
  invitations: Map<InvitationId, Invitation>;
  conflicts: Map<string, Conflict>;
  assignmentQueue: TaskAssignment[];
  progressTrackers: Map<MetaTaskId, ProgressTracker>;
  config: CollaborationConfig;
}

export interface TaskAssignment {
  id: string;
  metaTaskId: MetaTaskId;
  objectiveId: string;
  participantId: ParticipantId;
  priority: "critical" | "high" | "medium" | "low";
  status: "pending" | "assigned" | "in_progress" | "completed" | "blocked";
  assignedAt: string;
  dueDate?: string;
  blockedBy?: string[];
  notes?: string;
}

export interface ProgressTracker {
  metaTaskId: MetaTaskId;
  overallProgress: number;
  objectiveProgress: Record<string, number>;
  participantProgress: Record<ParticipantId, ParticipantProgress>;
  milestoneStatus: Record<string, boolean>;
  lastUpdated: string;
  healthStatus: "healthy" | "at_risk" | "blocked" | "off_track";
  alerts: ProgressAlert[];
}

export interface ParticipantProgress {
  participantId: ParticipantId;
  assignedObjectives: number;
  completedObjectives: number;
  contributionsCount: number;
  lastActive: string;
  activityScore: number;
}

export interface ProgressAlert {
  id: string;
  type: "deadline" | "blocked" | "inactive" | "scope" | "quality";
  severity: "info" | "warning" | "error";
  message: string;
  relatedObjective?: string;
  relatedParticipant?: ParticipantId;
  createdAt: string;
  acknowledged: boolean;
}

// ============================================================================
// Coordinator Creation
// ============================================================================

export function createWorkflowCoordinator(
  config: Partial<CollaborationConfig> = {}
): WorkflowCoordinator {
  return {
    invitations: new Map(),
    conflicts: new Map(),
    assignmentQueue: [],
    progressTrackers: new Map(),
    config: { ...DEFAULT_COLLABORATION_CONFIG, ...config },
  };
}

// ============================================================================
// Invitation System
// ============================================================================

export interface SendInvitationInput {
  metaTaskId: MetaTaskId;
  labId: string;
  labName: string;
  proposedRole: ParticipantRole;
  message: string;
  sentBy: ParticipantId;
}

export function sendInvitation(
  coordinator: WorkflowCoordinator,
  manager: MetaTaskManager,
  input: SendInvitationInput
): Invitation | null {
  const task = getTask(manager, input.metaTaskId);
  if (!task) return null;

  // Check if lab is already a participant
  if (task.participants.some((p) => p.labId === input.labId)) {
    throw new Error("Lab is already a participant");
  }

  // Check if there's already a pending invitation
  const existingInvitation = Array.from(coordinator.invitations.values()).find(
    (inv) =>
      inv.metaTaskId === input.metaTaskId &&
      inv.labId === input.labId &&
      inv.status === "pending"
  );
  if (existingInvitation) {
    throw new Error("Invitation already pending for this lab");
  }

  const now = new Date();
  const expiresAt = new Date(
    now.getTime() + coordinator.config.invitationExpiryDays * 24 * 60 * 60 * 1000
  );

  const invitation: Invitation = {
    id: createInvitationId(),
    metaTaskId: input.metaTaskId,
    labId: input.labId,
    labName: input.labName,
    proposedRole: input.proposedRole,
    message: input.message,
    status: "pending",
    sentBy: input.sentBy,
    sentAt: now.toISOString(),
    expiresAt: expiresAt.toISOString(),
  };

  coordinator.invitations.set(invitation.id, invitation);

  return invitation;
}

export function respondToInvitation(
  coordinator: WorkflowCoordinator,
  manager: MetaTaskManager,
  invitationId: InvitationId,
  accept: boolean,
  commitment?: ParticipantCommitment,
  expertise?: string[]
): Invitation | null {
  const invitation = coordinator.invitations.get(invitationId);
  if (!invitation) return null;

  if (invitation.status !== "pending") {
    throw new Error(`Invitation is ${invitation.status}, cannot respond`);
  }

  // Check if expired
  if (new Date() > new Date(invitation.expiresAt)) {
    invitation.status = "expired";
    return invitation;
  }

  const now = new Date().toISOString();
  invitation.respondedAt = now;

  if (accept) {
    if (!commitment) {
      throw new Error("Commitment required to accept invitation");
    }

    invitation.status = "accepted";

    // Add participant to task
    const joinInput: JoinTaskInput = {
      labId: invitation.labId,
      labName: invitation.labName,
      expertise: expertise || [],
      role: invitation.proposedRole,
      commitment,
    };

    addParticipant(manager, invitation.metaTaskId, joinInput);
  } else {
    invitation.status = "declined";
  }

  return invitation;
}

export function getInvitationsForLab(
  coordinator: WorkflowCoordinator,
  labId: string
): Invitation[] {
  return Array.from(coordinator.invitations.values()).filter(
    (inv) => inv.labId === labId
  );
}

export function getInvitationsForTask(
  coordinator: WorkflowCoordinator,
  metaTaskId: MetaTaskId
): Invitation[] {
  return Array.from(coordinator.invitations.values()).filter(
    (inv) => inv.metaTaskId === metaTaskId
  );
}

export function getPendingInvitations(
  coordinator: WorkflowCoordinator
): Invitation[] {
  return Array.from(coordinator.invitations.values()).filter(
    (inv) => inv.status === "pending"
  );
}

export function expireOldInvitations(
  coordinator: WorkflowCoordinator
): Invitation[] {
  const now = new Date();
  const expired: Invitation[] = [];

  for (const invitation of Array.from(coordinator.invitations.values())) {
    if (
      invitation.status === "pending" &&
      new Date(invitation.expiresAt) < now
    ) {
      invitation.status = "expired";
      expired.push(invitation);
    }
  }

  return expired;
}

// ============================================================================
// Task Assignment Coordination
// ============================================================================

export interface CreateAssignmentInput {
  metaTaskId: MetaTaskId;
  objectiveId: string;
  participantId: ParticipantId;
  priority: "critical" | "high" | "medium" | "low";
  dueDate?: string;
  notes?: string;
}

export function createAssignment(
  coordinator: WorkflowCoordinator,
  manager: MetaTaskManager,
  input: CreateAssignmentInput
): TaskAssignment | null {
  const task = getTask(manager, input.metaTaskId);
  if (!task) return null;

  const objective = task.objectives.find((o) => o.id === input.objectiveId);
  if (!objective) {
    throw new Error("Objective not found");
  }

  const participant = task.participants.find((p) => p.id === input.participantId);
  if (!participant) {
    throw new Error("Participant not found");
  }

  // Check dependencies
  const blockedBy: string[] = [];
  for (const depId of objective.dependencies) {
    const dep = task.objectives.find((o) => o.id === depId);
    if (dep && dep.status !== "completed") {
      blockedBy.push(depId);
    }
  }

  const assignment: TaskAssignment = {
    id: `assign-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    metaTaskId: input.metaTaskId,
    objectiveId: input.objectiveId,
    participantId: input.participantId,
    priority: input.priority,
    status: blockedBy.length > 0 ? "blocked" : "assigned",
    assignedAt: new Date().toISOString(),
    dueDate: input.dueDate,
    blockedBy: blockedBy.length > 0 ? blockedBy : undefined,
    notes: input.notes,
  };

  coordinator.assignmentQueue.push(assignment);

  // Update objective assignment
  if (!objective.assignedTo.includes(input.participantId)) {
    objective.assignedTo.push(input.participantId);
  }

  // Update participant's assigned objectives
  if (!participant.assignedObjectives.includes(input.objectiveId)) {
    participant.assignedObjectives.push(input.objectiveId);
  }

  return assignment;
}

export function updateAssignmentStatus(
  coordinator: WorkflowCoordinator,
  assignmentId: string,
  status: TaskAssignment["status"]
): TaskAssignment | null {
  const assignment = coordinator.assignmentQueue.find((a) => a.id === assignmentId);
  if (!assignment) return null;

  assignment.status = status;

  // If completed, unblock dependent assignments
  if (status === "completed") {
    unblockDependentAssignments(coordinator, assignment.objectiveId);
  }

  return assignment;
}

function unblockDependentAssignments(
  coordinator: WorkflowCoordinator,
  completedObjectiveId: string
): void {
  for (const assignment of coordinator.assignmentQueue) {
    if (assignment.blockedBy?.includes(completedObjectiveId)) {
      assignment.blockedBy = assignment.blockedBy.filter(
        (id) => id !== completedObjectiveId
      );
      if (assignment.blockedBy.length === 0) {
        assignment.blockedBy = undefined;
        if (assignment.status === "blocked") {
          assignment.status = "assigned";
        }
      }
    }
  }
}

export function getAssignmentsForParticipant(
  coordinator: WorkflowCoordinator,
  participantId: ParticipantId
): TaskAssignment[] {
  return coordinator.assignmentQueue.filter(
    (a) => a.participantId === participantId
  );
}

export function getAssignmentsForTask(
  coordinator: WorkflowCoordinator,
  metaTaskId: MetaTaskId
): TaskAssignment[] {
  return coordinator.assignmentQueue.filter((a) => a.metaTaskId === metaTaskId);
}

export function getBlockedAssignments(
  coordinator: WorkflowCoordinator
): TaskAssignment[] {
  return coordinator.assignmentQueue.filter((a) => a.status === "blocked");
}

export function getOverdueAssignments(
  coordinator: WorkflowCoordinator
): TaskAssignment[] {
  const now = new Date();
  return coordinator.assignmentQueue.filter(
    (a) =>
      a.dueDate &&
      new Date(a.dueDate) < now &&
      a.status !== "completed"
  );
}

// ============================================================================
// Progress Tracking
// ============================================================================

export function initializeProgressTracker(
  coordinator: WorkflowCoordinator,
  manager: MetaTaskManager,
  metaTaskId: MetaTaskId
): ProgressTracker | null {
  const task = getTask(manager, metaTaskId);
  if (!task) return null;

  const objectiveProgress: Record<string, number> = {};
  const milestoneStatus: Record<string, boolean> = {};
  const participantProgress: Record<ParticipantId, ParticipantProgress> = {};

  for (const objective of task.objectives) {
    objectiveProgress[objective.id] = objective.progress;
    for (const milestone of objective.milestones) {
      milestoneStatus[milestone.id] = milestone.completed;
    }
  }

  for (const participant of task.participants) {
    if (participant.status !== "active") continue;

    const assignedCount = participant.assignedObjectives.length;
    const completedCount = task.objectives.filter(
      (o) =>
        participant.assignedObjectives.includes(o.id) &&
        o.status === "completed"
    ).length;

    participantProgress[participant.id] = {
      participantId: participant.id,
      assignedObjectives: assignedCount,
      completedObjectives: completedCount,
      contributionsCount: participant.contributions.length,
      lastActive: participant.lastActiveAt,
      activityScore: calculateActivityScore(participant),
    };
  }

  const overallProgress = calculateOverallProgress(task);

  const tracker: ProgressTracker = {
    metaTaskId,
    overallProgress,
    objectiveProgress,
    participantProgress,
    milestoneStatus,
    lastUpdated: new Date().toISOString(),
    healthStatus: determineHealthStatus(task, overallProgress),
    alerts: generateProgressAlerts(task, coordinator),
  };

  coordinator.progressTrackers.set(metaTaskId, tracker);
  return tracker;
}

function calculateActivityScore(participant: Participant): number {
  const now = new Date().getTime();
  const lastActive = new Date(participant.lastActiveAt).getTime();
  const daysSinceActive = (now - lastActive) / (1000 * 60 * 60 * 24);

  // Decay activity score based on inactivity
  const recencyScore = Math.max(0, 100 - daysSinceActive * 10);

  // Boost for contributions
  const contributionScore = Math.min(50, participant.contributions.length * 10);

  return Math.min(100, (recencyScore + contributionScore) / 2);
}

function calculateOverallProgress(task: MetaTask): number {
  if (task.objectives.length === 0) return 0;

  // Weight by priority
  const priorityWeights = {
    critical: 4,
    high: 3,
    medium: 2,
    low: 1,
  };

  let weightedSum = 0;
  let totalWeight = 0;

  for (const objective of task.objectives) {
    const weight = priorityWeights[objective.priority];
    weightedSum += objective.progress * weight;
    totalWeight += weight * 100;
  }

  return totalWeight > 0 ? Math.round((weightedSum / totalWeight) * 100) : 0;
}

function determineHealthStatus(
  task: MetaTask,
  progress: number
): ProgressTracker["healthStatus"] {
  // Check for blocks
  const hasBlocked = task.objectives.some((o) => o.status === "blocked");
  if (hasBlocked) return "blocked";

  // Check timeline
  if (task.timeline.endDate) {
    const endDate = new Date(task.timeline.endDate);
    const now = new Date();
    const totalDuration = endDate.getTime() - new Date(task.createdAt).getTime();
    const elapsed = now.getTime() - new Date(task.createdAt).getTime();
    const expectedProgress = (elapsed / totalDuration) * 100;

    if (progress < expectedProgress - 20) return "off_track";
    if (progress < expectedProgress - 10) return "at_risk";
  }

  return "healthy";
}

function generateProgressAlerts(
  task: MetaTask,
  coordinator: WorkflowCoordinator
): ProgressAlert[] {
  const alerts: ProgressAlert[] = [];
  const now = new Date();

  // Check for deadline alerts
  for (const objective of task.objectives) {
    if (objective.deadline && objective.status !== "completed") {
      const deadline = new Date(objective.deadline);
      const daysUntil = (deadline.getTime() - now.getTime()) / (1000 * 60 * 60 * 24);

      if (daysUntil < 0) {
        alerts.push({
          id: `alert-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
          type: "deadline",
          severity: "error",
          message: `Objective "${objective.title}" is past due`,
          relatedObjective: objective.id,
          createdAt: now.toISOString(),
          acknowledged: false,
        });
      } else if (daysUntil < 3) {
        alerts.push({
          id: `alert-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
          type: "deadline",
          severity: "warning",
          message: `Objective "${objective.title}" due in ${Math.ceil(daysUntil)} days`,
          relatedObjective: objective.id,
          createdAt: now.toISOString(),
          acknowledged: false,
        });
      }
    }

    // Check for blocked objectives
    if (objective.status === "blocked") {
      alerts.push({
        id: `alert-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        type: "blocked",
        severity: "warning",
        message: `Objective "${objective.title}" is blocked`,
        relatedObjective: objective.id,
        createdAt: now.toISOString(),
        acknowledged: false,
      });
    }
  }

  // Check for inactive participants
  for (const participant of task.participants) {
    if (participant.status !== "active") continue;

    const lastActive = new Date(participant.lastActiveAt);
    const daysSinceActive =
      (now.getTime() - lastActive.getTime()) / (1000 * 60 * 60 * 24);

    if (daysSinceActive > 7) {
      alerts.push({
        id: `alert-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        type: "inactive",
        severity: daysSinceActive > 14 ? "error" : "warning",
        message: `${participant.labName} inactive for ${Math.floor(daysSinceActive)} days`,
        relatedParticipant: participant.id,
        createdAt: now.toISOString(),
        acknowledged: false,
      });
    }
  }

  return alerts;
}

export function updateProgress(
  coordinator: WorkflowCoordinator,
  manager: MetaTaskManager,
  metaTaskId: MetaTaskId,
  objectiveId: string,
  progress: number,
  actorId: ParticipantId
): ProgressTracker | null {
  // Update the objective progress in the task
  updateObjectiveProgress(manager, metaTaskId, objectiveId, progress, actorId);

  // Refresh the progress tracker
  return initializeProgressTracker(coordinator, manager, metaTaskId);
}

export function getProgressTracker(
  coordinator: WorkflowCoordinator,
  metaTaskId: MetaTaskId
): ProgressTracker | undefined {
  return coordinator.progressTrackers.get(metaTaskId);
}

export function acknowledgeAlert(
  coordinator: WorkflowCoordinator,
  metaTaskId: MetaTaskId,
  alertId: string
): boolean {
  const tracker = coordinator.progressTrackers.get(metaTaskId);
  if (!tracker) return false;

  const alert = tracker.alerts.find((a) => a.id === alertId);
  if (!alert) return false;

  alert.acknowledged = true;
  return true;
}

// ============================================================================
// Result Merging Coordination
// ============================================================================

export interface MergeRequest {
  id: string;
  metaTaskId: MetaTaskId;
  resultIds: string[];
  proposedBy: ParticipantId;
  proposedTitle: string;
  proposedDescription: string;
  status: "pending" | "approved" | "rejected" | "merged";
  approvals: ParticipantId[];
  rejections: ParticipantId[];
  requiredApprovals: number;
  createdAt: string;
  resolvedAt?: string;
}

const mergeRequests: Map<string, MergeRequest> = new Map();

export function proposeMerge(
  coordinator: WorkflowCoordinator,
  manager: MetaTaskManager,
  metaTaskId: MetaTaskId,
  resultIds: string[],
  proposedTitle: string,
  proposedDescription: string,
  proposedBy: ParticipantId
): MergeRequest | null {
  const task = getTask(manager, metaTaskId);
  if (!task) return null;

  // Validate all results exist
  for (const resultId of resultIds) {
    if (!task.results.find((r) => r.id === resultId)) {
      throw new Error(`Result ${resultId} not found`);
    }
  }

  // Get contributors to determine required approvals
  const contributors = new Set<ParticipantId>();
  for (const resultId of resultIds) {
    const result = task.results.find((r) => r.id === resultId);
    if (result) {
      for (const contrib of result.contributors) {
        contributors.add(contrib);
      }
    }
  }

  const request: MergeRequest = {
    id: `merge-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    metaTaskId,
    resultIds,
    proposedBy,
    proposedTitle,
    proposedDescription,
    status: "pending",
    approvals: [proposedBy], // Proposer auto-approves
    rejections: [],
    requiredApprovals: Math.ceil(contributors.size / 2), // Majority required
    createdAt: new Date().toISOString(),
  };

  mergeRequests.set(request.id, request);
  return request;
}

export function voteOnMerge(
  coordinator: WorkflowCoordinator,
  manager: MetaTaskManager,
  mergeRequestId: string,
  participantId: ParticipantId,
  approve: boolean
): MergeRequest | null {
  const request = mergeRequests.get(mergeRequestId);
  if (!request) return null;

  if (request.status !== "pending") {
    throw new Error(`Merge request is ${request.status}`);
  }

  // Remove any existing vote
  request.approvals = request.approvals.filter((id) => id !== participantId);
  request.rejections = request.rejections.filter((id) => id !== participantId);

  if (approve) {
    request.approvals.push(participantId);
  } else {
    request.rejections.push(participantId);
  }

  // Check if decision can be made
  if (request.approvals.length >= request.requiredApprovals) {
    request.status = "approved";
    request.resolvedAt = new Date().toISOString();

    // Execute the merge
    mergeTaskResults(
      manager,
      request.metaTaskId,
      request.resultIds,
      request.proposedTitle,
      request.proposedDescription,
      request.proposedBy
    );

    request.status = "merged";
  } else if (request.rejections.length > request.requiredApprovals) {
    request.status = "rejected";
    request.resolvedAt = new Date().toISOString();
  }

  return request;
}

export function getMergeRequests(
  metaTaskId: MetaTaskId
): MergeRequest[] {
  return Array.from(mergeRequests.values()).filter(
    (r) => r.metaTaskId === metaTaskId
  );
}

// ============================================================================
// Conflict Resolution
// ============================================================================

export interface CreateConflictInput {
  metaTaskId: MetaTaskId;
  type: ConflictType;
  description: string;
  severity: Conflict["severity"];
  parties: ParticipantId[];
  reportedBy: ParticipantId;
}

export function reportConflict(
  coordinator: WorkflowCoordinator,
  manager: MetaTaskManager,
  input: CreateConflictInput
): Conflict | null {
  const task = getTask(manager, input.metaTaskId);
  if (!task) return null;

  // Validate all parties are participants
  for (const partyId of input.parties) {
    if (!task.participants.find((p) => p.id === partyId)) {
      throw new Error(`Party ${partyId} is not a participant`);
    }
  }

  const conflict: Conflict = {
    id: `conflict-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    metaTaskId: input.metaTaskId,
    type: input.type,
    description: input.description,
    severity: input.severity,
    parties: input.parties,
    status: "open",
    proposedResolutions: [],
    createdAt: new Date().toISOString(),
  };

  coordinator.conflicts.set(conflict.id, conflict);
  return conflict;
}

export function proposeResolution(
  coordinator: WorkflowCoordinator,
  conflictId: string,
  proposedBy: ParticipantId,
  description: string,
  actions: string[]
): Resolution | null {
  const conflict = coordinator.conflicts.get(conflictId);
  if (!conflict) return null;

  if (conflict.status === "resolved") {
    throw new Error("Conflict is already resolved");
  }

  const resolution: Resolution = {
    id: `res-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    proposedBy,
    description,
    actions,
    votes: [{ participant: proposedBy, vote: "approve" }], // Proposer auto-votes
    status: "proposed",
  };

  conflict.proposedResolutions.push(resolution);
  conflict.status = "discussing";

  return resolution;
}

export function voteOnResolution(
  coordinator: WorkflowCoordinator,
  conflictId: string,
  resolutionId: string,
  participantId: ParticipantId,
  vote: "approve" | "reject" | "abstain"
): Resolution | null {
  const conflict = coordinator.conflicts.get(conflictId);
  if (!conflict) return null;

  const resolution = conflict.proposedResolutions.find(
    (r) => r.id === resolutionId
  );
  if (!resolution) return null;

  if (resolution.status !== "proposed") {
    throw new Error(`Resolution is ${resolution.status}`);
  }

  // Update or add vote
  const existingVote = resolution.votes.find(
    (v) => v.participant === participantId
  );
  if (existingVote) {
    existingVote.vote = vote;
  } else {
    resolution.votes.push({ participant: participantId, vote });
  }

  // Check if resolution is accepted (all parties approve)
  const partyVotes = resolution.votes.filter((v) =>
    conflict.parties.includes(v.participant)
  );
  const allPartiesVoted = partyVotes.length === conflict.parties.length;
  const allApproved = partyVotes.every((v) => v.vote === "approve");

  if (allPartiesVoted && allApproved) {
    resolution.status = "approved";
    conflict.selectedResolution = resolution;
    conflict.status = "resolved";
    conflict.resolvedAt = new Date().toISOString();
  } else if (partyVotes.some((v) => v.vote === "reject")) {
    resolution.status = "rejected";
  }

  return resolution;
}

export function escalateConflict(
  coordinator: WorkflowCoordinator,
  conflictId: string
): Conflict | null {
  const conflict = coordinator.conflicts.get(conflictId);
  if (!conflict) return null;

  conflict.status = "escalated";
  return conflict;
}

export function getConflictsForTask(
  coordinator: WorkflowCoordinator,
  metaTaskId: MetaTaskId
): Conflict[] {
  return Array.from(coordinator.conflicts.values()).filter(
    (c) => c.metaTaskId === metaTaskId
  );
}

export function getOpenConflicts(coordinator: WorkflowCoordinator): Conflict[] {
  return Array.from(coordinator.conflicts.values()).filter(
    (c) => c.status === "open" || c.status === "discussing"
  );
}

export function checkEscalationDeadlines(
  coordinator: WorkflowCoordinator
): Conflict[] {
  const now = new Date();
  const escalated: Conflict[] = [];

  for (const conflict of Array.from(coordinator.conflicts.values())) {
    if (conflict.status === "open" || conflict.status === "discussing") {
      const createdAt = new Date(conflict.createdAt);
      const daysSinceCreated =
        (now.getTime() - createdAt.getTime()) / (1000 * 60 * 60 * 24);

      if (daysSinceCreated > coordinator.config.maxConflictEscalationDays) {
        conflict.status = "escalated";
        escalated.push(conflict);
      }
    }
  }

  return escalated;
}

// ============================================================================
// Proposal Workflow
// ============================================================================

export function canStartRecruiting(
  manager: MetaTaskManager,
  metaTaskId: MetaTaskId
): { allowed: boolean; reason?: string } {
  const task = getTask(manager, metaTaskId);
  if (!task) return { allowed: false, reason: "Task not found" };

  if (task.status !== "proposed") {
    return { allowed: false, reason: `Task is ${task.status}, not proposed` };
  }

  if (task.objectives.length === 0) {
    return { allowed: false, reason: "Task must have at least one objective" };
  }

  return { allowed: true };
}

export function canAcceptTask(
  manager: MetaTaskManager,
  metaTaskId: MetaTaskId
): { allowed: boolean; reason?: string } {
  const task = getTask(manager, metaTaskId);
  if (!task) return { allowed: false, reason: "Task not found" };

  if (task.status !== "recruiting") {
    return { allowed: false, reason: `Task is ${task.status}, not recruiting` };
  }

  const activeParticipants = task.participants.filter(
    (p) => p.status === "active"
  );
  if (activeParticipants.length < task.requirements.minParticipants) {
    return {
      allowed: false,
      reason: `Need ${task.requirements.minParticipants} participants, have ${activeParticipants.length}`,
    };
  }

  // Check required expertise
  const teamExpertise = new Set<string>();
  for (const participant of activeParticipants) {
    for (const exp of participant.expertise) {
      teamExpertise.add(exp.toLowerCase());
    }
  }

  const missingExpertise = task.requirements.requiredExpertise.filter(
    (exp) => !teamExpertise.has(exp.toLowerCase())
  );
  if (missingExpertise.length > 0) {
    return {
      allowed: false,
      reason: `Missing required expertise: ${missingExpertise.join(", ")}`,
    };
  }

  return { allowed: true };
}

export function canStartTask(
  manager: MetaTaskManager,
  metaTaskId: MetaTaskId
): { allowed: boolean; reason?: string } {
  const task = getTask(manager, metaTaskId);
  if (!task) return { allowed: false, reason: "Task not found" };

  if (task.status !== "accepted") {
    return { allowed: false, reason: `Task is ${task.status}, not accepted` };
  }

  // Check if timeline start date has passed
  if (task.timeline.startDate) {
    if (new Date() < new Date(task.timeline.startDate)) {
      return {
        allowed: false,
        reason: `Start date ${task.timeline.startDate} has not arrived`,
      };
    }
  }

  return { allowed: true };
}

export function canSubmitForReview(
  manager: MetaTaskManager,
  metaTaskId: MetaTaskId
): { allowed: boolean; reason?: string } {
  const task = getTask(manager, metaTaskId);
  if (!task) return { allowed: false, reason: "Task not found" };

  if (task.status !== "active") {
    return { allowed: false, reason: `Task is ${task.status}, not active` };
  }

  const incompleteObjectives = task.objectives.filter(
    (o) => o.status !== "completed"
  );
  if (incompleteObjectives.length > 0) {
    return {
      allowed: false,
      reason: `${incompleteObjectives.length} objectives still incomplete`,
    };
  }

  // Check if there are any results
  if (task.results.length === 0) {
    return { allowed: false, reason: "No results submitted" };
  }

  return { allowed: true };
}

export function canCompleteTask(
  manager: MetaTaskManager,
  metaTaskId: MetaTaskId
): { allowed: boolean; reason?: string } {
  const task = getTask(manager, metaTaskId);
  if (!task) return { allowed: false, reason: "Task not found" };

  if (task.status !== "reviewing") {
    return { allowed: false, reason: `Task is ${task.status}, not reviewing` };
  }

  const unmetCriteria = task.completionCriteria.filter((c) => !c.met);
  if (unmetCriteria.length > 0) {
    return {
      allowed: false,
      reason: `${unmetCriteria.length} completion criteria not met`,
    };
  }

  // Check if all results are approved
  const unapprovedResults = task.results.filter(
    (r) => r.status !== "approved" && r.status !== "published"
  );
  if (unapprovedResults.length > 0) {
    return {
      allowed: false,
      reason: `${unapprovedResults.length} results not approved`,
    };
  }

  return { allowed: true };
}

// ============================================================================
// Workflow Summary
// ============================================================================

export interface WorkflowSummary {
  pendingInvitations: number;
  activeConflicts: number;
  blockedAssignments: number;
  overdueAssignments: number;
  tasksAtRisk: number;
  tasksBlocked: number;
  pendingMergeRequests: number;
  recentAlerts: ProgressAlert[];
}

export function getWorkflowSummary(
  coordinator: WorkflowCoordinator
): WorkflowSummary {
  const recentAlerts: ProgressAlert[] = [];

  for (const tracker of Array.from(coordinator.progressTrackers.values())) {
    recentAlerts.push(
      ...tracker.alerts.filter((a) => !a.acknowledged).slice(0, 5)
    );
  }

  // Count tasks at risk or blocked
  let tasksAtRisk = 0;
  let tasksBlocked = 0;
  for (const tracker of Array.from(coordinator.progressTrackers.values())) {
    if (tracker.healthStatus === "at_risk" || tracker.healthStatus === "off_track") {
      tasksAtRisk++;
    }
    if (tracker.healthStatus === "blocked") {
      tasksBlocked++;
    }
  }

  return {
    pendingInvitations: getPendingInvitations(coordinator).length,
    activeConflicts: getOpenConflicts(coordinator).length,
    blockedAssignments: getBlockedAssignments(coordinator).length,
    overdueAssignments: getOverdueAssignments(coordinator).length,
    tasksAtRisk,
    tasksBlocked,
    pendingMergeRequests: Array.from(mergeRequests.values()).filter(
      (r) => r.status === "pending"
    ).length,
    recentAlerts: recentAlerts.slice(0, 10),
  };
}

// ============================================================================
// Global Coordinator
// ============================================================================

let globalCoordinator: WorkflowCoordinator | null = null;

export function getGlobalWorkflowCoordinator(): WorkflowCoordinator {
  if (!globalCoordinator) {
    globalCoordinator = createWorkflowCoordinator();
  }
  return globalCoordinator;
}

export function resetGlobalWorkflowCoordinator(): void {
  globalCoordinator = null;
}
