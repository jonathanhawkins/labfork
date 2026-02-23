/**
 * Collaboration Types
 *
 * Type definitions for the collaboration system enabling multiple labs
 * to work together on meta-tasks spanning cross-domain research.
 */

// ============================================================================
// Core Types
// ============================================================================

export type MetaTaskId = `meta-task-${string}`;
export type ParticipantId = `participant-${string}`;
export type InvitationId = `invitation-${string}`;
export type CollaborationId = `collaboration-${string}`;

export function createMetaTaskId(): MetaTaskId {
  return `meta-task-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createParticipantId(): ParticipantId {
  return `participant-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createInvitationId(): InvitationId {
  return `invitation-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createCollaborationId(): CollaborationId {
  return `collaboration-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

// ============================================================================
// Meta-Task Status
// ============================================================================

export type MetaTaskStatus =
  | "proposed"      // Initial proposal, seeking participants
  | "recruiting"    // Actively seeking participants
  | "accepted"      // Minimum participants reached, preparing to start
  | "active"        // Work in progress
  | "reviewing"     // Work complete, under review
  | "completed"     // Successfully completed
  | "cancelled"     // Cancelled before completion
  | "failed";       // Failed to complete

export const META_TASK_STATUS_FLOW: Record<MetaTaskStatus, MetaTaskStatus[]> = {
  proposed: ["recruiting", "cancelled"],
  recruiting: ["accepted", "cancelled"],
  accepted: ["active", "cancelled"],
  active: ["reviewing", "cancelled", "failed"],
  reviewing: ["completed", "active", "failed"],
  completed: [],
  cancelled: [],
  failed: [],
};

// ============================================================================
// Participant Types
// ============================================================================

export type ParticipantRole =
  | "lead"          // Primary coordinator
  | "co-lead"       // Secondary coordinator
  | "contributor"   // Active contributor
  | "advisor"       // Provides guidance
  | "reviewer"      // Reviews work
  | "observer";     // Watches progress

export type ParticipantStatus =
  | "invited"       // Invitation sent
  | "pending"       // Accepted, awaiting confirmation
  | "active"        // Actively participating
  | "paused"        // Temporarily inactive
  | "withdrawn"     // Left the task
  | "completed";    // Completed their role

export interface Participant {
  id: ParticipantId;
  labId: string;
  labName: string;
  role: ParticipantRole;
  status: ParticipantStatus;
  expertise: string[];
  commitment: ParticipantCommitment;
  joinedAt: string;
  lastActiveAt: string;
  contributions: ParticipantContribution[];
  assignedObjectives: string[];
}

export interface ParticipantCommitment {
  hoursPerWeek: number;
  durationWeeks: number;
  resources: string[];
  responsibilities: string[];
}

export interface ParticipantContribution {
  id: string;
  type: ContributionType;
  description: string;
  timestamp: string;
  artifacts: ContributionArtifact[];
  impactScore: number;
}

export type ContributionType =
  | "research"
  | "implementation"
  | "data"
  | "analysis"
  | "review"
  | "documentation"
  | "coordination";

export interface ContributionArtifact {
  id: string;
  type: "code" | "paper" | "dataset" | "model" | "report" | "other";
  name: string;
  url?: string;
  metadata: Record<string, unknown>;
}

// ============================================================================
// Joint Objectives
// ============================================================================

export interface JointObjective {
  id: string;
  title: string;
  description: string;
  priority: "critical" | "high" | "medium" | "low";
  status: ObjectiveStatus;
  assignedTo: ParticipantId[];
  dependencies: string[];
  milestones: ObjectiveMilestone[];
  deliverables: Deliverable[];
  deadline?: string;
  progress: number;
  createdAt: string;
  updatedAt: string;
}

export type ObjectiveStatus =
  | "pending"
  | "in_progress"
  | "blocked"
  | "completed"
  | "deferred";

export interface ObjectiveMilestone {
  id: string;
  title: string;
  description: string;
  dueDate: string;
  completed: boolean;
  completedAt?: string;
  completedBy?: ParticipantId;
}

export interface Deliverable {
  id: string;
  name: string;
  description: string;
  type: "artifact" | "report" | "presentation" | "demo" | "other";
  status: "pending" | "in_progress" | "review" | "completed";
  owner: ParticipantId;
  reviewers: ParticipantId[];
  submittedAt?: string;
  approvedAt?: string;
  artifacts: ContributionArtifact[];
}

// ============================================================================
// Shared Results
// ============================================================================

export interface SharedResult {
  id: string;
  objectiveId: string;
  title: string;
  description: string;
  type: ResultType;
  contributors: ParticipantId[];
  data: ResultData;
  quality: ResultQuality;
  status: "draft" | "submitted" | "reviewed" | "approved" | "published";
  createdAt: string;
  updatedAt: string;
  mergedFrom?: string[];
}

export type ResultType =
  | "finding"
  | "technique"
  | "dataset"
  | "model"
  | "benchmark"
  | "paper"
  | "prototype";

export interface ResultData {
  summary: string;
  details: Record<string, unknown>;
  metrics?: Record<string, number>;
  artifacts: ContributionArtifact[];
  references: string[];
}

export interface ResultQuality {
  overallScore: number;
  reproducibility: number;
  novelty: number;
  significance: number;
  completeness: number;
  reviewNotes: string[];
}

// ============================================================================
// Meta-Task
// ============================================================================

export interface MetaTask {
  id: MetaTaskId;
  title: string;
  description: string;
  category: MetaTaskCategory;
  status: MetaTaskStatus;
  lead: Participant;
  participants: Participant[];
  objectives: JointObjective[];
  results: SharedResult[];
  requirements: MetaTaskRequirements;
  timeline: MetaTaskTimeline;
  completionCriteria: CompletionCriterion[];
  tags: string[];
  domains: string[];
  visibility: "public" | "private" | "invite_only";
  createdAt: string;
  updatedAt: string;
  startedAt?: string;
  completedAt?: string;
}

export type MetaTaskCategory =
  | "exploration"     // Exploring new research direction
  | "integration"     // Integrating techniques across domains
  | "benchmark"       // Creating shared benchmarks
  | "dataset"         // Building collaborative dataset
  | "replication"     // Replicating and validating results
  | "extension"       // Extending existing work
  | "application";    // Applying techniques to new problems

export interface MetaTaskRequirements {
  minParticipants: number;
  maxParticipants: number;
  requiredExpertise: string[];
  preferredExpertise: string[];
  resourceRequirements: ResourceRequirement[];
}

export interface ResourceRequirement {
  type: "compute" | "data" | "expertise" | "funding" | "equipment";
  description: string;
  amount?: number;
  unit?: string;
  priority: "required" | "preferred" | "optional";
}

export interface MetaTaskTimeline {
  proposalDeadline?: string;
  recruitmentDeadline?: string;
  startDate?: string;
  endDate?: string;
  phases: TimelinePhase[];
}

export interface TimelinePhase {
  id: string;
  name: string;
  description: string;
  startDate: string;
  endDate: string;
  objectives: string[];
  status: "pending" | "active" | "completed";
}

export interface CompletionCriterion {
  id: string;
  description: string;
  type: "objective" | "deliverable" | "metric" | "approval";
  target?: number;
  current?: number;
  met: boolean;
  verifiedBy?: ParticipantId;
  verifiedAt?: string;
}

// ============================================================================
// Invitations
// ============================================================================

export interface Invitation {
  id: InvitationId;
  metaTaskId: MetaTaskId;
  labId: string;
  labName: string;
  proposedRole: ParticipantRole;
  message: string;
  status: "pending" | "accepted" | "declined" | "expired";
  sentBy: ParticipantId;
  sentAt: string;
  respondedAt?: string;
  expiresAt: string;
}

// ============================================================================
// Collaboration Matching
// ============================================================================

export interface LabProfile {
  id: string;
  name: string;
  description: string;
  expertise: string[];
  domains: string[];
  resources: LabResource[];
  pastCollaborations: string[];
  availability: LabAvailability;
  preferences: CollaborationPreferences;
}

export interface LabResource {
  type: "compute" | "data" | "equipment" | "personnel";
  name: string;
  description: string;
  availability: "high" | "medium" | "low";
}

export interface LabAvailability {
  hoursPerWeek: number;
  availableFrom: string;
  availableUntil?: string;
  blackoutPeriods: Array<{ start: string; end: string }>;
}

export interface CollaborationPreferences {
  preferredRoles: ParticipantRole[];
  preferredCategories: MetaTaskCategory[];
  minTeamSize: number;
  maxTeamSize: number;
  preferredDuration: { min: number; max: number };
  openToNewCollaborators: boolean;
}

export interface CollaborationOpportunity {
  id: CollaborationId;
  type: "meta_task" | "lab_match" | "skill_complement";
  title: string;
  description: string;
  matchScore: number;
  benefits: OpportunityBenefit[];
  requirements: string[];
  estimatedEffort: { hours: number; weeks: number };
  deadline?: string;
  metaTask?: MetaTask;
  matchedLabs?: LabMatch[];
  createdAt: string;
}

export interface OpportunityBenefit {
  type: "skill_development" | "resource_access" | "network_expansion" | "publication" | "impact";
  description: string;
  value: number;
}

export interface LabMatch {
  lab: LabProfile;
  matchScore: number;
  expertiseOverlap: number;
  expertiseComplement: number;
  resourceMatch: number;
  availabilityMatch: number;
  recommendedRole: ParticipantRole;
  synergies: string[];
  challenges: string[];
}

export interface TeamComposition {
  labs: LabMatch[];
  overallScore: number;
  expertiseCoverage: number;
  resourceCoverage: number;
  roleBalance: Record<ParticipantRole, number>;
  strengths: string[];
  gaps: string[];
  recommendations: string[];
}

// ============================================================================
// Workflow Events
// ============================================================================

export interface WorkflowEvent {
  id: string;
  metaTaskId: MetaTaskId;
  type: WorkflowEventType;
  actor: ParticipantId | "system";
  description: string;
  data: Record<string, unknown>;
  timestamp: string;
}

export type WorkflowEventType =
  | "task_created"
  | "task_updated"
  | "status_changed"
  | "participant_joined"
  | "participant_left"
  | "invitation_sent"
  | "invitation_responded"
  | "objective_created"
  | "objective_updated"
  | "objective_completed"
  | "milestone_reached"
  | "deliverable_submitted"
  | "deliverable_approved"
  | "result_submitted"
  | "result_merged"
  | "conflict_detected"
  | "conflict_resolved"
  | "review_started"
  | "review_completed"
  | "task_completed";

// ============================================================================
// Conflict Resolution
// ============================================================================

export interface Conflict {
  id: string;
  metaTaskId: MetaTaskId;
  type: ConflictType;
  description: string;
  severity: "low" | "medium" | "high" | "critical";
  parties: ParticipantId[];
  status: "open" | "discussing" | "resolved" | "escalated";
  proposedResolutions: Resolution[];
  selectedResolution?: Resolution;
  createdAt: string;
  resolvedAt?: string;
}

export type ConflictType =
  | "resource_allocation"
  | "priority_disagreement"
  | "methodology_dispute"
  | "result_interpretation"
  | "credit_attribution"
  | "timeline_conflict"
  | "scope_creep";

export interface Resolution {
  id: string;
  proposedBy: ParticipantId;
  description: string;
  actions: string[];
  votes: Array<{ participant: ParticipantId; vote: "approve" | "reject" | "abstain" }>;
  status: "proposed" | "approved" | "rejected" | "implemented";
}

// ============================================================================
// Configuration
// ============================================================================

export interface CollaborationConfig {
  invitationExpiryDays: number;
  minParticipantsToStart: number;
  requireLeadApproval: boolean;
  allowMidTaskJoining: boolean;
  maxConflictEscalationDays: number;
  autoCompleteOnCriteriaMet: boolean;
  defaultVisibility: "public" | "private" | "invite_only";
}

export const DEFAULT_COLLABORATION_CONFIG: CollaborationConfig = {
  invitationExpiryDays: 14,
  minParticipantsToStart: 2,
  requireLeadApproval: true,
  allowMidTaskJoining: true,
  maxConflictEscalationDays: 7,
  autoCompleteOnCriteriaMet: false,
  defaultVisibility: "public",
};

// ============================================================================
// Factory Functions
// ============================================================================

export function createEmptyParticipantCommitment(): ParticipantCommitment {
  return {
    hoursPerWeek: 0,
    durationWeeks: 0,
    resources: [],
    responsibilities: [],
  };
}

export function createDefaultRequirements(): MetaTaskRequirements {
  return {
    minParticipants: 2,
    maxParticipants: 10,
    requiredExpertise: [],
    preferredExpertise: [],
    resourceRequirements: [],
  };
}

export function createDefaultTimeline(): MetaTaskTimeline {
  return {
    phases: [],
  };
}

export function createEmptyResultQuality(): ResultQuality {
  return {
    overallScore: 0,
    reproducibility: 0,
    novelty: 0,
    significance: 0,
    completeness: 0,
    reviewNotes: [],
  };
}

export function createDefaultLabAvailability(): LabAvailability {
  return {
    hoursPerWeek: 10,
    availableFrom: new Date().toISOString(),
    blackoutPeriods: [],
  };
}

export function createDefaultCollaborationPreferences(): CollaborationPreferences {
  return {
    preferredRoles: ["contributor"],
    preferredCategories: ["exploration", "integration"],
    minTeamSize: 2,
    maxTeamSize: 8,
    preferredDuration: { min: 4, max: 12 },
    openToNewCollaborators: true,
  };
}
