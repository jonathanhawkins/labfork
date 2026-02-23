/**
 * Collaboration Matcher
 *
 * Finds labs with complementary expertise, calculates match scores,
 * assesses benefits, and generates collaboration opportunities.
 */

import {
  MetaTask,
  MetaTaskCategory,
  LabProfile,
  LabMatch,
  TeamComposition,
  CollaborationOpportunity,
  OpportunityBenefit,
  ParticipantRole,
  createCollaborationId,
  createDefaultLabAvailability,
  createDefaultCollaborationPreferences,
} from "./types";
import { MetaTaskManager, getOpenTasks } from "./meta-task";

// ============================================================================
// Collaboration Matcher
// ============================================================================

export interface CollaborationMatcher {
  labs: Map<string, LabProfile>;
  opportunityCache: Map<string, CollaborationOpportunity[]>;
}

export function createCollaborationMatcher(): CollaborationMatcher {
  return {
    labs: new Map(),
    opportunityCache: new Map(),
  };
}

// ============================================================================
// Lab Registration
// ============================================================================

export function registerLab(
  matcher: CollaborationMatcher,
  lab: LabProfile
): void {
  matcher.labs.set(lab.id, lab);
  matcher.opportunityCache.delete(lab.id); // Invalidate cache
}

export function updateLabProfile(
  matcher: CollaborationMatcher,
  labId: string,
  updates: Partial<LabProfile>
): LabProfile | null {
  const lab = matcher.labs.get(labId);
  if (!lab) return null;

  const updated = { ...lab, ...updates };
  matcher.labs.set(labId, updated);
  matcher.opportunityCache.delete(labId);

  return updated;
}

export function getLab(
  matcher: CollaborationMatcher,
  labId: string
): LabProfile | undefined {
  return matcher.labs.get(labId);
}

export function getAllLabs(matcher: CollaborationMatcher): LabProfile[] {
  return Array.from(matcher.labs.values());
}

// ============================================================================
// Match Score Calculation
// ============================================================================

export interface MatchScoreFactors {
  expertiseOverlap: number;
  expertiseComplement: number;
  resourceMatch: number;
  availabilityMatch: number;
  preferenceAlignment: number;
  pastCollaborationBonus: number;
}

export function calculateMatchScore(
  lab1: LabProfile,
  lab2: LabProfile
): { score: number; factors: MatchScoreFactors } {
  // Expertise overlap (shared areas - good for communication)
  const sharedExpertise = lab1.expertise.filter((e) =>
    lab2.expertise.includes(e)
  );
  const expertiseOverlap =
    sharedExpertise.length /
    Math.max(1, Math.min(lab1.expertise.length, lab2.expertise.length));

  // Expertise complement (unique areas - good for coverage)
  const uniqueToLab1 = lab1.expertise.filter((e) => !lab2.expertise.includes(e));
  const uniqueToLab2 = lab2.expertise.filter((e) => !lab1.expertise.includes(e));
  const totalUnique = uniqueToLab1.length + uniqueToLab2.length;
  const totalExpertise = new Set([...lab1.expertise, ...lab2.expertise]).size;
  const expertiseComplement = totalExpertise > 0 ? totalUnique / totalExpertise : 0;

  // Resource match (complementary resources)
  const resourceTypes1 = new Set(lab1.resources.map((r) => r.type));
  const resourceTypes2 = new Set(lab2.resources.map((r) => r.type));
  const uniqueResources = new Set([...Array.from(resourceTypes1), ...Array.from(resourceTypes2)]).size;
  const resourceMatch = uniqueResources > 0 ? uniqueResources / 4 : 0; // 4 resource types

  // Availability match
  const availabilityMatch = calculateAvailabilityMatch(
    lab1.availability,
    lab2.availability
  );

  // Preference alignment
  const preferenceAlignment = calculatePreferenceAlignment(
    lab1.preferences,
    lab2.preferences
  );

  // Past collaboration bonus
  const pastCollaborationBonus =
    lab1.pastCollaborations.includes(lab2.id) ||
    lab2.pastCollaborations.includes(lab1.id)
      ? 0.1
      : 0;

  const factors: MatchScoreFactors = {
    expertiseOverlap,
    expertiseComplement,
    resourceMatch,
    availabilityMatch,
    preferenceAlignment,
    pastCollaborationBonus,
  };

  // Weighted score
  const score =
    expertiseOverlap * 0.15 +
    expertiseComplement * 0.25 +
    resourceMatch * 0.15 +
    availabilityMatch * 0.2 +
    preferenceAlignment * 0.15 +
    pastCollaborationBonus;

  return { score: Math.min(1, score), factors };
}

function calculateAvailabilityMatch(
  avail1: LabProfile["availability"],
  avail2: LabProfile["availability"]
): number {
  // Check if available periods overlap
  const start1 = new Date(avail1.availableFrom).getTime();
  const start2 = new Date(avail2.availableFrom).getTime();
  const maxStart = Math.max(start1, start2);

  const end1 = avail1.availableUntil
    ? new Date(avail1.availableUntil).getTime()
    : Infinity;
  const end2 = avail2.availableUntil
    ? new Date(avail2.availableUntil).getTime()
    : Infinity;
  const minEnd = Math.min(end1, end2);

  if (maxStart >= minEnd) return 0; // No overlap

  // Check hours per week compatibility
  const minHours = Math.min(avail1.hoursPerWeek, avail2.hoursPerWeek);
  const hoursMatch = minHours >= 5 ? Math.min(minHours / 20, 1) : 0;

  return hoursMatch;
}

function calculatePreferenceAlignment(
  pref1: LabProfile["preferences"],
  pref2: LabProfile["preferences"]
): number {
  let alignment = 0;

  // Role compatibility
  const roleOverlap = pref1.preferredRoles.filter((r) =>
    pref2.preferredRoles.includes(r)
  ).length;
  alignment += roleOverlap > 0 ? 0.25 : 0;

  // Category overlap
  const categoryOverlap = pref1.preferredCategories.filter((c) =>
    pref2.preferredCategories.includes(c)
  ).length;
  alignment += categoryOverlap > 0 ? 0.25 : 0;

  // Team size compatibility
  const sizeCompatible =
    Math.max(pref1.minTeamSize, pref2.minTeamSize) <=
    Math.min(pref1.maxTeamSize, pref2.maxTeamSize);
  alignment += sizeCompatible ? 0.25 : 0;

  // Duration compatibility
  const durationCompatible =
    Math.max(pref1.preferredDuration.min, pref2.preferredDuration.min) <=
    Math.min(pref1.preferredDuration.max, pref2.preferredDuration.max);
  alignment += durationCompatible ? 0.25 : 0;

  return alignment;
}

// ============================================================================
// Lab Matching
// ============================================================================

export function findMatchingLabs(
  matcher: CollaborationMatcher,
  labId: string,
  options?: {
    minScore?: number;
    maxResults?: number;
    requiredExpertise?: string[];
    preferredCategories?: MetaTaskCategory[];
  }
): LabMatch[] {
  const lab = matcher.labs.get(labId);
  if (!lab) return [];

  const minScore = options?.minScore ?? 0.3;
  const maxResults = options?.maxResults ?? 10;

  const matches: LabMatch[] = [];

  for (const [otherLabId, otherLab] of Array.from(matcher.labs)) {
    if (otherLabId === labId) continue;
    if (!otherLab.preferences.openToNewCollaborators) continue;

    // Check required expertise
    if (options?.requiredExpertise) {
      const hasRequired = options.requiredExpertise.every(
        (e) =>
          otherLab.expertise.includes(e) || lab.expertise.includes(e)
      );
      if (!hasRequired) continue;
    }

    // Check preferred categories
    if (options?.preferredCategories) {
      const hasCategory = options.preferredCategories.some((c) =>
        otherLab.preferences.preferredCategories.includes(c)
      );
      if (!hasCategory) continue;
    }

    const { score, factors } = calculateMatchScore(lab, otherLab);
    if (score < minScore) continue;

    matches.push({
      lab: otherLab,
      matchScore: score,
      expertiseOverlap: factors.expertiseOverlap,
      expertiseComplement: factors.expertiseComplement,
      resourceMatch: factors.resourceMatch,
      availabilityMatch: factors.availabilityMatch,
      recommendedRole: recommendRole(otherLab, lab),
      synergies: identifySynergies(lab, otherLab),
      challenges: identifyChallenges(lab, otherLab),
    });
  }

  // Sort by match score
  matches.sort((a, b) => b.matchScore - a.matchScore);

  return matches.slice(0, maxResults);
}

export function findLabsForTask(
  matcher: CollaborationMatcher,
  task: MetaTask
): LabMatch[] {
  const matches: LabMatch[] = [];
  const existingLabIds = new Set(task.participants.map((p) => p.labId));

  for (const [labId, lab] of Array.from(matcher.labs)) {
    if (existingLabIds.has(labId)) continue;
    if (!lab.preferences.openToNewCollaborators) continue;

    // Check required expertise
    const hasRequiredExpertise = task.requirements.requiredExpertise.every(
      (e) =>
        lab.expertise.includes(e) ||
        task.participants.some((p) => p.expertise.includes(e))
    );
    if (!hasRequiredExpertise) continue;

    // Calculate task-specific score
    const score = calculateTaskMatchScore(lab, task);
    if (score < 0.3) continue;

    matches.push({
      lab,
      matchScore: score,
      expertiseOverlap: calculateExpertiseOverlap(lab, task),
      expertiseComplement: calculateExpertiseComplement(lab, task),
      resourceMatch: calculateResourceMatch(lab, task),
      availabilityMatch: calculateTaskAvailabilityMatch(lab, task),
      recommendedRole: recommendRoleForTask(lab, task),
      synergies: identifyTaskSynergies(lab, task),
      challenges: identifyTaskChallenges(lab, task),
    });
  }

  matches.sort((a, b) => b.matchScore - a.matchScore);
  return matches;
}

function calculateTaskMatchScore(lab: LabProfile, task: MetaTask): number {
  let score = 0;

  // Expertise match
  const requiredMatch = task.requirements.requiredExpertise.filter((e) =>
    lab.expertise.includes(e)
  ).length;
  const preferredMatch = task.requirements.preferredExpertise.filter((e) =>
    lab.expertise.includes(e)
  ).length;

  score +=
    (requiredMatch / Math.max(1, task.requirements.requiredExpertise.length)) * 0.3;
  score +=
    (preferredMatch / Math.max(1, task.requirements.preferredExpertise.length)) * 0.2;

  // Domain match
  const domainMatch = task.domains.filter((d) => lab.domains.includes(d)).length;
  score += (domainMatch / Math.max(1, task.domains.length)) * 0.2;

  // Category preference
  if (lab.preferences.preferredCategories.includes(task.category)) {
    score += 0.15;
  }

  // Availability
  score += calculateTaskAvailabilityMatch(lab, task) * 0.15;

  return score;
}

function calculateExpertiseOverlap(lab: LabProfile, task: MetaTask): number {
  const taskExpertise = [
    ...task.requirements.requiredExpertise,
    ...task.requirements.preferredExpertise,
  ];
  const overlap = taskExpertise.filter((e) => lab.expertise.includes(e));
  return overlap.length / Math.max(1, taskExpertise.length);
}

function calculateExpertiseComplement(lab: LabProfile, task: MetaTask): number {
  const existingExpertise = new Set(
    task.participants.flatMap((p) => p.expertise)
  );
  const newExpertise = lab.expertise.filter((e) => !existingExpertise.has(e));
  return newExpertise.length / Math.max(1, lab.expertise.length);
}

function calculateResourceMatch(lab: LabProfile, task: MetaTask): number {
  const requiredResources = task.requirements.resourceRequirements.filter(
    (r) => r.priority === "required"
  );
  const matched = requiredResources.filter((req) =>
    lab.resources.some((r) => r.type === req.type)
  );
  return matched.length / Math.max(1, requiredResources.length);
}

function calculateTaskAvailabilityMatch(
  lab: LabProfile,
  task: MetaTask
): number {
  if (!task.timeline.startDate) return 0.5;

  const taskStart = new Date(task.timeline.startDate).getTime();
  const labStart = new Date(lab.availability.availableFrom).getTime();

  if (labStart > taskStart) return 0.2;

  return Math.min(lab.availability.hoursPerWeek / 10, 1);
}

// ============================================================================
// Role Recommendations
// ============================================================================

function recommendRole(
  targetLab: LabProfile,
  referenceLab: LabProfile
): ParticipantRole {
  // If target has more expertise, suggest advisor
  if (targetLab.expertise.length > referenceLab.expertise.length * 1.5) {
    return "advisor";
  }

  // Check preferred roles
  if (targetLab.preferences.preferredRoles.includes("lead")) {
    return "co-lead";
  }

  if (targetLab.preferences.preferredRoles.includes("reviewer")) {
    return "reviewer";
  }

  return "contributor";
}

function recommendRoleForTask(lab: LabProfile, task: MetaTask): ParticipantRole {
  // Check if task needs specific roles
  const existingRoles = new Set(task.participants.map((p) => p.role));

  if (!existingRoles.has("co-lead") && lab.preferences.preferredRoles.includes("lead")) {
    return "co-lead";
  }

  if (
    !existingRoles.has("reviewer") &&
    lab.preferences.preferredRoles.includes("reviewer")
  ) {
    return "reviewer";
  }

  // Match expertise to task needs
  const hasRequiredExpertise = task.requirements.requiredExpertise.some((e) =>
    lab.expertise.includes(e)
  );
  if (hasRequiredExpertise) {
    return "contributor";
  }

  if (lab.expertise.length > 5) {
    return "advisor";
  }

  return "contributor";
}

// ============================================================================
// Synergy & Challenge Identification
// ============================================================================

function identifySynergies(lab1: LabProfile, lab2: LabProfile): string[] {
  const synergies: string[] = [];

  // Expertise synergies
  const uniqueExpertise1 = lab1.expertise.filter(
    (e) => !lab2.expertise.includes(e)
  );
  const uniqueExpertise2 = lab2.expertise.filter(
    (e) => !lab1.expertise.includes(e)
  );

  if (uniqueExpertise1.length > 0 && uniqueExpertise2.length > 0) {
    synergies.push(
      `Complementary expertise: ${lab1.name} brings ${uniqueExpertise1.slice(0, 2).join(", ")}; ${lab2.name} brings ${uniqueExpertise2.slice(0, 2).join(", ")}`
    );
  }

  // Resource synergies
  const resources1 = new Set(lab1.resources.map((r) => r.type));
  const resources2 = new Set(lab2.resources.map((r) => r.type));
  const uniqueResources = [
    ...Array.from(resources1).filter((r) => !resources2.has(r)),
    ...Array.from(resources2).filter((r) => !resources1.has(r)),
  ];

  if (uniqueResources.length > 0) {
    synergies.push(`Combined resources: ${uniqueResources.join(", ")}`);
  }

  // Domain coverage
  const allDomains = new Set([...lab1.domains, ...lab2.domains]);
  if (allDomains.size > Math.max(lab1.domains.length, lab2.domains.length)) {
    synergies.push(`Expanded domain coverage: ${allDomains.size} domains`);
  }

  // Past collaboration
  if (
    lab1.pastCollaborations.includes(lab2.id) ||
    lab2.pastCollaborations.includes(lab1.id)
  ) {
    synergies.push("Prior successful collaboration experience");
  }

  return synergies;
}

function identifyChallenges(lab1: LabProfile, lab2: LabProfile): string[] {
  const challenges: string[] = [];

  // Availability challenges
  const hoursDiff = Math.abs(
    lab1.availability.hoursPerWeek - lab2.availability.hoursPerWeek
  );
  if (hoursDiff > 10) {
    challenges.push("Significant difference in time commitment availability");
  }

  // Team size preference mismatch
  if (
    lab1.preferences.maxTeamSize < lab2.preferences.minTeamSize ||
    lab2.preferences.maxTeamSize < lab1.preferences.minTeamSize
  ) {
    challenges.push("Mismatched team size preferences");
  }

  // Duration preference mismatch
  if (
    lab1.preferences.preferredDuration.max < lab2.preferences.preferredDuration.min ||
    lab2.preferences.preferredDuration.max < lab1.preferences.preferredDuration.min
  ) {
    challenges.push("Mismatched project duration preferences");
  }

  // No expertise overlap (communication challenge)
  const sharedExpertise = lab1.expertise.filter((e) =>
    lab2.expertise.includes(e)
  );
  if (sharedExpertise.length === 0) {
    challenges.push("No shared expertise areas - may require extra coordination");
  }

  return challenges;
}

function identifyTaskSynergies(lab: LabProfile, task: MetaTask): string[] {
  const synergies: string[] = [];

  // Expertise coverage
  const newExpertise = lab.expertise.filter(
    (e) =>
      !task.participants.some((p) => p.expertise.includes(e))
  );
  if (newExpertise.length > 0) {
    synergies.push(`Adds new expertise: ${newExpertise.slice(0, 3).join(", ")}`);
  }

  // Resource contribution
  const neededResources = task.requirements.resourceRequirements.filter(
    (r) => r.priority !== "optional"
  );
  const canProvide = neededResources.filter((req) =>
    lab.resources.some((r) => r.type === req.type)
  );
  if (canProvide.length > 0) {
    synergies.push(`Can provide: ${canProvide.map((r) => r.type).join(", ")}`);
  }

  // Domain alignment
  const sharedDomains = task.domains.filter((d) => lab.domains.includes(d));
  if (sharedDomains.length > 0) {
    synergies.push(`Domain alignment: ${sharedDomains.join(", ")}`);
  }

  return synergies;
}

function identifyTaskChallenges(lab: LabProfile, task: MetaTask): string[] {
  const challenges: string[] = [];

  // Missing required expertise
  const missingRequired = task.requirements.requiredExpertise.filter(
    (e) =>
      !lab.expertise.includes(e) &&
      !task.participants.some((p) => p.expertise.includes(e))
  );
  if (missingRequired.length > 0) {
    challenges.push(`Still needs: ${missingRequired.join(", ")}`);
  }

  // Timeline mismatch
  if (task.timeline.startDate) {
    const taskStart = new Date(task.timeline.startDate).getTime();
    const labStart = new Date(lab.availability.availableFrom).getTime();
    if (labStart > taskStart) {
      challenges.push("Lab not available until after task start date");
    }
  }

  return challenges;
}

// ============================================================================
// Team Composition
// ============================================================================

export function composeOptimalTeam(
  matcher: CollaborationMatcher,
  task: MetaTask,
  options?: {
    targetSize?: number;
    mustIncludeLabs?: string[];
  }
): TeamComposition {
  const targetSize = options?.targetSize ?? task.requirements.minParticipants + 2;
  const candidates = findLabsForTask(matcher, task);

  // Start with must-include labs
  const team: LabMatch[] = [];
  if (options?.mustIncludeLabs) {
    for (const labId of options.mustIncludeLabs) {
      const match = candidates.find((m) => m.lab.id === labId);
      if (match) {
        team.push(match);
      }
    }
  }

  // Greedily add labs that maximize coverage
  const selectedIds = new Set(team.map((m) => m.lab.id));
  const coveredExpertise = new Set(team.flatMap((m) => m.lab.expertise));

  while (team.length < targetSize && candidates.length > 0) {
    let bestCandidate: LabMatch | null = null;
    let bestScore = -1;

    for (const candidate of candidates) {
      if (selectedIds.has(candidate.lab.id)) continue;

      // Score based on new expertise contribution
      const newExpertise = candidate.lab.expertise.filter(
        (e) => !coveredExpertise.has(e)
      );
      const diversityBonus = newExpertise.length * 0.1;
      const score = candidate.matchScore + diversityBonus;

      if (score > bestScore) {
        bestScore = score;
        bestCandidate = candidate;
      }
    }

    if (!bestCandidate) break;

    team.push(bestCandidate);
    selectedIds.add(bestCandidate.lab.id);
    for (const e of bestCandidate.lab.expertise) {
      coveredExpertise.add(e);
    }
  }

  // Calculate team metrics
  const roleDistribution: Record<ParticipantRole, number> = {
    lead: 0,
    "co-lead": 0,
    contributor: 0,
    advisor: 0,
    reviewer: 0,
    observer: 0,
  };

  for (const match of team) {
    roleDistribution[match.recommendedRole]++;
  }

  // Add existing participants
  for (const participant of task.participants) {
    roleDistribution[participant.role]++;
  }

  // Calculate coverage
  const allRequiredExpertise = new Set(task.requirements.requiredExpertise);
  const coveredRequired = task.requirements.requiredExpertise.filter((e) =>
    coveredExpertise.has(e)
  );
  const expertiseCoverage = coveredRequired.length / Math.max(1, allRequiredExpertise.size);

  const requiredResources = task.requirements.resourceRequirements.filter(
    (r) => r.priority === "required"
  );
  const teamResources = new Set(team.flatMap((m) => m.lab.resources.map((r) => r.type)));
  const coveredResources = requiredResources.filter((r) =>
    teamResources.has(r.type as "data" | "compute" | "equipment" | "personnel")
  );
  const resourceCoverage = coveredResources.length / Math.max(1, requiredResources.length);

  return {
    labs: team,
    overallScore:
      team.reduce((sum, m) => sum + m.matchScore, 0) / Math.max(1, team.length),
    expertiseCoverage,
    resourceCoverage,
    roleBalance: roleDistribution,
    strengths: identifyTeamStrengths(team, task),
    gaps: identifyTeamGaps(team, task),
    recommendations: generateTeamRecommendations(team, task),
  };
}

function identifyTeamStrengths(team: LabMatch[], task: MetaTask): string[] {
  const strengths: string[] = [];

  if (team.length >= task.requirements.minParticipants) {
    strengths.push("Meets minimum participant requirement");
  }

  const allSynergies = team.flatMap((m) => m.synergies);
  const expertiseSynergies = allSynergies.filter((s) =>
    s.includes("expertise")
  );
  if (expertiseSynergies.length > 0) {
    strengths.push("Strong expertise complementarity");
  }

  const avgScore = team.reduce((sum, m) => sum + m.matchScore, 0) / team.length;
  if (avgScore > 0.7) {
    strengths.push("High overall match quality");
  }

  return strengths;
}

function identifyTeamGaps(team: LabMatch[], task: MetaTask): string[] {
  const gaps: string[] = [];

  const allExpertise = new Set(team.flatMap((m) => m.lab.expertise));
  const missingRequired = task.requirements.requiredExpertise.filter(
    (e) => !allExpertise.has(e)
  );
  if (missingRequired.length > 0) {
    gaps.push(`Missing required expertise: ${missingRequired.join(", ")}`);
  }

  if (team.length < task.requirements.minParticipants) {
    gaps.push(
      `Need ${task.requirements.minParticipants - team.length} more participants`
    );
  }

  const allChallenges = team.flatMap((m) => m.challenges);
  if (allChallenges.length > team.length) {
    gaps.push("Multiple coordination challenges identified");
  }

  return gaps;
}

function generateTeamRecommendations(
  team: LabMatch[],
  task: MetaTask
): string[] {
  const recommendations: string[] = [];

  // Role balance
  const roles = team.map((m) => m.recommendedRole);
  if (!roles.includes("reviewer")) {
    recommendations.push("Consider adding a reviewer for quality assurance");
  }

  // Expertise gaps
  const allExpertise = new Set(team.flatMap((m) => m.lab.expertise));
  const missingPreferred = task.requirements.preferredExpertise.filter(
    (e) => !allExpertise.has(e)
  );
  if (missingPreferred.length > 0) {
    recommendations.push(
      `Look for labs with: ${missingPreferred.slice(0, 3).join(", ")}`
    );
  }

  // Size optimization
  if (team.length > task.requirements.maxParticipants - 2) {
    recommendations.push("Team approaching maximum size - be selective");
  }

  return recommendations;
}

// ============================================================================
// Opportunity Generation
// ============================================================================

export function generateOpportunities(
  matcher: CollaborationMatcher,
  taskManager: MetaTaskManager,
  labId: string
): CollaborationOpportunity[] {
  const lab = matcher.labs.get(labId);
  if (!lab) return [];

  // Check cache
  const cached = matcher.opportunityCache.get(labId);
  if (cached) return cached;

  const opportunities: CollaborationOpportunity[] = [];

  // 1. Open meta-tasks
  const openTasks = getOpenTasks(taskManager);
  for (const task of openTasks) {
    const matches = findLabsForTask(matcher, task);
    const labMatch = matches.find((m) => m.lab.id === labId);

    if (!labMatch && calculateTaskMatchScore(lab, task) > 0.3) {
      opportunities.push({
        id: createCollaborationId(),
        type: "meta_task",
        title: `Join: ${task.title}`,
        description: task.description,
        matchScore: calculateTaskMatchScore(lab, task),
        benefits: assessBenefits(lab, task),
        requirements: task.requirements.requiredExpertise,
        estimatedEffort: estimateEffort(task),
        deadline: task.timeline.recruitmentDeadline,
        metaTask: task,
        createdAt: new Date().toISOString(),
      });
    }
  }

  // 2. Matching labs for new collaborations
  const matchingLabs = findMatchingLabs(matcher, labId, {
    minScore: 0.5,
    maxResults: 5,
  });

  if (matchingLabs.length > 0) {
    opportunities.push({
      id: createCollaborationId(),
      type: "lab_match",
      title: "Potential Collaborators Found",
      description: `${matchingLabs.length} labs with complementary expertise`,
      matchScore: matchingLabs[0].matchScore,
      benefits: [
        {
          type: "network_expansion",
          description: "Expand research network",
          value: matchingLabs.length * 0.1,
        },
        {
          type: "skill_development",
          description: "Learn from diverse expertise",
          value: 0.3,
        },
      ],
      requirements: [],
      estimatedEffort: { hours: 2, weeks: 1 },
      matchedLabs: matchingLabs,
      createdAt: new Date().toISOString(),
    });
  }

  // 3. Skill complement opportunities
  const skillGaps = identifySkillGaps(matcher, lab);
  if (skillGaps.length > 0) {
    const complementLabs = findLabsWithSkills(matcher, skillGaps, labId);
    if (complementLabs.length > 0) {
      opportunities.push({
        id: createCollaborationId(),
        type: "skill_complement",
        title: "Skill Enhancement Opportunity",
        description: `Find partners to cover: ${skillGaps.slice(0, 3).join(", ")}`,
        matchScore: 0.6,
        benefits: [
          {
            type: "skill_development",
            description: `Acquire skills in ${skillGaps.length} areas`,
            value: skillGaps.length * 0.15,
          },
        ],
        requirements: skillGaps,
        estimatedEffort: { hours: 10, weeks: 4 },
        matchedLabs: complementLabs,
        createdAt: new Date().toISOString(),
      });
    }
  }

  // Sort by match score
  opportunities.sort((a, b) => b.matchScore - a.matchScore);

  // Cache results
  matcher.opportunityCache.set(labId, opportunities);

  return opportunities;
}

function assessBenefits(
  lab: LabProfile,
  task: MetaTask
): OpportunityBenefit[] {
  const benefits: OpportunityBenefit[] = [];

  // Skill development
  const newSkills = task.requirements.requiredExpertise.filter(
    (e) => !lab.expertise.includes(e)
  );
  if (newSkills.length > 0) {
    benefits.push({
      type: "skill_development",
      description: `Learn: ${newSkills.slice(0, 2).join(", ")}`,
      value: newSkills.length * 0.1,
    });
  }

  // Network expansion
  benefits.push({
    type: "network_expansion",
    description: `Collaborate with ${task.participants.length} labs`,
    value: task.participants.length * 0.1,
  });

  // Impact potential
  if (task.category === "benchmark" || task.category === "dataset") {
    benefits.push({
      type: "impact",
      description: "Contribute to community resource",
      value: 0.4,
    });
  }

  // Publication potential
  if (task.category === "exploration" || task.category === "integration") {
    benefits.push({
      type: "publication",
      description: "Potential for joint publication",
      value: 0.5,
    });
  }

  return benefits;
}

function estimateEffort(task: MetaTask): { hours: number; weeks: number } {
  let hours = 20; // Base
  let weeks = 4; // Base

  // Adjust by category
  const categoryMultipliers: Record<MetaTaskCategory, number> = {
    exploration: 1.5,
    integration: 1.3,
    benchmark: 1.2,
    dataset: 1.4,
    replication: 1.0,
    extension: 1.2,
    application: 1.3,
  };

  const multiplier = categoryMultipliers[task.category];
  hours *= multiplier;

  // Adjust by objectives
  hours += task.objectives.length * 5;
  weeks += Math.ceil(task.objectives.length / 2);

  // Adjust by timeline
  if (task.timeline.endDate) {
    const duration =
      (new Date(task.timeline.endDate).getTime() -
        new Date(task.timeline.startDate || Date.now()).getTime()) /
      (1000 * 60 * 60 * 24 * 7);
    weeks = Math.max(weeks, Math.ceil(duration));
  }

  return { hours: Math.round(hours), weeks };
}

function identifySkillGaps(
  matcher: CollaborationMatcher,
  lab: LabProfile
): string[] {
  // Find common skills across all labs that this lab doesn't have
  const allSkills = new Map<string, number>();

  for (const [id, otherLab] of Array.from(matcher.labs)) {
    if (id === lab.id) continue;
    for (const skill of otherLab.expertise) {
      allSkills.set(skill, (allSkills.get(skill) || 0) + 1);
    }
  }

  // Find skills that appear in many labs but not in this one
  const gaps: string[] = [];
  for (const [skill, count] of Array.from(allSkills)) {
    if (count >= 3 && !lab.expertise.includes(skill)) {
      gaps.push(skill);
    }
  }

  return gaps.slice(0, 5);
}

function findLabsWithSkills(
  matcher: CollaborationMatcher,
  skills: string[],
  excludeLabId: string
): LabMatch[] {
  const matches: LabMatch[] = [];

  for (const [labId, lab] of Array.from(matcher.labs)) {
    if (labId === excludeLabId) continue;

    const matchingSkills = skills.filter((s) => lab.expertise.includes(s));
    if (matchingSkills.length === 0) continue;

    matches.push({
      lab,
      matchScore: matchingSkills.length / skills.length,
      expertiseOverlap: 0,
      expertiseComplement: matchingSkills.length / lab.expertise.length,
      resourceMatch: 0,
      availabilityMatch: 0.5,
      recommendedRole: "contributor",
      synergies: [`Can teach: ${matchingSkills.join(", ")}`],
      challenges: [],
    });
  }

  return matches.sort((a, b) => b.matchScore - a.matchScore).slice(0, 5);
}

// ============================================================================
// Global Matcher
// ============================================================================

let globalMatcher: CollaborationMatcherWrapper | null = null;

/**
 * Wrapper class that provides method-style access to matcher functions
 */
export class CollaborationMatcherWrapper {
  private matcher: CollaborationMatcher;

  constructor() {
    this.matcher = createCollaborationMatcher();
  }

  get labs(): Map<string, LabProfile> {
    return this.matcher.labs;
  }

  get opportunityCache(): Map<string, CollaborationOpportunity[]> {
    return this.matcher.opportunityCache;
  }

  registerLab(lab: LabProfile): void {
    registerLab(this.matcher, lab);
  }

  updateLabProfile(labId: string, updates: Partial<LabProfile>): LabProfile | null {
    return updateLabProfile(this.matcher, labId, updates);
  }

  getLab(labId: string): LabProfile | undefined {
    return getLab(this.matcher, labId);
  }

  getAllLabs(): LabProfile[] {
    return getAllLabs(this.matcher);
  }

  findMatchingLabs(
    labOrId: LabProfile | string,
    maxResults?: number
  ): LabMatch[] {
    const labId = typeof labOrId === 'string' ? labOrId : labOrId.id;
    return findMatchingLabs(this.matcher, labId, { maxResults });
  }

  findLabsForTask(task: MetaTask, maxResults?: number): LabMatch[] {
    return findLabsForTask(this.matcher, task).slice(0, maxResults);
  }

  composeOptimalTeam(
    task: MetaTask,
    targetSize?: number
  ): TeamComposition {
    return composeOptimalTeam(this.matcher, task, { targetSize });
  }

  generateOpportunities(
    lab: LabProfile,
    taskManager: MetaTaskManager,
    limit?: number
  ): CollaborationOpportunity[] {
    return generateOpportunities(this.matcher, taskManager, lab.id).slice(0, limit);
  }
}

export function getGlobalCollaborationMatcher(): CollaborationMatcherWrapper {
  if (!globalMatcher) {
    globalMatcher = new CollaborationMatcherWrapper();
  }
  return globalMatcher;
}

export function resetGlobalCollaborationMatcher(): void {
  globalMatcher = null;
}
