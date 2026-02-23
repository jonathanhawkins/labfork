/**
 * Opportunity Board
 *
 * Public board showing all research gaps with filtering,
 * bounty system, claim mechanism, progress tracking, and leaderboard.
 */

import {
  ResearchOpportunity,
  GapType,
  DifficultyLevel,
  EffortEstimate,
  SignificanceLevel,
  OpportunityStatus,
  ClaimInfo,
  BountyInfo,
  OpportunityLeaderboard,
  LeaderboardEntry,
  PaginatedResult,
  FilterOptions,
} from "./types";

// ============================================================================
// Opportunity Board Interface
// ============================================================================

export interface OpportunityBoard {
  opportunities: Map<string, ResearchOpportunity>;
  leaderboard: OpportunityLeaderboard;
  claims: Map<string, ClaimInfo>;
  bountyPool: number;
}

export interface BoardConfig {
  maxActiveClaimsPerLab: number;
  claimExpirationDays: number;
  minBountyAmount: number;
  defaultEffortMultiplier: number;
}

const DEFAULT_CONFIG: BoardConfig = {
  maxActiveClaimsPerLab: 3,
  claimExpirationDays: 30,
  minBountyAmount: 10,
  defaultEffortMultiplier: 1.5,
};

// ============================================================================
// Factory Functions
// ============================================================================

export function createOpportunityBoard(): OpportunityBoard {
  return {
    opportunities: new Map(),
    leaderboard: {
      entries: [],
      period: "all-time",
      updatedAt: new Date().toISOString(),
    },
    claims: new Map(),
    bountyPool: 0,
  };
}

export function createOpportunity(
  title: string,
  description: string,
  gapType: GapType,
  difficulty: DifficultyLevel,
  impact: SignificanceLevel,
  domain: string,
  options: Partial<{
    tags: string[];
    sourceGapId: string;
    relatedTechniques: string[];
    requiredExpertise: string[];
    suggestedApproaches: string[];
    bounty: BountyInfo;
    deadline: string;
  }> = {}
): ResearchOpportunity {
  const id = `opp-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  const now = new Date().toISOString();

  return {
    id,
    title,
    description,
    gapType,
    difficulty,
    estimatedEffort: estimateEffort(difficulty, gapType),
    impact,
    domain,
    tags: options.tags || [],
    status: "open",
    progress: 0,
    hasBounty: !!options.bounty,
    bounty: options.bounty,
    sourceGapId: options.sourceGapId,
    relatedTechniques: options.relatedTechniques || [],
    requiredExpertise: options.requiredExpertise || [],
    suggestedApproaches: options.suggestedApproaches || [],
    createdAt: now,
    updatedAt: now,
    deadline: options.deadline,
  };
}

function estimateEffort(
  difficulty: DifficultyLevel,
  gapType: GapType
): EffortEstimate {
  const difficultyHours: Record<DifficultyLevel, [number, number]> = {
    beginner: [2, 8],
    intermediate: [8, 24],
    advanced: [24, 80],
    expert: [80, 200],
  };

  const typeMultipliers: Record<GapType, number> = {
    "missing-technique": 1.5,
    "unexplored-combination": 1.2,
    "domain-adaptation": 1.3,
    "performance-optimization": 1.4,
    scalability: 1.6,
    "theoretical-gap": 2.0,
    tooling: 1.0,
    documentation: 0.5,
  };

  const [minBase, maxBase] = difficultyHours[difficulty];
  const multiplier = typeMultipliers[gapType];

  const teamSize =
    difficulty === "beginner"
      ? 1
      : difficulty === "intermediate"
      ? 1
      : difficulty === "advanced"
      ? 2
      : 3;

  return {
    minHours: Math.round(minBase * multiplier),
    maxHours: Math.round(maxBase * multiplier),
    recommendedTeamSize: teamSize,
  };
}

// ============================================================================
// Opportunity Management
// ============================================================================

export function addOpportunity(
  board: OpportunityBoard,
  opportunity: ResearchOpportunity
): ResearchOpportunity {
  board.opportunities.set(opportunity.id, opportunity);

  if (opportunity.bounty) {
    board.bountyPool += opportunity.bounty.amount;
  }

  return opportunity;
}

export function getOpportunity(
  board: OpportunityBoard,
  opportunityId: string
): ResearchOpportunity | null {
  return board.opportunities.get(opportunityId) || null;
}

export function updateOpportunity(
  board: OpportunityBoard,
  opportunityId: string,
  updates: Partial<ResearchOpportunity>
): ResearchOpportunity | null {
  const opportunity = board.opportunities.get(opportunityId);
  if (!opportunity) return null;

  const updated: ResearchOpportunity = {
    ...opportunity,
    ...updates,
    id: opportunity.id, // Preserve ID
    createdAt: opportunity.createdAt, // Preserve creation date
    updatedAt: new Date().toISOString(),
  };

  board.opportunities.set(opportunityId, updated);
  return updated;
}

export function removeOpportunity(
  board: OpportunityBoard,
  opportunityId: string
): boolean {
  const opportunity = board.opportunities.get(opportunityId);
  if (!opportunity) return false;

  if (opportunity.bounty) {
    board.bountyPool -= opportunity.bounty.amount;
  }

  return board.opportunities.delete(opportunityId);
}

// ============================================================================
// Claim Management
// ============================================================================

export function claimOpportunity(
  board: OpportunityBoard,
  opportunityId: string,
  labId: string,
  labName: string,
  expectedCompletion?: string,
  config: BoardConfig = DEFAULT_CONFIG
): ResearchOpportunity | null {
  const opportunity = board.opportunities.get(opportunityId);
  if (!opportunity) return null;

  if (opportunity.status !== "open") {
    return null; // Already claimed or completed
  }

  // Check claim limit
  const labClaims = Array.from(board.opportunities.values()).filter(
    (o) => o.claimedBy?.labId === labId && o.status === "claimed"
  );

  if (labClaims.length >= config.maxActiveClaimsPerLab) {
    return null; // Too many active claims
  }

  const claimInfo: ClaimInfo = {
    labId,
    labName,
    claimedAt: new Date().toISOString(),
    expectedCompletion,
  };

  const updated: ResearchOpportunity = {
    ...opportunity,
    status: "claimed",
    claimedBy: claimInfo,
    updatedAt: new Date().toISOString(),
  };

  board.opportunities.set(opportunityId, updated);
  board.claims.set(opportunityId, claimInfo);

  return updated;
}

export function unclaimOpportunity(
  board: OpportunityBoard,
  opportunityId: string,
  labId: string
): ResearchOpportunity | null {
  const opportunity = board.opportunities.get(opportunityId);
  if (!opportunity) return null;

  if (opportunity.claimedBy?.labId !== labId) {
    return null; // Not claimed by this lab
  }

  const updated: ResearchOpportunity = {
    ...opportunity,
    status: "open",
    claimedBy: undefined,
    progress: 0,
    updatedAt: new Date().toISOString(),
  };

  board.opportunities.set(opportunityId, updated);
  board.claims.delete(opportunityId);

  return updated;
}

export function startProgress(
  board: OpportunityBoard,
  opportunityId: string,
  labId: string
): ResearchOpportunity | null {
  const opportunity = board.opportunities.get(opportunityId);
  if (!opportunity) return null;

  if (
    opportunity.claimedBy?.labId !== labId ||
    opportunity.status !== "claimed"
  ) {
    return null;
  }

  const updated: ResearchOpportunity = {
    ...opportunity,
    status: "in-progress",
    updatedAt: new Date().toISOString(),
  };

  board.opportunities.set(opportunityId, updated);
  return updated;
}

export function updateProgress(
  board: OpportunityBoard,
  opportunityId: string,
  labId: string,
  progress: number
): ResearchOpportunity | null {
  const opportunity = board.opportunities.get(opportunityId);
  if (!opportunity) return null;

  if (opportunity.claimedBy?.labId !== labId) {
    return null;
  }

  const clampedProgress = Math.max(0, Math.min(100, progress));

  const updated: ResearchOpportunity = {
    ...opportunity,
    progress: clampedProgress,
    status: clampedProgress >= 100 ? "review" : opportunity.status,
    updatedAt: new Date().toISOString(),
  };

  board.opportunities.set(opportunityId, updated);
  return updated;
}

export function submitForReview(
  board: OpportunityBoard,
  opportunityId: string,
  labId: string
): ResearchOpportunity | null {
  const opportunity = board.opportunities.get(opportunityId);
  if (!opportunity) return null;

  if (
    opportunity.claimedBy?.labId !== labId ||
    opportunity.status !== "in-progress"
  ) {
    return null;
  }

  const updated: ResearchOpportunity = {
    ...opportunity,
    status: "review",
    progress: 100,
    updatedAt: new Date().toISOString(),
  };

  board.opportunities.set(opportunityId, updated);
  return updated;
}

export function completeOpportunity(
  board: OpportunityBoard,
  opportunityId: string
): ResearchOpportunity | null {
  const opportunity = board.opportunities.get(opportunityId);
  if (!opportunity) return null;

  if (opportunity.status !== "review") {
    return null;
  }

  const updated: ResearchOpportunity = {
    ...opportunity,
    status: "completed",
    progress: 100,
    updatedAt: new Date().toISOString(),
  };

  board.opportunities.set(opportunityId, updated);

  // Update leaderboard
  if (opportunity.claimedBy) {
    updateLeaderboardEntry(board, opportunity);
  }

  return updated;
}

export function expireOpportunity(
  board: OpportunityBoard,
  opportunityId: string
): ResearchOpportunity | null {
  const opportunity = board.opportunities.get(opportunityId);
  if (!opportunity) return null;

  const updated: ResearchOpportunity = {
    ...opportunity,
    status: "expired",
    updatedAt: new Date().toISOString(),
  };

  board.opportunities.set(opportunityId, updated);
  board.claims.delete(opportunityId);

  return updated;
}

// ============================================================================
// Filtering and Search
// ============================================================================

export function getOpportunities(
  board: OpportunityBoard,
  filters: FilterOptions = {},
  page: number = 0,
  pageSize: number = 20
): PaginatedResult<ResearchOpportunity> {
  let opportunities = Array.from(board.opportunities.values());

  // Apply filters
  if (filters.domain) {
    opportunities = opportunities.filter((o) => o.domain === filters.domain);
  }

  if (filters.difficulty) {
    opportunities = opportunities.filter(
      (o) => o.difficulty === filters.difficulty
    );
  }

  if (filters.impact) {
    opportunities = opportunities.filter((o) => o.impact === filters.impact);
  }

  if (filters.status) {
    opportunities = opportunities.filter((o) => o.status === filters.status);
  }

  if (filters.tags && filters.tags.length > 0) {
    opportunities = opportunities.filter((o) =>
      filters.tags!.some((tag) => o.tags.includes(tag))
    );
  }

  if (filters.dateRange) {
    const start = new Date(filters.dateRange.start);
    const end = new Date(filters.dateRange.end);
    opportunities = opportunities.filter((o) => {
      const created = new Date(o.createdAt);
      return created >= start && created <= end;
    });
  }

  // Sort by impact and creation date
  const impactOrder: Record<SignificanceLevel, number> = {
    critical: 0,
    high: 1,
    medium: 2,
    low: 3,
  };

  opportunities.sort((a, b) => {
    const impactDiff = impactOrder[a.impact] - impactOrder[b.impact];
    if (impactDiff !== 0) return impactDiff;
    return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
  });

  // Paginate
  const total = opportunities.length;
  const start = page * pageSize;
  const items = opportunities.slice(start, start + pageSize);

  return {
    items,
    total,
    page,
    pageSize,
    hasMore: start + pageSize < total,
  };
}

export function getOpenOpportunities(
  board: OpportunityBoard,
  limit: number = 20
): ResearchOpportunity[] {
  return getOpportunities(board, { status: "open" }, 0, limit).items;
}

export function getOpportunitiesByDomain(
  board: OpportunityBoard,
  domain: string
): ResearchOpportunity[] {
  return getOpportunities(board, { domain, status: "open" }, 0, 100).items;
}

export function getOpportunitiesWithBounty(
  board: OpportunityBoard
): ResearchOpportunity[] {
  return Array.from(board.opportunities.values())
    .filter((o) => o.hasBounty && o.status === "open")
    .sort((a, b) => (b.bounty?.amount || 0) - (a.bounty?.amount || 0));
}

export function getClaimedOpportunities(
  board: OpportunityBoard,
  labId: string
): ResearchOpportunity[] {
  return Array.from(board.opportunities.values()).filter(
    (o) =>
      o.claimedBy?.labId === labId &&
      (o.status === "claimed" || o.status === "in-progress")
  );
}

export function searchOpportunities(
  board: OpportunityBoard,
  query: string,
  limit: number = 20
): ResearchOpportunity[] {
  const lowerQuery = query.toLowerCase();

  return Array.from(board.opportunities.values())
    .filter(
      (o) =>
        o.title.toLowerCase().includes(lowerQuery) ||
        o.description.toLowerCase().includes(lowerQuery) ||
        o.tags.some((t) => t.toLowerCase().includes(lowerQuery)) ||
        o.domain.toLowerCase().includes(lowerQuery)
    )
    .slice(0, limit);
}

// ============================================================================
// Bounty Management
// ============================================================================

export function addBounty(
  board: OpportunityBoard,
  opportunityId: string,
  bounty: BountyInfo
): ResearchOpportunity | null {
  const opportunity = board.opportunities.get(opportunityId);
  if (!opportunity) return null;

  const updated: ResearchOpportunity = {
    ...opportunity,
    hasBounty: true,
    bounty,
    updatedAt: new Date().toISOString(),
  };

  board.opportunities.set(opportunityId, updated);
  board.bountyPool += bounty.amount;

  return updated;
}

export function removeBounty(
  board: OpportunityBoard,
  opportunityId: string
): ResearchOpportunity | null {
  const opportunity = board.opportunities.get(opportunityId);
  if (!opportunity || !opportunity.bounty) return null;

  board.bountyPool -= opportunity.bounty.amount;

  const updated: ResearchOpportunity = {
    ...opportunity,
    hasBounty: false,
    bounty: undefined,
    updatedAt: new Date().toISOString(),
  };

  board.opportunities.set(opportunityId, updated);
  return updated;
}

export function getTotalBountyPool(board: OpportunityBoard): number {
  return board.bountyPool;
}

// ============================================================================
// Leaderboard
// ============================================================================

function updateLeaderboardEntry(
  board: OpportunityBoard,
  opportunity: ResearchOpportunity
): void {
  if (!opportunity.claimedBy) return;

  const { labId, labName } = opportunity.claimedBy;
  const existingIndex = board.leaderboard.entries.findIndex(
    (e) => e.labId === labId
  );

  const impactScore: Record<SignificanceLevel, number> = {
    critical: 100,
    high: 50,
    medium: 25,
    low: 10,
  };

  if (existingIndex >= 0) {
    const existing = board.leaderboard.entries[existingIndex];
    board.leaderboard.entries[existingIndex] = {
      ...existing,
      opportunitiesCompleted: existing.opportunitiesCompleted + 1,
      totalBountyEarned:
        existing.totalBountyEarned + (opportunity.bounty?.amount || 0),
      impactScore: existing.impactScore + impactScore[opportunity.impact],
      streak: existing.streak + 1,
    };
  } else {
    board.leaderboard.entries.push({
      rank: 0,
      labId,
      labName,
      opportunitiesCompleted: 1,
      totalBountyEarned: opportunity.bounty?.amount || 0,
      impactScore: impactScore[opportunity.impact],
      streak: 1,
      badges: [],
    });
  }

  // Re-rank
  board.leaderboard.entries.sort((a, b) => b.impactScore - a.impactScore);
  board.leaderboard.entries.forEach((e, i) => {
    e.rank = i + 1;
  });

  // Award badges
  awardBadges(board);

  board.leaderboard.updatedAt = new Date().toISOString();
}

function awardBadges(board: OpportunityBoard): void {
  for (const entry of board.leaderboard.entries) {
    const badges: string[] = [];

    if (entry.opportunitiesCompleted >= 10) badges.push("veteran");
    if (entry.opportunitiesCompleted >= 50) badges.push("expert");
    if (entry.streak >= 5) badges.push("streak-master");
    if (entry.totalBountyEarned >= 1000) badges.push("bounty-hunter");
    if (entry.impactScore >= 500) badges.push("high-impact");
    if (entry.rank === 1) badges.push("top-contributor");

    entry.badges = badges;
  }
}

export function getLeaderboard(
  board: OpportunityBoard,
  period: "weekly" | "monthly" | "all-time" = "all-time",
  limit: number = 10
): OpportunityLeaderboard {
  // For simplicity, we return all-time leaderboard
  // In production, you'd filter by period
  return {
    entries: board.leaderboard.entries.slice(0, limit),
    period,
    updatedAt: board.leaderboard.updatedAt,
  };
}

export function getLabRank(
  board: OpportunityBoard,
  labId: string
): LeaderboardEntry | null {
  return board.leaderboard.entries.find((e) => e.labId === labId) || null;
}

// ============================================================================
// Statistics
// ============================================================================

export interface BoardStats {
  totalOpportunities: number;
  openOpportunities: number;
  claimedOpportunities: number;
  completedOpportunities: number;
  totalBountyPool: number;
  avgCompletionTime: number;
  byDomain: Record<string, number>;
  byDifficulty: Record<DifficultyLevel, number>;
}

export function getBoardStats(board: OpportunityBoard): BoardStats {
  const opportunities = Array.from(board.opportunities.values());

  const byDomain: Record<string, number> = {};
  const byDifficulty: Record<DifficultyLevel, number> = {
    beginner: 0,
    intermediate: 0,
    advanced: 0,
    expert: 0,
  };

  let openCount = 0;
  let claimedCount = 0;
  let completedCount = 0;
  let totalCompletionTime = 0;
  let completedWithTimeCount = 0;

  for (const opp of opportunities) {
    byDomain[opp.domain] = (byDomain[opp.domain] || 0) + 1;
    byDifficulty[opp.difficulty]++;

    if (opp.status === "open") openCount++;
    if (opp.status === "claimed" || opp.status === "in-progress") claimedCount++;
    if (opp.status === "completed") {
      completedCount++;
      if (opp.claimedBy) {
        const claimTime = new Date(opp.claimedBy.claimedAt).getTime();
        const completeTime = new Date(opp.updatedAt).getTime();
        totalCompletionTime += completeTime - claimTime;
        completedWithTimeCount++;
      }
    }
  }

  return {
    totalOpportunities: opportunities.length,
    openOpportunities: openCount,
    claimedOpportunities: claimedCount,
    completedOpportunities: completedCount,
    totalBountyPool: board.bountyPool,
    avgCompletionTime:
      completedWithTimeCount > 0
        ? totalCompletionTime / completedWithTimeCount / (1000 * 60 * 60 * 24) // Convert to days
        : 0,
    byDomain,
    byDifficulty,
  };
}

// ============================================================================
// Cleanup
// ============================================================================

export function cleanupExpiredClaims(
  board: OpportunityBoard,
  config: BoardConfig = DEFAULT_CONFIG
): number {
  const now = new Date();
  let count = 0;

  for (const [id, opportunity] of Array.from(board.opportunities.entries())) {
    if (opportunity.status !== "claimed" || !opportunity.claimedBy) continue;

    const claimDate = new Date(opportunity.claimedBy.claimedAt);
    const expirationDate = new Date(claimDate);
    expirationDate.setDate(
      expirationDate.getDate() + config.claimExpirationDays
    );

    if (now > expirationDate) {
      unclaimOpportunity(board, id, opportunity.claimedBy.labId);
      count++;
    }
  }

  return count;
}

// ============================================================================
// Export
// ============================================================================

export const boardConfig = DEFAULT_CONFIG;
