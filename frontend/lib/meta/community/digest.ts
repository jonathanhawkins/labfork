/**
 * Weekly Digest Generator
 *
 * Aggregates all activity across labs, identifies breakthroughs,
 * highlights trends, and generates newsletter-style digests.
 */

import {
  WeeklyDigest,
  DigestStatus,
  DigestSummary,
  Breakthrough,
  TrendingTechnique,
  SynergyHighlight,
  OpportunityHighlight,
  EvolutionHighlight,
  CollaborationUpdate,
  ContributorHighlight,
  WeeklyStats,
  SignificanceLevel,
} from "./types";

// ============================================================================
// Digest Generator Interface
// ============================================================================

export interface DigestGenerator {
  digests: Map<string, WeeklyDigest>;
  currentWeek: WeeklyDigest | null;
}

export interface DigestConfig {
  maxBreakthroughs: number;
  maxTrendingTechniques: number;
  maxSynergies: number;
  maxOpportunities: number;
  maxEvolutionHighlights: number;
  maxCollaborationUpdates: number;
  maxContributors: number;
  publishDayOfWeek: number; // 0 = Sunday
  publishHour: number;
}

const DEFAULT_CONFIG: DigestConfig = {
  maxBreakthroughs: 5,
  maxTrendingTechniques: 10,
  maxSynergies: 8,
  maxOpportunities: 6,
  maxEvolutionHighlights: 5,
  maxCollaborationUpdates: 5,
  maxContributors: 10,
  publishDayOfWeek: 0, // Sunday
  publishHour: 10, // 10 AM
};

// ============================================================================
// Factory Functions
// ============================================================================

export function createDigestGenerator(): DigestGenerator {
  return {
    digests: new Map(),
    currentWeek: null,
  };
}

export function getWeekNumber(date: Date): { week: number; year: number } {
  const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
  const dayNum = d.getUTCDay() || 7;
  d.setUTCDate(d.getUTCDate() + 4 - dayNum);
  const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
  const week = Math.ceil(((d.getTime() - yearStart.getTime()) / 86400000 + 1) / 7);
  return { week, year: d.getUTCFullYear() };
}

export function getWeekDateRange(week: number, year: number): { start: Date; end: Date } {
  const simple = new Date(year, 0, 1 + (week - 1) * 7);
  const dow = simple.getDay();
  const start = new Date(simple);
  if (dow <= 4) {
    start.setDate(simple.getDate() - dow + 1);
  } else {
    start.setDate(simple.getDate() + 8 - dow);
  }
  const end = new Date(start);
  end.setDate(start.getDate() + 6);
  end.setHours(23, 59, 59, 999);
  return { start, end };
}

// ============================================================================
// Digest Generation
// ============================================================================

export function generateDigest(
  generator: DigestGenerator,
  data: DigestSourceData,
  config: DigestConfig = DEFAULT_CONFIG
): WeeklyDigest {
  const now = new Date();
  const { week, year } = getWeekNumber(now);
  const { start, end } = getWeekDateRange(week, year);

  const id = `digest-${year}-W${week.toString().padStart(2, "0")}`;

  // Check if digest already exists
  const existing = generator.digests.get(id);
  if (existing && existing.status === "published") {
    return existing;
  }

  // Generate digest sections
  const breakthroughs = identifyBreakthroughs(data, config.maxBreakthroughs);
  const trendingTechniques = identifyTrendingTechniques(data, config.maxTrendingTechniques);
  const newSynergies = identifySynergies(data, config.maxSynergies);
  const researchOpportunities = identifyOpportunities(data, config.maxOpportunities);
  const evolutionHighlights = identifyEvolutionHighlights(data, config.maxEvolutionHighlights);
  const collaborationUpdates = getCollaborationUpdates(data, config.maxCollaborationUpdates);
  const topContributors = identifyTopContributors(data, config.maxContributors);
  const stats = aggregateWeeklyStats(data);

  // Generate summary
  const summary = generateSummary(
    breakthroughs,
    trendingTechniques,
    newSynergies,
    stats
  );

  const digest: WeeklyDigest = {
    id,
    weekNumber: week,
    year,
    startDate: start.toISOString(),
    endDate: end.toISOString(),
    generatedAt: now.toISOString(),
    status: "draft",
    summary,
    breakthroughs,
    trendingTechniques,
    newSynergies,
    researchOpportunities,
    evolutionHighlights,
    collaborationUpdates,
    topContributors,
    stats,
  };

  generator.digests.set(id, digest);
  generator.currentWeek = digest;

  return digest;
}

export interface DigestSourceData {
  discoveries: DiscoveryData[];
  techniques: TechniqueData[];
  synergies: SynergyData[];
  gaps: GapData[];
  evolutions: EvolutionData[];
  collaborations: CollaborationData[];
  labs: LabData[];
  papers: PaperData[];
}

export interface DiscoveryData {
  id: string;
  type: string;
  title: string;
  description: string;
  significance: SignificanceLevel;
  labId: string;
  labName: string;
  techniqueIds: string[];
  metrics: {
    improvementPercent?: number;
    affectedDomains?: string[];
    potentialApplications?: number;
  };
  timestamp: string;
}

export interface TechniqueData {
  id: string;
  name: string;
  category: string;
  usageCount: number;
  previousUsageCount: number;
  adoptedByLabs: string[];
  domains: string[];
  createdAt: string;
}

export interface SynergyData {
  id: string;
  techniques: string[];
  score: number;
  discoveredBy: string;
  labId: string;
  description: string;
  impact: SignificanceLevel;
  timestamp: string;
}

export interface GapData {
  id: string;
  title: string;
  type: string;
  difficulty: string;
  impact: SignificanceLevel;
  domain: string;
  status: string;
  claimedBy?: string;
  bountyAmount?: number;
}

export interface EvolutionData {
  id: string;
  techniqueId: string;
  techniqueName: string;
  generation: number;
  fitnessScore: number;
  previousFitness: number;
  parentTechniques: string[];
  newCapabilities: string[];
  timestamp: string;
}

export interface CollaborationData {
  id: string;
  title: string;
  status: string;
  participantCount: number;
  recentProgress: string;
  completedObjectives: number;
  totalObjectives: number;
}

export interface LabData {
  id: string;
  name: string;
  discoveries: number;
  contributions: number;
  collaborations: number;
  isActive: boolean;
}

export interface PaperData {
  id: string;
  status: string;
  source: string;
  processedAt?: string;
}

// ============================================================================
// Section Generators
// ============================================================================

function identifyBreakthroughs(
  data: DigestSourceData,
  max: number
): Breakthrough[] {
  return data.discoveries
    .filter((d) => d.significance === "critical" || d.significance === "high")
    .sort((a, b) => {
      const sigOrder = { critical: 0, high: 1, medium: 2, low: 3 };
      return sigOrder[a.significance] - sigOrder[b.significance];
    })
    .slice(0, max)
    .map((d) => ({
      id: d.id,
      title: d.title,
      description: d.description,
      significance: d.significance,
      discoveredBy: d.labName,
      labId: d.labId,
      labName: d.labName,
      techniqueIds: d.techniqueIds,
      metrics: {
        improvementPercent: d.metrics.improvementPercent || 0,
        affectedDomains: d.metrics.affectedDomains || [],
        potentialApplications: d.metrics.potentialApplications || 0,
        communityInterest: 0,
      },
      timestamp: d.timestamp,
    }));
}

function identifyTrendingTechniques(
  data: DigestSourceData,
  max: number
): TrendingTechnique[] {
  return data.techniques
    .map((t) => ({
      ...t,
      weeklyGrowth:
        t.previousUsageCount > 0
          ? ((t.usageCount - t.previousUsageCount) / t.previousUsageCount) * 100
          : t.usageCount > 0
          ? 100
          : 0,
      trendScore: calculateTrendScore(t),
    }))
    .sort((a, b) => b.trendScore - a.trendScore)
    .slice(0, max)
    .map((t) => ({
      id: t.id,
      name: t.name,
      category: t.category,
      trendScore: t.trendScore,
      adoptionRate: t.adoptedByLabs.length / Math.max(data.labs.length, 1),
      weeklyGrowth: t.weeklyGrowth,
      usageCount: t.usageCount,
      topLabs: t.adoptedByLabs.slice(0, 5),
      domains: t.domains,
    }));
}

function calculateTrendScore(technique: TechniqueData): number {
  const growth =
    technique.previousUsageCount > 0
      ? (technique.usageCount - technique.previousUsageCount) / technique.previousUsageCount
      : technique.usageCount > 0
      ? 1
      : 0;

  const adoption = technique.adoptedByLabs.length;
  const recency = Math.max(
    0,
    1 - (Date.now() - new Date(technique.createdAt).getTime()) / (30 * 24 * 60 * 60 * 1000)
  );

  return growth * 40 + adoption * 10 + recency * 20 + technique.usageCount * 0.5;
}

function identifySynergies(
  data: DigestSourceData,
  max: number
): SynergyHighlight[] {
  return data.synergies
    .sort((a, b) => b.score - a.score)
    .slice(0, max)
    .map((s) => ({
      id: s.id,
      techniques: s.techniques,
      synergyScore: s.score,
      discoveredBy: s.discoveredBy,
      labId: s.labId,
      description: s.description,
      potentialImpact: s.impact,
      timestamp: s.timestamp,
    }));
}

function identifyOpportunities(
  data: DigestSourceData,
  max: number
): OpportunityHighlight[] {
  return data.gaps
    .filter((g) => g.status === "open")
    .sort((a, b) => {
      const impactOrder = { critical: 0, high: 1, medium: 2, low: 3 };
      return impactOrder[a.impact] - impactOrder[b.impact];
    })
    .slice(0, max)
    .map((g) => ({
      id: g.id,
      title: g.title,
      gapType: g.type as OpportunityHighlight["gapType"],
      difficulty: g.difficulty as OpportunityHighlight["difficulty"],
      impact: g.impact,
      domain: g.domain,
      claimedBy: g.claimedBy,
      bountyAmount: g.bountyAmount,
    }));
}

function identifyEvolutionHighlights(
  data: DigestSourceData,
  max: number
): EvolutionHighlight[] {
  return data.evolutions
    .map((e) => ({
      ...e,
      fitnessImprovement:
        e.previousFitness > 0
          ? ((e.fitnessScore - e.previousFitness) / e.previousFitness) * 100
          : 0,
    }))
    .sort((a, b) => b.fitnessImprovement - a.fitnessImprovement)
    .slice(0, max)
    .map((e) => ({
      id: e.id,
      techniqueId: e.techniqueId,
      techniqueName: e.techniqueName,
      generation: e.generation,
      fitnessImprovement: e.fitnessImprovement,
      parentTechniques: e.parentTechniques,
      newCapabilities: e.newCapabilities,
      timestamp: e.timestamp,
    }));
}

function getCollaborationUpdates(
  data: DigestSourceData,
  max: number
): CollaborationUpdate[] {
  return data.collaborations
    .filter((c) => c.status === "active" || c.status === "completed")
    .sort((a, b) => b.completedObjectives - a.completedObjectives)
    .slice(0, max)
    .map((c) => ({
      id: c.id,
      metaTaskId: c.id,
      title: c.title,
      status: c.status,
      participantCount: c.participantCount,
      recentProgress: c.recentProgress,
      completedObjectives: c.completedObjectives,
      totalObjectives: c.totalObjectives,
    }));
}

function identifyTopContributors(
  data: DigestSourceData,
  max: number
): ContributorHighlight[] {
  return data.labs
    .filter((l) => l.isActive)
    .map((l) => ({
      labId: l.id,
      labName: l.name,
      contributions: l.contributions,
      discoveries: l.discoveries,
      collaborations: l.collaborations,
      weeklyScore: l.discoveries * 10 + l.contributions * 5 + l.collaborations * 3,
      rank: 0,
    }))
    .sort((a, b) => b.weeklyScore - a.weeklyScore)
    .slice(0, max)
    .map((c, index) => ({ ...c, rank: index + 1 }));
}

function aggregateWeeklyStats(data: DigestSourceData): WeeklyStats {
  const activeLabs = data.labs.filter((l) => l.isActive).length;
  const newPapers = data.papers.filter((p) => p.status === "processed").length;

  // Count by domain
  const domainCounts: Record<string, number> = {};
  data.techniques.forEach((t) => {
    t.domains.forEach((d) => {
      domainCounts[d] = (domainCounts[d] || 0) + 1;
    });
  });
  const topDomain = Object.entries(domainCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || "unknown";

  return {
    totalPapers: data.papers.length,
    newPapers,
    totalTechniques: data.techniques.length,
    newTechniques: data.techniques.filter(
      (t) => Date.now() - new Date(t.createdAt).getTime() < 7 * 24 * 60 * 60 * 1000
    ).length,
    totalLabs: data.labs.length,
    activeLabs,
    synergiesDiscovered: data.synergies.length,
    gapsFilled: data.gaps.filter((g) => g.status === "completed").length,
    collaborationsStarted: data.collaborations.filter((c) => c.status === "active").length,
    collaborationsCompleted: data.collaborations.filter((c) => c.status === "completed").length,
    evolutionGenerations: Math.max(...data.evolutions.map((e) => e.generation), 0),
    topDomain,
  };
}

function generateSummary(
  breakthroughs: Breakthrough[],
  trending: TrendingTechnique[],
  synergies: SynergyHighlight[],
  stats: WeeklyStats
): DigestSummary {
  const highlights: string[] = [];

  if (breakthroughs.length > 0) {
    highlights.push(
      `${breakthroughs.length} breakthrough${breakthroughs.length > 1 ? "s" : ""} discovered`
    );
  }

  if (trending.length > 0) {
    const topTrend = trending[0];
    highlights.push(`${topTrend.name} is trending with ${topTrend.weeklyGrowth.toFixed(0)}% growth`);
  }

  if (synergies.length > 0) {
    highlights.push(`${synergies.length} new synergies identified`);
  }

  if (stats.collaborationsStarted > 0) {
    highlights.push(`${stats.collaborationsStarted} new collaborations launched`);
  }

  if (stats.gapsFilled > 0) {
    highlights.push(`${stats.gapsFilled} research gaps filled`);
  }

  const headline = breakthroughs.length > 0
    ? `Week of Breakthroughs: ${breakthroughs[0].title}`
    : trending.length > 0
    ? `${trending[0].name} Leads This Week's Trends`
    : `Community Update: ${stats.activeLabs} Labs Active`;

  const wordCount = 500 + breakthroughs.length * 100 + trending.length * 50;

  return {
    headline,
    keyHighlights: highlights,
    wordCount,
    readTimeMinutes: Math.ceil(wordCount / 200),
  };
}

// ============================================================================
// Digest Management
// ============================================================================

export function publishDigest(
  generator: DigestGenerator,
  digestId: string
): WeeklyDigest | null {
  const digest = generator.digests.get(digestId);
  if (!digest) return null;

  const published: WeeklyDigest = {
    ...digest,
    status: "published",
    publishedAt: new Date().toISOString(),
  };

  generator.digests.set(digestId, published);
  return published;
}

export function scheduleDigest(
  generator: DigestGenerator,
  digestId: string,
  publishAt: Date
): WeeklyDigest | null {
  const digest = generator.digests.get(digestId);
  if (!digest) return null;

  const scheduled: WeeklyDigest = {
    ...digest,
    status: "scheduled",
  };

  generator.digests.set(digestId, scheduled);
  return scheduled;
}

export function archiveDigest(
  generator: DigestGenerator,
  digestId: string
): WeeklyDigest | null {
  const digest = generator.digests.get(digestId);
  if (!digest) return null;

  const archived: WeeklyDigest = {
    ...digest,
    status: "archived",
  };

  generator.digests.set(digestId, archived);
  return archived;
}

export function getDigest(
  generator: DigestGenerator,
  digestId: string
): WeeklyDigest | null {
  return generator.digests.get(digestId) || null;
}

export function getLatestDigest(generator: DigestGenerator): WeeklyDigest | null {
  const published = Array.from(generator.digests.values())
    .filter((d) => d.status === "published")
    .sort((a, b) => new Date(b.publishedAt!).getTime() - new Date(a.publishedAt!).getTime());

  return published[0] || generator.currentWeek;
}

export function getDigestArchive(
  generator: DigestGenerator,
  limit: number = 10
): WeeklyDigest[] {
  return Array.from(generator.digests.values())
    .filter((d) => d.status === "published" || d.status === "archived")
    .sort((a, b) => b.year * 100 + b.weekNumber - (a.year * 100 + a.weekNumber))
    .slice(0, limit);
}

export function shouldAutoPublish(config: DigestConfig = DEFAULT_CONFIG): boolean {
  const now = new Date();
  return now.getDay() === config.publishDayOfWeek && now.getHours() === config.publishHour;
}

// ============================================================================
// Export
// ============================================================================

export const digestGeneratorConfig = DEFAULT_CONFIG;
