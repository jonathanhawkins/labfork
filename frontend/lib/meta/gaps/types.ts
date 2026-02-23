/**
 * Gap Analysis Types
 *
 * Types for the gap analysis system that identifies missing connections,
 * unexplored combinations, and research opportunities in the knowledge graph.
 */

import { TechniqueNode, TechniqueCategory } from "../knowledge-graph/types";

/**
 * A research gap representing missing or unexplored area
 */
export interface ResearchGap {
  /** Unique gap ID */
  id: string;
  /** Gap type */
  type: GapType;
  /** Human-readable title */
  title: string;
  /** Detailed description of the gap */
  description: string;
  /** Severity of the gap (how important to fill) */
  severity: GapSeverity;
  /** Domains affected by this gap */
  domains: string[];
  /** Related technique IDs */
  relatedTechniques: string[];
  /** Evidence supporting this gap */
  evidence: GapEvidence[];
  /** When the gap was detected */
  detectedAt: Date;
  /** Confidence in gap detection (0-1) */
  confidence: number;
  /** Optional suggested approaches to fill the gap */
  suggestedApproaches?: string[];
}

/**
 * Types of research gaps
 */
export type GapType =
  | "missing_connection" // Two related techniques not yet connected
  | "unexplored_combination" // Promising combination not explored
  | "missing_technique" // Gap in technique coverage
  | "underserved_domain" // Domain with insufficient research
  | "missing_baseline" // No baseline for comparison
  | "evaluation_gap" // Lack of proper evaluation
  | "theoretical_gap" // Missing theoretical foundation
  | "practical_gap"; // Missing practical implementation

/**
 * Severity levels for gaps
 */
export type GapSeverity =
  | "critical" // Must address immediately
  | "high" // Important to address soon
  | "medium" // Should address
  | "low" // Nice to address
  | "informational"; // Just for awareness

/**
 * Evidence supporting a gap
 */
export interface GapEvidence {
  /** Type of evidence */
  type: "missing_edge" | "low_coverage" | "no_papers" | "user_feedback" | "analysis";
  /** Description of the evidence */
  description: string;
  /** Confidence score for this evidence (0-1) */
  confidence: number;
  /** Source of evidence (paper ID, technique ID, etc.) */
  sourceId?: string;
}

/**
 * A research opportunity arising from a gap
 */
export interface GapOpportunity {
  /** Unique opportunity ID */
  id: string;
  /** Parent gap ID */
  gapId: string;
  /** Opportunity title */
  title: string;
  /** Detailed description */
  description: string;
  /** Opportunity type */
  type: OpportunityType;
  /** Impact if pursued (0-1) */
  impactScore: number;
  /** Effort required to pursue */
  effort: EffortEstimate;
  /** Prerequisites */
  prerequisites: string[];
  /** Potential outcomes */
  potentialOutcomes: string[];
  /** Techniques to combine/use */
  suggestedTechniques: string[];
  /** Priority score (impact / effort) */
  priorityScore: number;
  /** When the opportunity was identified */
  identifiedAt: Date;
  /** Confidence in this opportunity (0-1) */
  confidence: number;
}

/**
 * Types of opportunities
 */
export type OpportunityType =
  | "new_technique" // Create a new technique
  | "technique_combination" // Combine existing techniques
  | "domain_transfer" // Transfer technique to new domain
  | "improvement" // Improve existing technique
  | "evaluation" // Create evaluation framework
  | "implementation" // Implement theoretical technique
  | "benchmark"; // Create benchmark dataset

/**
 * Effort estimate for pursuing an opportunity
 */
export interface EffortEstimate {
  /** Effort level */
  level: EffortLevel;
  /** Estimated person-weeks */
  personWeeks: number;
  /** Required skills */
  skills: string[];
  /** Required resources */
  resources: string[];
  /** Risk factors */
  risks: string[];
}

/**
 * Effort levels
 */
export type EffortLevel =
  | "trivial" // Less than 1 week
  | "small" // 1-2 weeks
  | "medium" // 2-4 weeks
  | "large" // 1-3 months
  | "research"; // 3+ months, significant research

/**
 * Landscape node for visualization
 */
export interface LandscapeNode {
  /** Node ID */
  id: string;
  /** Node label */
  label: string;
  /** Node type */
  type: "technique" | "domain" | "gap" | "opportunity";
  /** X position (0-1 normalized) */
  x: number;
  /** Y position (0-1 normalized) */
  y: number;
  /** Node size (based on importance) */
  size: number;
  /** Node color */
  color: string;
  /** Cluster ID */
  clusterId?: string;
  /** Metadata */
  metadata: Record<string, unknown>;
}

/**
 * Landscape edge for visualization
 */
export interface LandscapeEdge {
  /** Edge ID */
  id: string;
  /** Source node ID */
  source: string;
  /** Target node ID */
  target: string;
  /** Edge weight (0-1) */
  weight: number;
  /** Edge type */
  type: "connection" | "gap" | "opportunity";
  /** Is this edge dashed (represents gap) */
  dashed: boolean;
  /** Edge color */
  color: string;
}

/**
 * Research landscape for visualization
 */
export interface ResearchLandscape {
  /** Unique landscape ID */
  id: string;
  /** Domain being analyzed */
  domain: string;
  /** Nodes in the landscape */
  nodes: LandscapeNode[];
  /** Edges in the landscape */
  edges: LandscapeEdge[];
  /** Clusters detected */
  clusters: LandscapeCluster[];
  /** Gaps identified */
  gaps: ResearchGap[];
  /** Opportunities identified */
  opportunities: GapOpportunity[];
  /** Generation timestamp */
  generatedAt: Date;
  /** Coverage score (0-1) */
  coverageScore: number;
  /** Density score (0-1) */
  densityScore: number;
}

/**
 * Cluster in the landscape
 */
export interface LandscapeCluster {
  /** Cluster ID */
  id: string;
  /** Cluster label */
  label: string;
  /** Node IDs in this cluster */
  nodeIds: string[];
  /** Centroid position */
  centroid: { x: number; y: number };
  /** Cluster size */
  size: number;
  /** Dominant category */
  dominantCategory?: TechniqueCategory;
}

/**
 * Gap report summarizing analysis
 */
export interface GapReport {
  /** Report ID */
  id: string;
  /** Generation timestamp */
  generatedAt: Date;
  /** Domain analyzed (or "all") */
  domain: string;
  /** Total techniques analyzed */
  techniquesAnalyzed: number;
  /** Total gaps found */
  totalGaps: number;
  /** Gaps by type */
  gapsByType: Record<GapType, number>;
  /** Gaps by severity */
  gapsBySeverity: Record<GapSeverity, number>;
  /** Total opportunities */
  totalOpportunities: number;
  /** Top opportunities (sorted by priority) */
  topOpportunities: GapOpportunity[];
  /** Coverage metrics */
  coverage: CoverageMetrics;
  /** Recommendations */
  recommendations: string[];
}

/**
 * Coverage metrics for gap analysis
 */
export interface CoverageMetrics {
  /** Domain coverage (0-1) */
  domainCoverage: number;
  /** Category coverage (0-1) */
  categoryCoverage: number;
  /** Connection density (0-1) */
  connectionDensity: number;
  /** Average techniques per domain */
  avgTechniquesPerDomain: number;
  /** Domains with gaps */
  domainsWithGaps: string[];
  /** Categories with gaps */
  categoriesWithGaps: TechniqueCategory[];
}

/**
 * Configuration for gap analysis
 */
export interface GapAnalysisConfig {
  /** Minimum connection count to not be a gap */
  minConnections: number;
  /** Minimum techniques per domain */
  minTechniquesPerDomain: number;
  /** Maximum gap age to consider (days) */
  maxGapAgeDays: number;
  /** Minimum confidence to report */
  minConfidence: number;
  /** Domains to analyze (empty = all) */
  domains: string[];
  /** Categories to analyze (empty = all) */
  categories: TechniqueCategory[];
  /** Include severity levels */
  severityFilter: GapSeverity[];
  /** Maximum gaps to return */
  maxGaps: number;
  /** Maximum opportunities to return */
  maxOpportunities: number;
}

/**
 * Default gap analysis configuration
 */
export const DEFAULT_GAP_CONFIG: GapAnalysisConfig = {
  minConnections: 2,
  minTechniquesPerDomain: 3,
  maxGapAgeDays: 365,
  minConfidence: 0.5,
  domains: [],
  categories: [],
  severityFilter: ["critical", "high", "medium"],
  maxGaps: 50,
  maxOpportunities: 20,
};

/**
 * Weekly gap summary
 */
export interface WeeklyGapSummary {
  /** Week start date */
  weekStart: Date;
  /** Week end date */
  weekEnd: Date;
  /** New gaps detected */
  newGaps: number;
  /** Gaps resolved */
  resolvedGaps: number;
  /** New opportunities */
  newOpportunities: number;
  /** Top gaps for the week */
  topGaps: ResearchGap[];
  /** Top opportunities for the week */
  topOpportunities: GapOpportunity[];
  /** Overall health score (0-1) */
  healthScore: number;
  /** Trend compared to last week */
  trend: "improving" | "stable" | "declining";
}

// ============================================================================
// Type Guards
// ============================================================================

export function isGapType(value: string): value is GapType {
  return [
    "missing_connection",
    "unexplored_combination",
    "missing_technique",
    "underserved_domain",
    "missing_baseline",
    "evaluation_gap",
    "theoretical_gap",
    "practical_gap",
  ].includes(value);
}

export function isGapSeverity(value: string): value is GapSeverity {
  return ["critical", "high", "medium", "low", "informational"].includes(value);
}

export function isOpportunityType(value: string): value is OpportunityType {
  return [
    "new_technique",
    "technique_combination",
    "domain_transfer",
    "improvement",
    "evaluation",
    "implementation",
    "benchmark",
  ].includes(value);
}

export function isEffortLevel(value: string): value is EffortLevel {
  return ["trivial", "small", "medium", "large", "research"].includes(value);
}

export function isResearchGap(obj: unknown): obj is ResearchGap {
  return (
    typeof obj === "object" &&
    obj !== null &&
    "id" in obj &&
    "type" in obj &&
    "title" in obj &&
    "severity" in obj
  );
}

export function isGapOpportunity(obj: unknown): obj is GapOpportunity {
  return (
    typeof obj === "object" &&
    obj !== null &&
    "id" in obj &&
    "gapId" in obj &&
    "type" in obj &&
    "impactScore" in obj
  );
}

// ============================================================================
// Factory Functions
// ============================================================================

export function createGapId(): string {
  return `gap-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createOpportunityId(): string {
  return `opp-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createLandscapeId(): string {
  return `landscape-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createReportId(): string {
  return `gap-report-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * Create default effort estimate
 */
export function createDefaultEffort(level: EffortLevel = "medium"): EffortEstimate {
  const weeksByLevel: Record<EffortLevel, number> = {
    trivial: 0.5,
    small: 1.5,
    medium: 3,
    large: 8,
    research: 16,
  };

  return {
    level,
    personWeeks: weeksByLevel[level],
    skills: [],
    resources: [],
    risks: [],
  };
}

/**
 * Calculate priority score from impact and effort
 */
export function calculatePriorityScore(
  impactScore: number,
  effort: EffortEstimate
): number {
  const effortWeight: Record<EffortLevel, number> = {
    trivial: 1.0,
    small: 0.8,
    medium: 0.6,
    large: 0.4,
    research: 0.2,
  };

  return impactScore * effortWeight[effort.level];
}

// ============================================================================
// Display Helpers
// ============================================================================

export function getGapTypeLabel(type: GapType): string {
  const labels: Record<GapType, string> = {
    missing_connection: "Missing Connection",
    unexplored_combination: "Unexplored Combination",
    missing_technique: "Missing Technique",
    underserved_domain: "Underserved Domain",
    missing_baseline: "Missing Baseline",
    evaluation_gap: "Evaluation Gap",
    theoretical_gap: "Theoretical Gap",
    practical_gap: "Practical Gap",
  };
  return labels[type];
}

export function getGapSeverityColor(severity: GapSeverity): string {
  const colors: Record<GapSeverity, string> = {
    critical: "#ef4444", // red
    high: "#f97316", // orange
    medium: "#eab308", // yellow
    low: "#22c55e", // green
    informational: "#6b7280", // gray
  };
  return colors[severity];
}

export function getOpportunityTypeIcon(type: OpportunityType): string {
  const icons: Record<OpportunityType, string> = {
    new_technique: "sparkles",
    technique_combination: "merge",
    domain_transfer: "arrow-right-circle",
    improvement: "trending-up",
    evaluation: "clipboard-check",
    implementation: "code",
    benchmark: "bar-chart",
  };
  return icons[type];
}

export function getEffortLevelLabel(level: EffortLevel): string {
  const labels: Record<EffortLevel, string> = {
    trivial: "Trivial (< 1 week)",
    small: "Small (1-2 weeks)",
    medium: "Medium (2-4 weeks)",
    large: "Large (1-3 months)",
    research: "Research (3+ months)",
  };
  return labels[level];
}
