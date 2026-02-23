/**
 * Pattern Recognition Types
 *
 * Types for the pattern recognition system that detects research trends
 * and recurring patterns across techniques and papers.
 */

/**
 * A detected research trend
 */
export interface ResearchTrend {
  /** Unique trend ID */
  id: string;
  /** Trend name/title */
  name: string;
  /** Trend description */
  description: string;
  /** Category of the trend */
  category: TrendCategory;
  /** Keywords associated with this trend */
  keywords: string[];
  /** Domains affected by this trend */
  domains: string[];
  /** Trend strength (0-1) */
  strength: number;
  /** Trend momentum (positive = growing, negative = declining) */
  momentum: number;
  /** First detected timestamp */
  firstDetected: Date;
  /** Last updated timestamp */
  lastUpdated: Date;
  /** Time series data points */
  timeSeries: TrendDataPoint[];
  /** Related technique IDs */
  relatedTechniques: string[];
  /** Related paper IDs */
  relatedPapers: string[];
  /** Confidence score (0-1) */
  confidence: number;
}

/**
 * Categories of research trends
 */
export type TrendCategory =
  | "architecture" // New architectural patterns
  | "training" // Training methodology trends
  | "application" // Application domain trends
  | "efficiency" // Efficiency/optimization trends
  | "quality" // Quality improvement trends
  | "data" // Data-related trends
  | "hardware" // Hardware/deployment trends
  | "theoretical"; // Theoretical advances

/**
 * Time series data point for trend tracking
 */
export interface TrendDataPoint {
  /** Timestamp */
  timestamp: Date;
  /** Value (e.g., paper count, adoption rate) */
  value: number;
  /** Optional breakdown by source */
  breakdown?: Record<string, number>;
}

/**
 * Recurring architectural pattern
 */
export interface ArchitecturePattern {
  /** Pattern ID */
  id: string;
  /** Pattern name */
  name: string;
  /** Pattern description */
  description: string;
  /** Component types in this pattern */
  components: PatternComponent[];
  /** How components connect */
  connections: PatternConnection[];
  /** Frequency of occurrence */
  frequency: number;
  /** Domains where this pattern appears */
  domains: string[];
  /** Example techniques using this pattern */
  examples: string[];
  /** When first detected */
  firstDetected: Date;
  /** Confidence score */
  confidence: number;
}

/**
 * Component in an architecture pattern
 */
export interface PatternComponent {
  /** Component type */
  type: string;
  /** Role in the pattern */
  role: string;
  /** Optional variants */
  variants?: string[];
  /** Is this component required or optional */
  required: boolean;
}

/**
 * Connection between pattern components
 */
export interface PatternConnection {
  /** Source component type */
  from: string;
  /** Target component type */
  to: string;
  /** Connection type */
  type: "sequential" | "parallel" | "residual" | "attention" | "conditioning";
}

/**
 * Technique adoption metrics
 */
export interface TechniqueAdoption {
  /** Technique ID */
  techniqueId: string;
  /** Technique name */
  techniqueName: string;
  /** Current adoption score (0-1) */
  adoptionScore: number;
  /** Adoption trend (positive = increasing) */
  adoptionTrend: number;
  /** Time to mainstream adoption (estimated weeks) */
  timeToMainstream?: number;
  /** Adoption stage */
  stage: AdoptionStage;
  /** Adoption time series */
  timeSeries: AdoptionDataPoint[];
  /** Papers citing this technique */
  citationCount: number;
  /** Implementations/repos using this technique */
  implementationCount: number;
}

/**
 * Adoption stages
 */
export type AdoptionStage =
  | "emerging" // Just appeared, few adopters
  | "growing" // Gaining traction
  | "mainstream" // Widely adopted
  | "mature" // Fully established
  | "declining"; // Being replaced

/**
 * Adoption data point
 */
export interface AdoptionDataPoint {
  /** Timestamp */
  timestamp: Date;
  /** Adoption score at this time */
  adoptionScore: number;
  /** New citations in this period */
  newCitations: number;
  /** New implementations in this period */
  newImplementations: number;
}

/**
 * Cross-domain transfer pattern
 */
export interface CrossDomainTransfer {
  /** Transfer ID */
  id: string;
  /** Technique being transferred */
  techniqueId: string;
  /** Technique name */
  techniqueName: string;
  /** Original domain */
  sourceDomain: string;
  /** Target domain */
  targetDomain: string;
  /** Success of the transfer (0-1) */
  successScore: number;
  /** Adaptations made for the transfer */
  adaptations: string[];
  /** When the transfer was detected */
  detectedAt: Date;
  /** Papers demonstrating this transfer */
  evidencePapers: string[];
  /** Confidence in this transfer pattern */
  confidence: number;
}

/**
 * Pattern recognition configuration
 */
export interface PatternRecognitionConfig {
  /** Minimum occurrences to detect a pattern */
  minOccurrences: number;
  /** Time window for trend detection (days) */
  trendWindowDays: number;
  /** Minimum confidence to report */
  minConfidence: number;
  /** Domains to focus on (empty = all) */
  focusDomains: string[];
  /** Enable cross-domain pattern detection */
  detectCrossDomain: boolean;
  /** Enable architecture pattern detection */
  detectArchitecturePatterns: boolean;
  /** Enable adoption tracking */
  trackAdoption: boolean;
}

/**
 * Default configuration
 */
export const DEFAULT_PATTERN_CONFIG: PatternRecognitionConfig = {
  minOccurrences: 3,
  trendWindowDays: 90,
  minConfidence: 0.5,
  focusDomains: [],
  detectCrossDomain: true,
  detectArchitecturePatterns: true,
  trackAdoption: true,
};

/**
 * Comprehensive pattern report
 */
export interface PatternReport {
  /** Report ID */
  id: string;
  /** Generation timestamp */
  generatedAt: Date;
  /** Report period start */
  periodStart: Date;
  /** Report period end */
  periodEnd: Date;
  /** Active trends */
  trends: ResearchTrend[];
  /** Emerging patterns */
  emergingPatterns: ArchitecturePattern[];
  /** Adoption metrics */
  adoptionMetrics: TechniqueAdoption[];
  /** Cross-domain transfers */
  crossDomainTransfers: CrossDomainTransfer[];
  /** Summary statistics */
  summary: ReportSummary;
}

/**
 * Report summary statistics
 */
export interface ReportSummary {
  /** Total techniques analyzed */
  techniquesAnalyzed: number;
  /** Total papers analyzed */
  papersAnalyzed: number;
  /** New trends detected */
  newTrends: number;
  /** Growing trends */
  growingTrends: number;
  /** Declining trends */
  decliningTrends: number;
  /** Patterns detected */
  patternsDetected: number;
  /** Cross-domain transfers detected */
  transfersDetected: number;
  /** Top trending keywords */
  topKeywords: Array<{ keyword: string; count: number }>;
  /** Most active domains */
  activeDomains: Array<{ domain: string; activity: number }>;
}

// Type guards
export function isResearchTrend(obj: unknown): obj is ResearchTrend {
  return (
    typeof obj === "object" &&
    obj !== null &&
    "id" in obj &&
    "name" in obj &&
    "category" in obj &&
    "strength" in obj
  );
}

export function isTrendCategory(value: string): value is TrendCategory {
  return [
    "architecture",
    "training",
    "application",
    "efficiency",
    "quality",
    "data",
    "hardware",
    "theoretical",
  ].includes(value);
}

export function isAdoptionStage(value: string): value is AdoptionStage {
  return ["emerging", "growing", "mainstream", "mature", "declining"].includes(
    value
  );
}

// Factory functions
export function createTrendId(): string {
  return `trend-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createPatternId(): string {
  return `pattern-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createTransferId(): string {
  return `transfer-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createReportId(): string {
  return `report-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}
