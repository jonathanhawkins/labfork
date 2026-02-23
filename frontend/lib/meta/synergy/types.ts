/**
 * Synergy Discovery Types
 *
 * Types for the synergy discovery system that identifies promising
 * combinations of research techniques.
 */

import { TechniqueNode } from "../knowledge-graph/types";

/**
 * Synergy score components
 */
export interface SynergyScoreComponents {
  /** Similarity score (0-1) - how related the techniques are */
  similarity: number;
  /** Complementarity score (0-1) - how well they fill each other's gaps */
  complementarity: number;
  /** Novelty score (0-1) - how unexplored this combination is */
  novelty: number;
  /** Feasibility score (0-1) - how practical to implement */
  feasibility: number;
  /** Impact score (0-1) - expected improvement potential */
  impact: number;
}

/**
 * Overall synergy score with breakdown
 */
export interface SynergyScore {
  /** Overall synergy score (0-1) */
  overall: number;
  /** Component breakdown */
  components: SynergyScoreComponents;
  /** Confidence in the score (0-1) */
  confidence: number;
}

/**
 * A proposed combination of techniques
 */
export interface SynergyProposal {
  /** Unique proposal ID */
  id: string;
  /** First technique in the combination */
  techniqueA: TechniqueNode;
  /** Second technique in the combination */
  techniqueB: TechniqueNode;
  /** Synergy score */
  score: SynergyScore;
  /** Why this combination is promising */
  justification: string;
  /** Specific aspects to combine */
  combinationAspects: CombinationAspect[];
  /** Expected outcomes */
  expectedOutcomes: ExpectedOutcome[];
  /** Status of this proposal */
  status: ProposalStatus;
  /** When the proposal was created */
  createdAt: Date;
  /** When the proposal was last updated */
  updatedAt: Date;
  /** Who created this proposal (user ID or 'system') */
  createdBy: string;
  /** Optional notes from reviewers */
  notes?: string;
}

/**
 * Specific aspect of how techniques combine
 */
export interface CombinationAspect {
  /** What from technique A */
  fromA: string;
  /** What from technique B */
  fromB: string;
  /** How they combine */
  combination: string;
  /** Why this is beneficial */
  benefit: string;
}

/**
 * Expected outcome of combining techniques
 */
export interface ExpectedOutcome {
  /** Metric being improved */
  metric: string;
  /** Baseline value (current best) */
  baseline: number;
  /** Expected value after combination */
  expected: number;
  /** Unit of measurement */
  unit: string;
  /** Confidence in this prediction (0-1) */
  confidence: number;
}

/**
 * Status of a synergy proposal
 */
export type ProposalStatus =
  | "pending" // Awaiting review
  | "accepted" // Accepted for exploration
  | "rejected" // Rejected with reason
  | "exploring" // Currently being explored
  | "validated" // Combination validated as beneficial
  | "invalidated"; // Combination tried but not beneficial

/**
 * Explored combination result
 */
export interface ExploredCombination {
  /** Original proposal */
  proposal: SynergyProposal;
  /** When exploration started */
  startedAt: Date;
  /** When exploration completed (if done) */
  completedAt?: Date;
  /** Exploration status */
  status: "in_progress" | "completed" | "failed" | "abandoned";
  /** Actual results */
  results?: CombinationResult[];
  /** Implementation notes */
  implementationNotes?: string;
  /** Link to resulting technique (if created) */
  resultingTechniqueId?: string;
}

/**
 * Result of exploring a combination
 */
export interface CombinationResult {
  /** Metric measured */
  metric: string;
  /** Actual value achieved */
  actual: number;
  /** Expected value from proposal */
  expected: number;
  /** Unit of measurement */
  unit: string;
  /** Whether this met expectations */
  meetsExpectation: boolean;
}

/**
 * Configuration for synergy discovery
 */
export interface SynergyDiscoveryConfig {
  /** Minimum similarity threshold for consideration */
  minSimilarity: number;
  /** Minimum overall score to propose */
  minScore: number;
  /** Maximum proposals to return */
  maxProposals: number;
  /** Weight for similarity in overall score */
  similarityWeight: number;
  /** Weight for complementarity in overall score */
  complementarityWeight: number;
  /** Weight for novelty in overall score */
  noveltyWeight: number;
  /** Weight for feasibility in overall score */
  feasibilityWeight: number;
  /** Weight for impact in overall score */
  impactWeight: number;
  /** Domains to focus on (empty = all) */
  focusDomains: string[];
  /** Tags to filter techniques (empty = all) */
  filterTags: string[];
  /** Exclude already explored combinations */
  excludeExplored: boolean;
}

/**
 * Default configuration
 */
export const DEFAULT_SYNERGY_CONFIG: SynergyDiscoveryConfig = {
  minSimilarity: 0.2,
  minScore: 0.5,
  maxProposals: 20,
  similarityWeight: 0.15,
  complementarityWeight: 0.25,
  noveltyWeight: 0.2,
  feasibilityWeight: 0.15,
  impactWeight: 0.25,
  focusDomains: [],
  filterTags: [],
  excludeExplored: true,
};

/**
 * Feature vector for a technique (for similarity computation)
 */
export interface TechniqueFeatures {
  /** Technique ID */
  id: string;
  /** Domain features (one-hot encoded) */
  domains: number[];
  /** Architecture features */
  architecture: ArchitectureFeatures;
  /** Training features */
  training: TrainingFeatures;
  /** Performance features */
  performance: PerformanceFeatures;
  /** Tag features (normalized) */
  tags: Map<string, number>;
}

/**
 * Architecture-related features
 */
export interface ArchitectureFeatures {
  /** Uses transformer */
  usesTransformer: boolean;
  /** Uses convolution */
  usesConvolution: boolean;
  /** Uses attention */
  usesAttention: boolean;
  /** Uses diffusion */
  usesDiffusion: boolean;
  /** Uses VAE */
  usesVAE: boolean;
  /** Uses GAN */
  usesGAN: boolean;
  /** Uses flow */
  usesFlow: boolean;
  /** Parameter count (normalized) */
  parameterScale: number;
}

/**
 * Training-related features
 */
export interface TrainingFeatures {
  /** Uses supervised learning */
  supervised: boolean;
  /** Uses self-supervised learning */
  selfSupervised: boolean;
  /** Uses reinforcement learning */
  reinforcement: boolean;
  /** Uses contrastive learning */
  contrastive: boolean;
  /** Data efficiency (0-1) */
  dataEfficiency: number;
  /** Training stability (0-1) */
  stability: number;
}

/**
 * Performance-related features
 */
export interface PerformanceFeatures {
  /** Quality score (0-1) */
  quality: number;
  /** Speed score (0-1) */
  speed: number;
  /** Memory efficiency (0-1) */
  memoryEfficiency: number;
  /** Scalability (0-1) */
  scalability: number;
}

// Type guards
export function isSynergyProposal(obj: unknown): obj is SynergyProposal {
  return (
    typeof obj === "object" &&
    obj !== null &&
    "id" in obj &&
    "techniqueA" in obj &&
    "techniqueB" in obj &&
    "score" in obj
  );
}

export function isProposalStatus(value: string): value is ProposalStatus {
  return [
    "pending",
    "accepted",
    "rejected",
    "exploring",
    "validated",
    "invalidated",
  ].includes(value);
}

// Factory functions
export function createProposalId(): string {
  return `synergy-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createEmptyFeatures(id: string): TechniqueFeatures {
  return {
    id,
    domains: [],
    architecture: {
      usesTransformer: false,
      usesConvolution: false,
      usesAttention: false,
      usesDiffusion: false,
      usesVAE: false,
      usesGAN: false,
      usesFlow: false,
      parameterScale: 0,
    },
    training: {
      supervised: false,
      selfSupervised: false,
      reinforcement: false,
      contrastive: false,
      dataEfficiency: 0.5,
      stability: 0.5,
    },
    performance: {
      quality: 0.5,
      speed: 0.5,
      memoryEfficiency: 0.5,
      scalability: 0.5,
    },
    tags: new Map(),
  };
}
