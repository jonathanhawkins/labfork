/**
 * Genetic Evolution Types
 *
 * Types for the genetic algorithm system that evolves and optimizes
 * research technique combinations through simulated evolution.
 */

import { TechniqueNode, TechniqueCategory } from "../knowledge-graph/types";

/**
 * A gene representing a single technique trait
 */
export interface Gene {
  /** Gene identifier */
  id: string;
  /** Gene name */
  name: string;
  /** Gene type/category */
  type: GeneType;
  /** Gene value (can be numeric or categorical) */
  value: number | string | boolean;
  /** Minimum value (for numeric genes) */
  minValue?: number;
  /** Maximum value (for numeric genes) */
  maxValue?: number;
  /** Allowed values (for categorical genes) */
  allowedValues?: (string | boolean)[];
  /** Mutation rate for this gene (0-1) */
  mutationRate: number;
  /** Weight for fitness calculation */
  weight: number;
  /** Is this gene dominant? */
  dominant: boolean;
  /** Source technique ID if inherited */
  sourceId?: string;
}

/**
 * Types of genes
 */
export type GeneType =
  | "architecture" // Architectural choices
  | "training" // Training methodology
  | "conditioning" // Conditioning approach
  | "loss" // Loss function
  | "hyperparameter" // Hyperparameter value
  | "data" // Data processing
  | "performance"; // Performance characteristic

/**
 * A chromosome representing a complete technique configuration
 */
export interface Chromosome {
  /** Unique chromosome ID */
  id: string;
  /** Human-readable name */
  name: string;
  /** All genes in this chromosome */
  genes: Gene[];
  /** Fitness score (0-1) */
  fitness: number;
  /** Component fitness scores */
  fitnessComponents: FitnessComponents;
  /** Generation this chromosome was created */
  generation: number;
  /** Parent chromosome IDs */
  parentIds: string[];
  /** Mutation history */
  mutations: MutationRecord[];
  /** When created */
  createdAt: Date;
  /** Associated technique ID if materialized */
  techniqueId?: string;
  /** Is this chromosome elite? */
  isElite: boolean;
}

/**
 * Fitness score components
 */
export interface FitnessComponents {
  /** Quality score (0-1) */
  quality: number;
  /** Efficiency score (0-1) */
  efficiency: number;
  /** Novelty score (0-1) */
  novelty: number;
  /** Feasibility score (0-1) */
  feasibility: number;
  /** Compatibility score (0-1) */
  compatibility: number;
}

/**
 * Record of a mutation
 */
export interface MutationRecord {
  /** Mutated gene ID */
  geneId: string;
  /** Original value */
  originalValue: number | string | boolean;
  /** New value after mutation */
  newValue: number | string | boolean;
  /** Mutation type */
  type: MutationType;
  /** When the mutation occurred */
  timestamp: Date;
  /** Impact on fitness */
  fitnessImpact?: number;
}

/**
 * Types of mutations
 */
export type MutationType =
  | "point" // Single value change
  | "insertion" // New gene added
  | "deletion" // Gene removed
  | "inversion" // Value inverted
  | "duplication"; // Gene duplicated

/**
 * A population of chromosomes
 */
export interface Population {
  /** Population ID */
  id: string;
  /** Population name */
  name: string;
  /** Current generation number */
  generation: number;
  /** All chromosomes in the population */
  chromosomes: Chromosome[];
  /** Population size */
  size: number;
  /** Best chromosome ID */
  bestChromosomeId: string;
  /** Average fitness */
  averageFitness: number;
  /** Fitness diversity */
  fitnessDiversity: number;
  /** Generation history */
  generationHistory: GenerationStats[];
  /** Configuration used */
  config: EvolutionConfig;
  /** When population was created */
  createdAt: Date;
  /** Last evolution timestamp */
  lastEvolved: Date;
  /** Status */
  status: PopulationStatus;
}

/**
 * Population status
 */
export type PopulationStatus =
  | "initializing" // Being set up
  | "evolving" // Currently evolving
  | "converged" // Reached convergence
  | "paused" // Evolution paused
  | "completed"; // Evolution completed

/**
 * Statistics for a generation
 */
export interface GenerationStats {
  /** Generation number */
  generation: number;
  /** Best fitness in generation */
  bestFitness: number;
  /** Average fitness */
  averageFitness: number;
  /** Worst fitness */
  worstFitness: number;
  /** Fitness standard deviation */
  fitnessStdDev: number;
  /** Number of mutations */
  mutationCount: number;
  /** Number of crossovers */
  crossoverCount: number;
  /** Number of elite preserved */
  eliteCount: number;
  /** Timestamp */
  timestamp: Date;
}

/**
 * Configuration for evolution
 */
export interface EvolutionConfig {
  /** Population size */
  populationSize: number;
  /** Number of generations to run */
  maxGenerations: number;
  /** Mutation probability (0-1) */
  mutationRate: number;
  /** Crossover probability (0-1) */
  crossoverRate: number;
  /** Elite preservation percentage (0-1) */
  elitePercentage: number;
  /** Tournament size for selection */
  tournamentSize: number;
  /** Selection strategy */
  selectionStrategy: SelectionStrategy;
  /** Crossover strategy */
  crossoverStrategy: CrossoverStrategy;
  /** Fitness weights */
  fitnessWeights: FitnessWeights;
  /** Convergence threshold */
  convergenceThreshold: number;
  /** Convergence window (generations) */
  convergenceWindow: number;
  /** Enable adaptive mutation */
  adaptiveMutation: boolean;
  /** Enable niching for diversity */
  enableNiching: boolean;
}

/**
 * Selection strategies
 */
export type SelectionStrategy =
  | "tournament" // Tournament selection
  | "roulette" // Roulette wheel selection
  | "rank" // Rank-based selection
  | "elitist"; // Elitist selection

/**
 * Crossover strategies
 */
export type CrossoverStrategy =
  | "single_point" // Single point crossover
  | "two_point" // Two point crossover
  | "uniform" // Uniform crossover
  | "blend"; // Blend crossover for numeric genes

/**
 * Weights for fitness components
 */
export interface FitnessWeights {
  quality: number;
  efficiency: number;
  novelty: number;
  feasibility: number;
  compatibility: number;
}

/**
 * Default evolution configuration
 */
export const DEFAULT_EVOLUTION_CONFIG: EvolutionConfig = {
  populationSize: 50,
  maxGenerations: 100,
  mutationRate: 0.1,
  crossoverRate: 0.8,
  elitePercentage: 0.1,
  tournamentSize: 3,
  selectionStrategy: "tournament",
  crossoverStrategy: "uniform",
  fitnessWeights: {
    quality: 0.3,
    efficiency: 0.2,
    novelty: 0.2,
    feasibility: 0.15,
    compatibility: 0.15,
  },
  convergenceThreshold: 0.001,
  convergenceWindow: 10,
  adaptiveMutation: true,
  enableNiching: false,
};

/**
 * Lineage information for a chromosome
 */
export interface ChromosomeLineage {
  /** Root chromosome ID */
  rootId: string;
  /** All ancestor IDs */
  ancestors: string[];
  /** Generation depth */
  depth: number;
  /** Lineage tree structure */
  tree: LineageNode;
  /** Key mutations in lineage */
  keyMutations: MutationRecord[];
  /** Fitness trajectory */
  fitnessTrajectory: Array<{ generation: number; fitness: number }>;
}

/**
 * Node in lineage tree
 */
export interface LineageNode {
  /** Chromosome ID */
  id: string;
  /** Chromosome name */
  name: string;
  /** Generation */
  generation: number;
  /** Fitness */
  fitness: number;
  /** Child nodes */
  children: LineageNode[];
}

/**
 * Result of evolution run
 */
export interface EvolutionResult {
  /** Final population */
  population: Population;
  /** Best chromosome found */
  bestChromosome: Chromosome;
  /** All best chromosomes per generation */
  bestPerGeneration: Chromosome[];
  /** Total generations run */
  totalGenerations: number;
  /** Did evolution converge? */
  converged: boolean;
  /** Evolution duration (ms) */
  durationMs: number;
  /** Final statistics */
  finalStats: GenerationStats;
}

/**
 * Gene template for encoding techniques
 */
export interface GeneTemplate {
  /** Template ID */
  id: string;
  /** Gene name */
  name: string;
  /** Gene type */
  type: GeneType;
  /** Default value */
  defaultValue: number | string | boolean;
  /** Value range for numeric */
  range?: { min: number; max: number };
  /** Allowed values for categorical */
  options?: (string | boolean)[];
  /** Mutation rate */
  mutationRate: number;
  /** Importance weight */
  weight: number;
  /** Is typically dominant */
  dominantByDefault: boolean;
}

/**
 * Standard gene templates
 */
export const STANDARD_GENE_TEMPLATES: GeneTemplate[] = [
  {
    id: "uses_transformer",
    name: "Uses Transformer",
    type: "architecture",
    defaultValue: false,
    options: [true, false],
    mutationRate: 0.05,
    weight: 0.15,
    dominantByDefault: true,
  },
  {
    id: "uses_attention",
    name: "Uses Attention",
    type: "architecture",
    defaultValue: true,
    options: [true, false],
    mutationRate: 0.05,
    weight: 0.12,
    dominantByDefault: true,
  },
  {
    id: "uses_diffusion",
    name: "Uses Diffusion",
    type: "architecture",
    defaultValue: false,
    options: [true, false],
    mutationRate: 0.05,
    weight: 0.15,
    dominantByDefault: false,
  },
  {
    id: "uses_vae",
    name: "Uses VAE",
    type: "architecture",
    defaultValue: false,
    options: [true, false],
    mutationRate: 0.05,
    weight: 0.1,
    dominantByDefault: false,
  },
  {
    id: "learning_rate",
    name: "Learning Rate",
    type: "hyperparameter",
    defaultValue: 0.001,
    range: { min: 0.00001, max: 0.1 },
    mutationRate: 0.15,
    weight: 0.08,
    dominantByDefault: false,
  },
  {
    id: "batch_size",
    name: "Batch Size",
    type: "hyperparameter",
    defaultValue: 32,
    range: { min: 1, max: 512 },
    mutationRate: 0.1,
    weight: 0.05,
    dominantByDefault: false,
  },
  {
    id: "uses_self_supervised",
    name: "Uses Self-Supervised",
    type: "training",
    defaultValue: false,
    options: [true, false],
    mutationRate: 0.08,
    weight: 0.1,
    dominantByDefault: false,
  },
  {
    id: "uses_contrastive",
    name: "Uses Contrastive Learning",
    type: "training",
    defaultValue: false,
    options: [true, false],
    mutationRate: 0.08,
    weight: 0.1,
    dominantByDefault: false,
  },
  {
    id: "data_augmentation",
    name: "Data Augmentation Level",
    type: "data",
    defaultValue: "medium",
    options: ["none", "light", "medium", "heavy"],
    mutationRate: 0.1,
    weight: 0.05,
    dominantByDefault: false,
  },
  {
    id: "model_size",
    name: "Model Size",
    type: "architecture",
    defaultValue: "medium",
    options: ["small", "medium", "large", "xlarge"],
    mutationRate: 0.08,
    weight: 0.1,
    dominantByDefault: false,
  },
];

// ============================================================================
// Type Guards
// ============================================================================

export function isGeneType(value: string): value is GeneType {
  return [
    "architecture",
    "training",
    "conditioning",
    "loss",
    "hyperparameter",
    "data",
    "performance",
  ].includes(value);
}

export function isMutationType(value: string): value is MutationType {
  return ["point", "insertion", "deletion", "inversion", "duplication"].includes(
    value
  );
}

export function isSelectionStrategy(value: string): value is SelectionStrategy {
  return ["tournament", "roulette", "rank", "elitist"].includes(value);
}

export function isCrossoverStrategy(value: string): value is CrossoverStrategy {
  return ["single_point", "two_point", "uniform", "blend"].includes(value);
}

export function isPopulationStatus(value: string): value is PopulationStatus {
  return ["initializing", "evolving", "converged", "paused", "completed"].includes(
    value
  );
}

export function isChromosome(obj: unknown): obj is Chromosome {
  return (
    typeof obj === "object" &&
    obj !== null &&
    "id" in obj &&
    "genes" in obj &&
    "fitness" in obj &&
    "generation" in obj
  );
}

export function isPopulation(obj: unknown): obj is Population {
  return (
    typeof obj === "object" &&
    obj !== null &&
    "id" in obj &&
    "chromosomes" in obj &&
    "generation" in obj &&
    "size" in obj
  );
}

// ============================================================================
// Factory Functions
// ============================================================================

export function createGeneId(): string {
  return `gene-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createChromosomeId(): string {
  return `chr-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createPopulationId(): string {
  return `pop-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * Create a gene from a template
 */
export function createGeneFromTemplate(
  template: GeneTemplate,
  value?: number | string | boolean
): Gene {
  return {
    id: createGeneId(),
    name: template.name,
    type: template.type,
    value: value ?? template.defaultValue,
    minValue: template.range?.min,
    maxValue: template.range?.max,
    allowedValues: template.options,
    mutationRate: template.mutationRate,
    weight: template.weight,
    dominant: template.dominantByDefault,
  };
}

/**
 * Create default fitness components
 */
export function createDefaultFitnessComponents(): FitnessComponents {
  return {
    quality: 0.5,
    efficiency: 0.5,
    novelty: 0.5,
    feasibility: 0.5,
    compatibility: 0.5,
  };
}

/**
 * Calculate overall fitness from components
 */
export function calculateFitness(
  components: FitnessComponents,
  weights: FitnessWeights = DEFAULT_EVOLUTION_CONFIG.fitnessWeights
): number {
  const totalWeight =
    weights.quality +
    weights.efficiency +
    weights.novelty +
    weights.feasibility +
    weights.compatibility;

  return (
    (components.quality * weights.quality +
      components.efficiency * weights.efficiency +
      components.novelty * weights.novelty +
      components.feasibility * weights.feasibility +
      components.compatibility * weights.compatibility) /
    totalWeight
  );
}

/**
 * Create empty generation stats
 */
export function createEmptyGenerationStats(generation: number): GenerationStats {
  return {
    generation,
    bestFitness: 0,
    averageFitness: 0,
    worstFitness: 0,
    fitnessStdDev: 0,
    mutationCount: 0,
    crossoverCount: 0,
    eliteCount: 0,
    timestamp: new Date(),
  };
}

// ============================================================================
// Display Helpers
// ============================================================================

export function getGeneTypeLabel(type: GeneType): string {
  const labels: Record<GeneType, string> = {
    architecture: "Architecture",
    training: "Training",
    conditioning: "Conditioning",
    loss: "Loss Function",
    hyperparameter: "Hyperparameter",
    data: "Data Processing",
    performance: "Performance",
  };
  return labels[type];
}

export function getGeneTypeColor(type: GeneType): string {
  const colors: Record<GeneType, string> = {
    architecture: "#3b82f6",
    training: "#10b981",
    conditioning: "#8b5cf6",
    loss: "#ef4444",
    hyperparameter: "#f59e0b",
    data: "#06b6d4",
    performance: "#ec4899",
  };
  return colors[type];
}

export function getStatusColor(status: PopulationStatus): string {
  const colors: Record<PopulationStatus, string> = {
    initializing: "#6b7280",
    evolving: "#3b82f6",
    converged: "#22c55e",
    paused: "#f59e0b",
    completed: "#10b981",
  };
  return colors[status];
}

export function formatFitness(fitness: number): string {
  return (fitness * 100).toFixed(1) + "%";
}
