/**
 * Lineage Tracker
 *
 * Tracks technique ancestry through generations, stores parent-child relationships,
 * calculates genetic distance, and maintains full evolution history.
 */

import {
  Chromosome,
  Population,
  GenerationStats,
  MutationRecord,
  Gene,
  ChromosomeLineage,
  LineageNode,
} from "./types";

// ============================================================================
// Lineage Types
// ============================================================================

export interface LineageTracker {
  /** Population ID being tracked */
  populationId: string;
  /** All recorded chromosomes */
  chromosomes: Map<string, ChromosomeRecord>;
  /** Parent-child relationships */
  parentChildMap: Map<string, string[]>;
  /** Child-parent relationships */
  childParentMap: Map<string, string[]>;
  /** Generation snapshots */
  generationSnapshots: Map<number, GenerationSnapshot>;
  /** Evolution history */
  evolutionHistory: EvolutionEvent[];
  /** Genetic distance cache */
  distanceCache: Map<string, number>;
}

export interface ChromosomeRecord {
  /** Chromosome data */
  chromosome: Chromosome;
  /** When recorded */
  recordedAt: Date;
  /** Generation recorded */
  generation: number;
  /** Survival count (generations survived) */
  survivalCount: number;
  /** Offspring count */
  offspringCount: number;
  /** Contribution score */
  contributionScore: number;
}

export interface GenerationSnapshot {
  /** Generation number */
  generation: number;
  /** Timestamp */
  timestamp: Date;
  /** All chromosome IDs in this generation */
  chromosomeIds: string[];
  /** Statistics */
  stats: GenerationStats;
  /** Best chromosome ID */
  bestId: string;
  /** New chromosomes (not from previous gen) */
  newChromosomeIds: string[];
  /** Extinct chromosomes (not in next gen) */
  extinctChromosomeIds: string[];
}

export interface EvolutionEvent {
  /** Event type */
  type: EvolutionEventType;
  /** Generation */
  generation: number;
  /** Timestamp */
  timestamp: Date;
  /** Related chromosome IDs */
  chromosomeIds: string[];
  /** Event details */
  details: Record<string, unknown>;
}

export type EvolutionEventType =
  | "birth" // New chromosome created
  | "death" // Chromosome removed from population
  | "mutation" // Significant mutation
  | "crossover" // Crossover event
  | "elite_selection" // Selected as elite
  | "fitness_improvement" // Fitness improved
  | "convergence" // Population converged
  | "divergence" // Population diverged
  | "extinction" // Lineage went extinct;

export interface LineageAnalysis {
  /** Root chromosome */
  rootId: string;
  /** Total descendants */
  totalDescendants: number;
  /** Living descendants (in current generation) */
  livingDescendants: number;
  /** Lineage depth */
  depth: number;
  /** Average fitness over time */
  averageFitnessOverTime: number[];
  /** Key mutations that improved fitness */
  keyMutations: KeyMutation[];
  /** Dominant genes (most preserved) */
  dominantGenes: DominantGene[];
  /** Lineage contribution to population */
  populationContribution: number;
}

export interface KeyMutation {
  /** Generation when occurred */
  generation: number;
  /** Chromosome before */
  beforeId: string;
  /** Chromosome after */
  afterId: string;
  /** Fitness improvement */
  fitnessImprovement: number;
  /** Mutation details */
  mutation: MutationRecord;
}

export interface DominantGene {
  /** Gene name */
  geneName: string;
  /** Gene type */
  geneType: string;
  /** Dominant value */
  dominantValue: number | string | boolean;
  /** Preservation rate (0-1) */
  preservationRate: number;
  /** Generations observed */
  generationsObserved: number;
}

export interface GeneticDistance {
  /** Chromosome 1 ID */
  chr1Id: string;
  /** Chromosome 2 ID */
  chr2Id: string;
  /** Overall distance (0-1) */
  distance: number;
  /** Distance per gene */
  geneDistances: Map<string, number>;
  /** Shared ancestry depth */
  sharedAncestryDepth: number;
  /** Common ancestor ID */
  commonAncestorId?: string;
}

export interface LineageTree {
  /** Root node */
  root: ExtendedLineageNode;
  /** Total nodes */
  totalNodes: number;
  /** Max depth */
  maxDepth: number;
  /** Breadth at each level */
  breadthPerLevel: number[];
}

export interface ExtendedLineageNode extends LineageNode {
  /** Parent node */
  parent?: ExtendedLineageNode;
  /** Depth in tree */
  depth: number;
  /** Is alive (in current population) */
  isAlive: boolean;
  /** Contribution to descendants */
  contribution: number;
  /** Mutations at this node */
  mutations: MutationRecord[];
}

// ============================================================================
// Lineage Tracker Implementation
// ============================================================================

export function createLineageTracker(populationId: string): LineageTracker {
  return {
    populationId,
    chromosomes: new Map(),
    parentChildMap: new Map(),
    childParentMap: new Map(),
    generationSnapshots: new Map(),
    evolutionHistory: [],
    distanceCache: new Map(),
  };
}

/**
 * Record a chromosome in the tracker
 */
export function recordChromosome(
  tracker: LineageTracker,
  chromosome: Chromosome
): void {
  const existing = tracker.chromosomes.get(chromosome.id);

  if (existing) {
    existing.survivalCount++;
    return;
  }

  tracker.chromosomes.set(chromosome.id, {
    chromosome: { ...chromosome },
    recordedAt: new Date(),
    generation: chromosome.generation,
    survivalCount: 1,
    offspringCount: 0,
    contributionScore: 0,
  });

  // Record parent-child relationships
  for (const parentId of chromosome.parentIds) {
    // Parent -> Child
    const children = tracker.parentChildMap.get(parentId) || [];
    if (!children.includes(chromosome.id)) {
      children.push(chromosome.id);
      tracker.parentChildMap.set(parentId, children);
    }

    // Child -> Parent
    const parents = tracker.childParentMap.get(chromosome.id) || [];
    if (!parents.includes(parentId)) {
      parents.push(parentId);
      tracker.childParentMap.set(chromosome.id, parents);
    }

    // Update parent's offspring count
    const parentRecord = tracker.chromosomes.get(parentId);
    if (parentRecord) {
      parentRecord.offspringCount++;
    }
  }

  // Record birth event
  tracker.evolutionHistory.push({
    type: "birth",
    generation: chromosome.generation,
    timestamp: new Date(),
    chromosomeIds: [chromosome.id],
    details: {
      parentIds: chromosome.parentIds,
      fitness: chromosome.fitness,
    },
  });
}

/**
 * Record a generation snapshot
 */
export function recordGeneration(
  tracker: LineageTracker,
  population: Population,
  stats: GenerationStats
): void {
  const chromosomeIds = population.chromosomes.map((c) => c.id);
  const previousSnapshot = tracker.generationSnapshots.get(population.generation - 1);

  const newIds = previousSnapshot
    ? chromosomeIds.filter((id) => !previousSnapshot.chromosomeIds.includes(id))
    : chromosomeIds;

  const extinctIds = previousSnapshot
    ? previousSnapshot.chromosomeIds.filter((id) => !chromosomeIds.includes(id))
    : [];

  // Record all chromosomes
  for (const chr of population.chromosomes) {
    recordChromosome(tracker, chr);
  }

  // Create snapshot
  tracker.generationSnapshots.set(population.generation, {
    generation: population.generation,
    timestamp: new Date(),
    chromosomeIds,
    stats,
    bestId: population.bestChromosomeId,
    newChromosomeIds: newIds,
    extinctChromosomeIds: extinctIds,
  });

  // Record extinction events
  for (const extinctId of extinctIds) {
    tracker.evolutionHistory.push({
      type: "death",
      generation: population.generation,
      timestamp: new Date(),
      chromosomeIds: [extinctId],
      details: {},
    });
  }
}

/**
 * Get all descendants of a chromosome
 */
export function getDescendants(
  tracker: LineageTracker,
  chromosomeId: string,
  maxDepth: number = Infinity
): string[] {
  const descendants: string[] = [];
  const visited = new Set<string>();

  function traverse(id: string, depth: number): void {
    if (visited.has(id)) return;
    visited.add(id);

    const children = tracker.parentChildMap.get(id) || [];
    for (const childId of children) {
      if (!visited.has(childId) && depth < maxDepth) {
        descendants.push(childId);
        traverse(childId, depth + 1);
      }
    }
  }

  traverse(chromosomeId, 0);
  return descendants;
}

/**
 * Get all ancestors of a chromosome
 */
export function getAncestors(
  tracker: LineageTracker,
  chromosomeId: string,
  maxDepth: number = Infinity
): string[] {
  const ancestors: string[] = [];
  const visited = new Set<string>();

  function traverse(id: string, depth: number): void {
    if (depth > maxDepth || visited.has(id)) return;
    visited.add(id);

    const parents = tracker.childParentMap.get(id) || [];
    for (const parentId of parents) {
      if (!visited.has(parentId)) {
        ancestors.push(parentId);
        traverse(parentId, depth + 1);
      }
    }
  }

  traverse(chromosomeId, 0);
  return ancestors;
}

/**
 * Find common ancestor between two chromosomes
 */
export function findCommonAncestor(
  tracker: LineageTracker,
  chr1Id: string,
  chr2Id: string
): string | null {
  const ancestors1 = new Set(getAncestors(tracker, chr1Id));
  const ancestors2 = getAncestors(tracker, chr2Id);

  for (const ancestor of ancestors2) {
    if (ancestors1.has(ancestor)) {
      return ancestor;
    }
  }

  return null;
}

/**
 * Calculate genetic distance between two chromosomes
 */
export function calculateGeneticDistance(
  tracker: LineageTracker,
  chr1Id: string,
  chr2Id: string
): GeneticDistance {
  const cacheKey = [chr1Id, chr2Id].sort().join("-");
  const cachedDistance = tracker.distanceCache.get(cacheKey);

  const record1 = tracker.chromosomes.get(chr1Id);
  const record2 = tracker.chromosomes.get(chr2Id);

  if (!record1 || !record2) {
    return {
      chr1Id,
      chr2Id,
      distance: 1,
      geneDistances: new Map(),
      sharedAncestryDepth: 0,
    };
  }

  const chr1 = record1.chromosome;
  const chr2 = record2.chromosome;

  const geneDistances = new Map<string, number>();
  let totalDistance = 0;

  for (let i = 0; i < chr1.genes.length; i++) {
    const g1 = chr1.genes[i];
    const g2 = chr2.genes[i];
    let distance = 0;

    if (typeof g1.value === "number" && typeof g2.value === "number") {
      const range = (g1.maxValue ?? 1) - (g1.minValue ?? 0);
      distance = Math.abs(g1.value - g2.value) / (range || 1);
    } else if (g1.value !== g2.value) {
      distance = 1;
    }

    geneDistances.set(g1.name, distance);
    totalDistance += distance;
  }

  const avgDistance = chr1.genes.length > 0 ? totalDistance / chr1.genes.length : 0;

  // Find shared ancestry
  const commonAncestor = findCommonAncestor(tracker, chr1Id, chr2Id);
  let sharedAncestryDepth = 0;
  if (commonAncestor) {
    const ancestors1 = getAncestors(tracker, chr1Id);
    sharedAncestryDepth = ancestors1.indexOf(commonAncestor) + 1;
  }

  const result: GeneticDistance = {
    chr1Id,
    chr2Id,
    distance: avgDistance,
    geneDistances,
    sharedAncestryDepth,
    commonAncestorId: commonAncestor ?? undefined,
  };

  tracker.distanceCache.set(cacheKey, avgDistance);
  return result;
}

/**
 * Build lineage tree for a chromosome
 */
export function buildLineageTree(
  tracker: LineageTracker,
  rootId: string,
  currentPopulation: Population
): LineageTree {
  const currentIds = new Set(currentPopulation.chromosomes.map((c) => c.id));
  const breadthPerLevel: number[] = [];

  function buildNode(
    id: string,
    parent: ExtendedLineageNode | undefined,
    depth: number
  ): ExtendedLineageNode | null {
    const record = tracker.chromosomes.get(id);
    if (!record) return null;

    // Update breadth
    while (breadthPerLevel.length <= depth) {
      breadthPerLevel.push(0);
    }
    breadthPerLevel[depth]++;

    const descendants = getDescendants(tracker, id);
    const livingDescendants = descendants.filter((d) => currentIds.has(d));

    const node: ExtendedLineageNode = {
      id,
      name: record.chromosome.name,
      generation: record.chromosome.generation,
      fitness: record.chromosome.fitness,
      children: [],
      parent,
      depth,
      isAlive: currentIds.has(id),
      contribution: livingDescendants.length / Math.max(1, descendants.length),
      mutations: record.chromosome.mutations,
    };

    const childIds = tracker.parentChildMap.get(id) || [];
    for (const childId of childIds) {
      const childNode = buildNode(childId, node, depth + 1);
      if (childNode) {
        node.children.push(childNode);
      }
    }

    return node;
  }

  const root = buildNode(rootId, undefined, 0);

  if (!root) {
    return {
      root: {
        id: rootId,
        name: "Unknown",
        generation: 0,
        fitness: 0,
        children: [],
        depth: 0,
        isAlive: false,
        contribution: 0,
        mutations: [],
      },
      totalNodes: 0,
      maxDepth: 0,
      breadthPerLevel: [],
    };
  }

  return {
    root,
    totalNodes: countNodes(root),
    maxDepth: breadthPerLevel.length,
    breadthPerLevel,
  };
}

function countNodes(node: ExtendedLineageNode): number {
  let count = 1;
  for (const child of node.children) {
    count += countNodes(child as ExtendedLineageNode);
  }
  return count;
}

/**
 * Analyze lineage of a chromosome
 */
export function analyzeLineage(
  tracker: LineageTracker,
  rootId: string,
  currentPopulation: Population
): LineageAnalysis {
  const currentIds = new Set(currentPopulation.chromosomes.map((c) => c.id));
  const descendants = getDescendants(tracker, rootId);
  const livingDescendants = descendants.filter((d) => currentIds.has(d));

  // Calculate average fitness over time
  const fitnessOverTime: number[] = [];
  const allLineage = [rootId, ...descendants];

  const byGeneration = new Map<number, number[]>();
  for (const id of allLineage) {
    const record = tracker.chromosomes.get(id);
    if (record) {
      const gen = record.chromosome.generation;
      const fitnesses = byGeneration.get(gen) || [];
      fitnesses.push(record.chromosome.fitness);
      byGeneration.set(gen, fitnesses);
    }
  }

  const sortedGens = Array.from(byGeneration.keys()).sort((a, b) => a - b);
  for (const gen of sortedGens) {
    const fitnesses = byGeneration.get(gen) || [];
    const avg = fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length;
    fitnessOverTime.push(avg);
  }

  // Find key mutations
  const keyMutations: KeyMutation[] = [];
  for (const id of allLineage) {
    const record = tracker.chromosomes.get(id);
    if (!record) continue;

    const parents = tracker.childParentMap.get(id) || [];
    for (const parentId of parents) {
      const parentRecord = tracker.chromosomes.get(parentId);
      if (!parentRecord) continue;

      const fitnessImprovement = record.chromosome.fitness - parentRecord.chromosome.fitness;
      if (fitnessImprovement > 0.05) {
        for (const mutation of record.chromosome.mutations) {
          keyMutations.push({
            generation: record.chromosome.generation,
            beforeId: parentId,
            afterId: id,
            fitnessImprovement,
            mutation,
          });
        }
      }
    }
  }

  // Sort by fitness improvement
  keyMutations.sort((a, b) => b.fitnessImprovement - a.fitnessImprovement);

  // Find dominant genes
  const geneValueCounts = new Map<string, Map<string, number>>();

  for (const id of allLineage) {
    const record = tracker.chromosomes.get(id);
    if (!record) continue;

    for (const gene of record.chromosome.genes) {
      const valueCounts = geneValueCounts.get(gene.name) || new Map<string, number>();
      const valueKey = JSON.stringify(gene.value);
      const count = valueCounts.get(valueKey) || 0;
      valueCounts.set(valueKey, count + 1);
      geneValueCounts.set(gene.name, valueCounts);
    }
  }

  const dominantGenes: DominantGene[] = [];
  for (const [geneName, valueCounts] of Array.from(geneValueCounts)) {
    let maxCount = 0;
    let dominantValue: number | string | boolean = "";

    for (const [valueKey, count] of Array.from(valueCounts)) {
      if (count > maxCount) {
        maxCount = count;
        dominantValue = JSON.parse(valueKey);
      }
    }

    const totalCount = Array.from(valueCounts.values()).reduce((a, b) => a + b, 0);

    dominantGenes.push({
      geneName,
      geneType: tracker.chromosomes.get(rootId)?.chromosome.genes.find((g) => g.name === geneName)?.type || "unknown",
      dominantValue,
      preservationRate: maxCount / totalCount,
      generationsObserved: sortedGens.length,
    });
  }

  // Sort by preservation rate
  dominantGenes.sort((a, b) => b.preservationRate - a.preservationRate);

  // Calculate population contribution
  const populationContribution = livingDescendants.length / currentPopulation.size;

  // Calculate depth
  let maxDepth = 0;
  for (const id of allLineage) {
    const record = tracker.chromosomes.get(id);
    if (record) {
      const depth = record.chromosome.generation - (tracker.chromosomes.get(rootId)?.chromosome.generation || 0);
      maxDepth = Math.max(maxDepth, depth);
    }
  }

  return {
    rootId,
    totalDescendants: descendants.length,
    livingDescendants: livingDescendants.length,
    depth: maxDepth,
    averageFitnessOverTime: fitnessOverTime,
    keyMutations: keyMutations.slice(0, 10),
    dominantGenes: dominantGenes.slice(0, 10),
    populationContribution,
  };
}

/**
 * Get evolution timeline
 */
export function getEvolutionTimeline(
  tracker: LineageTracker,
  startGeneration?: number,
  endGeneration?: number
): EvolutionEvent[] {
  return tracker.evolutionHistory.filter((event) => {
    if (startGeneration !== undefined && event.generation < startGeneration) return false;
    if (endGeneration !== undefined && event.generation > endGeneration) return false;
    return true;
  });
}

/**
 * Get chromosome offspring
 */
export function getOffspring(
  tracker: LineageTracker,
  chromosomeId: string,
  directOnly: boolean = true
): Chromosome[] {
  const childIds = directOnly
    ? tracker.parentChildMap.get(chromosomeId) || []
    : getDescendants(tracker, chromosomeId);

  return childIds
    .map((id) => tracker.chromosomes.get(id)?.chromosome)
    .filter((c): c is Chromosome => c !== undefined);
}

/**
 * Get lineage statistics
 */
export function getLineageStats(tracker: LineageTracker): {
  totalChromosomes: number;
  totalGenerations: number;
  totalBirths: number;
  totalDeaths: number;
  averageSurvival: number;
  averageOffspring: number;
  longestLineage: number;
} {
  const births = tracker.evolutionHistory.filter((e) => e.type === "birth").length;
  const deaths = tracker.evolutionHistory.filter((e) => e.type === "death").length;

  let totalSurvival = 0;
  let totalOffspring = 0;
  let longestLineage = 0;

  for (const [id, record] of Array.from(tracker.chromosomes)) {
    totalSurvival += record.survivalCount;
    totalOffspring += record.offspringCount;

    const descendants = getDescendants(tracker, id);
    longestLineage = Math.max(longestLineage, descendants.length);
  }

  const count = tracker.chromosomes.size;

  return {
    totalChromosomes: count,
    totalGenerations: tracker.generationSnapshots.size,
    totalBirths: births,
    totalDeaths: deaths,
    averageSurvival: count > 0 ? totalSurvival / count : 0,
    averageOffspring: count > 0 ? totalOffspring / count : 0,
    longestLineage,
  };
}

/**
 * Export lineage to ChromosomeLineage format (for compatibility)
 */
export function toChromosomeLineage(
  tracker: LineageTracker,
  chromosomeId: string
): ChromosomeLineage {
  const record = tracker.chromosomes.get(chromosomeId);
  if (!record) {
    return {
      rootId: chromosomeId,
      ancestors: [],
      depth: 0,
      tree: {
        id: chromosomeId,
        name: "Unknown",
        generation: 0,
        fitness: 0,
        children: [],
      },
      keyMutations: [],
      fitnessTrajectory: [],
    };
  }

  const ancestors = getAncestors(tracker, chromosomeId);
  const fitnessTrajectory: Array<{ generation: number; fitness: number }> = [];

  for (const ancestorId of [chromosomeId, ...ancestors]) {
    const ancestorRecord = tracker.chromosomes.get(ancestorId);
    if (ancestorRecord) {
      fitnessTrajectory.push({
        generation: ancestorRecord.chromosome.generation,
        fitness: ancestorRecord.chromosome.fitness,
      });
    }
  }

  fitnessTrajectory.sort((a, b) => a.generation - b.generation);

  const buildTree = (id: string): LineageNode => {
    const r = tracker.chromosomes.get(id);
    if (!r) {
      return {
        id,
        name: "Unknown",
        generation: 0,
        fitness: 0,
        children: [],
      };
    }

    const parentIds = tracker.childParentMap.get(id) || [];
    const children = parentIds.map((pid) => buildTree(pid));

    return {
      id,
      name: r.chromosome.name,
      generation: r.chromosome.generation,
      fitness: r.chromosome.fitness,
      children,
    };
  };

  const keyMutations = record.chromosome.mutations.filter(
    (m) => m.fitnessImpact !== undefined && Math.abs(m.fitnessImpact) > 0.05
  );

  return {
    rootId: chromosomeId,
    ancestors,
    depth: ancestors.length,
    tree: buildTree(chromosomeId),
    keyMutations,
    fitnessTrajectory,
  };
}
