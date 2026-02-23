/**
 * Evolution Operators
 *
 * Complete implementations of selection, crossover, and mutation operators
 * for the genetic algorithm evolution system.
 */

import {
  Gene,
  Chromosome,
  MutationRecord,
  MutationType,
  FitnessComponents,
  createGeneId,
  createChromosomeId,
  createDefaultFitnessComponents,
} from "./types";

// ============================================================================
// Operator Statistics
// ============================================================================

export interface OperatorStats {
  selection: SelectionStats;
  crossover: CrossoverStats;
  mutation: MutationStats;
}

export interface SelectionStats {
  totalSelections: number;
  tournamentWins: Map<string, number>;
  averageSelectionPressure: number;
  diversityPreserved: number;
}

export interface CrossoverStats {
  totalCrossovers: number;
  successfulCrossovers: number;
  averageOffspringFitness: number;
  crossoverPointDistribution: number[];
  geneticContributions: Map<string, number>;
}

export interface MutationStats {
  totalMutations: number;
  beneficialMutations: number;
  neutralMutations: number;
  deleteriousMutations: number;
  mutationTypeDistribution: Map<MutationType, number>;
  averageFitnessImpact: number;
}

export function createEmptyOperatorStats(): OperatorStats {
  return {
    selection: {
      totalSelections: 0,
      tournamentWins: new Map(),
      averageSelectionPressure: 0,
      diversityPreserved: 0,
    },
    crossover: {
      totalCrossovers: 0,
      successfulCrossovers: 0,
      averageOffspringFitness: 0,
      crossoverPointDistribution: [],
      geneticContributions: new Map(),
    },
    mutation: {
      totalMutations: 0,
      beneficialMutations: 0,
      neutralMutations: 0,
      deleteriousMutations: 0,
      mutationTypeDistribution: new Map(),
      averageFitnessImpact: 0,
    },
  };
}

// ============================================================================
// Selection Operators
// ============================================================================

export interface SelectionConfig {
  tournamentSize: number;
  elitePressure: number;
  diversityWeight: number;
  rankExponent: number;
  boltzmannTemperature: number;
}

export const DEFAULT_SELECTION_CONFIG: SelectionConfig = {
  tournamentSize: 3,
  elitePressure: 2.0,
  diversityWeight: 0.1,
  rankExponent: 1.5,
  boltzmannTemperature: 1.0,
};

/**
 * Tournament Selection with configurable pressure
 */
export function tournamentSelect(
  chromosomes: Chromosome[],
  config: SelectionConfig = DEFAULT_SELECTION_CONFIG,
  stats?: SelectionStats
): Chromosome {
  const tournament: Chromosome[] = [];
  const indices = new Set<number>();

  // Select unique individuals for tournament
  while (tournament.length < config.tournamentSize && indices.size < chromosomes.length) {
    const idx = Math.floor(Math.random() * chromosomes.length);
    if (!indices.has(idx)) {
      indices.add(idx);
      tournament.push(chromosomes[idx]);
    }
  }

  // Sort by fitness (descending)
  tournament.sort((a, b) => b.fitness - a.fitness);

  // Apply selection pressure - probability of selecting best decreases with pressure
  for (let i = 0; i < tournament.length; i++) {
    const selectProbability = Math.pow(1 / config.elitePressure, i);
    if (Math.random() < selectProbability || i === tournament.length - 1) {
      if (stats) {
        stats.totalSelections++;
        const wins = stats.tournamentWins.get(tournament[i].id) || 0;
        stats.tournamentWins.set(tournament[i].id, wins + 1);
      }
      return tournament[i];
    }
  }

  return tournament[0];
}

/**
 * Roulette Wheel Selection (Fitness Proportionate)
 */
export function rouletteSelect(
  chromosomes: Chromosome[],
  stats?: SelectionStats
): Chromosome {
  // Handle negative fitness values by shifting
  const minFitness = Math.min(...chromosomes.map((c) => c.fitness));
  const shift = minFitness < 0 ? Math.abs(minFitness) + 0.1 : 0;

  const totalFitness = chromosomes.reduce((sum, chr) => sum + chr.fitness + shift, 0);

  if (totalFitness === 0) {
    const selected = chromosomes[Math.floor(Math.random() * chromosomes.length)];
    if (stats) stats.totalSelections++;
    return selected;
  }

  let target = Math.random() * totalFitness;
  for (const chr of chromosomes) {
    target -= chr.fitness + shift;
    if (target <= 0) {
      if (stats) {
        stats.totalSelections++;
        const wins = stats.tournamentWins.get(chr.id) || 0;
        stats.tournamentWins.set(chr.id, wins + 1);
      }
      return chr;
    }
  }

  if (stats) stats.totalSelections++;
  return chromosomes[chromosomes.length - 1];
}

/**
 * Rank-Based Selection with configurable exponent
 */
export function rankSelect(
  chromosomes: Chromosome[],
  config: SelectionConfig = DEFAULT_SELECTION_CONFIG,
  stats?: SelectionStats
): Chromosome {
  // Ensure sorted by fitness (descending)
  const sorted = [...chromosomes].sort((a, b) => b.fitness - a.fitness);
  const n = sorted.length;

  // Calculate rank probabilities with exponent
  const ranks: number[] = [];
  let totalRank = 0;
  for (let i = 0; i < n; i++) {
    const rank = Math.pow(n - i, config.rankExponent);
    ranks.push(rank);
    totalRank += rank;
  }

  let target = Math.random() * totalRank;
  for (let i = 0; i < n; i++) {
    target -= ranks[i];
    if (target <= 0) {
      if (stats) {
        stats.totalSelections++;
        const wins = stats.tournamentWins.get(sorted[i].id) || 0;
        stats.tournamentWins.set(sorted[i].id, wins + 1);
      }
      return sorted[i];
    }
  }

  if (stats) stats.totalSelections++;
  return sorted[0];
}

/**
 * Elitist Selection - Only selects from top percentage
 */
export function elitistSelect(
  chromosomes: Chromosome[],
  elitePercentage: number = 0.2,
  stats?: SelectionStats
): Chromosome {
  // Ensure sorted by fitness (descending)
  const sorted = [...chromosomes].sort((a, b) => b.fitness - a.fitness);
  const eliteCount = Math.max(1, Math.ceil(sorted.length * elitePercentage));
  const elite = sorted.slice(0, eliteCount);

  const selected = elite[Math.floor(Math.random() * elite.length)];
  if (stats) {
    stats.totalSelections++;
    const wins = stats.tournamentWins.get(selected.id) || 0;
    stats.tournamentWins.set(selected.id, wins + 1);
  }
  return selected;
}

/**
 * Boltzmann Selection - Temperature-based probability
 */
export function boltzmannSelect(
  chromosomes: Chromosome[],
  temperature: number = 1.0,
  stats?: SelectionStats
): Chromosome {
  // Calculate Boltzmann probabilities
  const expValues = chromosomes.map((chr) =>
    Math.exp(chr.fitness / temperature)
  );
  const sumExp = expValues.reduce((a, b) => a + b, 0);

  let target = Math.random() * sumExp;
  for (let i = 0; i < chromosomes.length; i++) {
    target -= expValues[i];
    if (target <= 0) {
      if (stats) stats.totalSelections++;
      return chromosomes[i];
    }
  }

  if (stats) stats.totalSelections++;
  return chromosomes[chromosomes.length - 1];
}

/**
 * Stochastic Universal Sampling (SUS)
 * Selects multiple individuals in a single spin
 */
export function stochasticUniversalSampling(
  chromosomes: Chromosome[],
  count: number,
  stats?: SelectionStats
): Chromosome[] {
  const selected: Chromosome[] = [];

  // Calculate fitness proportionate positions
  const minFitness = Math.min(...chromosomes.map((c) => c.fitness));
  const shift = minFitness < 0 ? Math.abs(minFitness) + 0.1 : 0;
  const totalFitness = chromosomes.reduce((sum, chr) => sum + chr.fitness + shift, 0);

  if (totalFitness === 0) {
    // Random selection if all fitness is zero
    while (selected.length < count) {
      selected.push(chromosomes[Math.floor(Math.random() * chromosomes.length)]);
    }
    return selected;
  }

  const distance = totalFitness / count;
  let start = Math.random() * distance;

  let cumFitness = 0;
  let chrIndex = 0;

  for (let i = 0; i < count; i++) {
    const pointer = start + i * distance;

    while (cumFitness + chromosomes[chrIndex].fitness + shift < pointer) {
      cumFitness += chromosomes[chrIndex].fitness + shift;
      chrIndex++;
      if (chrIndex >= chromosomes.length) chrIndex = 0;
    }

    selected.push(chromosomes[chrIndex]);
    if (stats) stats.totalSelections++;
  }

  return selected;
}

// ============================================================================
// Crossover Operators
// ============================================================================

export interface CrossoverConfig {
  blendAlpha: number;
  uniformBias: number;
  arithmeticWeight: number;
  sbxEta: number; // For Simulated Binary Crossover
}

export const DEFAULT_CROSSOVER_CONFIG: CrossoverConfig = {
  blendAlpha: 0.5,
  uniformBias: 0.5,
  arithmeticWeight: 0.5,
  sbxEta: 2.0,
};

/**
 * Single Point Crossover
 */
export function singlePointCrossover(
  parent1: Chromosome,
  parent2: Chromosome,
  generation: number,
  stats?: CrossoverStats
): Chromosome {
  const genes: Gene[] = [];
  const point = Math.floor(Math.random() * parent1.genes.length);

  if (stats) {
    stats.totalCrossovers++;
    stats.crossoverPointDistribution.push(point / parent1.genes.length);
  }

  for (let i = 0; i < parent1.genes.length; i++) {
    genes.push(cloneGene(i < point ? parent1.genes[i] : parent2.genes[i]));
  }

  return createOffspring(parent1, parent2, genes, generation);
}

/**
 * Two Point Crossover
 */
export function twoPointCrossover(
  parent1: Chromosome,
  parent2: Chromosome,
  generation: number,
  stats?: CrossoverStats
): Chromosome {
  const genes: Gene[] = [];
  const len = parent1.genes.length;

  let point1 = Math.floor(Math.random() * len);
  let point2 = Math.floor(Math.random() * len);

  if (point1 > point2) {
    [point1, point2] = [point2, point1];
  }

  if (stats) {
    stats.totalCrossovers++;
    stats.crossoverPointDistribution.push(point1 / len, point2 / len);
  }

  for (let i = 0; i < len; i++) {
    if (i < point1 || i >= point2) {
      genes.push(cloneGene(parent1.genes[i]));
    } else {
      genes.push(cloneGene(parent2.genes[i]));
    }
  }

  return createOffspring(parent1, parent2, genes, generation);
}

/**
 * Uniform Crossover with configurable bias
 */
export function uniformCrossover(
  parent1: Chromosome,
  parent2: Chromosome,
  generation: number,
  config: CrossoverConfig = DEFAULT_CROSSOVER_CONFIG,
  stats?: CrossoverStats
): Chromosome {
  const genes: Gene[] = [];
  let parent1Contributions = 0;
  let parent2Contributions = 0;

  for (let i = 0; i < parent1.genes.length; i++) {
    if (Math.random() < config.uniformBias) {
      genes.push(cloneGene(parent1.genes[i]));
      parent1Contributions++;
    } else {
      genes.push(cloneGene(parent2.genes[i]));
      parent2Contributions++;
    }
  }

  if (stats) {
    stats.totalCrossovers++;
    const p1Contrib = stats.geneticContributions.get(parent1.id) || 0;
    const p2Contrib = stats.geneticContributions.get(parent2.id) || 0;
    stats.geneticContributions.set(parent1.id, p1Contrib + parent1Contributions);
    stats.geneticContributions.set(parent2.id, p2Contrib + parent2Contributions);
  }

  return createOffspring(parent1, parent2, genes, generation);
}

/**
 * Blend Crossover (BLX-alpha) for numeric genes
 */
export function blendCrossover(
  parent1: Chromosome,
  parent2: Chromosome,
  generation: number,
  config: CrossoverConfig = DEFAULT_CROSSOVER_CONFIG,
  stats?: CrossoverStats
): Chromosome {
  const genes: Gene[] = [];
  const alpha = config.blendAlpha;

  for (let i = 0; i < parent1.genes.length; i++) {
    const gene1 = parent1.genes[i];
    const gene2 = parent2.genes[i];

    if (typeof gene1.value === "number" && typeof gene2.value === "number") {
      const min = Math.min(gene1.value, gene2.value);
      const max = Math.max(gene1.value, gene2.value);
      const range = max - min;

      // BLX-alpha extends the range by alpha on each side
      const lowerBound = min - alpha * range;
      const upperBound = max + alpha * range;

      const newValue = lowerBound + Math.random() * (upperBound - lowerBound);
      const clampedValue = Math.max(
        gene1.minValue ?? -Infinity,
        Math.min(gene1.maxValue ?? Infinity, newValue)
      );

      const newGene = cloneGene(gene1);
      newGene.value = clampedValue;
      genes.push(newGene);
    } else {
      // For non-numeric, use uniform crossover
      genes.push(cloneGene(Math.random() < 0.5 ? gene1 : gene2));
    }
  }

  if (stats) stats.totalCrossovers++;
  return createOffspring(parent1, parent2, genes, generation);
}

/**
 * Arithmetic Crossover - Weighted average of parents
 */
export function arithmeticCrossover(
  parent1: Chromosome,
  parent2: Chromosome,
  generation: number,
  config: CrossoverConfig = DEFAULT_CROSSOVER_CONFIG,
  stats?: CrossoverStats
): Chromosome {
  const genes: Gene[] = [];
  const weight = config.arithmeticWeight;

  for (let i = 0; i < parent1.genes.length; i++) {
    const gene1 = parent1.genes[i];
    const gene2 = parent2.genes[i];

    if (typeof gene1.value === "number" && typeof gene2.value === "number") {
      const newValue = weight * gene1.value + (1 - weight) * gene2.value;
      const clampedValue = Math.max(
        gene1.minValue ?? -Infinity,
        Math.min(gene1.maxValue ?? Infinity, newValue)
      );

      const newGene = cloneGene(gene1);
      newGene.value = clampedValue;
      genes.push(newGene);
    } else {
      genes.push(cloneGene(Math.random() < weight ? gene1 : gene2));
    }
  }

  if (stats) stats.totalCrossovers++;
  return createOffspring(parent1, parent2, genes, generation);
}

/**
 * Simulated Binary Crossover (SBX)
 * Maintains distribution of genes similar to binary crossover
 */
export function sbxCrossover(
  parent1: Chromosome,
  parent2: Chromosome,
  generation: number,
  config: CrossoverConfig = DEFAULT_CROSSOVER_CONFIG,
  stats?: CrossoverStats
): Chromosome {
  const genes: Gene[] = [];
  const eta = config.sbxEta;

  for (let i = 0; i < parent1.genes.length; i++) {
    const gene1 = parent1.genes[i];
    const gene2 = parent2.genes[i];

    if (typeof gene1.value === "number" && typeof gene2.value === "number") {
      const u = Math.random();
      let beta: number;

      if (u <= 0.5) {
        beta = Math.pow(2 * u, 1 / (eta + 1));
      } else {
        beta = Math.pow(1 / (2 * (1 - u)), 1 / (eta + 1));
      }

      const child1Value = 0.5 * ((1 + beta) * gene1.value + (1 - beta) * gene2.value);
      const clampedValue = Math.max(
        gene1.minValue ?? -Infinity,
        Math.min(gene1.maxValue ?? Infinity, child1Value)
      );

      const newGene = cloneGene(gene1);
      newGene.value = clampedValue;
      genes.push(newGene);
    } else {
      genes.push(cloneGene(Math.random() < 0.5 ? gene1 : gene2));
    }
  }

  if (stats) stats.totalCrossovers++;
  return createOffspring(parent1, parent2, genes, generation);
}

/**
 * Order Crossover (OX) - For ordered/permutation genes
 */
export function orderCrossover(
  parent1: Chromosome,
  parent2: Chromosome,
  generation: number,
  stats?: CrossoverStats
): Chromosome {
  const len = parent1.genes.length;
  const genes: (Gene | null)[] = new Array(len).fill(null);

  // Select random subsequence
  let start = Math.floor(Math.random() * len);
  let end = Math.floor(Math.random() * len);
  if (start > end) [start, end] = [end, start];

  // Copy subsequence from parent1
  for (let i = start; i <= end; i++) {
    genes[i] = cloneGene(parent1.genes[i]);
  }

  // Fill remaining from parent2 in order
  const usedValues = new Set(
    genes.filter((g): g is Gene => g !== null).map((g) => JSON.stringify(g.value))
  );

  let p2Index = 0;
  for (let i = 0; i < len; i++) {
    if (genes[i] !== null) continue;

    while (
      p2Index < len &&
      usedValues.has(JSON.stringify(parent2.genes[p2Index].value))
    ) {
      p2Index++;
    }

    if (p2Index < len) {
      genes[i] = cloneGene(parent2.genes[p2Index]);
      usedValues.add(JSON.stringify(parent2.genes[p2Index].value));
      p2Index++;
    }
  }

  // Fill any remaining nulls
  for (let i = 0; i < len; i++) {
    if (genes[i] === null) {
      genes[i] = cloneGene(parent1.genes[i]);
    }
  }

  if (stats) stats.totalCrossovers++;
  return createOffspring(parent1, parent2, genes as Gene[], generation);
}

// ============================================================================
// Mutation Operators
// ============================================================================

export interface MutationConfig {
  gaussianSigma: number;
  polynomialEta: number;
  creepRange: number;
  uniformRange: number;
  adaptiveRate: boolean;
  minMutationRate: number;
  maxMutationRate: number;
}

export const DEFAULT_MUTATION_CONFIG: MutationConfig = {
  gaussianSigma: 0.1,
  polynomialEta: 20,
  creepRange: 0.1,
  uniformRange: 0.2,
  adaptiveRate: true,
  minMutationRate: 0.01,
  maxMutationRate: 0.5,
};

/**
 * Point Mutation - Simple value change
 */
export function pointMutation(
  gene: Gene,
  config: MutationConfig = DEFAULT_MUTATION_CONFIG
): MutationRecord {
  const originalValue = gene.value;

  if (typeof gene.value === "number") {
    const range = (gene.maxValue ?? 1) - (gene.minValue ?? 0);
    const change = (Math.random() - 0.5) * range * config.uniformRange;
    gene.value = Math.max(
      gene.minValue ?? -Infinity,
      Math.min(gene.maxValue ?? Infinity, gene.value + change)
    );
  } else if (typeof gene.value === "boolean") {
    gene.value = !gene.value;
  } else if (gene.allowedValues) {
    const otherValues = gene.allowedValues.filter((v) => v !== gene.value);
    if (otherValues.length > 0) {
      gene.value = otherValues[Math.floor(Math.random() * otherValues.length)];
    }
  }

  return {
    geneId: gene.id,
    originalValue,
    newValue: gene.value,
    type: "point",
    timestamp: new Date(),
  };
}

/**
 * Gaussian Mutation - Adds Gaussian noise
 */
export function gaussianMutation(
  gene: Gene,
  config: MutationConfig = DEFAULT_MUTATION_CONFIG
): MutationRecord {
  const originalValue = gene.value;

  if (typeof gene.value === "number") {
    // Box-Muller transform for Gaussian random
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);

    const range = (gene.maxValue ?? 1) - (gene.minValue ?? 0);
    const noise = z * config.gaussianSigma * range;

    gene.value = Math.max(
      gene.minValue ?? -Infinity,
      Math.min(gene.maxValue ?? Infinity, gene.value + noise)
    );
  } else {
    // Fall back to point mutation for non-numeric
    return pointMutation(gene, config);
  }

  return {
    geneId: gene.id,
    originalValue,
    newValue: gene.value,
    type: "point",
    timestamp: new Date(),
  };
}

/**
 * Polynomial Mutation
 */
export function polynomialMutation(
  gene: Gene,
  config: MutationConfig = DEFAULT_MUTATION_CONFIG
): MutationRecord {
  const originalValue = gene.value;

  if (typeof gene.value === "number") {
    const eta = config.polynomialEta;
    const u = Math.random();
    let delta: number;

    if (u < 0.5) {
      delta = Math.pow(2 * u, 1 / (eta + 1)) - 1;
    } else {
      delta = 1 - Math.pow(2 * (1 - u), 1 / (eta + 1));
    }

    const range = (gene.maxValue ?? 1) - (gene.minValue ?? 0);
    gene.value = Math.max(
      gene.minValue ?? -Infinity,
      Math.min(gene.maxValue ?? Infinity, gene.value + delta * range)
    );
  } else {
    return pointMutation(gene, config);
  }

  return {
    geneId: gene.id,
    originalValue,
    newValue: gene.value,
    type: "point",
    timestamp: new Date(),
  };
}

/**
 * Swap Mutation - Swaps two genes in chromosome
 */
export function swapMutation(
  chromosome: Chromosome
): MutationRecord[] {
  if (chromosome.genes.length < 2) return [];

  const idx1 = Math.floor(Math.random() * chromosome.genes.length);
  let idx2 = Math.floor(Math.random() * chromosome.genes.length);
  while (idx2 === idx1) {
    idx2 = Math.floor(Math.random() * chromosome.genes.length);
  }

  const temp = chromosome.genes[idx1];
  chromosome.genes[idx1] = chromosome.genes[idx2];
  chromosome.genes[idx2] = temp;

  return [
    {
      geneId: chromosome.genes[idx1].id,
      originalValue: idx1,
      newValue: idx2,
      type: "inversion",
      timestamp: new Date(),
    },
    {
      geneId: chromosome.genes[idx2].id,
      originalValue: idx2,
      newValue: idx1,
      type: "inversion",
      timestamp: new Date(),
    },
  ];
}

/**
 * Scramble Mutation - Scrambles a subsequence of genes
 */
export function scrambleMutation(
  chromosome: Chromosome
): MutationRecord[] {
  const len = chromosome.genes.length;
  if (len < 2) return [];

  let start = Math.floor(Math.random() * len);
  let end = Math.floor(Math.random() * len);
  if (start > end) [start, end] = [end, start];

  const records: MutationRecord[] = [];

  // Fisher-Yates shuffle on subsequence
  for (let i = end; i > start; i--) {
    const j = start + Math.floor(Math.random() * (i - start + 1));
    const temp = chromosome.genes[i];
    chromosome.genes[i] = chromosome.genes[j];
    chromosome.genes[j] = temp;

    records.push({
      geneId: chromosome.genes[i].id,
      originalValue: i,
      newValue: j,
      type: "inversion",
      timestamp: new Date(),
    });
  }

  return records;
}

/**
 * Inversion Mutation - Reverses a subsequence
 */
export function inversionMutation(
  chromosome: Chromosome
): MutationRecord[] {
  const len = chromosome.genes.length;
  if (len < 2) return [];

  let start = Math.floor(Math.random() * len);
  let end = Math.floor(Math.random() * len);
  if (start > end) [start, end] = [end, start];

  const records: MutationRecord[] = [];

  // Reverse the subsequence
  while (start < end) {
    const temp = chromosome.genes[start];
    chromosome.genes[start] = chromosome.genes[end];
    chromosome.genes[end] = temp;

    records.push({
      geneId: chromosome.genes[start].id,
      originalValue: start,
      newValue: end,
      type: "inversion",
      timestamp: new Date(),
    });

    start++;
    end--;
  }

  return records;
}

/**
 * Creep Mutation - Small changes to numeric values
 */
export function creepMutation(
  gene: Gene,
  config: MutationConfig = DEFAULT_MUTATION_CONFIG
): MutationRecord {
  const originalValue = gene.value;

  if (typeof gene.value === "number") {
    const range = (gene.maxValue ?? 1) - (gene.minValue ?? 0);
    const creep = (Math.random() - 0.5) * 2 * config.creepRange * range;

    gene.value = Math.max(
      gene.minValue ?? -Infinity,
      Math.min(gene.maxValue ?? Infinity, gene.value + creep)
    );
  } else {
    return pointMutation(gene, config);
  }

  return {
    geneId: gene.id,
    originalValue,
    newValue: gene.value,
    type: "point",
    timestamp: new Date(),
  };
}

/**
 * Adaptive Mutation - Adjusts mutation strength based on fitness
 */
export function adaptiveMutation(
  chromosome: Chromosome,
  populationBestFitness: number,
  populationAvgFitness: number,
  config: MutationConfig = DEFAULT_MUTATION_CONFIG
): MutationRecord[] {
  const records: MutationRecord[] = [];

  // Calculate adaptive rate based on relative fitness
  let adaptiveRate: number;
  if (chromosome.fitness >= populationAvgFitness) {
    // High fitness: lower mutation rate
    adaptiveRate = config.minMutationRate;
  } else {
    // Low fitness: higher mutation rate
    const relativeUnfitness =
      (populationAvgFitness - chromosome.fitness) /
      (populationBestFitness - populationAvgFitness + 0.001);
    adaptiveRate = Math.min(
      config.maxMutationRate,
      config.minMutationRate + relativeUnfitness * (config.maxMutationRate - config.minMutationRate)
    );
  }

  for (const gene of chromosome.genes) {
    if (Math.random() < adaptiveRate) {
      const record = gaussianMutation(gene, config);
      records.push(record);
    }
  }

  return records;
}

/**
 * Boundary Mutation - Sets gene to min or max boundary
 */
export function boundaryMutation(gene: Gene): MutationRecord {
  const originalValue = gene.value;

  if (typeof gene.value === "number" && gene.minValue !== undefined && gene.maxValue !== undefined) {
    gene.value = Math.random() < 0.5 ? gene.minValue : gene.maxValue;
  } else if (typeof gene.value === "boolean") {
    gene.value = !gene.value;
  } else if (gene.allowedValues && gene.allowedValues.length > 0) {
    gene.value = Math.random() < 0.5 ? gene.allowedValues[0] : gene.allowedValues[gene.allowedValues.length - 1];
  }

  return {
    geneId: gene.id,
    originalValue,
    newValue: gene.value,
    type: "point",
    timestamp: new Date(),
  };
}

// ============================================================================
// Helper Functions
// ============================================================================

function cloneGene(gene: Gene): Gene {
  return {
    ...gene,
    id: createGeneId(),
  };
}

function createOffspring(
  parent1: Chromosome,
  parent2: Chromosome,
  genes: Gene[],
  generation: number
): Chromosome {
  const name = `${parent1.name.split("-")[0]}-${parent2.name.split("-")[0]}-G${generation}`;

  return {
    id: createChromosomeId(),
    name,
    genes,
    fitness: 0,
    fitnessComponents: createDefaultFitnessComponents(),
    generation,
    parentIds: [parent1.id, parent2.id],
    mutations: [],
    createdAt: new Date(),
    isElite: false,
  };
}

// ============================================================================
// Operator Application Functions
// ============================================================================

export type SelectionOperator = "tournament" | "roulette" | "rank" | "elitist" | "boltzmann" | "sus";
export type CrossoverOperator = "single_point" | "two_point" | "uniform" | "blend" | "arithmetic" | "sbx" | "order";
export type MutationOperator = "point" | "gaussian" | "polynomial" | "swap" | "scramble" | "inversion" | "creep" | "boundary" | "adaptive";

/**
 * Apply selection operator
 */
export function applySelection(
  operator: SelectionOperator,
  chromosomes: Chromosome[],
  config: SelectionConfig = DEFAULT_SELECTION_CONFIG,
  stats?: SelectionStats
): Chromosome {
  switch (operator) {
    case "tournament":
      return tournamentSelect(chromosomes, config, stats);
    case "roulette":
      return rouletteSelect(chromosomes, stats);
    case "rank":
      return rankSelect(chromosomes, config, stats);
    case "elitist":
      return elitistSelect(chromosomes, 0.2, stats);
    case "boltzmann":
      return boltzmannSelect(chromosomes, config.boltzmannTemperature, stats);
    case "sus":
      return stochasticUniversalSampling(chromosomes, 1, stats)[0];
    default:
      return tournamentSelect(chromosomes, config, stats);
  }
}

/**
 * Apply crossover operator
 */
export function applyCrossover(
  operator: CrossoverOperator,
  parent1: Chromosome,
  parent2: Chromosome,
  generation: number,
  config: CrossoverConfig = DEFAULT_CROSSOVER_CONFIG,
  stats?: CrossoverStats
): Chromosome {
  switch (operator) {
    case "single_point":
      return singlePointCrossover(parent1, parent2, generation, stats);
    case "two_point":
      return twoPointCrossover(parent1, parent2, generation, stats);
    case "uniform":
      return uniformCrossover(parent1, parent2, generation, config, stats);
    case "blend":
      return blendCrossover(parent1, parent2, generation, config, stats);
    case "arithmetic":
      return arithmeticCrossover(parent1, parent2, generation, config, stats);
    case "sbx":
      return sbxCrossover(parent1, parent2, generation, config, stats);
    case "order":
      return orderCrossover(parent1, parent2, generation, stats);
    default:
      return uniformCrossover(parent1, parent2, generation, config, stats);
  }
}

/**
 * Apply mutation operator to a gene
 */
export function applyGeneMutation(
  operator: MutationOperator,
  gene: Gene,
  config: MutationConfig = DEFAULT_MUTATION_CONFIG
): MutationRecord {
  switch (operator) {
    case "point":
      return pointMutation(gene, config);
    case "gaussian":
      return gaussianMutation(gene, config);
    case "polynomial":
      return polynomialMutation(gene, config);
    case "creep":
      return creepMutation(gene, config);
    case "boundary":
      return boundaryMutation(gene);
    default:
      return pointMutation(gene, config);
  }
}

/**
 * Apply mutation operator to a chromosome
 */
export function applyChromosomeMutation(
  operator: MutationOperator,
  chromosome: Chromosome,
  config: MutationConfig = DEFAULT_MUTATION_CONFIG,
  populationBestFitness?: number,
  populationAvgFitness?: number
): MutationRecord[] {
  switch (operator) {
    case "swap":
      return swapMutation(chromosome);
    case "scramble":
      return scrambleMutation(chromosome);
    case "inversion":
      return inversionMutation(chromosome);
    case "adaptive":
      return adaptiveMutation(
        chromosome,
        populationBestFitness ?? 1,
        populationAvgFitness ?? 0.5,
        config
      );
    default: {
      const records: MutationRecord[] = [];
      for (const gene of chromosome.genes) {
        if (Math.random() < gene.mutationRate) {
          records.push(applyGeneMutation(operator, gene, config));
        }
      }
      return records;
    }
  }
}
