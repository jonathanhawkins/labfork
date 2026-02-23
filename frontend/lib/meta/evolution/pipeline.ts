/**
 * Evolution Pipeline
 *
 * Orchestrates continuous evolution cycles with monitoring, callbacks,
 * and state management. Supports pause/resume and convergence detection.
 */

import {
  Population,
  PopulationStatus,
  Chromosome,
  GenerationStats,
  EvolutionConfig,
  EvolutionResult,
  MutationRecord,
  DEFAULT_EVOLUTION_CONFIG,
  createPopulationId,
  createEmptyGenerationStats,
  calculateFitness,
} from "./types";
import {
  OperatorStats,
  createEmptyOperatorStats,
  applySelection,
  applyCrossover,
  applyChromosomeMutation,
  SelectionOperator,
  CrossoverOperator,
  MutationOperator,
  SelectionConfig,
  CrossoverConfig,
  MutationConfig,
  DEFAULT_SELECTION_CONFIG,
  DEFAULT_CROSSOVER_CONFIG,
  DEFAULT_MUTATION_CONFIG,
} from "./operators";
import { EvolutionEngine } from "./engine";

// ============================================================================
// Pipeline Configuration
// ============================================================================

export interface PipelineConfig extends EvolutionConfig {
  /** Run mode */
  mode: PipelineMode;
  /** Generations per batch (for batch mode) */
  batchSize: number;
  /** Delay between generations (ms) */
  generationDelay: number;
  /** Enable auto-save */
  autoSave: boolean;
  /** Auto-save interval (generations) */
  autoSaveInterval: number;
  /** Selection operator */
  selectionOperator: SelectionOperator;
  /** Crossover operator */
  crossoverOperator: CrossoverOperator;
  /** Mutation operator */
  mutationOperator: MutationOperator;
  /** Operator configs */
  selectionConfig: SelectionConfig;
  crossoverConfig: CrossoverConfig;
  mutationConfig: MutationConfig;
  /** Niching radius */
  nichingRadius: number;
  /** Minimum population diversity */
  minDiversity: number;
  /** Enable island model */
  enableIslands: boolean;
  /** Number of islands */
  islandCount: number;
  /** Migration interval */
  migrationInterval: number;
  /** Migration rate */
  migrationRate: number;
}

export type PipelineMode = "continuous" | "batch" | "step";

export const DEFAULT_PIPELINE_CONFIG: PipelineConfig = {
  ...DEFAULT_EVOLUTION_CONFIG,
  mode: "continuous",
  batchSize: 10,
  generationDelay: 0,
  autoSave: true,
  autoSaveInterval: 10,
  selectionOperator: "tournament",
  crossoverOperator: "uniform",
  mutationOperator: "gaussian",
  selectionConfig: DEFAULT_SELECTION_CONFIG,
  crossoverConfig: DEFAULT_CROSSOVER_CONFIG,
  mutationConfig: DEFAULT_MUTATION_CONFIG,
  nichingRadius: 0.1,
  minDiversity: 0.05,
  enableIslands: false,
  islandCount: 4,
  migrationInterval: 10,
  migrationRate: 0.1,
};

// ============================================================================
// Pipeline Events
// ============================================================================

export interface PipelineEvents {
  onGenerationStart?: (generation: number, population: Population) => void;
  onGenerationEnd?: (generation: number, stats: GenerationStats, population: Population) => void;
  onEliteSelected?: (elite: Chromosome[]) => void;
  onOffspringCreated?: (offspring: Chromosome, parents: [Chromosome, Chromosome]) => void;
  onMutation?: (chromosome: Chromosome, mutations: MutationRecord[]) => void;
  onConvergence?: (generation: number, population: Population) => void;
  onPaused?: (generation: number) => void;
  onResumed?: (generation: number) => void;
  onCompleted?: (result: EvolutionResult) => void;
  onError?: (error: Error, generation: number) => void;
  onSave?: (generation: number, population: Population) => void;
  onDiversityWarning?: (diversity: number, threshold: number) => void;
  onIslandMigration?: (fromIsland: number, toIsland: number, migrants: Chromosome[]) => void;
}

// ============================================================================
// Pipeline State
// ============================================================================

export interface PipelineState {
  /** Current status */
  status: PipelineStatus;
  /** Current generation */
  currentGeneration: number;
  /** Total elapsed time (ms) */
  elapsedTime: number;
  /** Start time */
  startTime?: Date;
  /** Pause time */
  pauseTime?: Date;
  /** Total paused time (ms) */
  pausedTime: number;
  /** Operator statistics */
  operatorStats: OperatorStats;
  /** Generation snapshots */
  snapshots: Map<number, PopulationSnapshot>;
  /** Checkpoints */
  checkpoints: PipelineCheckpoint[];
  /** Error count */
  errorCount: number;
  /** Last error */
  lastError?: Error;
}

export type PipelineStatus = "idle" | "running" | "paused" | "completed" | "error";

export interface PopulationSnapshot {
  generation: number;
  timestamp: Date;
  bestFitness: number;
  averageFitness: number;
  diversity: number;
  bestChromosome: Chromosome;
  topChromosomes: Chromosome[];
}

export interface PipelineCheckpoint {
  generation: number;
  timestamp: Date;
  population: Population;
  state: Partial<PipelineState>;
}

// ============================================================================
// Evolution Pipeline
// ============================================================================

export class EvolutionPipeline {
  private config: PipelineConfig;
  private events: PipelineEvents;
  private state: PipelineState;
  private population: Population | null;
  private engine: EvolutionEngine;
  private abortController: AbortController | null;
  private islands: Population[];

  constructor(
    config: Partial<PipelineConfig> = {},
    events: PipelineEvents = {}
  ) {
    this.config = { ...DEFAULT_PIPELINE_CONFIG, ...config };
    this.events = events;
    this.state = this.createInitialState();
    this.population = null;
    this.engine = new EvolutionEngine(this.config);
    this.abortController = null;
    this.islands = [];
  }

  /**
   * Initialize population and start pipeline
   */
  initialize(population: Population): void {
    this.population = population;
    this.state = this.createInitialState();
    this.state.currentGeneration = population.generation;

    if (this.config.enableIslands) {
      this.initializeIslands();
    }
  }

  /**
   * Start continuous evolution
   */
  async start(): Promise<EvolutionResult> {
    if (!this.population) {
      throw new Error("Pipeline not initialized. Call initialize() first.");
    }

    this.state.status = "running";
    this.state.startTime = new Date();
    this.abortController = new AbortController();

    try {
      switch (this.config.mode) {
        case "continuous":
          return await this.runContinuous();
        case "batch":
          return await this.runBatch();
        case "step":
          return await this.runStep();
        default:
          return await this.runContinuous();
      }
    } catch (error) {
      this.state.status = "error";
      this.state.lastError = error as Error;
      this.state.errorCount++;
      this.events.onError?.(error as Error, this.state.currentGeneration);
      throw error;
    }
  }

  /**
   * Pause evolution
   */
  pause(): void {
    if (this.state.status !== "running") return;

    this.state.status = "paused";
    this.state.pauseTime = new Date();
    this.abortController?.abort();
    this.events.onPaused?.(this.state.currentGeneration);
  }

  /**
   * Resume evolution
   */
  async resume(): Promise<EvolutionResult> {
    if (this.state.status !== "paused" || !this.population) {
      throw new Error("Cannot resume: pipeline is not paused");
    }

    // Calculate paused duration
    if (this.state.pauseTime) {
      this.state.pausedTime += Date.now() - this.state.pauseTime.getTime();
      this.state.pauseTime = undefined;
    }

    this.state.status = "running";
    this.abortController = new AbortController();
    this.events.onResumed?.(this.state.currentGeneration);

    return this.start();
  }

  /**
   * Stop evolution
   */
  stop(): void {
    this.abortController?.abort();
    this.state.status = "completed";
  }

  /**
   * Get current state
   */
  getState(): PipelineState {
    return { ...this.state };
  }

  /**
   * Get current population
   */
  getPopulation(): Population | null {
    return this.population;
  }

  /**
   * Create checkpoint
   */
  createCheckpoint(): PipelineCheckpoint | null {
    if (!this.population) return null;

    const checkpoint: PipelineCheckpoint = {
      generation: this.state.currentGeneration,
      timestamp: new Date(),
      population: this.clonePopulation(this.population),
      state: {
        currentGeneration: this.state.currentGeneration,
        elapsedTime: this.state.elapsedTime,
        operatorStats: { ...this.state.operatorStats },
      },
    };

    this.state.checkpoints.push(checkpoint);
    return checkpoint;
  }

  /**
   * Restore from checkpoint
   */
  restoreFromCheckpoint(checkpoint: PipelineCheckpoint): void {
    this.population = this.clonePopulation(checkpoint.population);
    this.state.currentGeneration = checkpoint.state.currentGeneration ?? 0;
    this.state.elapsedTime = checkpoint.state.elapsedTime ?? 0;
    if (checkpoint.state.operatorStats) {
      this.state.operatorStats = { ...checkpoint.state.operatorStats };
    }
  }

  /**
   * Get snapshot for a generation
   */
  getSnapshot(generation: number): PopulationSnapshot | undefined {
    return this.state.snapshots.get(generation);
  }

  /**
   * Evolve single step (one generation)
   */
  async evolveStep(): Promise<GenerationStats | null> {
    if (!this.population || this.state.status === "paused") {
      return null;
    }

    const generation = this.state.currentGeneration + 1;
    this.events.onGenerationStart?.(generation, this.population);

    // Perform evolution
    const newChromosomes: Chromosome[] = [];
    const eliteCount = Math.floor(this.population.size * this.config.elitePercentage);

    // Preserve elite
    const elite = this.population.chromosomes
      .slice(0, eliteCount)
      .map((chr) => ({
        ...chr,
        isElite: true,
        generation,
      }));
    newChromosomes.push(...elite);
    this.events.onEliteSelected?.(elite);

    // Generate offspring
    let mutationCount = 0;
    let crossoverCount = 0;

    while (newChromosomes.length < this.population.size) {
      // Selection
      const parent1 = applySelection(
        this.config.selectionOperator,
        this.population.chromosomes,
        this.config.selectionConfig,
        this.state.operatorStats.selection
      );
      const parent2 = applySelection(
        this.config.selectionOperator,
        this.population.chromosomes,
        this.config.selectionConfig,
        this.state.operatorStats.selection
      );

      // Crossover
      let offspring: Chromosome;
      if (Math.random() < this.config.crossoverRate) {
        offspring = applyCrossover(
          this.config.crossoverOperator,
          parent1,
          parent2,
          generation,
          this.config.crossoverConfig,
          this.state.operatorStats.crossover
        );
        crossoverCount++;
        this.events.onOffspringCreated?.(offspring, [parent1, parent2]);
      } else {
        offspring = this.cloneChromosome(parent1, generation);
      }

      // Mutation
      const mutationRate = this.config.adaptiveMutation
        ? this.getAdaptiveMutationRate()
        : this.config.mutationRate;

      if (Math.random() < mutationRate) {
        const mutations = applyChromosomeMutation(
          this.config.mutationOperator,
          offspring,
          this.config.mutationConfig,
          this.population.chromosomes[0].fitness,
          this.population.averageFitness
        );
        offspring.mutations.push(...mutations);
        mutationCount++;
        this.events.onMutation?.(offspring, mutations);

        // Update mutation stats
        for (const mutation of mutations) {
          const typeCount = this.state.operatorStats.mutation.mutationTypeDistribution.get(mutation.type) || 0;
          this.state.operatorStats.mutation.mutationTypeDistribution.set(mutation.type, typeCount + 1);
        }
        this.state.operatorStats.mutation.totalMutations += mutations.length;
      }

      // Evaluate fitness
      this.evaluateFitness(offspring);
      newChromosomes.push(offspring);
    }

    // Apply niching if enabled
    if (this.config.enableNiching) {
      this.applyNiching(newChromosomes);
    }

    // Sort by fitness
    newChromosomes.sort((a, b) => b.fitness - a.fitness);

    // Create stats
    const stats = this.calculateStats(newChromosomes, generation);
    stats.mutationCount = mutationCount;
    stats.crossoverCount = crossoverCount;
    stats.eliteCount = eliteCount;

    // Check diversity
    if (stats.fitnessStdDev < this.config.minDiversity) {
      this.events.onDiversityWarning?.(stats.fitnessStdDev, this.config.minDiversity);
    }

    // Check convergence
    const converged = this.checkConvergence(stats);

    // Update population
    this.population = {
      ...this.population,
      generation,
      chromosomes: newChromosomes,
      bestChromosomeId: newChromosomes[0].id,
      averageFitness: stats.averageFitness,
      fitnessDiversity: stats.fitnessStdDev,
      generationHistory: [...this.population.generationHistory, stats],
      lastEvolved: new Date(),
      status: converged ? "converged" : "evolving",
    };

    // Update state
    this.state.currentGeneration = generation;
    this.updateElapsedTime();

    // Create snapshot
    this.createSnapshot(generation);

    // Auto-save
    if (this.config.autoSave && generation % this.config.autoSaveInterval === 0) {
      this.createCheckpoint();
      this.events.onSave?.(generation, this.population);
    }

    // Handle island migration
    if (this.config.enableIslands && generation % this.config.migrationInterval === 0) {
      this.performMigration();
    }

    this.events.onGenerationEnd?.(generation, stats, this.population);

    if (converged) {
      this.events.onConvergence?.(generation, this.population);
    }

    return stats;
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private createInitialState(): PipelineState {
    return {
      status: "idle",
      currentGeneration: 0,
      elapsedTime: 0,
      pausedTime: 0,
      operatorStats: createEmptyOperatorStats(),
      snapshots: new Map(),
      checkpoints: [],
      errorCount: 0,
    };
  }

  private async runContinuous(): Promise<EvolutionResult> {
    while (
      this.state.currentGeneration < this.config.maxGenerations &&
      this.state.status === "running" &&
      this.population?.status !== "converged"
    ) {
      if (this.abortController?.signal.aborted) break;

      await this.evolveStep();

      if (this.config.generationDelay > 0) {
        await this.delay(this.config.generationDelay);
      }
    }

    return this.createResult();
  }

  private async runBatch(): Promise<EvolutionResult> {
    let batchCount = 0;

    while (
      this.state.currentGeneration < this.config.maxGenerations &&
      this.state.status === "running" &&
      this.population?.status !== "converged"
    ) {
      if (this.abortController?.signal.aborted) break;

      await this.evolveStep();
      batchCount++;

      if (batchCount >= this.config.batchSize) {
        break;
      }
    }

    return this.createResult();
  }

  private async runStep(): Promise<EvolutionResult> {
    await this.evolveStep();
    return this.createResult();
  }

  private createResult(): EvolutionResult {
    if (!this.population) {
      throw new Error("No population available");
    }

    this.state.status = "completed";

    const result: EvolutionResult = {
      population: this.population,
      bestChromosome: this.population.chromosomes[0],
      bestPerGeneration: this.getBestPerGeneration(),
      totalGenerations: this.state.currentGeneration,
      converged: this.population.status === "converged",
      durationMs: this.state.elapsedTime,
      finalStats: this.population.generationHistory[this.population.generationHistory.length - 1],
    };

    this.events.onCompleted?.(result);
    return result;
  }

  private getBestPerGeneration(): Chromosome[] {
    const best: Chromosome[] = [];
    for (const [, snapshot] of Array.from(this.state.snapshots)) {
      best.push(snapshot.bestChromosome);
    }
    return best.sort((a, b) => a.generation - b.generation);
  }

  private calculateStats(chromosomes: Chromosome[], generation: number): GenerationStats {
    const fitnesses = chromosomes.map((c) => c.fitness);
    const sum = fitnesses.reduce((a, b) => a + b, 0);
    const avg = sum / fitnesses.length;

    const variance = fitnesses.reduce((a, b) => a + Math.pow(b - avg, 2), 0) / fitnesses.length;
    const stdDev = Math.sqrt(variance);

    return {
      generation,
      bestFitness: Math.max(...fitnesses),
      averageFitness: avg,
      worstFitness: Math.min(...fitnesses),
      fitnessStdDev: stdDev,
      mutationCount: 0,
      crossoverCount: 0,
      eliteCount: 0,
      timestamp: new Date(),
    };
  }

  private checkConvergence(stats: GenerationStats): boolean {
    if (!this.population || this.population.generationHistory.length < this.config.convergenceWindow) {
      return false;
    }

    const recentStats = this.population.generationHistory.slice(-this.config.convergenceWindow);
    const fitnessChanges = recentStats.slice(1).map((s, i) =>
      Math.abs(s.bestFitness - recentStats[i].bestFitness)
    );

    const maxChange = Math.max(...fitnessChanges);
    return maxChange < this.config.convergenceThreshold;
  }

  private getAdaptiveMutationRate(): number {
    if (!this.population) return this.config.mutationRate;

    const diversity = this.population.fitnessDiversity;
    const baseMutation = this.config.mutationRate;

    if (diversity < 0.05) {
      return Math.min(baseMutation * 2, 0.5);
    } else if (diversity < 0.1) {
      return baseMutation * 1.5;
    }
    return baseMutation;
  }

  private evaluateFitness(chromosome: Chromosome): void {
    // Simple fitness evaluation based on gene values
    const geneMap = new Map<string, number | string | boolean>();
    for (const gene of chromosome.genes) {
      geneMap.set(gene.name, gene.value);
    }

    let quality = 0.5;
    let efficiency = 0.6;
    let novelty = 0.4;
    let feasibility = 0.7;
    let compatibility = 0.6;

    // Architecture bonuses
    if (geneMap.get("Uses Transformer") === true) quality += 0.1;
    if (geneMap.get("Uses Attention") === true) quality += 0.08;
    if (geneMap.get("Uses Diffusion") === true) {
      quality += 0.12;
      efficiency -= 0.1;
    }

    const modelSize = geneMap.get("Model Size");
    if (modelSize === "small") efficiency += 0.2;
    if (modelSize === "large") {
      quality += 0.1;
      efficiency -= 0.1;
    }
    if (modelSize === "xlarge") {
      quality += 0.15;
      efficiency -= 0.2;
      feasibility -= 0.2;
    }

    // Training methodology
    if (geneMap.get("Uses Self-Supervised") === true) compatibility += 0.15;
    if (geneMap.get("Uses Contrastive Learning") === true) compatibility += 0.1;

    // Calculate novelty from unique combinations
    const architectureCount = [
      geneMap.get("Uses Transformer"),
      geneMap.get("Uses Attention"),
      geneMap.get("Uses Diffusion"),
      geneMap.get("Uses VAE"),
    ].filter(Boolean).length;
    novelty += architectureCount * 0.1;

    chromosome.fitnessComponents = {
      quality: Math.max(0, Math.min(1, quality)),
      efficiency: Math.max(0, Math.min(1, efficiency)),
      novelty: Math.max(0, Math.min(1, novelty)),
      feasibility: Math.max(0, Math.min(1, feasibility)),
      compatibility: Math.max(0, Math.min(1, compatibility)),
    };

    chromosome.fitness = calculateFitness(chromosome.fitnessComponents, this.config.fitnessWeights);
  }

  private cloneChromosome(chr: Chromosome, generation: number): Chromosome {
    return {
      ...chr,
      id: `chr-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      genes: chr.genes.map((g) => ({ ...g, id: `gene-${Date.now()}-${Math.random().toString(36).slice(2, 8)}` })),
      generation,
      parentIds: [chr.id],
      mutations: [],
      createdAt: new Date(),
      isElite: false,
    };
  }

  private clonePopulation(pop: Population): Population {
    return {
      ...pop,
      chromosomes: pop.chromosomes.map((chr) => ({
        ...chr,
        genes: chr.genes.map((g) => ({ ...g })),
        fitnessComponents: { ...chr.fitnessComponents },
        mutations: [...chr.mutations],
      })),
      generationHistory: pop.generationHistory.map((s) => ({ ...s })),
      config: { ...pop.config },
    };
  }

  private createSnapshot(generation: number): void {
    if (!this.population) return;

    const snapshot: PopulationSnapshot = {
      generation,
      timestamp: new Date(),
      bestFitness: this.population.chromosomes[0].fitness,
      averageFitness: this.population.averageFitness,
      diversity: this.population.fitnessDiversity,
      bestChromosome: { ...this.population.chromosomes[0] },
      topChromosomes: this.population.chromosomes.slice(0, 5).map((c) => ({ ...c })),
    };

    this.state.snapshots.set(generation, snapshot);
  }

  private updateElapsedTime(): void {
    if (this.state.startTime) {
      this.state.elapsedTime = Date.now() - this.state.startTime.getTime() - this.state.pausedTime;
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private applyNiching(chromosomes: Chromosome[]): void {
    // Fitness sharing for niching
    for (let i = 0; i < chromosomes.length; i++) {
      let nicheCount = 0;
      for (let j = 0; j < chromosomes.length; j++) {
        if (i === j) continue;
        const distance = this.calculateGeneticDistance(chromosomes[i], chromosomes[j]);
        if (distance < this.config.nichingRadius) {
          nicheCount += 1 - distance / this.config.nichingRadius;
        }
      }
      if (nicheCount > 0) {
        chromosomes[i].fitness /= nicheCount + 1;
      }
    }
  }

  private calculateGeneticDistance(chr1: Chromosome, chr2: Chromosome): number {
    let distance = 0;
    for (let i = 0; i < chr1.genes.length; i++) {
      const g1 = chr1.genes[i];
      const g2 = chr2.genes[i];

      if (typeof g1.value === "number" && typeof g2.value === "number") {
        const range = (g1.maxValue ?? 1) - (g1.minValue ?? 0);
        distance += Math.abs(g1.value - g2.value) / range;
      } else if (g1.value !== g2.value) {
        distance += 1;
      }
    }
    return distance / chr1.genes.length;
  }

  private initializeIslands(): void {
    if (!this.population) return;

    const islandSize = Math.floor(this.population.size / this.config.islandCount);
    this.islands = [];

    for (let i = 0; i < this.config.islandCount; i++) {
      const start = i * islandSize;
      const end = i === this.config.islandCount - 1 ? this.population.size : start + islandSize;

      this.islands.push({
        ...this.population,
        id: createPopulationId(),
        name: `${this.population.name}-Island${i}`,
        chromosomes: this.population.chromosomes.slice(start, end).map((c) => ({ ...c })),
        size: end - start,
      });
    }
  }

  private performMigration(): void {
    if (this.islands.length < 2) return;

    const migrantCount = Math.ceil(this.islands[0].size * this.config.migrationRate);

    for (let i = 0; i < this.islands.length; i++) {
      const targetIsland = (i + 1) % this.islands.length;
      const migrants = this.islands[i].chromosomes.slice(0, migrantCount);

      // Remove worst from target, add migrants
      this.islands[targetIsland].chromosomes = this.islands[targetIsland].chromosomes
        .slice(0, -migrantCount)
        .concat(migrants.map((m) => ({ ...m })));

      this.events.onIslandMigration?.(i, targetIsland, migrants);
    }

    // Merge islands back to main population
    if (this.population) {
      this.population.chromosomes = this.islands.flatMap((island) => island.chromosomes);
      this.population.chromosomes.sort((a, b) => b.fitness - a.fitness);
    }
  }
}

// ============================================================================
// Pipeline Factory
// ============================================================================

export function createEvolutionPipeline(
  config?: Partial<PipelineConfig>,
  events?: PipelineEvents
): EvolutionPipeline {
  return new EvolutionPipeline(config, events);
}
