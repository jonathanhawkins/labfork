/**
 * Genetic Evolution Engine
 *
 * Core engine for evolving technique combinations using genetic algorithms.
 * Handles population management, selection, crossover, mutation, and fitness evaluation.
 */

import {
  Gene,
  GeneType,
  Chromosome,
  Population,
  PopulationStatus,
  GenerationStats,
  EvolutionConfig,
  EvolutionResult,
  MutationRecord,
  MutationType,
  FitnessComponents,
  ChromosomeLineage,
  LineageNode,
  GeneTemplate,
  STANDARD_GENE_TEMPLATES,
  DEFAULT_EVOLUTION_CONFIG,
  createGeneId,
  createChromosomeId,
  createPopulationId,
  createGeneFromTemplate,
  createDefaultFitnessComponents,
  calculateFitness,
  createEmptyGenerationStats,
} from "./types";
import { TechniqueNode, isTechniqueNode } from "../knowledge-graph";

/**
 * Genetic Evolution Engine
 */
export class EvolutionEngine {
  private config: EvolutionConfig;
  private fitnessCache: Map<string, FitnessComponents>;

  constructor(config: Partial<EvolutionConfig> = {}) {
    this.config = { ...DEFAULT_EVOLUTION_CONFIG, ...config };
    this.fitnessCache = new Map();
  }

  /**
   * Initialize a new population
   */
  initializePopulation(
    name: string,
    seedTechniques?: TechniqueNode[]
  ): Population {
    const chromosomes: Chromosome[] = [];

    // Create chromosomes from seed techniques
    if (seedTechniques && seedTechniques.length > 0) {
      for (const tech of seedTechniques.slice(0, this.config.populationSize)) {
        chromosomes.push(this.encodeToChromosome(tech));
      }
    }

    // Fill remaining with random chromosomes
    while (chromosomes.length < this.config.populationSize) {
      chromosomes.push(this.createRandomChromosome());
    }

    // Evaluate initial fitness
    for (const chr of chromosomes) {
      this.evaluateFitness(chr);
    }

    // Sort by fitness
    chromosomes.sort((a, b) => b.fitness - a.fitness);

    const stats = this.calculateGenerationStats(chromosomes, 0);

    return {
      id: createPopulationId(),
      name,
      generation: 0,
      chromosomes,
      size: chromosomes.length,
      bestChromosomeId: chromosomes[0].id,
      averageFitness: stats.averageFitness,
      fitnessDiversity: stats.fitnessStdDev,
      generationHistory: [stats],
      config: this.config,
      createdAt: new Date(),
      lastEvolved: new Date(),
      status: "initializing",
    };
  }

  /**
   * Evolve population for one generation
   */
  evolveGeneration(population: Population): Population {
    const newChromosomes: Chromosome[] = [];
    const eliteCount = Math.floor(
      population.size * this.config.elitePercentage
    );

    // Preserve elite chromosomes
    const elite = population.chromosomes
      .slice(0, eliteCount)
      .map((chr) => ({
        ...chr,
        isElite: true,
        generation: population.generation + 1,
      }));
    newChromosomes.push(...elite);

    // Generate new offspring
    let mutationCount = 0;
    let crossoverCount = 0;

    while (newChromosomes.length < population.size) {
      // Select parents
      const parent1 = this.selectParent(population.chromosomes);
      const parent2 = this.selectParent(population.chromosomes);

      // Crossover
      let offspring: Chromosome;
      if (Math.random() < this.config.crossoverRate) {
        offspring = this.crossover(parent1, parent2, population.generation + 1);
        crossoverCount++;
      } else {
        offspring = this.cloneChromosome(parent1, population.generation + 1);
      }

      // Mutation
      const mutationRate = this.config.adaptiveMutation
        ? this.getAdaptiveMutationRate(population)
        : this.config.mutationRate;

      if (Math.random() < mutationRate) {
        this.mutate(offspring);
        mutationCount++;
      }

      // Evaluate fitness
      this.evaluateFitness(offspring);

      newChromosomes.push(offspring);
    }

    // Trim to exact size
    newChromosomes.length = population.size;

    // Sort by fitness
    newChromosomes.sort((a, b) => b.fitness - a.fitness);

    // Calculate stats
    const stats = this.calculateGenerationStats(
      newChromosomes,
      population.generation + 1
    );
    stats.mutationCount = mutationCount;
    stats.crossoverCount = crossoverCount;
    stats.eliteCount = eliteCount;

    // Check convergence
    const converged = this.checkConvergence(population, stats);

    return {
      ...population,
      generation: population.generation + 1,
      chromosomes: newChromosomes,
      bestChromosomeId: newChromosomes[0].id,
      averageFitness: stats.averageFitness,
      fitnessDiversity: stats.fitnessStdDev,
      generationHistory: [...population.generationHistory, stats],
      lastEvolved: new Date(),
      status: converged ? "converged" : "evolving",
    };
  }

  /**
   * Run full evolution
   */
  runEvolution(population: Population): EvolutionResult {
    const startTime = Date.now();
    let currentPopulation = { ...population, status: "evolving" as PopulationStatus };
    const bestPerGeneration: Chromosome[] = [];

    // Store initial best
    bestPerGeneration.push(currentPopulation.chromosomes[0]);

    // Evolve for max generations or until convergence
    while (
      currentPopulation.generation < this.config.maxGenerations &&
      currentPopulation.status === "evolving"
    ) {
      currentPopulation = this.evolveGeneration(currentPopulation);
      bestPerGeneration.push(currentPopulation.chromosomes[0]);
    }

    if (currentPopulation.status !== "converged") {
      currentPopulation.status = "completed";
    }

    return {
      population: currentPopulation,
      bestChromosome: currentPopulation.chromosomes[0],
      bestPerGeneration,
      totalGenerations: currentPopulation.generation,
      converged: currentPopulation.status === "converged",
      durationMs: Date.now() - startTime,
      finalStats:
        currentPopulation.generationHistory[
          currentPopulation.generationHistory.length - 1
        ],
    };
  }

  /**
   * Encode a technique into a chromosome
   */
  encodeToChromosome(technique: TechniqueNode): Chromosome {
    const genes: Gene[] = [];

    // Extract genes from technique properties
    for (const template of STANDARD_GENE_TEMPLATES) {
      let value = template.defaultValue;

      // Map technique properties to gene values
      switch (template.id) {
        case "uses_transformer":
          value =
            technique.architecture?.toLowerCase().includes("transformer") ||
            technique.tags.some((t) => t.toLowerCase().includes("transformer"));
          break;
        case "uses_attention":
          value =
            technique.tags.some((t) => t.toLowerCase().includes("attention")) ||
            technique.architecture?.toLowerCase().includes("attention");
          break;
        case "uses_diffusion":
          value =
            technique.architecture?.toLowerCase().includes("diffusion") ||
            technique.tags.some((t) => t.toLowerCase().includes("diffusion"));
          break;
        case "uses_vae":
          value =
            technique.architecture?.toLowerCase().includes("vae") ||
            technique.tags.some((t) => t.toLowerCase().includes("vae"));
          break;
        case "uses_self_supervised":
          value = technique.tags.some((t) =>
            t.toLowerCase().includes("self-supervised")
          );
          break;
        case "uses_contrastive":
          value = technique.tags.some((t) =>
            t.toLowerCase().includes("contrastive")
          );
          break;
        case "model_size":
          if (technique.complexity === "simple") value = "small";
          else if (technique.complexity === "moderate") value = "medium";
          else if (technique.complexity === "complex") value = "large";
          else value = "xlarge";
          break;
      }

      const gene = createGeneFromTemplate(template, value);
      gene.sourceId = technique.id;
      genes.push(gene);
    }

    const components = this.evaluateFitnessComponents(genes);
    const fitness = calculateFitness(components, this.config.fitnessWeights);

    return {
      id: createChromosomeId(),
      name: technique.name,
      genes,
      fitness,
      fitnessComponents: components,
      generation: 0,
      parentIds: [],
      mutations: [],
      createdAt: new Date(),
      techniqueId: technique.id,
      isElite: false,
    };
  }

  /**
   * Create a random chromosome
   */
  createRandomChromosome(): Chromosome {
    const genes: Gene[] = [];

    for (const template of STANDARD_GENE_TEMPLATES) {
      let value: number | string | boolean;

      if (template.range) {
        // Random numeric value in range
        value =
          template.range.min +
          Math.random() * (template.range.max - template.range.min);
      } else if (template.options) {
        // Random option
        value = template.options[Math.floor(Math.random() * template.options.length)];
      } else {
        value = template.defaultValue;
      }

      genes.push(createGeneFromTemplate(template, value));
    }

    return {
      id: createChromosomeId(),
      name: `Random-${Date.now().toString(36)}`,
      genes,
      fitness: 0,
      fitnessComponents: createDefaultFitnessComponents(),
      generation: 0,
      parentIds: [],
      mutations: [],
      createdAt: new Date(),
      isElite: false,
    };
  }

  /**
   * Select a parent using configured strategy
   */
  private selectParent(chromosomes: Chromosome[]): Chromosome {
    switch (this.config.selectionStrategy) {
      case "tournament":
        return this.tournamentSelect(chromosomes);
      case "roulette":
        return this.rouletteSelect(chromosomes);
      case "rank":
        return this.rankSelect(chromosomes);
      case "elitist":
        return chromosomes[Math.floor(Math.random() * Math.ceil(chromosomes.length * 0.2))];
      default:
        return this.tournamentSelect(chromosomes);
    }
  }

  /**
   * Tournament selection
   */
  private tournamentSelect(chromosomes: Chromosome[]): Chromosome {
    const tournament: Chromosome[] = [];
    for (let i = 0; i < this.config.tournamentSize; i++) {
      const idx = Math.floor(Math.random() * chromosomes.length);
      tournament.push(chromosomes[idx]);
    }
    tournament.sort((a, b) => b.fitness - a.fitness);
    return tournament[0];
  }

  /**
   * Roulette wheel selection
   */
  private rouletteSelect(chromosomes: Chromosome[]): Chromosome {
    const totalFitness = chromosomes.reduce((sum, chr) => sum + chr.fitness, 0);
    if (totalFitness === 0) {
      return chromosomes[Math.floor(Math.random() * chromosomes.length)];
    }

    let target = Math.random() * totalFitness;
    for (const chr of chromosomes) {
      target -= chr.fitness;
      if (target <= 0) return chr;
    }
    return chromosomes[chromosomes.length - 1];
  }

  /**
   * Rank-based selection
   */
  private rankSelect(chromosomes: Chromosome[]): Chromosome {
    // Chromosomes are already sorted by fitness
    const n = chromosomes.length;
    const totalRank = (n * (n + 1)) / 2;
    let target = Math.random() * totalRank;

    for (let i = 0; i < n; i++) {
      target -= n - i;
      if (target <= 0) return chromosomes[i];
    }
    return chromosomes[0];
  }

  /**
   * Perform crossover between two parents
   */
  private crossover(
    parent1: Chromosome,
    parent2: Chromosome,
    generation: number
  ): Chromosome {
    const genes: Gene[] = [];

    switch (this.config.crossoverStrategy) {
      case "single_point": {
        const point = Math.floor(Math.random() * parent1.genes.length);
        for (let i = 0; i < parent1.genes.length; i++) {
          genes.push(
            this.cloneGene(i < point ? parent1.genes[i] : parent2.genes[i])
          );
        }
        break;
      }

      case "two_point": {
        const point1 = Math.floor(Math.random() * parent1.genes.length);
        const point2 =
          point1 + Math.floor(Math.random() * (parent1.genes.length - point1));
        for (let i = 0; i < parent1.genes.length; i++) {
          genes.push(
            this.cloneGene(
              i < point1 || i >= point2 ? parent1.genes[i] : parent2.genes[i]
            )
          );
        }
        break;
      }

      case "uniform": {
        for (let i = 0; i < parent1.genes.length; i++) {
          genes.push(
            this.cloneGene(
              Math.random() < 0.5 ? parent1.genes[i] : parent2.genes[i]
            )
          );
        }
        break;
      }

      case "blend": {
        for (let i = 0; i < parent1.genes.length; i++) {
          const gene1 = parent1.genes[i];
          const gene2 = parent2.genes[i];

          if (typeof gene1.value === "number" && typeof gene2.value === "number") {
            // Blend numeric genes
            const alpha = Math.random();
            const value = gene1.value * alpha + gene2.value * (1 - alpha);
            const newGene = this.cloneGene(gene1);
            newGene.value = Math.max(
              gene1.minValue ?? -Infinity,
              Math.min(gene1.maxValue ?? Infinity, value)
            );
            genes.push(newGene);
          } else {
            genes.push(
              this.cloneGene(Math.random() < 0.5 ? gene1 : gene2)
            );
          }
        }
        break;
      }
    }

    return {
      id: createChromosomeId(),
      name: `${parent1.name.split("-")[0]}-${parent2.name.split("-")[0]}-${generation}`,
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

  /**
   * Clone a chromosome
   */
  private cloneChromosome(chr: Chromosome, generation: number): Chromosome {
    return {
      id: createChromosomeId(),
      name: chr.name,
      genes: chr.genes.map((g) => this.cloneGene(g)),
      fitness: chr.fitness,
      fitnessComponents: { ...chr.fitnessComponents },
      generation,
      parentIds: [chr.id],
      mutations: [],
      createdAt: new Date(),
      isElite: false,
    };
  }

  /**
   * Clone a gene
   */
  private cloneGene(gene: Gene): Gene {
    return {
      ...gene,
      id: createGeneId(),
    };
  }

  /**
   * Mutate a chromosome
   */
  private mutate(chromosome: Chromosome): void {
    for (const gene of chromosome.genes) {
      if (Math.random() < gene.mutationRate) {
        const originalValue = gene.value;
        this.mutateGene(gene);

        chromosome.mutations.push({
          geneId: gene.id,
          originalValue,
          newValue: gene.value,
          type: "point",
          timestamp: new Date(),
        });
      }
    }
  }

  /**
   * Mutate a single gene
   */
  private mutateGene(gene: Gene): void {
    if (typeof gene.value === "number") {
      // Numeric mutation with Gaussian noise
      const range = (gene.maxValue ?? 1) - (gene.minValue ?? 0);
      const noise = (Math.random() - 0.5) * range * 0.2;
      gene.value = Math.max(
        gene.minValue ?? -Infinity,
        Math.min(gene.maxValue ?? Infinity, gene.value + noise)
      );
    } else if (typeof gene.value === "boolean") {
      // Boolean flip
      gene.value = !gene.value;
    } else if (gene.allowedValues) {
      // Categorical: random other value
      const otherValues = gene.allowedValues.filter((v) => v !== gene.value);
      if (otherValues.length > 0) {
        gene.value = otherValues[Math.floor(Math.random() * otherValues.length)];
      }
    }
  }

  /**
   * Evaluate fitness of a chromosome
   */
  private evaluateFitness(chromosome: Chromosome): void {
    const components = this.evaluateFitnessComponents(chromosome.genes);
    chromosome.fitnessComponents = components;
    chromosome.fitness = calculateFitness(components, this.config.fitnessWeights);
  }

  /**
   * Evaluate fitness components from genes
   */
  private evaluateFitnessComponents(genes: Gene[]): FitnessComponents {
    // Extract gene values
    const geneMap = new Map<string, Gene>();
    for (const gene of genes) {
      geneMap.set(gene.name, gene);
    }

    // Quality: based on architecture choices
    let quality = 0.5;
    if (geneMap.get("Uses Transformer")?.value === true) quality += 0.1;
    if (geneMap.get("Uses Attention")?.value === true) quality += 0.08;
    if (geneMap.get("Uses Diffusion")?.value === true) quality += 0.12;
    const modelSize = geneMap.get("Model Size")?.value;
    if (modelSize === "large") quality += 0.1;
    if (modelSize === "xlarge") quality += 0.15;

    // Efficiency: inverse relationship with complexity
    let efficiency = 0.6;
    if (modelSize === "small") efficiency += 0.2;
    if (modelSize === "large") efficiency -= 0.1;
    if (modelSize === "xlarge") efficiency -= 0.2;
    if (geneMap.get("Uses Diffusion")?.value === true) efficiency -= 0.1;

    // Novelty: based on unique combinations
    let novelty = 0.4;
    const architectureCount = [
      geneMap.get("Uses Transformer")?.value,
      geneMap.get("Uses Attention")?.value,
      geneMap.get("Uses Diffusion")?.value,
      geneMap.get("Uses VAE")?.value,
    ].filter(Boolean).length;
    novelty += architectureCount * 0.1;

    // Feasibility: based on resource requirements
    let feasibility = 0.7;
    if (modelSize === "xlarge") feasibility -= 0.2;
    if (geneMap.get("Uses Diffusion")?.value === true) feasibility -= 0.1;

    // Compatibility: based on training methodology
    let compatibility = 0.6;
    if (geneMap.get("Uses Self-Supervised")?.value === true) compatibility += 0.15;
    if (geneMap.get("Uses Contrastive Learning")?.value === true) compatibility += 0.1;

    return {
      quality: Math.max(0, Math.min(1, quality)),
      efficiency: Math.max(0, Math.min(1, efficiency)),
      novelty: Math.max(0, Math.min(1, novelty)),
      feasibility: Math.max(0, Math.min(1, feasibility)),
      compatibility: Math.max(0, Math.min(1, compatibility)),
    };
  }

  /**
   * Calculate generation statistics
   */
  private calculateGenerationStats(
    chromosomes: Chromosome[],
    generation: number
  ): GenerationStats {
    const fitnesses = chromosomes.map((c) => c.fitness);
    const sum = fitnesses.reduce((a, b) => a + b, 0);
    const avg = sum / fitnesses.length;

    const variance =
      fitnesses.reduce((a, b) => a + Math.pow(b - avg, 2), 0) / fitnesses.length;
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

  /**
   * Check for convergence
   */
  private checkConvergence(
    population: Population,
    currentStats: GenerationStats
  ): boolean {
    if (population.generationHistory.length < this.config.convergenceWindow) {
      return false;
    }

    const recentStats = population.generationHistory.slice(
      -this.config.convergenceWindow
    );
    const fitnessChanges = recentStats
      .slice(1)
      .map((s, i) => Math.abs(s.bestFitness - recentStats[i].bestFitness));

    const maxChange = Math.max(...fitnessChanges);
    return maxChange < this.config.convergenceThreshold;
  }

  /**
   * Get adaptive mutation rate based on population diversity
   */
  private getAdaptiveMutationRate(population: Population): number {
    // Increase mutation if diversity is low
    const baseMutation = this.config.mutationRate;
    const diversity = population.fitnessDiversity;

    if (diversity < 0.05) {
      return Math.min(baseMutation * 2, 0.5);
    } else if (diversity < 0.1) {
      return baseMutation * 1.5;
    }
    return baseMutation;
  }

  /**
   * Get lineage for a chromosome
   */
  getLineage(chromosome: Chromosome, population: Population): ChromosomeLineage {
    const ancestors = new Set<string>();
    const fitnessTrajectory: Array<{ generation: number; fitness: number }> = [];

    // Build lineage tree
    const buildTree = (chrId: string, depth: number): LineageNode | null => {
      const chr = population.chromosomes.find((c) => c.id === chrId);
      if (!chr) {
        // Check in history if available
        return null;
      }

      ancestors.add(chr.id);
      fitnessTrajectory.push({
        generation: chr.generation,
        fitness: chr.fitness,
      });

      const children: LineageNode[] = [];
      for (const parentId of chr.parentIds) {
        const parentNode = buildTree(parentId, depth + 1);
        if (parentNode) {
          children.push(parentNode);
        }
      }

      return {
        id: chr.id,
        name: chr.name,
        generation: chr.generation,
        fitness: chr.fitness,
        children,
      };
    };

    const tree = buildTree(chromosome.id, 0);
    const keyMutations = chromosome.mutations.filter(
      (m) => m.fitnessImpact && Math.abs(m.fitnessImpact) > 0.05
    );

    return {
      rootId: chromosome.id,
      ancestors: Array.from(ancestors),
      depth: ancestors.size,
      tree: tree || {
        id: chromosome.id,
        name: chromosome.name,
        generation: chromosome.generation,
        fitness: chromosome.fitness,
        children: [],
      },
      keyMutations,
      fitnessTrajectory: fitnessTrajectory.sort(
        (a, b) => a.generation - b.generation
      ),
    };
  }
}

/**
 * Global evolution engine instance
 */
let globalEvolutionEngine: EvolutionEngine | null = null;

export function getGlobalEvolutionEngine(): EvolutionEngine {
  if (!globalEvolutionEngine) {
    globalEvolutionEngine = new EvolutionEngine();
  }
  return globalEvolutionEngine;
}

export function resetGlobalEvolutionEngine(): void {
  globalEvolutionEngine = null;
}
