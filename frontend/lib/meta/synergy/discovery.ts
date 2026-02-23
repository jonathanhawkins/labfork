/**
 * Synergy Discovery Algorithm
 *
 * Discovers promising combinations of research techniques by analyzing
 * their features, complementarity, and potential synergies.
 */

import { TechniqueNode, KnowledgeGraph } from "../knowledge-graph";
import {
  SynergyProposal,
  SynergyScore,
  SynergyScoreComponents,
  CombinationAspect,
  ExpectedOutcome,
  SynergyDiscoveryConfig,
  DEFAULT_SYNERGY_CONFIG,
  TechniqueFeatures,
  ArchitectureFeatures,
  TrainingFeatures,
  createProposalId,
  createEmptyFeatures,
  ExploredCombination,
} from "./types";

/**
 * Synergy Discovery Engine
 */
export class SynergyDiscovery {
  private graph: KnowledgeGraph;
  private config: SynergyDiscoveryConfig;
  private exploredCombinations: Map<string, ExploredCombination>;
  private featureCache: Map<string, TechniqueFeatures>;

  constructor(graph: KnowledgeGraph, config?: Partial<SynergyDiscoveryConfig>) {
    this.graph = graph;
    this.config = { ...DEFAULT_SYNERGY_CONFIG, ...config };
    this.exploredCombinations = new Map();
    this.featureCache = new Map();
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<SynergyDiscoveryConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   */
  getConfig(): SynergyDiscoveryConfig {
    return { ...this.config };
  }

  /**
   * Discover synergy proposals between techniques
   */
  discover(): SynergyProposal[] {
    // Get all techniques
    let techniques = this.graph.getNodesByType("technique") as TechniqueNode[];

    // Apply filters
    techniques = this.filterTechniques(techniques);

    if (techniques.length < 2) {
      return [];
    }

    // Extract features for all techniques
    const features = techniques.map((t) => this.extractFeatures(t));

    // Compute pairwise synergies
    const proposals: SynergyProposal[] = [];

    for (let i = 0; i < techniques.length; i++) {
      for (let j = i + 1; j < techniques.length; j++) {
        const techniqueA = techniques[i];
        const techniqueB = techniques[j];
        const featuresA = features[i];
        const featuresB = features[j];

        // Skip if already explored
        if (
          this.config.excludeExplored &&
          this.isExplored(techniqueA.id, techniqueB.id)
        ) {
          continue;
        }

        // Compute synergy score
        const score = this.computeSynergyScore(
          techniqueA,
          techniqueB,
          featuresA,
          featuresB
        );

        // Check thresholds
        if (
          score.components.similarity < this.config.minSimilarity ||
          score.overall < this.config.minScore
        ) {
          continue;
        }

        // Generate proposal
        const proposal = this.generateProposal(
          techniqueA,
          techniqueB,
          score,
          featuresA,
          featuresB
        );

        proposals.push(proposal);
      }
    }

    // Sort by score and limit
    return proposals
      .sort((a, b) => b.score.overall - a.score.overall)
      .slice(0, this.config.maxProposals);
  }

  /**
   * Filter techniques based on configuration
   */
  private filterTechniques(techniques: TechniqueNode[]): TechniqueNode[] {
    let filtered = techniques;

    // Filter by domains
    if (this.config.focusDomains.length > 0) {
      filtered = filtered.filter((t) =>
        t.domains.some((d) => this.config.focusDomains.includes(d))
      );
    }

    // Filter by tags
    if (this.config.filterTags.length > 0) {
      filtered = filtered.filter((t) =>
        t.tags.some((tag) => this.config.filterTags.includes(tag))
      );
    }

    return filtered;
  }

  /**
   * Extract features from a technique
   */
  extractFeatures(technique: TechniqueNode): TechniqueFeatures {
    // Check cache
    const cached = this.featureCache.get(technique.id);
    if (cached) {
      return cached;
    }

    const features = createEmptyFeatures(technique.id);

    // Extract architecture features from description and tags
    features.architecture = this.extractArchitectureFeatures(technique);

    // Extract training features
    features.training = this.extractTrainingFeatures(technique);

    // Extract performance features
    features.performance = this.extractPerformanceFeatures(technique);

    // Extract tag features
    for (const tag of technique.tags) {
      features.tags.set(tag.toLowerCase(), 1);
    }

    // Cache and return
    this.featureCache.set(technique.id, features);
    return features;
  }

  /**
   * Extract architecture features from technique
   */
  private extractArchitectureFeatures(
    technique: TechniqueNode
  ): ArchitectureFeatures {
    const text = `${technique.name} ${technique.description} ${technique.tags.join(" ")}`.toLowerCase();

    return {
      usesTransformer:
        text.includes("transformer") || text.includes("attention"),
      usesConvolution: text.includes("conv") || text.includes("cnn"),
      usesAttention:
        text.includes("attention") || text.includes("self-attention"),
      usesDiffusion:
        text.includes("diffusion") ||
        text.includes("denoising") ||
        text.includes("ddpm"),
      usesVAE:
        text.includes("vae") ||
        text.includes("variational") ||
        text.includes("encoder-decoder"),
      usesGAN:
        text.includes("gan") ||
        text.includes("generative adversarial") ||
        text.includes("discriminator"),
      usesFlow:
        text.includes("flow") ||
        text.includes("normalizing") ||
        text.includes("invertible"),
      parameterScale: this.estimateParameterScale(technique),
    };
  }

  /**
   * Estimate parameter scale from technique info
   */
  private estimateParameterScale(technique: TechniqueNode): number {
    const text =
      `${technique.name} ${technique.description}`.toLowerCase();

    // Look for parameter counts
    const billions = text.match(/(\d+(?:\.\d+)?)\s*b(?:illion)?\s*param/);
    if (billions) {
      const b = parseFloat(billions[1]);
      return Math.min(1, b / 100); // Normalize to 0-1 (100B as max)
    }

    const millions = text.match(/(\d+(?:\.\d+)?)\s*m(?:illion)?\s*param/);
    if (millions) {
      const m = parseFloat(millions[1]);
      return Math.min(1, m / 100000); // Normalize to 0-1
    }

    // Default based on type hints
    if (text.includes("large") || text.includes("xl")) return 0.7;
    if (text.includes("base")) return 0.4;
    if (text.includes("small") || text.includes("tiny")) return 0.2;

    return 0.5; // Default
  }

  /**
   * Extract training features from technique
   */
  private extractTrainingFeatures(technique: TechniqueNode): TrainingFeatures {
    const text = `${technique.name} ${technique.description} ${technique.tags.join(" ")}`.toLowerCase();

    return {
      supervised:
        text.includes("supervised") ||
        text.includes("labeled") ||
        text.includes("classification"),
      selfSupervised:
        text.includes("self-supervised") ||
        text.includes("unsupervised") ||
        text.includes("pretraining"),
      reinforcement:
        text.includes("reinforcement") ||
        text.includes("rl") ||
        text.includes("reward"),
      contrastive:
        text.includes("contrastive") ||
        text.includes("triplet") ||
        text.includes("siamese"),
      dataEfficiency: this.estimateDataEfficiency(technique),
      stability: this.estimateStability(technique),
    };
  }

  /**
   * Estimate data efficiency
   */
  private estimateDataEfficiency(technique: TechniqueNode): number {
    const text =
      `${technique.name} ${technique.description}`.toLowerCase();

    if (text.includes("few-shot") || text.includes("zero-shot")) return 0.9;
    if (text.includes("efficient") || text.includes("data-efficient"))
      return 0.8;
    if (text.includes("pretrained") || text.includes("transfer")) return 0.7;
    if (text.includes("large-scale") || text.includes("massive")) return 0.3;

    return 0.5;
  }

  /**
   * Estimate training stability
   */
  private estimateStability(technique: TechniqueNode): number {
    const text =
      `${technique.name} ${technique.description}`.toLowerCase();

    // GANs are typically less stable
    if (text.includes("gan") || text.includes("adversarial")) return 0.4;
    // RL can be unstable
    if (text.includes("reinforcement") || text.includes("rl")) return 0.5;
    // Transformers with good practices are stable
    if (text.includes("transformer") && text.includes("pretrain")) return 0.8;
    // Diffusion models are generally stable
    if (text.includes("diffusion")) return 0.8;

    return 0.6;
  }

  /**
   * Extract performance features from technique
   */
  private extractPerformanceFeatures(
    technique: TechniqueNode
  ): TechniqueFeatures["performance"] {
    const text =
      `${technique.name} ${technique.description}`.toLowerCase();

    return {
      quality: this.estimateQuality(technique),
      speed: this.estimateSpeed(text),
      memoryEfficiency: this.estimateMemoryEfficiency(text),
      scalability: this.estimateScalability(text),
    };
  }

  /**
   * Estimate quality from metrics
   */
  private estimateQuality(technique: TechniqueNode): number {
    // If we have metrics, use them
    if (technique.metrics) {
      const values = Object.values(technique.metrics);
      if (values.length > 0) {
        // Normalize metrics (assuming higher is better, 0-100 scale)
        const avg = values.reduce((a, b) => a + b, 0) / values.length;
        return Math.min(1, avg / 100);
      }
    }

    return 0.5;
  }

  /**
   * Estimate speed from description
   */
  private estimateSpeed(text: string): number {
    if (text.includes("real-time") || text.includes("fast")) return 0.9;
    if (text.includes("efficient") || text.includes("lightweight")) return 0.8;
    if (text.includes("slow") || text.includes("heavy")) return 0.3;
    if (text.includes("iterative") || text.includes("diffusion")) return 0.4;

    return 0.5;
  }

  /**
   * Estimate memory efficiency from description
   */
  private estimateMemoryEfficiency(text: string): number {
    if (text.includes("efficient") || text.includes("lightweight")) return 0.8;
    if (text.includes("quantized") || text.includes("pruned")) return 0.9;
    if (text.includes("large") || text.includes("massive")) return 0.3;

    return 0.5;
  }

  /**
   * Estimate scalability from description
   */
  private estimateScalability(text: string): number {
    if (text.includes("scalable") || text.includes("distributed")) return 0.9;
    if (text.includes("parallel") || text.includes("batch")) return 0.7;
    if (text.includes("sequential") || text.includes("autoregressive"))
      return 0.4;

    return 0.5;
  }

  /**
   * Compute synergy score between two techniques
   */
  computeSynergyScore(
    techniqueA: TechniqueNode,
    techniqueB: TechniqueNode,
    featuresA: TechniqueFeatures,
    featuresB: TechniqueFeatures
  ): SynergyScore {
    const components: SynergyScoreComponents = {
      similarity: this.computeSimilarity(featuresA, featuresB),
      complementarity: this.computeComplementarity(featuresA, featuresB),
      novelty: this.computeNovelty(techniqueA, techniqueB),
      feasibility: this.computeFeasibility(featuresA, featuresB),
      impact: this.computeImpact(techniqueA, techniqueB, featuresA, featuresB),
    };

    // Weighted sum
    const overall =
      components.similarity * this.config.similarityWeight +
      components.complementarity * this.config.complementarityWeight +
      components.novelty * this.config.noveltyWeight +
      components.feasibility * this.config.feasibilityWeight +
      components.impact * this.config.impactWeight;

    // Confidence based on data quality
    const confidence = this.computeConfidence(techniqueA, techniqueB);

    return { overall, components, confidence };
  }

  /**
   * Compute similarity between techniques
   */
  computeSimilarity(
    featuresA: TechniqueFeatures,
    featuresB: TechniqueFeatures
  ): number {
    let similarity = 0;
    let count = 0;

    // Architecture similarity
    const archA = featuresA.architecture;
    const archB = featuresB.architecture;

    const archFeatures: (keyof ArchitectureFeatures)[] = [
      "usesTransformer",
      "usesConvolution",
      "usesAttention",
      "usesDiffusion",
      "usesVAE",
      "usesGAN",
      "usesFlow",
    ];

    let archMatch = 0;
    for (const feat of archFeatures) {
      if (archA[feat] === archB[feat]) archMatch++;
    }
    similarity += archMatch / archFeatures.length;
    count++;

    // Tag overlap (Jaccard similarity)
    const tagsA = new Set(Array.from(featuresA.tags.keys()));
    const tagsB = new Set(Array.from(featuresB.tags.keys()));
    const intersection = new Set(Array.from(tagsA).filter((t) => tagsB.has(t)));
    const union = new Set([...Array.from(tagsA), ...Array.from(tagsB)]);

    if (union.size > 0) {
      similarity += intersection.size / union.size;
      count++;
    }

    // Parameter scale similarity
    similarity +=
      1 - Math.abs(archA.parameterScale - archB.parameterScale);
    count++;

    return similarity / count;
  }

  /**
   * Compute complementarity between techniques
   */
  computeComplementarity(
    featuresA: TechniqueFeatures,
    featuresB: TechniqueFeatures
  ): number {
    let complementarity = 0;
    let count = 0;

    // Architecture complementarity (different approaches that can combine)
    const archA = featuresA.architecture;
    const archB = featuresB.architecture;

    // Complementary pairs
    const complementaryPairs: [keyof ArchitectureFeatures, keyof ArchitectureFeatures][] = [
      ["usesTransformer", "usesConvolution"], // Different processing
      ["usesDiffusion", "usesGAN"], // Different generation
      ["usesVAE", "usesFlow"], // Different latent spaces
    ];

    for (const [feat1, feat2] of complementaryPairs) {
      const a1 = archA[feat1] as boolean;
      const a2 = archA[feat2] as boolean;
      const b1 = archB[feat1] as boolean;
      const b2 = archB[feat2] as boolean;

      // One has feat1, other has feat2
      if ((a1 && b2 && !a2 && !b1) || (a2 && b1 && !a1 && !b2)) {
        complementarity += 1;
      }
      count++;
    }

    // Training complementarity
    const trainA = featuresA.training;
    const trainB = featuresB.training;

    // Self-supervised + Supervised is complementary
    if (
      (trainA.selfSupervised && trainB.supervised) ||
      (trainA.supervised && trainB.selfSupervised)
    ) {
      complementarity += 1;
    }
    count++;

    // Performance complementarity (one strong where other is weak)
    const perfA = featuresA.performance;
    const perfB = featuresB.performance;

    const perfMetrics: (keyof typeof perfA)[] = [
      "quality",
      "speed",
      "memoryEfficiency",
      "scalability",
    ];

    for (const metric of perfMetrics) {
      const diff = Math.abs(perfA[metric] - perfB[metric]);
      // High difference in performance = potential for one to help the other
      if (diff > 0.3) {
        complementarity += diff;
        count++;
      }
    }

    return count > 0 ? Math.min(1, complementarity / count) : 0;
  }

  /**
   * Compute novelty of combining two techniques
   */
  computeNovelty(
    techniqueA: TechniqueNode,
    techniqueB: TechniqueNode
  ): number {
    // Check graph for existing connections
    const edgesFromA = this.graph.getOutgoingEdges(techniqueA.id);
    const edgesFromB = this.graph.getOutgoingEdges(techniqueB.id);

    // Direct connection reduces novelty
    const directConnection =
      edgesFromA.some((e) => e.targetId === techniqueB.id) ||
      edgesFromB.some((e) => e.targetId === techniqueA.id);

    if (directConnection) {
      return 0.2; // Already connected, low novelty
    }

    // Common neighbors reduce novelty
    const neighborsA = new Set(edgesFromA.map((e) => e.targetId));
    const neighborsB = new Set(edgesFromB.map((e) => e.targetId));
    const commonNeighbors = Array.from(neighborsA).filter((n) => neighborsB.has(n));

    if (commonNeighbors.length > 2) {
      return 0.4; // Many common neighbors
    }

    if (commonNeighbors.length > 0) {
      return 0.6; // Some common neighbors
    }

    // Check if combination has been explored before
    if (this.isExplored(techniqueA.id, techniqueB.id)) {
      return 0.1;
    }

    // High novelty - no direct connection, few common neighbors
    return 0.9;
  }

  /**
   * Compute feasibility of combining two techniques
   */
  computeFeasibility(
    featuresA: TechniqueFeatures,
    featuresB: TechniqueFeatures
  ): number {
    let feasibility = 0.5; // Base feasibility

    // Same general approach is more feasible
    const archA = featuresA.architecture;
    const archB = featuresB.architecture;

    // Both use similar base architectures
    if (archA.usesTransformer && archB.usesTransformer) feasibility += 0.1;
    if (archA.usesConvolution && archB.usesConvolution) feasibility += 0.1;

    // Similar scale is more feasible
    const scaleDiff = Math.abs(archA.parameterScale - archB.parameterScale);
    feasibility += (1 - scaleDiff) * 0.2;

    // Both stable = more feasible
    const stability =
      (featuresA.training.stability + featuresB.training.stability) / 2;
    feasibility += stability * 0.1;

    return Math.min(1, feasibility);
  }

  /**
   * Compute expected impact of combining two techniques
   */
  computeImpact(
    techniqueA: TechniqueNode,
    techniqueB: TechniqueNode,
    featuresA: TechniqueFeatures,
    featuresB: TechniqueFeatures
  ): number {
    let impact = 0;

    // Quality improvement potential
    const qualityA = featuresA.performance.quality;
    const qualityB = featuresB.performance.quality;
    const qualityPotential = Math.max(qualityA, qualityB) + 0.1; // Best + improvement
    impact += Math.min(1, qualityPotential) * 0.4;

    // Speed improvement potential
    const speedA = featuresA.performance.speed;
    const speedB = featuresB.performance.speed;
    // If one is fast and one is quality-focused, potential for both
    if ((speedA > 0.7 && qualityB > 0.7) || (speedB > 0.7 && qualityA > 0.7)) {
      impact += 0.3;
    }

    // Domain coverage expansion
    const domainsA = new Set(techniqueA.domains);
    const domainsB = new Set(techniqueB.domains);
    const combinedDomains = new Set([...Array.from(domainsA), ...Array.from(domainsB)]);
    if (combinedDomains.size > Math.max(domainsA.size, domainsB.size)) {
      impact += 0.2; // Expands to new domains
    }

    // Adoption potential (both well-established = higher impact)
    const incomingA = this.graph.getIncomingEdges(techniqueA.id).length;
    const incomingB = this.graph.getIncomingEdges(techniqueB.id).length;
    const adoption = Math.min(1, (incomingA + incomingB) / 10);
    impact += adoption * 0.1;

    return Math.min(1, impact);
  }

  /**
   * Compute confidence in the synergy score
   */
  private computeConfidence(
    techniqueA: TechniqueNode,
    techniqueB: TechniqueNode
  ): number {
    let confidence = 0.5;

    // More description = higher confidence
    const descLengthA = techniqueA.description?.length || 0;
    const descLengthB = techniqueB.description?.length || 0;
    confidence += Math.min(0.2, (descLengthA + descLengthB) / 2000);

    // More tags = higher confidence
    const tagCountA = techniqueA.tags.length;
    const tagCountB = techniqueB.tags.length;
    confidence += Math.min(0.15, (tagCountA + tagCountB) / 20);

    // Has metrics = higher confidence
    if (techniqueA.metrics && Object.keys(techniqueA.metrics).length > 0) {
      confidence += 0.1;
    }
    if (techniqueB.metrics && Object.keys(techniqueB.metrics).length > 0) {
      confidence += 0.1;
    }

    return Math.min(1, confidence);
  }

  /**
   * Generate a synergy proposal
   */
  private generateProposal(
    techniqueA: TechniqueNode,
    techniqueB: TechniqueNode,
    score: SynergyScore,
    featuresA: TechniqueFeatures,
    featuresB: TechniqueFeatures
  ): SynergyProposal {
    const now = new Date();

    return {
      id: createProposalId(),
      techniqueA,
      techniqueB,
      score,
      justification: this.generateJustification(
        techniqueA,
        techniqueB,
        score,
        featuresA,
        featuresB
      ),
      combinationAspects: this.generateCombinationAspects(
        techniqueA,
        techniqueB,
        featuresA,
        featuresB
      ),
      expectedOutcomes: this.generateExpectedOutcomes(
        techniqueA,
        techniqueB,
        featuresA,
        featuresB
      ),
      status: "pending",
      createdAt: now,
      updatedAt: now,
      createdBy: "system",
    };
  }

  /**
   * Generate justification text
   */
  private generateJustification(
    techniqueA: TechniqueNode,
    techniqueB: TechniqueNode,
    score: SynergyScore,
    featuresA: TechniqueFeatures,
    featuresB: TechniqueFeatures
  ): string {
    const parts: string[] = [];

    // Similarity justification
    if (score.components.similarity > 0.6) {
      parts.push(
        `${techniqueA.name} and ${techniqueB.name} share similar architectural foundations`
      );
    }

    // Complementarity justification
    if (score.components.complementarity > 0.5) {
      const archA = featuresA.architecture;
      const archB = featuresB.architecture;

      if (archA.usesTransformer && archB.usesConvolution) {
        parts.push("combining transformer attention with convolutional processing");
      }
      if (archA.usesDiffusion && archB.usesGAN) {
        parts.push("merging diffusion stability with GAN sharpness");
      }
      if (archA.usesVAE && archB.usesFlow) {
        parts.push("unifying VAE flexibility with flow invertibility");
      }
    }

    // Novelty justification
    if (score.components.novelty > 0.7) {
      parts.push("this combination has not been explored in the literature");
    }

    // Impact justification
    if (score.components.impact > 0.6) {
      parts.push("expected to significantly improve quality and efficiency");
    }

    if (parts.length === 0) {
      return `Combining ${techniqueA.name} with ${techniqueB.name} shows promising synergy potential.`;
    }

    return (
      parts[0].charAt(0).toUpperCase() +
      parts[0].slice(1) +
      (parts.length > 1 ? ", " + parts.slice(1).join(", ") : "") +
      "."
    );
  }

  /**
   * Generate combination aspects
   */
  private generateCombinationAspects(
    techniqueA: TechniqueNode,
    techniqueB: TechniqueNode,
    featuresA: TechniqueFeatures,
    featuresB: TechniqueFeatures
  ): CombinationAspect[] {
    const aspects: CombinationAspect[] = [];
    const archA = featuresA.architecture;
    const archB = featuresB.architecture;

    // Architecture combinations
    if (archA.usesTransformer && archB.usesConvolution) {
      aspects.push({
        fromA: "transformer attention mechanism",
        fromB: "convolutional feature extraction",
        combination: "attention-augmented convolutions",
        benefit: "better local-global feature integration",
      });
    }

    if (archA.usesDiffusion && archB.usesGAN) {
      aspects.push({
        fromA: "diffusion denoising process",
        fromB: "GAN discriminator feedback",
        combination: "discriminator-guided diffusion",
        benefit: "sharper outputs with stable training",
      });
    }

    if (archA.usesVAE && archB.usesFlow) {
      aspects.push({
        fromA: "VAE latent space",
        fromB: "flow-based transformation",
        combination: "flow-enhanced VAE",
        benefit: "more expressive latent distributions",
      });
    }

    // Training combinations
    if (featuresA.training.selfSupervised && featuresB.training.supervised) {
      aspects.push({
        fromA: "self-supervised pretraining",
        fromB: "supervised fine-tuning",
        combination: "two-stage training pipeline",
        benefit: "better generalization with less labeled data",
      });
    }

    // If no specific aspects found, add generic ones
    if (aspects.length === 0) {
      aspects.push({
        fromA: `core mechanism from ${techniqueA.name}`,
        fromB: `innovations from ${techniqueB.name}`,
        combination: "hybrid architecture",
        benefit: "combines strengths of both approaches",
      });
    }

    return aspects;
  }

  /**
   * Generate expected outcomes
   */
  private generateExpectedOutcomes(
    techniqueA: TechniqueNode,
    techniqueB: TechniqueNode,
    featuresA: TechniqueFeatures,
    featuresB: TechniqueFeatures
  ): ExpectedOutcome[] {
    const outcomes: ExpectedOutcome[] = [];

    // Quality improvement
    const baseQuality = Math.max(
      featuresA.performance.quality,
      featuresB.performance.quality
    );
    outcomes.push({
      metric: "Quality Score",
      baseline: baseQuality * 100,
      expected: Math.min(100, baseQuality * 100 + 5),
      unit: "%",
      confidence: 0.6,
    });

    // Speed improvement if one is fast
    if (
      featuresA.performance.speed > 0.7 ||
      featuresB.performance.speed > 0.7
    ) {
      const baseSpeed = Math.max(
        featuresA.performance.speed,
        featuresB.performance.speed
      );
      outcomes.push({
        metric: "Inference Speed",
        baseline: baseSpeed * 100,
        expected: Math.min(100, baseSpeed * 100 + 10),
        unit: "% of real-time",
        confidence: 0.5,
      });
    }

    // Memory efficiency if combining efficient techniques
    if (
      featuresA.performance.memoryEfficiency > 0.6 &&
      featuresB.performance.memoryEfficiency > 0.6
    ) {
      const baseMemory =
        (featuresA.performance.memoryEfficiency +
          featuresB.performance.memoryEfficiency) /
        2;
      outcomes.push({
        metric: "Memory Usage",
        baseline: (1 - baseMemory) * 100,
        expected: (1 - baseMemory) * 100 - 10,
        unit: "% of baseline",
        confidence: 0.4,
      });
    }

    return outcomes;
  }

  /**
   * Check if a combination has been explored
   */
  isExplored(idA: string, idB: string): boolean {
    const key = this.getCombinationKey(idA, idB);
    return this.exploredCombinations.has(key);
  }

  /**
   * Mark a combination as explored
   */
  markExplored(proposal: SynergyProposal): ExploredCombination {
    const key = this.getCombinationKey(
      proposal.techniqueA.id,
      proposal.techniqueB.id
    );

    const exploration: ExploredCombination = {
      proposal,
      startedAt: new Date(),
      status: "in_progress",
    };

    this.exploredCombinations.set(key, exploration);
    return exploration;
  }

  /**
   * Update exploration status
   */
  updateExploration(
    idA: string,
    idB: string,
    update: Partial<ExploredCombination>
  ): ExploredCombination | null {
    const key = this.getCombinationKey(idA, idB);
    const existing = this.exploredCombinations.get(key);

    if (!existing) {
      return null;
    }

    const updated = { ...existing, ...update };
    this.exploredCombinations.set(key, updated);
    return updated;
  }

  /**
   * Get explored combinations
   */
  getExploredCombinations(): ExploredCombination[] {
    return Array.from(this.exploredCombinations.values());
  }

  /**
   * Get combination key (order-independent)
   */
  private getCombinationKey(idA: string, idB: string): string {
    return [idA, idB].sort().join(":");
  }

  /**
   * Clear feature cache
   */
  clearCache(): void {
    this.featureCache.clear();
  }
}

/**
 * Create a synergy discovery engine
 */
export function createSynergyDiscovery(
  graph: KnowledgeGraph,
  config?: Partial<SynergyDiscoveryConfig>
): SynergyDiscovery {
  return new SynergyDiscovery(graph, config);
}
