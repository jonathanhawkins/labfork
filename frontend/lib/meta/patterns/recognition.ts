/**
 * Pattern Recognition Service
 *
 * Detects research trends, recurring patterns, and cross-domain transfers
 * by analyzing the knowledge graph.
 */

import {
  KnowledgeGraph,
  TechniqueNode,
  PaperNode,
  GraphNode,
} from "../knowledge-graph";
import {
  ResearchTrend,
  TrendCategory,
  TrendDataPoint,
  ArchitecturePattern,
  PatternComponent,
  PatternConnection,
  TechniqueAdoption,
  AdoptionStage,
  AdoptionDataPoint,
  CrossDomainTransfer,
  PatternRecognitionConfig,
  DEFAULT_PATTERN_CONFIG,
  PatternReport,
  ReportSummary,
  createTrendId,
  createPatternId,
  createTransferId,
  createReportId,
} from "./types";

/**
 * Pattern Recognition Engine
 */
export class PatternRecognition {
  private graph: KnowledgeGraph;
  private config: PatternRecognitionConfig;
  private detectedTrends: Map<string, ResearchTrend>;
  private detectedPatterns: Map<string, ArchitecturePattern>;
  private adoptionMetrics: Map<string, TechniqueAdoption>;
  private crossDomainTransfers: Map<string, CrossDomainTransfer>;

  constructor(
    graph: KnowledgeGraph,
    config?: Partial<PatternRecognitionConfig>
  ) {
    this.graph = graph;
    this.config = { ...DEFAULT_PATTERN_CONFIG, ...config };
    this.detectedTrends = new Map();
    this.detectedPatterns = new Map();
    this.adoptionMetrics = new Map();
    this.crossDomainTransfers = new Map();
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<PatternRecognitionConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration
   */
  getConfig(): PatternRecognitionConfig {
    return { ...this.config };
  }

  /**
   * Run full pattern recognition analysis
   */
  analyze(): PatternReport {
    const now = new Date();
    const periodStart = new Date(
      now.getTime() - this.config.trendWindowDays * 24 * 60 * 60 * 1000
    );

    // Detect trends
    const trends = this.detectTrends();

    // Detect architecture patterns
    const patterns = this.config.detectArchitecturePatterns
      ? this.detectArchitecturePatterns()
      : [];

    // Track adoption
    const adoption = this.config.trackAdoption
      ? this.trackTechniqueAdoption()
      : [];

    // Detect cross-domain transfers
    const transfers = this.config.detectCrossDomain
      ? this.detectCrossDomainTransfers()
      : [];

    // Generate summary
    const summary = this.generateSummary(trends, patterns, adoption, transfers);

    return {
      id: createReportId(),
      generatedAt: now,
      periodStart,
      periodEnd: now,
      trends,
      emergingPatterns: patterns,
      adoptionMetrics: adoption,
      crossDomainTransfers: transfers,
      summary,
    };
  }

  /**
   * Detect research trends from the knowledge graph
   */
  detectTrends(): ResearchTrend[] {
    const techniques = this.graph.getNodesByType("technique") as TechniqueNode[];
    const papers = this.graph.getNodesByType("paper") as PaperNode[];

    // Group by keywords and categories
    const keywordCounts = new Map<string, { count: number; techniques: string[]; papers: string[] }>();
    const categoryCounts = new Map<TrendCategory, number>();

    // Count keywords from techniques
    for (const tech of techniques) {
      for (const tag of tech.tags) {
        const lower = tag.toLowerCase();
        const existing = keywordCounts.get(lower) || { count: 0, techniques: [], papers: [] };
        existing.count++;
        existing.techniques.push(tech.id);
        keywordCounts.set(lower, existing);
      }

      // Detect category from description/tags
      const category = this.detectCategory(tech);
      categoryCounts.set(category, (categoryCounts.get(category) || 0) + 1);
    }

    // Count keywords from papers
    for (const paper of papers) {
      for (const tag of paper.tags) {
        const lower = tag.toLowerCase();
        const existing = keywordCounts.get(lower) || { count: 0, techniques: [], papers: [] };
        existing.count++;
        existing.papers.push(paper.id);
        keywordCounts.set(lower, existing);
      }
    }

    // Convert to trends (filter by minimum occurrences)
    const trends: ResearchTrend[] = [];
    const now = new Date();

    for (const [keyword, data] of Array.from(keywordCounts)) {
      if (data.count < this.config.minOccurrences) {
        continue;
      }

      // Calculate strength based on occurrences
      const maxCount = Math.max(...Array.from(keywordCounts.values()).map((d) => d.count));
      const strength = data.count / maxCount;

      // Calculate momentum (simplified - would need historical data in production)
      const momentum = this.calculateMomentum(data.techniques, data.papers);

      // Get domains from related techniques
      const domains = this.getDomainsFromTechniques(data.techniques);

      // Determine category
      const category = this.detectCategoryFromKeyword(keyword);

      const trend: ResearchTrend = {
        id: createTrendId(),
        name: this.formatKeywordAsName(keyword),
        description: `Research trend around "${keyword}" with ${data.count} related items.`,
        category,
        keywords: [keyword],
        domains,
        strength,
        momentum,
        firstDetected: now,
        lastUpdated: now,
        timeSeries: this.generateTimeSeries(data.count),
        relatedTechniques: data.techniques,
        relatedPapers: data.papers,
        confidence: Math.min(1, data.count / 10),
      };

      trends.push(trend);
      this.detectedTrends.set(trend.id, trend);
    }

    // Filter by confidence and sort by strength
    return trends
      .filter((t) => t.confidence >= this.config.minConfidence)
      .sort((a, b) => b.strength - a.strength);
  }

  /**
   * Detect category from technique
   */
  private detectCategory(tech: TechniqueNode): TrendCategory {
    const text = `${tech.name} ${tech.description} ${tech.tags.join(" ")}`.toLowerCase();

    if (text.includes("transformer") || text.includes("architecture") || text.includes("attention")) {
      return "architecture";
    }
    if (text.includes("training") || text.includes("learning") || text.includes("optimization")) {
      return "training";
    }
    if (text.includes("efficient") || text.includes("fast") || text.includes("lightweight")) {
      return "efficiency";
    }
    if (text.includes("quality") || text.includes("naturalness") || text.includes("fidelity")) {
      return "quality";
    }
    if (text.includes("data") || text.includes("dataset") || text.includes("corpus")) {
      return "data";
    }
    if (text.includes("gpu") || text.includes("hardware") || text.includes("deployment")) {
      return "hardware";
    }
    if (text.includes("theoretical") || text.includes("proof") || text.includes("bound")) {
      return "theoretical";
    }

    return "application";
  }

  /**
   * Detect category from keyword
   */
  private detectCategoryFromKeyword(keyword: string): TrendCategory {
    const lower = keyword.toLowerCase();

    if (["transformer", "attention", "conv", "layer", "network"].some((k) => lower.includes(k))) {
      return "architecture";
    }
    if (["train", "learn", "loss", "optim", "gradient"].some((k) => lower.includes(k))) {
      return "training";
    }
    if (["efficient", "fast", "small", "light", "mobile"].some((k) => lower.includes(k))) {
      return "efficiency";
    }
    if (["quality", "natural", "mos", "pesq", "stoi"].some((k) => lower.includes(k))) {
      return "quality";
    }
    if (["data", "dataset", "corpus", "sample"].some((k) => lower.includes(k))) {
      return "data";
    }

    return "application";
  }

  /**
   * Format keyword as a readable name
   */
  private formatKeywordAsName(keyword: string): string {
    return keyword
      .split(/[-_]/)
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  }

  /**
   * Calculate momentum for a trend
   */
  private calculateMomentum(techniqueIds: string[], paperIds: string[]): number {
    // Simplified momentum based on creation dates
    // Positive = recent activity, negative = older items
    const now = Date.now();
    const recentThreshold = 30 * 24 * 60 * 60 * 1000; // 30 days

    let recentCount = 0;
    let oldCount = 0;

    for (const id of techniqueIds) {
      const node = this.graph.getNode(id);
      if (node) {
        const createdAt = typeof node.createdAt === "string" ? new Date(node.createdAt) : node.createdAt;
        const age = now - createdAt.getTime();
        if (age < recentThreshold) {
          recentCount++;
        } else {
          oldCount++;
        }
      }
    }

    for (const id of paperIds) {
      const node = this.graph.getNode(id);
      if (node) {
        const createdAt = typeof node.createdAt === "string" ? new Date(node.createdAt) : node.createdAt;
        const age = now - createdAt.getTime();
        if (age < recentThreshold) {
          recentCount++;
        } else {
          oldCount++;
        }
      }
    }

    const total = recentCount + oldCount;
    if (total === 0) return 0;

    // Momentum: positive if more recent, negative if more old
    return (recentCount - oldCount) / total;
  }

  /**
   * Get domains from technique IDs
   */
  private getDomainsFromTechniques(techniqueIds: string[]): string[] {
    const domains = new Set<string>();

    for (const id of techniqueIds) {
      const node = this.graph.getNode(id) as TechniqueNode | undefined;
      if (node?.domains) {
        for (const domain of node.domains) {
          domains.add(domain);
        }
      }
    }

    return Array.from(domains);
  }

  /**
   * Generate synthetic time series (simplified)
   */
  private generateTimeSeries(currentCount: number): TrendDataPoint[] {
    const points: TrendDataPoint[] = [];
    const now = new Date();

    // Generate weekly data points for the past 12 weeks
    for (let i = 11; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * 7 * 24 * 60 * 60 * 1000);
      // Simple linear growth with noise
      const value = Math.max(0, (currentCount * (12 - i)) / 12 + Math.random() * 2 - 1);
      points.push({ timestamp, value: Math.round(value) });
    }

    return points;
  }

  /**
   * Detect recurring architecture patterns
   */
  detectArchitecturePatterns(): ArchitecturePattern[] {
    const techniques = this.graph.getNodesByType("technique") as TechniqueNode[];
    const patternCounts = new Map<string, { components: PatternComponent[]; count: number; examples: string[] }>();

    // Common patterns to detect
    const patterns: Array<{
      name: string;
      keywords: string[];
      components: PatternComponent[];
      connections: PatternConnection[];
    }> = [
      {
        name: "Encoder-Decoder",
        keywords: ["encoder", "decoder", "seq2seq"],
        components: [
          { type: "encoder", role: "input processing", required: true },
          { type: "decoder", role: "output generation", required: true },
        ],
        connections: [{ from: "encoder", to: "decoder", type: "sequential" }],
      },
      {
        name: "Transformer with Cross-Attention",
        keywords: ["transformer", "cross-attention", "multi-head"],
        components: [
          { type: "self-attention", role: "context encoding", required: true },
          { type: "cross-attention", role: "alignment", required: true },
          { type: "feedforward", role: "transformation", required: true },
        ],
        connections: [
          { from: "self-attention", to: "feedforward", type: "residual" },
          { from: "cross-attention", to: "feedforward", type: "residual" },
        ],
      },
      {
        name: "VAE with Latent Conditioning",
        keywords: ["vae", "latent", "conditioning"],
        components: [
          { type: "encoder", role: "latent encoding", required: true },
          { type: "latent", role: "representation", required: true },
          { type: "decoder", role: "generation", required: true },
          { type: "conditioning", role: "control", required: false },
        ],
        connections: [
          { from: "encoder", to: "latent", type: "sequential" },
          { from: "latent", to: "decoder", type: "sequential" },
          { from: "conditioning", to: "decoder", type: "conditioning" },
        ],
      },
      {
        name: "Diffusion with Conditioning",
        keywords: ["diffusion", "denoising", "conditioning"],
        components: [
          { type: "noise-predictor", role: "denoising", required: true },
          { type: "conditioning", role: "control", required: true },
          { type: "scheduler", role: "diffusion process", required: true },
        ],
        connections: [
          { from: "conditioning", to: "noise-predictor", type: "conditioning" },
        ],
      },
      {
        name: "GAN with Discriminator Feedback",
        keywords: ["gan", "discriminator", "generator"],
        components: [
          { type: "generator", role: "synthesis", required: true },
          { type: "discriminator", role: "feedback", required: true },
        ],
        connections: [
          { from: "generator", to: "discriminator", type: "sequential" },
        ],
      },
    ];

    // Check each technique for pattern matches
    for (const tech of techniques) {
      const text = `${tech.name} ${tech.description} ${tech.tags.join(" ")}`.toLowerCase();

      for (const pattern of patterns) {
        const matchCount = pattern.keywords.filter((k) => text.includes(k)).length;

        if (matchCount >= 2) {
          // Matches this pattern
          const existing = patternCounts.get(pattern.name) || {
            components: pattern.components,
            count: 0,
            examples: [],
          };
          existing.count++;
          existing.examples.push(tech.id);
          patternCounts.set(pattern.name, existing);
        }
      }
    }

    // Convert to ArchitecturePattern objects
    const detectedPatterns: ArchitecturePattern[] = [];
    const now = new Date();

    for (const [name, data] of Array.from(patternCounts)) {
      if (data.count < this.config.minOccurrences) {
        continue;
      }

      const patternDef = patterns.find((p) => p.name === name)!;
      const domains = this.getDomainsFromTechniques(data.examples);

      const pattern: ArchitecturePattern = {
        id: createPatternId(),
        name,
        description: `${name} architecture pattern found in ${data.count} techniques.`,
        components: data.components,
        connections: patternDef.connections,
        frequency: data.count,
        domains,
        examples: data.examples.slice(0, 5),
        firstDetected: now,
        confidence: Math.min(1, data.count / 5),
      };

      detectedPatterns.push(pattern);
      this.detectedPatterns.set(pattern.id, pattern);
    }

    return detectedPatterns.sort((a, b) => b.frequency - a.frequency);
  }

  /**
   * Track technique adoption over time
   */
  trackTechniqueAdoption(): TechniqueAdoption[] {
    const techniques = this.graph.getNodesByType("technique") as TechniqueNode[];
    const adoptionList: TechniqueAdoption[] = [];

    for (const tech of techniques) {
      // Count incoming edges (citations, implementations)
      const incomingEdges = this.graph.getIncomingEdges(tech.id);
      const citationEdges = incomingEdges.filter((e) => e.type === "cites" || e.type === "derived_from");
      const implementEdges = incomingEdges.filter((e) => e.type === "implements");

      const citationCount = citationEdges.length;
      const implementationCount = implementEdges.length;

      // Calculate adoption score
      const adoptionScore = Math.min(1, (citationCount * 0.3 + implementationCount * 0.7) / 10);

      // Determine stage
      const stage = this.determineAdoptionStage(adoptionScore, tech.createdAt);

      // Calculate trend
      const adoptionTrend = this.calculateAdoptionTrend(tech);

      // Generate time series
      const timeSeries = this.generateAdoptionTimeSeries(citationCount, implementationCount);

      const adoption: TechniqueAdoption = {
        techniqueId: tech.id,
        techniqueName: tech.name,
        adoptionScore,
        adoptionTrend,
        stage,
        timeSeries,
        citationCount,
        implementationCount,
      };

      if (stage === "emerging" || stage === "growing") {
        adoption.timeToMainstream = this.estimateTimeToMainstream(adoptionScore, adoptionTrend);
      }

      adoptionList.push(adoption);
      this.adoptionMetrics.set(tech.id, adoption);
    }

    return adoptionList.sort((a, b) => b.adoptionScore - a.adoptionScore);
  }

  /**
   * Determine adoption stage
   */
  private determineAdoptionStage(score: number, createdAt: Date | string): AdoptionStage {
    const createdDate = typeof createdAt === "string" ? new Date(createdAt) : createdAt;
    const ageWeeks = (Date.now() - createdDate.getTime()) / (7 * 24 * 60 * 60 * 1000);

    if (score < 0.1 && ageWeeks < 12) return "emerging";
    if (score < 0.3) return "growing";
    if (score < 0.7) return "mainstream";
    if (score >= 0.7 && ageWeeks > 52) return "mature";

    return "growing";
  }

  /**
   * Calculate adoption trend
   */
  private calculateAdoptionTrend(tech: TechniqueNode): number {
    // Simplified: based on how recent the technique is and its connections
    const createdAt = typeof tech.createdAt === "string" ? new Date(tech.createdAt) : tech.createdAt;
    const ageWeeks = (Date.now() - createdAt.getTime()) / (7 * 24 * 60 * 60 * 1000);
    const connections = this.graph.getIncomingEdges(tech.id).length;

    // Newer techniques with more connections = positive trend
    if (ageWeeks < 12) {
      return connections > 3 ? 0.5 : 0.2;
    } else if (ageWeeks < 52) {
      return connections > 5 ? 0.3 : 0;
    } else {
      return connections > 10 ? 0.1 : -0.1;
    }
  }

  /**
   * Estimate time to mainstream adoption
   */
  private estimateTimeToMainstream(currentScore: number, trend: number): number | undefined {
    if (trend <= 0) return undefined;

    // Simple linear estimate
    const targetScore = 0.5;
    const scoreNeeded = targetScore - currentScore;

    if (scoreNeeded <= 0) return 0;

    // Assume trend represents weekly progress
    return Math.ceil(scoreNeeded / (trend / 10));
  }

  /**
   * Generate adoption time series
   */
  private generateAdoptionTimeSeries(citations: number, implementations: number): AdoptionDataPoint[] {
    const points: AdoptionDataPoint[] = [];
    const now = new Date();

    // Generate weekly data points for the past 12 weeks
    for (let i = 11; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * 7 * 24 * 60 * 60 * 1000);
      const progress = (12 - i) / 12;
      const adoptionScore = Math.min(1, ((citations + implementations) * progress) / 10);
      const weeklyRate = (citations + implementations) / 12;

      points.push({
        timestamp,
        adoptionScore,
        newCitations: Math.round(weeklyRate * Math.random()),
        newImplementations: Math.round(weeklyRate * 0.3 * Math.random()),
      });
    }

    return points;
  }

  /**
   * Detect cross-domain transfers
   */
  detectCrossDomainTransfers(): CrossDomainTransfer[] {
    const techniques = this.graph.getNodesByType("technique") as TechniqueNode[];
    const transfers: CrossDomainTransfer[] = [];

    for (const tech of techniques) {
      if (tech.domains.length < 2) continue;

      // Check for derived_from edges that cross domains
      const derivedEdges = this.graph.getIncomingEdges(tech.id).filter((e) => e.type === "derived_from");

      for (const edge of derivedEdges) {
        const sourceNode = this.graph.getNode(edge.sourceId) as TechniqueNode | undefined;
        if (!sourceNode || sourceNode.type !== "technique") continue;

        const sourceDomains = new Set(sourceNode.domains);
        const targetDomains = new Set(tech.domains);

        // Find domains that are new (in target but not in source)
        const newDomains = Array.from(targetDomains).filter((d) => !sourceDomains.has(d));

        if (newDomains.length > 0 && sourceDomains.size > 0) {
          // Cross-domain transfer detected
          for (const newDomain of newDomains) {
            for (const sourceDomain of Array.from(sourceDomains)) {
              const transfer: CrossDomainTransfer = {
                id: createTransferId(),
                techniqueId: tech.id,
                techniqueName: tech.name,
                sourceDomain,
                targetDomain: newDomain,
                successScore: this.estimateTransferSuccess(tech),
                adaptations: this.detectAdaptations(sourceNode, tech),
                detectedAt: new Date(),
                evidencePapers: [],
                confidence: 0.6,
              };

              transfers.push(transfer);
              this.crossDomainTransfers.set(transfer.id, transfer);
            }
          }
        }
      }
    }

    return transfers.sort((a, b) => b.successScore - a.successScore);
  }

  /**
   * Estimate success of a cross-domain transfer
   */
  private estimateTransferSuccess(tech: TechniqueNode): number {
    // Based on connections and metrics
    const connections = this.graph.getIncomingEdges(tech.id).length;
    let score = Math.min(0.5, connections * 0.1);

    // Add metric-based score if available
    if (tech.metrics) {
      const values = Object.values(tech.metrics);
      if (values.length > 0) {
        const avgMetric = values.reduce((a, b) => a + b, 0) / values.length;
        score += Math.min(0.5, avgMetric / 200);
      }
    }

    return Math.min(1, score);
  }

  /**
   * Detect adaptations made during transfer
   */
  private detectAdaptations(source: TechniqueNode, target: TechniqueNode): string[] {
    const adaptations: string[] = [];

    // Compare tags to find differences
    const sourceTags = new Set(source.tags.map((t) => t.toLowerCase()));
    const targetTags = new Set(target.tags.map((t) => t.toLowerCase()));

    const newTags = Array.from(targetTags).filter((t) => !sourceTags.has(t));

    for (const tag of newTags) {
      if (tag.includes("adapt") || tag.includes("modif") || tag.includes("custom")) {
        adaptations.push(`Added ${tag} capability`);
      }
    }

    // Check for domain-specific adaptations
    const sourceDomains = new Set(source.domains);
    const newDomains = target.domains.filter((d) => !sourceDomains.has(d));

    for (const domain of newDomains) {
      adaptations.push(`Adapted for ${domain} domain`);
    }

    if (adaptations.length === 0) {
      adaptations.push("Direct transfer with minimal modifications");
    }

    return adaptations;
  }

  /**
   * Generate summary statistics
   */
  private generateSummary(
    trends: ResearchTrend[],
    patterns: ArchitecturePattern[],
    adoption: TechniqueAdoption[],
    transfers: CrossDomainTransfer[]
  ): ReportSummary {
    const techniques = this.graph.getNodesByType("technique");
    const papers = this.graph.getNodesByType("paper");

    // Count trends by momentum
    const growingTrends = trends.filter((t) => t.momentum > 0).length;
    const decliningTrends = trends.filter((t) => t.momentum < 0).length;

    // Top keywords
    const keywordCounts = new Map<string, number>();
    for (const trend of trends) {
      for (const keyword of trend.keywords) {
        keywordCounts.set(keyword, (keywordCounts.get(keyword) || 0) + 1);
      }
    }
    const topKeywords = Array.from(keywordCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([keyword, count]) => ({ keyword, count }));

    // Active domains
    const domainActivity = new Map<string, number>();
    for (const trend of trends) {
      for (const domain of trend.domains) {
        domainActivity.set(domain, (domainActivity.get(domain) || 0) + trend.strength);
      }
    }
    const activeDomains = Array.from(domainActivity.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([domain, activity]) => ({ domain, activity }));

    return {
      techniquesAnalyzed: techniques.length,
      papersAnalyzed: papers.length,
      newTrends: trends.filter((t) => t.firstDetected > new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)).length,
      growingTrends,
      decliningTrends,
      patternsDetected: patterns.length,
      transfersDetected: transfers.length,
      topKeywords,
      activeDomains,
    };
  }

  /**
   * Get detected trends
   */
  getTrends(): ResearchTrend[] {
    return Array.from(this.detectedTrends.values());
  }

  /**
   * Get emerging trends (high momentum, recent)
   */
  getEmergingTrends(limit = 10): ResearchTrend[] {
    return this.getTrends()
      .filter((t) => t.momentum > 0.2)
      .sort((a, b) => b.momentum - a.momentum)
      .slice(0, limit);
  }

  /**
   * Get detected patterns
   */
  getPatterns(): ArchitecturePattern[] {
    return Array.from(this.detectedPatterns.values());
  }

  /**
   * Get adoption metrics
   */
  getAdoptionMetrics(): TechniqueAdoption[] {
    return Array.from(this.adoptionMetrics.values());
  }

  /**
   * Get cross-domain transfers
   */
  getCrossDomainTransfers(): CrossDomainTransfer[] {
    return Array.from(this.crossDomainTransfers.values());
  }

  /**
   * Clear all detected patterns
   */
  clear(): void {
    this.detectedTrends.clear();
    this.detectedPatterns.clear();
    this.adoptionMetrics.clear();
    this.crossDomainTransfers.clear();
  }
}

/**
 * Create a pattern recognition engine
 */
export function createPatternRecognition(
  graph: KnowledgeGraph,
  config?: Partial<PatternRecognitionConfig>
): PatternRecognition {
  return new PatternRecognition(graph, config);
}
