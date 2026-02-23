/**
 * Gap Analysis Service
 *
 * Analyzes the knowledge graph to identify research gaps, missing connections,
 * unexplored combinations, and opportunities for new research.
 */

import {
  ResearchGap,
  GapOpportunity,
  GapType,
  GapSeverity,
  OpportunityType,
  EffortLevel,
  GapEvidence,
  ResearchLandscape,
  LandscapeNode,
  LandscapeEdge,
  LandscapeCluster,
  GapReport,
  CoverageMetrics,
  GapAnalysisConfig,
  DEFAULT_GAP_CONFIG,
  createGapId,
  createOpportunityId,
  createLandscapeId,
  createReportId,
  createDefaultEffort,
  calculatePriorityScore,
} from "./types";
import {
  KnowledgeGraph,
  TechniqueNode,
  TechniqueCategory,
  GraphEdge,
  isTechniqueNode,
} from "../knowledge-graph";

/**
 * Gap Analyzer class for detecting research gaps and opportunities
 */
export class GapAnalyzer {
  private config: GapAnalysisConfig;

  constructor(config: Partial<GapAnalysisConfig> = {}) {
    this.config = { ...DEFAULT_GAP_CONFIG, ...config };
  }

  /**
   * Analyze a knowledge graph and return all gaps
   */
  analyze(graph: KnowledgeGraph): {
    gaps: ResearchGap[];
    opportunities: GapOpportunity[];
  } {
    const gaps: ResearchGap[] = [];
    const opportunities: GapOpportunity[] = [];

    // Get all techniques
    const techniques = graph
      .getAllNodes()
      .filter(isTechniqueNode) as TechniqueNode[];

    // Detect different types of gaps
    const missingConnections = this.detectMissingConnections(graph, techniques);
    const unexploredCombinations = this.detectUnexploredCombinations(
      graph,
      techniques
    );
    const underservedDomains = this.detectUnderservedDomains(techniques);
    const evaluationGaps = this.detectEvaluationGaps(techniques);

    gaps.push(
      ...missingConnections,
      ...unexploredCombinations,
      ...underservedDomains,
      ...evaluationGaps
    );

    // Filter by configuration
    const filteredGaps = this.filterGaps(gaps);

    // Generate opportunities from gaps
    for (const gap of filteredGaps) {
      const gapOpportunities = this.generateOpportunities(gap, techniques);
      opportunities.push(...gapOpportunities);
    }

    // Sort and limit
    const sortedGaps = filteredGaps
      .sort((a, b) => this.getGapPriority(b) - this.getGapPriority(a))
      .slice(0, this.config.maxGaps);

    const sortedOpportunities = opportunities
      .sort((a, b) => b.priorityScore - a.priorityScore)
      .slice(0, this.config.maxOpportunities);

    return { gaps: sortedGaps, opportunities: sortedOpportunities };
  }

  /**
   * Detect missing connections between related techniques
   */
  private detectMissingConnections(
    graph: KnowledgeGraph,
    techniques: TechniqueNode[]
  ): ResearchGap[] {
    const gaps: ResearchGap[] = [];

    for (let i = 0; i < techniques.length; i++) {
      for (let j = i + 1; j < techniques.length; j++) {
        const techA = techniques[i];
        const techB = techniques[j];

        // Check if they share domains but aren't connected
        const sharedDomains = techA.domains.filter((d) =>
          techB.domains.includes(d)
        );

        if (sharedDomains.length > 0) {
          // Check if there's an edge between these nodes
          const edgesBetween = graph.getEdgesBetween(techA.id, techB.id);
          const reverseEdges = graph.getEdgesBetween(techB.id, techA.id);
          const connected = edgesBetween.length > 0 || reverseEdges.length > 0;

          if (!connected) {
            // Check if they share tags (indicating similarity)
            const sharedTags = techA.tags.filter((t) => techB.tags.includes(t));

            if (sharedTags.length >= 2) {
              const confidence = Math.min(
                0.5 + sharedTags.length * 0.1 + sharedDomains.length * 0.1,
                1.0
              );

              gaps.push({
                id: createGapId(),
                type: "missing_connection",
                title: `Missing connection: ${techA.name} - ${techB.name}`,
                description: `These techniques share ${sharedDomains.length} domain(s) and ${sharedTags.length} tag(s) but have no documented relationship.`,
                severity: this.calculateSeverity(confidence, sharedDomains.length),
                domains: sharedDomains,
                relatedTechniques: [techA.id, techB.id],
                evidence: [
                  {
                    type: "missing_edge",
                    description: `No edge exists between ${techA.name} and ${techB.name}`,
                    confidence,
                    sourceId: techA.id,
                  },
                ],
                detectedAt: new Date(),
                confidence,
                suggestedApproaches: [
                  `Research if ${techA.name} can enhance ${techB.name}`,
                  `Explore combining ${sharedTags.join(", ")} approaches`,
                ],
              });
            }
          }
        }
      }
    }

    return gaps;
  }

  /**
   * Detect unexplored technique combinations
   */
  private detectUnexploredCombinations(
    graph: KnowledgeGraph,
    techniques: TechniqueNode[]
  ): ResearchGap[] {
    const gaps: ResearchGap[] = [];

    // Find techniques that would complement each other
    const architectureTypes = new Map<string, TechniqueNode[]>();

    for (const tech of techniques) {
      if (tech.architecture) {
        const existing = architectureTypes.get(tech.architecture) || [];
        existing.push(tech);
        architectureTypes.set(tech.architecture, existing);
      }
    }

    // Look for complementary architectures not yet combined
    const complementaryPairs: [string, string][] = [
      ["transformer", "diffusion"],
      ["transformer", "vae"],
      ["vae", "gan"],
      ["diffusion", "flow"],
      ["attention", "convolution"],
    ];

    for (const [archA, archB] of complementaryPairs) {
      const techsA = architectureTypes.get(archA) || [];
      const techsB = architectureTypes.get(archB) || [];

      // Check if any technique combines these architectures
      const combined = techniques.filter(
        (t) =>
          t.tags.some((tag) => tag.toLowerCase().includes(archA)) &&
          t.tags.some((tag) => tag.toLowerCase().includes(archB))
      );

      if (combined.length === 0 && techsA.length > 0 && techsB.length > 0) {
        gaps.push({
          id: createGapId(),
          type: "unexplored_combination",
          title: `Unexplored: ${archA} + ${archB} combination`,
          description: `No techniques combine ${archA} and ${archB} architectures, which could be beneficial.`,
          severity: "medium",
          domains: Array.from(new Set([
              ...techsA.flatMap((t) => t.domains),
              ...techsB.flatMap((t) => t.domains),
            ])),
          relatedTechniques: [...techsA.map((t) => t.id), ...techsB.map((t) => t.id)],
          evidence: [
            {
              type: "analysis",
              description: `Found ${techsA.length} ${archA} techniques and ${techsB.length} ${archB} techniques but no combination`,
              confidence: 0.7,
            },
          ],
          detectedAt: new Date(),
          confidence: 0.7,
          suggestedApproaches: [
            `Research ${archA}-${archB} hybrid architectures`,
            `Experiment with ${archA} encoder and ${archB} decoder`,
          ],
        });
      }
    }

    return gaps;
  }

  /**
   * Detect underserved domains
   */
  private detectUnderservedDomains(techniques: TechniqueNode[]): ResearchGap[] {
    const gaps: ResearchGap[] = [];

    // Count techniques per domain
    const domainCounts = new Map<string, number>();
    for (const tech of techniques) {
      for (const domain of tech.domains) {
        domainCounts.set(domain, (domainCounts.get(domain) || 0) + 1);
      }
    }

    // Find underserved domains
    for (const [domain, count] of Array.from(domainCounts)) {
      if (count < this.config.minTechniquesPerDomain) {
        const relatedTechs = techniques.filter((t) => t.domains.includes(domain));

        gaps.push({
          id: createGapId(),
          type: "underserved_domain",
          title: `Underserved domain: ${domain}`,
          description: `The ${domain} domain has only ${count} technique(s), which is below the recommended minimum of ${this.config.minTechniquesPerDomain}.`,
          severity: count === 1 ? "high" : "medium",
          domains: [domain],
          relatedTechniques: relatedTechs.map((t) => t.id),
          evidence: [
            {
              type: "low_coverage",
              description: `Only ${count} technique(s) in ${domain}`,
              confidence: 0.9,
            },
          ],
          detectedAt: new Date(),
          confidence: 0.9,
          suggestedApproaches: [
            `Research existing ${domain} techniques from other sources`,
            `Transfer successful techniques from related domains`,
          ],
        });
      }
    }

    return gaps;
  }

  /**
   * Detect evaluation gaps
   */
  private detectEvaluationGaps(techniques: TechniqueNode[]): ResearchGap[] {
    const gaps: ResearchGap[] = [];

    // Find techniques without proper metrics
    const techsWithoutMetrics = techniques.filter(
      (t) => !t.metrics || Object.keys(t.metrics).length === 0
    );

    if (techsWithoutMetrics.length > techniques.length * 0.3) {
      gaps.push({
        id: createGapId(),
        type: "evaluation_gap",
        title: "Widespread lack of evaluation metrics",
        description: `${techsWithoutMetrics.length} out of ${techniques.length} techniques (${Math.round((techsWithoutMetrics.length / techniques.length) * 100)}%) lack proper evaluation metrics.`,
        severity: "high",
        domains: Array.from(new Set(techsWithoutMetrics.flatMap((t) => t.domains))),
        relatedTechniques: techsWithoutMetrics.map((t) => t.id),
        evidence: [
          {
            type: "analysis",
            description: `${techsWithoutMetrics.length} techniques without metrics`,
            confidence: 0.95,
          },
        ],
        detectedAt: new Date(),
        confidence: 0.95,
        suggestedApproaches: [
          "Create standardized evaluation benchmarks",
          "Develop automated evaluation pipelines",
        ],
      });
    }

    return gaps;
  }

  /**
   * Generate opportunities from a gap
   */
  private generateOpportunities(
    gap: ResearchGap,
    techniques: TechniqueNode[]
  ): GapOpportunity[] {
    const opportunities: GapOpportunity[] = [];

    switch (gap.type) {
      case "missing_connection": {
        const relatedTechs = techniques.filter((t) =>
          gap.relatedTechniques.includes(t.id)
        );
        if (relatedTechs.length >= 2) {
          const effort = createDefaultEffort("small");
          const impactScore = gap.confidence * 0.7;

          opportunities.push({
            id: createOpportunityId(),
            gapId: gap.id,
            title: `Connect ${relatedTechs[0].name} with ${relatedTechs[1].name}`,
            description: `Investigate the relationship between these techniques and document findings.`,
            type: "improvement",
            impactScore,
            effort,
            prerequisites: [],
            potentialOutcomes: [
              "Better understanding of technique relationships",
              "Potential synergy discovery",
            ],
            suggestedTechniques: gap.relatedTechniques,
            priorityScore: calculatePriorityScore(impactScore, effort),
            identifiedAt: new Date(),
            confidence: gap.confidence * 0.9,
          });
        }
        break;
      }

      case "unexplored_combination": {
        const effort = createDefaultEffort("medium");
        effort.skills = ["deep learning", "research"];
        const impactScore = 0.8;

        opportunities.push({
          id: createOpportunityId(),
          gapId: gap.id,
          title: `Explore ${gap.title.replace("Unexplored: ", "")}`,
          description: gap.description,
          type: "technique_combination",
          impactScore,
          effort,
          prerequisites: ["Understanding of both architectures"],
          potentialOutcomes: [
            "Novel hybrid technique",
            "Potential quality improvements",
          ],
          suggestedTechniques: gap.relatedTechniques,
          priorityScore: calculatePriorityScore(impactScore, effort),
          identifiedAt: new Date(),
          confidence: gap.confidence,
        });
        break;
      }

      case "underserved_domain": {
        const effort = createDefaultEffort("large");
        effort.skills = ["domain expertise", "research"];
        const impactScore = 0.75;

        opportunities.push({
          id: createOpportunityId(),
          gapId: gap.id,
          title: `Expand ${gap.domains[0]} coverage`,
          description: `Develop new techniques or transfer existing ones to the ${gap.domains[0]} domain.`,
          type: "domain_transfer",
          impactScore,
          effort,
          prerequisites: ["Domain knowledge"],
          potentialOutcomes: [
            "Better domain coverage",
            "Cross-domain innovation",
          ],
          suggestedTechniques: gap.relatedTechniques,
          priorityScore: calculatePriorityScore(impactScore, effort),
          identifiedAt: new Date(),
          confidence: gap.confidence * 0.8,
        });
        break;
      }

      case "evaluation_gap": {
        const effort = createDefaultEffort("medium");
        effort.skills = ["evaluation", "benchmarking"];
        const impactScore = 0.85;

        opportunities.push({
          id: createOpportunityId(),
          gapId: gap.id,
          title: "Create evaluation framework",
          description: "Develop standardized evaluation metrics and benchmarks.",
          type: "evaluation",
          impactScore,
          effort,
          prerequisites: ["Understanding of quality metrics"],
          potentialOutcomes: [
            "Standardized evaluation",
            "Better technique comparison",
          ],
          suggestedTechniques: gap.relatedTechniques.slice(0, 5),
          priorityScore: calculatePriorityScore(impactScore, effort),
          identifiedAt: new Date(),
          confidence: gap.confidence,
        });
        break;
      }
    }

    return opportunities;
  }

  /**
   * Generate research landscape
   */
  generateLandscape(
    graph: KnowledgeGraph,
    domain?: string
  ): ResearchLandscape {
    const techniques = graph
      .getAllNodes()
      .filter(isTechniqueNode)
      .filter((t) => !domain || t.domains.includes(domain)) as TechniqueNode[];

    const { gaps, opportunities } = this.analyze(graph);
    const filteredGaps = domain
      ? gaps.filter((g) => g.domains.includes(domain))
      : gaps;
    const filteredOpportunities = domain
      ? opportunities.filter((o) =>
          gaps.find((g) => g.id === o.gapId && g.domains.includes(domain))
        )
      : opportunities;

    // Generate nodes
    const nodes: LandscapeNode[] = [];
    const edges: LandscapeEdge[] = [];

    // Add technique nodes
    for (let i = 0; i < techniques.length; i++) {
      const tech = techniques[i];
      const angle = (2 * Math.PI * i) / techniques.length;
      const radius = 0.3 + Math.random() * 0.2;

      nodes.push({
        id: tech.id,
        label: tech.name,
        type: "technique",
        x: 0.5 + radius * Math.cos(angle),
        y: 0.5 + radius * Math.sin(angle),
        size: tech.hasImplementation ? 1.2 : 1.0,
        color: this.getCategoryColor(tech.category),
        metadata: { category: tech.category, domains: tech.domains },
      });
    }

    // Add gap nodes
    for (const gap of filteredGaps.slice(0, 10)) {
      nodes.push({
        id: gap.id,
        label: gap.title,
        type: "gap",
        x: 0.5 + (Math.random() - 0.5) * 0.4,
        y: 0.5 + (Math.random() - 0.5) * 0.4,
        size: gap.severity === "critical" ? 1.5 : gap.severity === "high" ? 1.2 : 1.0,
        color: "#ef4444",
        metadata: { type: gap.type, severity: gap.severity },
      });
    }

    // Add edges from graph
    const graphEdges = graph.getAllEdges();
    for (const edge of graphEdges) {
      if (
        nodes.find((n) => n.id === edge.sourceId) &&
        nodes.find((n) => n.id === edge.targetId)
      ) {
        edges.push({
          id: edge.id,
          source: edge.sourceId,
          target: edge.targetId,
          weight: edge.weight,
          type: "connection",
          dashed: false,
          color: "#94a3b8",
        });
      }
    }

    // Add gap edges (dashed)
    for (const gap of filteredGaps.slice(0, 10)) {
      if (gap.relatedTechniques.length >= 2) {
        edges.push({
          id: `gap-edge-${gap.id}`,
          source: gap.relatedTechniques[0],
          target: gap.relatedTechniques[1],
          weight: gap.confidence,
          type: "gap",
          dashed: true,
          color: "#ef4444",
        });
      }
    }

    // Simple clustering by category
    const clusters = this.generateClusters(nodes, techniques);

    return {
      id: createLandscapeId(),
      domain: domain || "all",
      nodes,
      edges,
      clusters,
      gaps: filteredGaps,
      opportunities: filteredOpportunities,
      generatedAt: new Date(),
      coverageScore: this.calculateCoverageScore(techniques),
      densityScore: this.calculateDensityScore(graphEdges.length, techniques.length),
    };
  }

  /**
   * Generate gap report
   */
  generateReport(graph: KnowledgeGraph, domain?: string): GapReport {
    const { gaps, opportunities } = this.analyze(graph);

    const filteredGaps = domain
      ? gaps.filter((g) => g.domains.includes(domain))
      : gaps;

    const gapsByType: Record<GapType, number> = {
      missing_connection: 0,
      unexplored_combination: 0,
      missing_technique: 0,
      underserved_domain: 0,
      missing_baseline: 0,
      evaluation_gap: 0,
      theoretical_gap: 0,
      practical_gap: 0,
    };

    const gapsBySeverity: Record<GapSeverity, number> = {
      critical: 0,
      high: 0,
      medium: 0,
      low: 0,
      informational: 0,
    };

    for (const gap of filteredGaps) {
      gapsByType[gap.type]++;
      gapsBySeverity[gap.severity]++;
    }

    const techniques = graph
      .getAllNodes()
      .filter(isTechniqueNode) as TechniqueNode[];

    const coverage = this.calculateCoverageMetrics(graph, techniques);

    return {
      id: createReportId(),
      generatedAt: new Date(),
      domain: domain || "all",
      techniquesAnalyzed: techniques.length,
      totalGaps: filteredGaps.length,
      gapsByType,
      gapsBySeverity,
      totalOpportunities: opportunities.length,
      topOpportunities: opportunities.slice(0, 5),
      coverage,
      recommendations: this.generateRecommendations(filteredGaps, coverage),
    };
  }

  /**
   * Filter gaps based on configuration
   */
  private filterGaps(gaps: ResearchGap[]): ResearchGap[] {
    return gaps.filter((gap) => {
      // Filter by confidence
      if (gap.confidence < this.config.minConfidence) return false;

      // Filter by severity
      if (!this.config.severityFilter.includes(gap.severity)) return false;

      // Filter by domain
      if (this.config.domains.length > 0) {
        const hasMatchingDomain = gap.domains.some((d) =>
          this.config.domains.includes(d)
        );
        if (!hasMatchingDomain) return false;
      }

      return true;
    });
  }

  /**
   * Calculate severity based on confidence and other factors
   */
  private calculateSeverity(
    confidence: number,
    factorCount: number
  ): GapSeverity {
    const score = confidence * 0.6 + (factorCount / 5) * 0.4;
    if (score >= 0.8) return "critical";
    if (score >= 0.6) return "high";
    if (score >= 0.4) return "medium";
    if (score >= 0.2) return "low";
    return "informational";
  }

  /**
   * Get gap priority for sorting
   */
  private getGapPriority(gap: ResearchGap): number {
    const severityWeight: Record<GapSeverity, number> = {
      critical: 1.0,
      high: 0.8,
      medium: 0.6,
      low: 0.4,
      informational: 0.2,
    };

    return severityWeight[gap.severity] * gap.confidence;
  }

  /**
   * Get color for technique category
   */
  private getCategoryColor(category: TechniqueCategory): string {
    const colors: Record<TechniqueCategory, string> = {
      architecture: "#3b82f6",
      conditioning: "#8b5cf6",
      "loss-function": "#ec4899",
      training: "#10b981",
      inference: "#f59e0b",
      preprocessing: "#06b6d4",
      postprocessing: "#84cc16",
      evaluation: "#6366f1",
      "data-augmentation": "#f97316",
      other: "#6b7280",
    };
    return colors[category] || "#6b7280";
  }

  /**
   * Generate clusters from nodes
   */
  private generateClusters(
    nodes: LandscapeNode[],
    techniques: TechniqueNode[]
  ): LandscapeCluster[] {
    const clusters: LandscapeCluster[] = [];
    const categoryNodes = new Map<TechniqueCategory, string[]>();

    for (const tech of techniques) {
      const existing = categoryNodes.get(tech.category) || [];
      existing.push(tech.id);
      categoryNodes.set(tech.category, existing);
    }

    let clusterId = 0;
    for (const [category, nodeIds] of Array.from(categoryNodes)) {
      if (nodeIds.length >= 2) {
        const clusterNodes = nodes.filter((n) => nodeIds.includes(n.id));
        const centroid = {
          x: clusterNodes.reduce((sum, n) => sum + n.x, 0) / clusterNodes.length,
          y: clusterNodes.reduce((sum, n) => sum + n.y, 0) / clusterNodes.length,
        };

        clusters.push({
          id: `cluster-${clusterId++}`,
          label: category,
          nodeIds,
          centroid,
          size: nodeIds.length,
          dominantCategory: category,
        });
      }
    }

    return clusters;
  }

  /**
   * Calculate coverage score
   */
  private calculateCoverageScore(techniques: TechniqueNode[]): number {
    if (techniques.length === 0) return 0;

    // Count unique domains and categories
    const domains = new Set(techniques.flatMap((t) => t.domains));
    const categories = new Set(techniques.map((t) => t.category));

    // Expected minimums
    const expectedDomains = 5;
    const expectedCategories = 8;

    const domainScore = Math.min(domains.size / expectedDomains, 1);
    const categoryScore = Math.min(categories.size / expectedCategories, 1);

    return (domainScore + categoryScore) / 2;
  }

  /**
   * Calculate density score
   */
  private calculateDensityScore(edgeCount: number, nodeCount: number): number {
    if (nodeCount < 2) return 0;

    // Maximum possible edges in undirected graph
    const maxEdges = (nodeCount * (nodeCount - 1)) / 2;
    return Math.min(edgeCount / maxEdges, 1);
  }

  /**
   * Calculate coverage metrics
   */
  private calculateCoverageMetrics(
    graph: KnowledgeGraph,
    techniques: TechniqueNode[]
  ): CoverageMetrics {
    const domains = new Set<string>();
    const categories = new Set<TechniqueCategory>();
    const domainCounts = new Map<string, number>();

    for (const tech of techniques) {
      categories.add(tech.category);
      for (const domain of tech.domains) {
        domains.add(domain);
        domainCounts.set(domain, (domainCounts.get(domain) || 0) + 1);
      }
    }

    const edges = graph.getAllEdges();
    const density =
      techniques.length > 1
        ? edges.length / ((techniques.length * (techniques.length - 1)) / 2)
        : 0;

    const domainsWithGaps = Array.from(domainCounts.entries())
      .filter(([, count]) => count < this.config.minTechniquesPerDomain)
      .map(([domain]) => domain);

    const allCategories: TechniqueCategory[] = [
      "architecture",
      "conditioning",
      "loss-function",
      "training",
      "inference",
      "preprocessing",
      "postprocessing",
      "evaluation",
      "data-augmentation",
    ];

    const categoriesWithGaps = allCategories.filter(
      (cat) => !categories.has(cat)
    );

    return {
      domainCoverage: Math.min(domains.size / 5, 1),
      categoryCoverage: categories.size / allCategories.length,
      connectionDensity: Math.min(density, 1),
      avgTechniquesPerDomain:
        domains.size > 0 ? techniques.length / domains.size : 0,
      domainsWithGaps,
      categoriesWithGaps,
    };
  }

  /**
   * Generate recommendations based on analysis
   */
  private generateRecommendations(
    gaps: ResearchGap[],
    coverage: CoverageMetrics
  ): string[] {
    const recommendations: string[] = [];

    const criticalGaps = gaps.filter((g) => g.severity === "critical").length;
    const highGaps = gaps.filter((g) => g.severity === "high").length;

    if (criticalGaps > 0) {
      recommendations.push(
        `Address ${criticalGaps} critical gap(s) immediately`
      );
    }

    if (highGaps > 5) {
      recommendations.push(
        `Prioritize resolving ${highGaps} high-severity gaps`
      );
    }

    if (coverage.domainCoverage < 0.5) {
      recommendations.push("Expand research coverage to more domains");
    }

    if (coverage.categoryCoverage < 0.6) {
      recommendations.push(
        `Add techniques in missing categories: ${coverage.categoriesWithGaps.join(", ")}`
      );
    }

    if (coverage.connectionDensity < 0.1) {
      recommendations.push(
        "Improve documentation of technique relationships"
      );
    }

    if (coverage.domainsWithGaps.length > 0) {
      recommendations.push(
        `Increase technique coverage in: ${coverage.domainsWithGaps.join(", ")}`
      );
    }

    return recommendations;
  }
}

/**
 * Global gap analyzer instance
 */
let globalGapAnalyzer: GapAnalyzer | null = null;

export function getGlobalGapAnalyzer(): GapAnalyzer {
  if (!globalGapAnalyzer) {
    globalGapAnalyzer = new GapAnalyzer();
  }
  return globalGapAnalyzer;
}

export function resetGlobalGapAnalyzer(): void {
  globalGapAnalyzer = null;
}
