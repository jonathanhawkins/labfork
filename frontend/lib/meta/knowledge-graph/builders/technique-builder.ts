/**
 * Technique Node Builder
 *
 * Extracts technique information from papers and other sources
 * to build TechniqueNode objects for the knowledge graph.
 */

import { Paper, ExtractedTechnique } from "../../../papers/types";
import {
  TechniqueNode,
  TechniqueCategory,
  TechniqueMetrics,
  GraphEdge,
  createTechniqueNode,
  createEdge,
  generateNodeId,
} from "../types";

// ============================================================================
// Types
// ============================================================================

export interface TechniqueExtraction {
  /** Extracted techniques */
  techniques: TechniqueNode[];
  /** Relationships between techniques */
  edges: GraphEdge[];
  /** Extraction metadata */
  metadata: {
    sourcePaperId: string;
    extractedAt: string;
    confidence: number;
    method: "analysis" | "llm" | "manual";
  };
}

export interface TechniquePattern {
  /** Pattern name */
  name: string;
  /** Regex or string patterns to match */
  patterns: RegExp[];
  /** Category if matched */
  category: TechniqueCategory;
  /** Complexity hint */
  complexity: TechniqueNode["complexity"];
}

// ============================================================================
// Pattern Definitions
// ============================================================================

const ARCHITECTURE_PATTERNS: TechniquePattern[] = [
  {
    name: "Transformer",
    patterns: [/transformer/i, /attention\s+mechanism/i, /self-attention/i],
    category: "architecture",
    complexity: "complex",
  },
  {
    name: "VAE",
    patterns: [/variational\s+auto-?encoder/i, /\bvae\b/i, /latent\s+space/i],
    category: "architecture",
    complexity: "complex",
  },
  {
    name: "GAN",
    patterns: [
      /generative\s+adversarial/i,
      /\bgan\b/i,
      /discriminator/i,
      /generator\s+network/i,
    ],
    category: "architecture",
    complexity: "complex",
  },
  {
    name: "Diffusion",
    patterns: [
      /diffusion\s+model/i,
      /denoising/i,
      /score\s+matching/i,
      /ddpm/i,
    ],
    category: "architecture",
    complexity: "complex",
  },
  {
    name: "CNN",
    patterns: [
      /convolutional\s+neural/i,
      /\bcnn\b/i,
      /conv\d+d/i,
      /convolution/i,
    ],
    category: "architecture",
    complexity: "moderate",
  },
  {
    name: "RNN",
    patterns: [/recurrent\s+neural/i, /\brnn\b/i, /\blstm\b/i, /\bgru\b/i],
    category: "architecture",
    complexity: "moderate",
  },
  {
    name: "Flow",
    patterns: [/normalizing\s+flow/i, /flow-based/i, /invertible\s+network/i],
    category: "architecture",
    complexity: "complex",
  },
];

const CONDITIONING_PATTERNS: TechniquePattern[] = [
  {
    name: "Cross-Attention Conditioning",
    patterns: [/cross-attention/i, /cross\s+attention/i],
    category: "conditioning",
    complexity: "moderate",
  },
  {
    name: "AdaIN",
    patterns: [/adaptive\s+instance\s+norm/i, /adain/i, /style\s+transfer/i],
    category: "conditioning",
    complexity: "moderate",
  },
  {
    name: "FiLM",
    patterns: [/film\s+layer/i, /feature-wise\s+linear/i, /\bfilm\b/i],
    category: "conditioning",
    complexity: "simple",
  },
  {
    name: "Prompt Conditioning",
    patterns: [
      /prompt\s+condition/i,
      /text\s+embedding/i,
      /language\s+condition/i,
    ],
    category: "conditioning",
    complexity: "moderate",
  },
  {
    name: "Reference Conditioning",
    patterns: [
      /reference\s+audio/i,
      /speaker\s+embedding/i,
      /voice\s+condition/i,
    ],
    category: "conditioning",
    complexity: "moderate",
  },
];

const LOSS_PATTERNS: TechniquePattern[] = [
  {
    name: "Adversarial Loss",
    patterns: [/adversarial\s+loss/i, /discriminator\s+loss/i, /gan\s+loss/i],
    category: "loss-function",
    complexity: "moderate",
  },
  {
    name: "Reconstruction Loss",
    patterns: [/reconstruction\s+loss/i, /mse\s+loss/i, /l[12]\s+loss/i],
    category: "loss-function",
    complexity: "simple",
  },
  {
    name: "Perceptual Loss",
    patterns: [/perceptual\s+loss/i, /vgg\s+loss/i, /feature\s+matching/i],
    category: "loss-function",
    complexity: "moderate",
  },
  {
    name: "Contrastive Loss",
    patterns: [/contrastive\s+loss/i, /infonce/i, /triplet\s+loss/i],
    category: "loss-function",
    complexity: "moderate",
  },
  {
    name: "KL Divergence",
    patterns: [/kl\s+divergence/i, /kullback-leibler/i, /elbo/i],
    category: "loss-function",
    complexity: "moderate",
  },
];

const TRAINING_PATTERNS: TechniquePattern[] = [
  {
    name: "Multi-Task Learning",
    patterns: [/multi-task/i, /auxiliary\s+task/i, /joint\s+training/i],
    category: "training",
    complexity: "moderate",
  },
  {
    name: "Curriculum Learning",
    patterns: [/curriculum/i, /progressive\s+training/i, /easy-to-hard/i],
    category: "training",
    complexity: "moderate",
  },
  {
    name: "Self-Supervised",
    patterns: [
      /self-supervised/i,
      /pretext\s+task/i,
      /unsupervised\s+pretrain/i,
    ],
    category: "training",
    complexity: "complex",
  },
  {
    name: "Reinforcement Learning",
    patterns: [/reinforcement\s+learning/i, /\brl\b/i, /reward\s+model/i],
    category: "training",
    complexity: "research",
  },
  {
    name: "Fine-tuning",
    patterns: [/fine-tun/i, /adapter/i, /lora/i, /parameter-efficient/i],
    category: "training",
    complexity: "simple",
  },
];

const ALL_PATTERNS = [
  ...ARCHITECTURE_PATTERNS,
  ...CONDITIONING_PATTERNS,
  ...LOSS_PATTERNS,
  ...TRAINING_PATTERNS,
];

// ============================================================================
// Builder Class
// ============================================================================

export class TechniqueBuilder {
  private existingTechniques: Map<string, TechniqueNode>;
  private paperTechniqueMap: Map<string, string[]>;

  constructor() {
    this.existingTechniques = new Map();
    this.paperTechniqueMap = new Map();
  }

  /**
   * Register existing techniques from the graph
   */
  registerExisting(techniques: TechniqueNode[]): void {
    for (const technique of techniques) {
      this.existingTechniques.set(this.normalizeKey(technique.name), technique);
    }
  }

  /**
   * Extract techniques from a paper
   */
  extractFromPaper(paper: Paper): TechniqueExtraction {
    const techniques: TechniqueNode[] = [];
    const edges: GraphEdge[] = [];
    const seenNames = new Set<string>();

    // Combine text sources for analysis
    const textToAnalyze = [
      paper.metadata.title,
      paper.metadata.abstract,
      ...(paper.analysis?.techniques?.map((t) => t.description) || []),
    ]
      .filter(Boolean)
      .join(" ");

    // Extract from patterns
    for (const pattern of ALL_PATTERNS) {
      if (this.matchesPattern(textToAnalyze, pattern)) {
        const normalizedName = this.normalizeKey(pattern.name);

        if (seenNames.has(normalizedName)) continue;
        seenNames.add(normalizedName);

        // Check if technique already exists
        const existing = this.existingTechniques.get(normalizedName);
        if (existing) {
          // Just link paper to existing technique
          if (!existing.sourcePaperIds.includes(paper.id)) {
            existing.sourcePaperIds.push(paper.id);
          }
          techniques.push(existing);
        } else {
          // Create new technique
          const technique = this.createFromPattern(pattern, paper);
          techniques.push(technique);
          this.existingTechniques.set(normalizedName, technique);
        }
      }
    }

    // Extract from paper's analyzed techniques
    if (paper.analysis?.techniques) {
      for (const extracted of paper.analysis.techniques) {
        const normalizedName = this.normalizeKey(extracted.name);

        if (seenNames.has(normalizedName)) continue;
        seenNames.add(normalizedName);

        const existing = this.existingTechniques.get(normalizedName);
        if (existing) {
          if (!existing.sourcePaperIds.includes(paper.id)) {
            existing.sourcePaperIds.push(paper.id);
          }
          techniques.push(existing);
        } else {
          const technique = this.createFromExtracted(extracted, paper);
          techniques.push(technique);
          this.existingTechniques.set(normalizedName, technique);
        }
      }
    }

    // Infer relationships between techniques
    for (let i = 0; i < techniques.length; i++) {
      for (let j = i + 1; j < techniques.length; j++) {
        const edge = this.inferRelationship(techniques[i], techniques[j], paper);
        if (edge) {
          edges.push(edge);
        }
      }
    }

    // Track paper-technique mapping
    this.paperTechniqueMap.set(
      paper.id,
      techniques.map((t) => t.id)
    );

    return {
      techniques,
      edges,
      metadata: {
        sourcePaperId: paper.id,
        extractedAt: new Date().toISOString(),
        confidence: this.calculateConfidence(paper, techniques),
        method: paper.analysis ? "analysis" : "llm",
      },
    };
  }

  /**
   * Create a technique from a pattern match
   */
  private createFromPattern(pattern: TechniquePattern, paper: Paper): TechniqueNode {
    return createTechniqueNode(pattern.name, pattern.category, {
      description: `${pattern.name} technique extracted from ${paper.metadata.title}`,
      tags: this.inferTags(pattern.name, pattern.category),
      complexity: pattern.complexity,
      hasImplementation: false,
      sourcePaperIds: [paper.id],
      metadata: {
        extractionSource: "pattern",
        patternName: pattern.name,
      },
    });
  }

  /**
   * Create a technique from paper's extracted technique
   */
  private createFromExtracted(
    extracted: ExtractedTechnique,
    paper: Paper
  ): TechniqueNode {
    const category = this.inferCategory(extracted.name, extracted.description);

    return createTechniqueNode(extracted.name, category, {
      description: extracted.description,
      tags: this.inferTags(extracted.name, category),
      complexity: extracted.isMainContribution ? "complex" : "moderate",
      hasImplementation: false,
      sourcePaperIds: [paper.id],
      metadata: {
        extractionSource: "analysis",
        isMainContribution: extracted.isMainContribution,
        relatedTo: extracted.relatedTo,
      },
    });
  }

  /**
   * Infer relationship between two techniques
   */
  private inferRelationship(
    t1: TechniqueNode,
    t2: TechniqueNode,
    paper: Paper
  ): GraphEdge | null {
    // Different categories often combine
    if (t1.category !== t2.category) {
      // Architecture + Conditioning = combines_with
      if (
        (t1.category === "architecture" && t2.category === "conditioning") ||
        (t1.category === "conditioning" && t2.category === "architecture")
      ) {
        return createEdge("combines_with", t1.id, t2.id, {
          weight: 0.8,
          confidence: 0.7,
          isInferred: true,
          evidence: [
            {
              type: "analysis",
              sourceId: paper.id,
              description: `Both techniques appear in ${paper.metadata.title}`,
              confidence: 0.7,
              timestamp: new Date().toISOString(),
            },
          ],
        });
      }

      // Architecture + Loss = uses
      if (
        (t1.category === "architecture" && t2.category === "loss-function") ||
        (t1.category === "loss-function" && t2.category === "architecture")
      ) {
        const [arch, loss] =
          t1.category === "architecture" ? [t1, t2] : [t2, t1];

        return createEdge("uses", arch.id, loss.id, {
          weight: 0.6,
          confidence: 0.6,
          isInferred: true,
          evidence: [
            {
              type: "analysis",
              sourceId: paper.id,
              description: `Architecture uses loss function in ${paper.metadata.title}`,
              confidence: 0.6,
              timestamp: new Date().toISOString(),
            },
          ],
        });
      }
    }

    // Same category - could be similar or one extends another
    if (t1.category === t2.category) {
      // Check for naming patterns that suggest extension
      const name1Lower = t1.name.toLowerCase();
      const name2Lower = t2.name.toLowerCase();

      if (
        name2Lower.includes(name1Lower) ||
        name1Lower.includes(name2Lower)
      ) {
        const [base, extension] =
          name1Lower.length < name2Lower.length ? [t1, t2] : [t2, t1];

        return createEdge("extends", extension.id, base.id, {
          weight: 0.7,
          confidence: 0.5,
          isInferred: true,
          evidence: [
            {
              type: "analysis",
              sourceId: paper.id,
              description: `Name similarity suggests extension`,
              confidence: 0.5,
              timestamp: new Date().toISOString(),
            },
          ],
        });
      }
    }

    return null;
  }

  /**
   * Infer category from technique name and description
   */
  private inferCategory(name: string, description: string): TechniqueCategory {
    const combined = `${name} ${description}`.toLowerCase();

    // Check patterns
    for (const pattern of ALL_PATTERNS) {
      for (const regex of pattern.patterns) {
        if (regex.test(combined)) {
          return pattern.category;
        }
      }
    }

    // Keyword heuristics
    if (
      combined.includes("loss") ||
      combined.includes("objective") ||
      combined.includes("criterion")
    ) {
      return "loss-function";
    }

    if (
      combined.includes("train") ||
      combined.includes("learn") ||
      combined.includes("optim")
    ) {
      return "training";
    }

    if (
      combined.includes("condition") ||
      combined.includes("control") ||
      combined.includes("guide")
    ) {
      return "conditioning";
    }

    if (
      combined.includes("network") ||
      combined.includes("model") ||
      combined.includes("layer")
    ) {
      return "architecture";
    }

    return "other";
  }

  /**
   * Infer tags for a technique
   */
  private inferTags(name: string, category: TechniqueCategory): string[] {
    const tags: string[] = [category];
    const nameLower = name.toLowerCase();

    // Add specific tags based on keywords
    if (nameLower.includes("audio") || nameLower.includes("speech")) {
      tags.push("audio");
    }
    if (nameLower.includes("text") || nameLower.includes("language")) {
      tags.push("text");
    }
    if (nameLower.includes("image") || nameLower.includes("visual")) {
      tags.push("vision");
    }
    if (nameLower.includes("embedding") || nameLower.includes("latent")) {
      tags.push("representation");
    }
    if (nameLower.includes("generative") || nameLower.includes("synthesis")) {
      tags.push("generative");
    }

    return Array.from(new Set(tags));
  }

  /**
   * Match text against a pattern
   */
  private matchesPattern(text: string, pattern: TechniquePattern): boolean {
    return pattern.patterns.some((regex) => regex.test(text));
  }

  /**
   * Normalize technique name for deduplication
   */
  private normalizeKey(name: string): string {
    return name
      .toLowerCase()
      .replace(/[^a-z0-9]/g, "")
      .trim();
  }

  /**
   * Calculate extraction confidence
   */
  private calculateConfidence(paper: Paper, techniques: TechniqueNode[]): number {
    let confidence = 0.5;

    // Higher confidence if paper has analysis
    if (paper.analysis) {
      confidence += 0.2;
    }

    // Higher confidence for more techniques (up to a point)
    if (techniques.length >= 2 && techniques.length <= 5) {
      confidence += 0.1;
    }

    // Higher confidence if paper has citation count
    if (paper.metadata.citationCount && paper.metadata.citationCount > 10) {
      confidence += 0.1;
    }

    return Math.min(confidence, 1.0);
  }

  /**
   * Get techniques for a paper
   */
  getTechniquesForPaper(paperId: string): string[] {
    return this.paperTechniqueMap.get(paperId) || [];
  }

  /**
   * Get all extracted techniques
   */
  getAllTechniques(): TechniqueNode[] {
    return Array.from(this.existingTechniques.values());
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a new technique builder
 */
export function createTechniqueBuilder(): TechniqueBuilder {
  return new TechniqueBuilder();
}

/**
 * Extract techniques from multiple papers
 */
export function extractTechniquesFromPapers(
  papers: Paper[]
): {
  techniques: TechniqueNode[];
  edges: GraphEdge[];
  paperMap: Map<string, string[]>;
} {
  const builder = createTechniqueBuilder();
  const allEdges: GraphEdge[] = [];

  for (const paper of papers) {
    const extraction = builder.extractFromPaper(paper);
    allEdges.push(...extraction.edges);
  }

  // Deduplicate edges
  const edgeMap = new Map<string, GraphEdge>();
  for (const edge of allEdges) {
    const key = `${edge.type}:${edge.sourceId}:${edge.targetId}`;
    if (!edgeMap.has(key)) {
      edgeMap.set(key, edge);
    }
  }

  return {
    techniques: builder.getAllTechniques(),
    edges: Array.from(edgeMap.values()),
    paperMap: new Map(
      papers.map((p) => [p.id, builder.getTechniquesForPaper(p.id)])
    ),
  };
}

/**
 * Create a technique node from manual input
 */
export function createManualTechnique(
  name: string,
  category: TechniqueCategory,
  description: string,
  options: {
    tags?: string[];
    complexity?: TechniqueNode["complexity"];
    architecture?: string;
    conditioning?: string;
    loss?: string;
    dataFlow?: string;
    sourcePaperIds?: string[];
    metrics?: TechniqueMetrics;
  } = {}
): TechniqueNode {
  return createTechniqueNode(name, category, {
    description,
    tags: options.tags || [],
    complexity: options.complexity || "moderate",
    architecture: options.architecture,
    conditioning: options.conditioning,
    loss: options.loss,
    dataFlow: options.dataFlow,
    hasImplementation: false,
    sourcePaperIds: options.sourcePaperIds || [],
    metrics: options.metrics,
    metadata: {
      createdManually: true,
    },
  });
}
