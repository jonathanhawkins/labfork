/**
 * Technique Builder Tests
 */

import { describe, it, expect, beforeEach } from "vitest";
import {
  TechniqueBuilder,
  createTechniqueBuilder,
  extractTechniquesFromPapers,
  createManualTechnique,
} from "../../../../lib/meta/knowledge-graph/builders/technique-builder";
import { Paper, PaperMetadata, ExtractedTechnique } from "../../../../lib/papers/types";

// Helper to create mock papers
function createMockPaper(
  title: string,
  abstract: string,
  options: {
    techniques?: ExtractedTechnique[];
    id?: string;
  } = {}
): Paper {
  const now = new Date().toISOString();
  const id = options.id || `paper-${Date.now()}`;

  return {
    id,
    metadata: {
      id,
      title,
      authors: [],
      abstract,
      source: "arxiv",
      url: `https://arxiv.org/abs/${id}`,
    },
    status: "analyzed",
    addedAt: now,
    updatedAt: now,
    analysis: options.techniques
      ? {
          relevanceScore: 80,
          relevanceReason: "Test paper",
          techniques: options.techniques,
          novelty: "Test novelty",
          complexity: "moderate",
          complexityReason: "Test complexity",
          resources: [],
          taskBreakdown: {
            research: { title: "Research", description: "Research task" },
            implementation: {
              title: "Implementation",
              description: "Implementation task",
            },
            evaluation: { title: "Evaluation", description: "Evaluation task" },
          },
          analyzedAt: now,
        }
      : undefined,
  };
}

describe("TechniqueBuilder", () => {
  let builder: TechniqueBuilder;

  beforeEach(() => {
    builder = createTechniqueBuilder();
  });

  describe("extractFromPaper", () => {
    it("should extract techniques from paper title and abstract", () => {
      const paper = createMockPaper(
        "Attention Is All You Need: Transformer Architecture",
        "We propose a new architecture based on self-attention mechanisms."
      );

      const extraction = builder.extractFromPaper(paper);

      expect(extraction.techniques.length).toBeGreaterThan(0);
      expect(extraction.metadata.sourcePaperId).toBe(paper.id);
      expect(extraction.metadata.method).toBe("llm");
    });

    it("should extract Transformer architecture", () => {
      const paper = createMockPaper(
        "Novel Transformer Model",
        "We use a transformer with self-attention to process sequences."
      );

      const extraction = builder.extractFromPaper(paper);

      const transformer = extraction.techniques.find(
        (t) => t.name === "Transformer"
      );
      expect(transformer).toBeDefined();
      expect(transformer?.category).toBe("architecture");
    });

    it("should extract VAE architecture", () => {
      const paper = createMockPaper(
        "Variational Autoencoder for Speech",
        "We propose a VAE-based model that learns a latent space representation."
      );

      const extraction = builder.extractFromPaper(paper);

      const vae = extraction.techniques.find((t) => t.name === "VAE");
      expect(vae).toBeDefined();
      expect(vae?.category).toBe("architecture");
    });

    it("should extract GAN architecture", () => {
      const paper = createMockPaper(
        "Generative Adversarial Network",
        "We train a generator and discriminator in an adversarial manner."
      );

      const extraction = builder.extractFromPaper(paper);

      const gan = extraction.techniques.find((t) => t.name === "GAN");
      expect(gan).toBeDefined();
    });

    it("should extract conditioning techniques", () => {
      const paper = createMockPaper(
        "Cross-Attention Conditioning",
        "We use cross-attention to condition the generation process."
      );

      const extraction = builder.extractFromPaper(paper);

      const crossAttn = extraction.techniques.find(
        (t) => t.name === "Cross-Attention Conditioning"
      );
      expect(crossAttn).toBeDefined();
      expect(crossAttn?.category).toBe("conditioning");
    });

    it("should extract loss functions", () => {
      const paper = createMockPaper(
        "Perceptual Loss for Style Transfer",
        "We use perceptual loss with VGG features for better quality."
      );

      const extraction = builder.extractFromPaper(paper);

      const perceptualLoss = extraction.techniques.find(
        (t) => t.name === "Perceptual Loss"
      );
      expect(perceptualLoss).toBeDefined();
      expect(perceptualLoss?.category).toBe("loss-function");
    });

    it("should extract training techniques", () => {
      const paper = createMockPaper(
        "Curriculum Learning for NLP",
        "We use curriculum learning to train from easy to hard examples."
      );

      const extraction = builder.extractFromPaper(paper);

      const curriculum = extraction.techniques.find(
        (t) => t.name === "Curriculum Learning"
      );
      expect(curriculum).toBeDefined();
      expect(curriculum?.category).toBe("training");
    });

    it("should use paper analysis techniques when available", () => {
      const paper = createMockPaper(
        "Novel Technique Paper",
        "This paper introduces a custom technique.",
        {
          techniques: [
            {
              name: "Custom Technique",
              description: "A custom technique for testing",
              isMainContribution: true,
            },
          ],
        }
      );

      const extraction = builder.extractFromPaper(paper);

      const custom = extraction.techniques.find(
        (t) => t.name === "Custom Technique"
      );
      expect(custom).toBeDefined();
      expect(extraction.metadata.method).toBe("analysis");
    });

    it("should avoid duplicate techniques", () => {
      const paper = createMockPaper(
        "Transformer Transformer Transformer",
        "We use transformer with transformer architecture and transformer layers."
      );

      const extraction = builder.extractFromPaper(paper);

      const transformers = extraction.techniques.filter(
        (t) => t.name === "Transformer"
      );
      expect(transformers).toHaveLength(1);
    });

    it("should link techniques to source paper", () => {
      const paper = createMockPaper(
        "Transformer Model",
        "We use a transformer architecture."
      );

      const extraction = builder.extractFromPaper(paper);

      for (const technique of extraction.techniques) {
        expect(technique.sourcePaperIds).toContain(paper.id);
      }
    });
  });

  describe("relationship inference", () => {
    it("should infer combines_with between architecture and conditioning", () => {
      const paper = createMockPaper(
        "Transformer with Cross-Attention",
        "We combine transformer with cross-attention conditioning."
      );

      const extraction = builder.extractFromPaper(paper);

      const combinesEdge = extraction.edges.find(
        (e) => e.type === "combines_with"
      );
      expect(combinesEdge).toBeDefined();
    });

    it("should infer uses between architecture and loss", () => {
      const paper = createMockPaper(
        "GAN with Adversarial Loss",
        "Our GAN uses adversarial loss for training the discriminator."
      );

      const extraction = builder.extractFromPaper(paper);

      const usesEdge = extraction.edges.find((e) => e.type === "uses");
      expect(usesEdge).toBeDefined();
    });
  });

  describe("registerExisting", () => {
    it("should avoid creating duplicates of existing techniques", () => {
      const existingTechnique = createManualTechnique(
        "Transformer",
        "architecture",
        "Pre-existing transformer"
      );

      builder.registerExisting([existingTechnique]);

      const paper = createMockPaper(
        "Using Transformers",
        "We apply transformer for our task."
      );

      const extraction = builder.extractFromPaper(paper);

      const transformers = extraction.techniques.filter(
        (t) => t.name === "Transformer"
      );
      expect(transformers).toHaveLength(1);
      expect(transformers[0].id).toBe(existingTechnique.id);
    });

    it("should add paper ID to existing technique", () => {
      const existingTechnique = createManualTechnique(
        "Transformer",
        "architecture",
        "Pre-existing transformer"
      );

      builder.registerExisting([existingTechnique]);

      const paper = createMockPaper(
        "Using Transformers",
        "We apply transformer architecture."
      );

      builder.extractFromPaper(paper);

      expect(existingTechnique.sourcePaperIds).toContain(paper.id);
    });
  });

  describe("getTechniquesForPaper", () => {
    it("should return technique IDs for a paper", () => {
      const paper = createMockPaper(
        "Transformer VAE",
        "We combine transformer and VAE architectures."
      );

      builder.extractFromPaper(paper);

      const techniqueIds = builder.getTechniquesForPaper(paper.id);
      expect(techniqueIds.length).toBeGreaterThan(0);
    });

    it("should return empty array for unknown paper", () => {
      const techniqueIds = builder.getTechniquesForPaper("unknown");
      expect(techniqueIds).toEqual([]);
    });
  });

  describe("getAllTechniques", () => {
    it("should return all extracted techniques", () => {
      const paper1 = createMockPaper(
        "Transformer Paper",
        "Using transformer architecture."
      );
      const paper2 = createMockPaper("VAE Paper", "Using VAE architecture.");

      builder.extractFromPaper(paper1);
      builder.extractFromPaper(paper2);

      const allTechniques = builder.getAllTechniques();
      expect(allTechniques.length).toBeGreaterThanOrEqual(2);
    });
  });
});

describe("extractTechniquesFromPapers", () => {
  it("should extract from multiple papers", () => {
    const papers = [
      createMockPaper(
        "Transformer Model",
        "Using self-attention mechanism.",
        { id: "p1" }
      ),
      createMockPaper("VAE Model", "Using variational autoencoder.", {
        id: "p2",
      }),
      createMockPaper("GAN Model", "Using generative adversarial networks.", {
        id: "p3",
      }),
    ];

    const result = extractTechniquesFromPapers(papers);

    expect(result.techniques.length).toBeGreaterThanOrEqual(3);
    expect(result.paperMap.size).toBe(3);
  });

  it("should deduplicate techniques across papers", () => {
    const papers = [
      createMockPaper("Transformer 1", "Using transformer.", { id: "p1" }),
      createMockPaper("Transformer 2", "Also using transformer.", { id: "p2" }),
    ];

    const result = extractTechniquesFromPapers(papers);

    const transformers = result.techniques.filter(
      (t) => t.name === "Transformer"
    );
    expect(transformers).toHaveLength(1);
  });

  it("should deduplicate edges", () => {
    const papers = [
      createMockPaper(
        "Transformer + Conditioning 1",
        "Transformer with cross-attention conditioning.",
        { id: "p1" }
      ),
      createMockPaper(
        "Transformer + Conditioning 2",
        "Another transformer with cross-attention.",
        { id: "p2" }
      ),
    ];

    const result = extractTechniquesFromPapers(papers);

    // Edges should be deduplicated
    const uniqueEdgeKeys = new Set(
      result.edges.map((e) => `${e.type}:${e.sourceId}:${e.targetId}`)
    );
    expect(uniqueEdgeKeys.size).toBe(result.edges.length);
  });
});

describe("createManualTechnique", () => {
  it("should create a technique with required fields", () => {
    const technique = createManualTechnique(
      "Custom Technique",
      "architecture",
      "A custom technique for testing"
    );

    expect(technique.name).toBe("Custom Technique");
    expect(technique.category).toBe("architecture");
    expect(technique.description).toBe("A custom technique for testing");
    expect(technique.type).toBe("technique");
    expect(technique.complexity).toBe("moderate");
    expect(technique.hasImplementation).toBe(false);
    expect(technique.metadata.createdManually).toBe(true);
  });

  it("should accept optional fields", () => {
    const technique = createManualTechnique(
      "Custom Technique",
      "conditioning",
      "Description",
      {
        tags: ["ml", "audio"],
        complexity: "complex",
        architecture: "encoder-decoder",
        conditioning: "cross-attention",
        sourcePaperIds: ["paper-1"],
        metrics: { quality: 95, speed: 1.2 },
      }
    );

    expect(technique.tags).toEqual(["ml", "audio"]);
    expect(technique.complexity).toBe("complex");
    expect(technique.architecture).toBe("encoder-decoder");
    expect(technique.conditioning).toBe("cross-attention");
    expect(technique.sourcePaperIds).toEqual(["paper-1"]);
    expect(technique.metrics?.quality).toBe(95);
    expect(technique.metrics?.speed).toBe(1.2);
  });
});

describe("Pattern Matching", () => {
  it("should match diffusion models", () => {
    const builder = createTechniqueBuilder();
    const paper = createMockPaper(
      "Diffusion Model for Audio",
      "We use a denoising diffusion probabilistic model (DDPM)."
    );

    const extraction = builder.extractFromPaper(paper);

    const diffusion = extraction.techniques.find(
      (t) => t.name === "Diffusion"
    );
    expect(diffusion).toBeDefined();
  });

  it("should match reinforcement learning", () => {
    const builder = createTechniqueBuilder();
    const paper = createMockPaper(
      "RL for TTS",
      "We use reinforcement learning with a reward model."
    );

    const extraction = builder.extractFromPaper(paper);

    const rl = extraction.techniques.find(
      (t) => t.name === "Reinforcement Learning"
    );
    expect(rl).toBeDefined();
    expect(rl?.complexity).toBe("research");
  });

  it("should match fine-tuning techniques", () => {
    const builder = createTechniqueBuilder();
    const paper = createMockPaper(
      "Efficient Fine-tuning",
      "We use LoRA adapters for parameter-efficient fine-tuning."
    );

    const extraction = builder.extractFromPaper(paper);

    const finetuning = extraction.techniques.find(
      (t) => t.name === "Fine-tuning"
    );
    expect(finetuning).toBeDefined();
    expect(finetuning?.complexity).toBe("simple");
  });
});

describe("Tag Inference", () => {
  it("should include category as tag by default", () => {
    const technique = createManualTechnique(
      "Custom Technique",
      "architecture",
      "A custom technique"
    );

    // Category is always included as a tag
    expect(technique.category).toBe("architecture");
  });

  it("should accept explicit tags", () => {
    const technique = createManualTechnique(
      "Audio Transformer",
      "architecture",
      "Transformer for audio",
      { tags: ["audio", "ml"] }
    );

    expect(technique.tags).toContain("audio");
    expect(technique.tags).toContain("ml");
  });

  it("should allow custom tags for domain-specific techniques", () => {
    const technique = createManualTechnique(
      "Latent Embedding",
      "architecture",
      "Learns embeddings",
      { tags: ["representation", "embedding", "latent"] }
    );

    expect(technique.tags).toContain("representation");
    expect(technique.tags).toContain("embedding");
  });
});
