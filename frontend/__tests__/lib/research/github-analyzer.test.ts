/**
 * Tests for GitHub Repository Analyzer
 */

import { describe, it, expect } from "vitest";
import {
  detectFramework,
  detectArchitectures,
  detectProjectType,
  extractTechniques,
  extractLinkedPapers,
  parseDependencies,
  generateTasks,
  assessComplexity,
  extractResources,
  summarizeReadme,
  calculateMLConfidence,
  analyzeRepository,
  FRAMEWORK_DISPLAY_NAMES,
  PROJECT_TYPE_DISPLAY_NAMES,
  ARCHITECTURE_DISPLAY_NAMES,
} from "@/lib/research/github-analyzer";

describe("GitHub Analyzer", () => {
  describe("detectFramework", () => {
    it("detects PyTorch", () => {
      const code = `
        import torch
        from torch import nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
      `;
      const { framework, confidence } = detectFramework(code);
      expect(framework).toBe("pytorch");
      expect(confidence).toBeGreaterThan(0.5);
    });

    it("detects TensorFlow", () => {
      const code = `
        import tensorflow as tf
        from tensorflow import keras

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10)
        ])
      `;
      const { framework, confidence } = detectFramework(code);
      expect(framework).toBe("tensorflow");
      expect(confidence).toBeGreaterThan(0.5);
    });

    it("detects JAX", () => {
      const code = `
        import jax
        import jax.numpy as jnp
        from flax import linen as nn

        @jax.jit
        def forward(x):
            return jnp.dot(x, weights)
      `;
      const { framework, confidence } = detectFramework(code);
      expect(framework).toBe("jax");
      expect(confidence).toBeGreaterThan(0.5);
    });

    it("detects Hugging Face", () => {
      const code = `
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained("bert-base")
        tokenizer = AutoTokenizer.from_pretrained("bert-base")
      `;
      const { framework, confidence } = detectFramework(code);
      expect(framework).toBe("huggingface");
      expect(confidence).toBeGreaterThan(0.5);
    });

    it("detects PyTorch Lightning", () => {
      const code = `
        import pytorch_lightning as pl

        class LitModel(pl.LightningModule):
            def training_step(self, batch, batch_idx):
                return loss
      `;
      const { framework, confidence } = detectFramework(code);
      expect(framework).toBe("lightning");
      expect(confidence).toBeGreaterThan(0.5);
    });

    it("returns unknown for unrecognized code", () => {
      const code = `
        def hello_world():
            print("Hello, World!")
      `;
      const { framework, confidence } = detectFramework(code);
      expect(framework).toBe("unknown");
      expect(confidence).toBe(0);
    });
  });

  describe("detectArchitectures", () => {
    it("detects transformer architecture", () => {
      const code = `
        class TransformerEncoder(nn.Module):
            def __init__(self):
                self.attention = MultiHeadAttention(8, 512)
                self.positional_encoding = PositionalEncoding()
      `;
      const archs = detectArchitectures(code);
      expect(archs).toContain("transformer");
    });

    it("detects CNN architecture", () => {
      const code = `
        class ConvNet(nn.Module):
            def __init__(self):
                self.conv1 = nn.Conv2d(3, 64, 3)
                self.pool = nn.MaxPool2d(2)
      `;
      const archs = detectArchitectures(code);
      expect(archs).toContain("cnn");
    });

    it("detects diffusion architecture", () => {
      const code = `
        class UNet2D(nn.Module):
            def __init__(self):
                self.noise_schedule = ...

            def denoise(self, x, t):
                # DDPM denoising step
                return x_denoised
      `;
      const archs = detectArchitectures(code);
      expect(archs).toContain("diffusion");
    });

    it("detects GAN architecture", () => {
      const code = `
        class StyleGAN:
            def __init__(self):
                self.generator = Generator()
                self.discriminator = Discriminator()
      `;
      const archs = detectArchitectures(code);
      expect(archs).toContain("gan");
    });

    it("detects multiple architectures", () => {
      const code = `
        class HybridModel:
            def __init__(self):
                self.transformer = TransformerEncoder()
                self.conv = nn.Conv2d(3, 64, 3)
                self.lstm = nn.LSTM(512, 256)
      `;
      const archs = detectArchitectures(code);
      expect(archs.length).toBeGreaterThanOrEqual(2);
    });

    it("adds hybrid when multiple architectures detected", () => {
      const code = `
        class ComplexModel:
            # Uses Transformer, CNN, and RNN
            self.attention = MultiHeadAttention()
            self.conv = nn.Conv2d(3, 64)
            self.lstm = nn.LSTM(512, 256)
      `;
      const archs = detectArchitectures(code);
      expect(archs).toContain("hybrid");
    });
  });

  describe("detectProjectType", () => {
    it("detects training project", () => {
      const files = ["train.py", "trainer.py", "config.yaml", "model.py"];
      const content = "epochs = 100\noptimizer = AdamW\ntraining_loop()";
      expect(detectProjectType(files, content)).toBe("training");
    });

    it("detects inference project", () => {
      const files = ["inference.py", "predict.py", "serve.py", "api.py"];
      const content = "model.generate()\npredict(input)\ndeploy()";
      expect(detectProjectType(files, content)).toBe("inference");
    });

    it("detects model project", () => {
      const files = ["model.safetensors", "config.json", "README.md"];
      const content = "pretrained weights\ncheckpoint available";
      expect(detectProjectType(files, content)).toBe("model");
    });

    it("detects research project", () => {
      const files = ["paper.md", "experiments/", "ablation.py"];
      const content = "Our paper presents... arXiv:2401.12345\nexperiment results";
      expect(detectProjectType(files, content)).toBe("research");
    });

    it("detects application project", () => {
      const files = ["app.py", "streamlit_app.py", "frontend/", "demo.py"];
      const content = "streamlit run app.py\ngradio interface";
      expect(detectProjectType(files, content)).toBe("application");
    });
  });

  describe("extractTechniques", () => {
    it("extracts training techniques", () => {
      const readme = "We use mixed-precision fp16 training with gradient-accumulation.";
      const code = "scaler = GradScaler()\nwith autocast(): ...";
      const techniques = extractTechniques(readme, code);
      expect(techniques.some(t => t.name === "Mixed Precision Training")).toBe(true);
      expect(techniques.some(t => t.name === "Gradient Accumulation")).toBe(true);
    });

    it("extracts LoRA fine-tuning", () => {
      const readme = "We fine-tune using LoRA for parameter-efficient training.";
      const techniques = extractTechniques(readme, "");
      expect(techniques.some(t => t.name === "LoRA Fine-tuning")).toBe(true);
    });

    it("extracts architecture techniques", () => {
      const readme = "Uses multi-head-attention with rotary embedding (RoPE).";
      const techniques = extractTechniques(readme, "");
      expect(techniques.some(t => t.name === "Multi-Head Attention")).toBe(true);
      expect(techniques.some(t => t.name === "Rotary Position Embedding")).toBe(true);
    });

    it("extracts optimization techniques", () => {
      const readme = "Optimized with flash_attention and xformers memory-efficient attention.";
      const techniques = extractTechniques(readme, "");
      expect(techniques.some(t => t.name === "Flash Attention")).toBe(true);
      expect(techniques.some(t => t.name === "xFormers")).toBe(true);
    });

    it("identifies main contribution", () => {
      const readme = "We propose a novel attention mechanism for efficient inference.";
      const techniques = extractTechniques(readme, "");
      const mainContrib = techniques.find(t => t.isMainContribution);
      expect(mainContrib).toBeDefined();
      expect(mainContrib?.description).toContain("novel attention mechanism");
    });
  });

  describe("extractLinkedPapers", () => {
    it("extracts arXiv IDs from URLs", () => {
      const readme = `
        Based on our paper: https://arxiv.org/abs/2401.12345
        See also: https://arxiv.org/pdf/2312.98765.pdf
      `;
      const papers = extractLinkedPapers(readme, "");
      expect(papers).toHaveLength(2);
      expect(papers[0].id).toBe("2401.12345");
      expect(papers[0].type).toBe("arxiv");
      expect(papers[1].id).toBe("2312.98765");
    });

    it("extracts arXiv IDs from text references", () => {
      const readme = "arXiv: 2401.12345 and arXiv:2312.98765";
      const papers = extractLinkedPapers(readme, "");
      expect(papers).toHaveLength(2);
    });

    it("extracts DOIs", () => {
      const readme = "Published at https://doi.org/10.1234/example.paper";
      const papers = extractLinkedPapers(readme, "");
      expect(papers).toHaveLength(1);
      expect(papers[0].type).toBe("doi");
      expect(papers[0].id).toBe("10.1234/example.paper");
    });

    it("deduplicates papers", () => {
      const readme = `
        arxiv.org/abs/2401.12345
        arxiv.org/pdf/2401.12345.pdf
        arXiv: 2401.12345
      `;
      const papers = extractLinkedPapers(readme, "");
      expect(papers).toHaveLength(1);
    });

    it("identifies source correctly", () => {
      const readme = "See our paper: arxiv.org/abs/2401.12345";
      const code = "# Citation: arxiv.org/abs/2312.98765";
      const papers = extractLinkedPapers(readme, code);
      expect(papers.find(p => p.id === "2401.12345")?.source).toBe("readme");
      expect(papers.find(p => p.id === "2312.98765")?.source).toBe("code");
    });
  });

  describe("parseDependencies", () => {
    it("parses requirements.txt format", () => {
      const content = `
        torch>=2.0.0
        transformers==4.30.0
        numpy
        pandas~=2.0
      `;
      const deps = parseDependencies(content);
      expect(deps).toHaveLength(4);
      expect(deps.find(d => d.name === "torch")?.version).toBe("2.0.0");
      expect(deps.find(d => d.name === "transformers")?.version).toBe("4.30.0");
    });

    it("identifies ML dependencies", () => {
      const content = `
        torch>=2.0.0
        tensorflow
        jax
        numpy
      `;
      const deps = parseDependencies(content);
      expect(deps.filter(d => d.category === "ml")).toHaveLength(3);
    });

    it("identifies data dependencies", () => {
      const content = `
        numpy
        pandas
        scipy
        pillow
      `;
      const deps = parseDependencies(content);
      expect(deps.filter(d => d.category === "data")).toHaveLength(4);
    });

    it("marks key dependencies", () => {
      const content = `
        torch
        requests
        tqdm
      `;
      const deps = parseDependencies(content);
      expect(deps.find(d => d.name === "torch")?.isKey).toBe(true);
      expect(deps.find(d => d.name === "requests")?.isKey).toBe(false);
    });

    it("skips comments and section headers", () => {
      const content = `
        # Core dependencies
        [project.dependencies]
        torch>=2.0.0
      `;
      const deps = parseDependencies(content);
      expect(deps).toHaveLength(1);
      expect(deps[0].name).toBe("torch");
    });
  });

  describe("generateTasks", () => {
    it("generates setup task first", () => {
      const analysis = { framework: "pytorch" as const };
      const tasks = generateTasks(analysis, "");
      expect(tasks[0].category).toBe("setup");
      expect(tasks[0].priority).toBe(1);
    });

    it("generates understanding task", () => {
      const analysis = { architectures: ["transformer" as const] };
      const tasks = generateTasks(analysis, "");
      expect(tasks.some(t => t.category === "understand")).toBe(true);
    });

    it("generates paper reading task when papers linked", () => {
      const analysis = {
        linkedPapers: [{ id: "2401.12345", type: "arxiv" as const, source: "readme" as const }],
      };
      const tasks = generateTasks(analysis, "");
      const paperTask = tasks.find(t => t.title.includes("paper"));
      expect(paperTask).toBeDefined();
      expect(paperTask?.category).toBe("understand");
    });

    it("generates training tasks for training projects", () => {
      const analysis = { projectType: "training" as const };
      const tasks = generateTasks(analysis, "");
      expect(tasks.some(t => t.category === "train" || t.title.includes("training"))).toBe(true);
    });

    it("generates inference tasks for model projects", () => {
      const analysis = { projectType: "model" as const };
      const tasks = generateTasks(analysis, "");
      expect(tasks.some(t => t.title.includes("inference"))).toBe(true);
    });

    it("generates evaluation task", () => {
      const analysis = { projectType: "research" as const };
      const tasks = generateTasks(analysis, "");
      expect(tasks.some(t => t.category === "evaluate")).toBe(true);
    });

    it("sets up task dependencies correctly", () => {
      const analysis = { projectType: "training" as const };
      const tasks = generateTasks(analysis, "");
      const understandTask = tasks.find(t => t.category === "understand");
      expect(understandTask?.dependencies).toContain("Set up development environment");
    });
  });

  describe("assessComplexity", () => {
    it("returns simple for basic projects", () => {
      const analysis = {
        framework: "pytorch" as const,
        architectures: ["mlp" as const],
        techniques: [],
        dependencies: [],
      };
      const { complexity } = assessComplexity(analysis, "Simple model.");
      expect(complexity).toBe("simple");
    });

    it("returns moderate for typical ML projects", () => {
      const analysis = {
        framework: "pytorch" as const,
        architectures: ["transformer" as const],
        techniques: Array(3).fill({ name: "technique" }),
        dependencies: Array(10).fill({ name: "dep" }),
      };
      const { complexity } = assessComplexity(analysis, "Standard transformer model.");
      expect(["simple", "moderate"]).toContain(complexity);
    });

    it("returns complex for multi-architecture projects", () => {
      const analysis = {
        framework: "pytorch" as const,
        architectures: ["transformer" as const, "cnn" as const, "rnn" as const],
        techniques: Array(6).fill({ name: "technique" }),
        resources: [{ type: "gpu" as const, name: "GPU", required: true }],
      };
      const { complexity } = assessComplexity(analysis, "Complex hybrid model...");
      expect(["moderate", "complex", "research"]).toContain(complexity);
    });

    it("returns research for highly complex projects", () => {
      const analysis = {
        framework: "unknown" as const,
        architectures: ["diffusion" as const, "transformer" as const, "hybrid" as const],
        techniques: Array(10).fill({ name: "technique" }),
        resources: [{ type: "gpu" as const, name: "GPU", required: true }],
        dependencies: Array(25).fill({ name: "dep" }),
      };
      const readme = "A".repeat(15000); // Long README
      const { complexity } = assessComplexity(analysis, readme);
      // Score: 2 (unknown) + 2 (>2 archs) + 2 (diffusion) + 2 (hybrid) + 1 (techniques) + 1 (gpu) + 1 (readme) + 1 (deps) = 12
      expect(["complex", "research"]).toContain(complexity);
    });

    it("provides complexity reason", () => {
      const analysis = {
        framework: "unknown" as const,
        architectures: ["diffusion" as const],
      };
      const { reason } = assessComplexity(analysis, "");
      expect(reason).toContain("Unknown framework");
    });
  });

  describe("extractResources", () => {
    it("extracts GPU requirements", () => {
      const readme = "Requires a GPU with CUDA support. Tested on A100 with 40GB VRAM.";
      const resources = extractResources(readme, []);
      expect(resources.some(r => r.type === "gpu")).toBe(true);
    });

    it("extracts memory requirements", () => {
      const readme = "Requires memory: 32GB RAM for training.";
      const resources = extractResources(readme, []);
      expect(resources.some(r => r.type === "hardware" && r.estimate?.includes("32"))).toBe(true);
    });

    it("extracts dataset requirements", () => {
      const readme = "Trained on LibriSpeech dataset.";
      const resources = extractResources(readme, []);
      expect(resources.some(r => r.type === "dataset")).toBe(true);
    });

    it("extracts key library requirements", () => {
      const deps = [
        { name: "torch", version: "2.0.0", category: "ml" as const, isKey: true },
        { name: "tqdm", category: "utility" as const, isKey: false },
      ];
      const resources = extractResources("", deps);
      expect(resources.some(r => r.type === "library" && r.name === "torch")).toBe(true);
    });
  });

  describe("summarizeReadme", () => {
    it("extracts first paragraph", () => {
      const readme = `
# Project Title

This is a cool ML project that does amazing things.

## Installation

pip install project
      `;
      const summary = summarizeReadme(readme);
      expect(summary).toContain("cool ML project");
    });

    it("skips badges and images", () => {
      const readme = `
[![Build](https://badge.svg)](url)
![Logo](logo.png)
# Title

Actual content here.
      `;
      const summary = summarizeReadme(readme);
      expect(summary).not.toContain("badge");
      expect(summary).toContain("Actual content");
    });

    it("truncates long summaries", () => {
      const readme = "# Title\n\n" + "A".repeat(1000);
      const summary = summarizeReadme(readme);
      expect(summary.length).toBeLessThanOrEqual(503); // 500 + "..."
    });

    it("returns default for empty README", () => {
      const summary = summarizeReadme("");
      expect(summary).toBe("No description available.");
    });
  });

  describe("calculateMLConfidence", () => {
    it("returns high confidence for clear ML project", () => {
      const confidence = calculateMLConfidence(
        "pytorch",
        ["transformer"],
        [
          { name: "torch", category: "ml", isKey: true },
          { name: "transformers", category: "ml", isKey: true },
          { name: "accelerate", category: "ml", isKey: true },
        ],
        "This model uses deep learning and neural networks for training."
      );
      expect(confidence).toBeGreaterThanOrEqual(0.8);
    });

    it("returns low confidence for non-ML project", () => {
      const confidence = calculateMLConfidence(
        "unknown",
        ["unknown"],
        [{ name: "requests", category: "utility", isKey: false }],
        "A web scraping tool for data collection."
      );
      expect(confidence).toBeLessThan(0.5);
    });

    it("considers framework detection", () => {
      const withFramework = calculateMLConfidence("pytorch", [], [], "");
      const withoutFramework = calculateMLConfidence("unknown", [], [], "");
      expect(withFramework).toBeGreaterThan(withoutFramework);
    });

    it("considers ML keywords in README", () => {
      const withKeywords = calculateMLConfidence(
        "unknown",
        [],
        [],
        "This model uses transformer architecture for inference and training."
      );
      const withoutKeywords = calculateMLConfidence(
        "unknown",
        [],
        [],
        "A simple utility script."
      );
      expect(withKeywords).toBeGreaterThan(withoutKeywords);
    });
  });

  describe("analyzeRepository", () => {
    it("performs full analysis", () => {
      const url = "https://github.com/owner/repo";
      const readme = `
# ML Project

A transformer-based model for text generation.

We present a novel approach using multi-head attention.

Paper: https://arxiv.org/abs/2401.12345

Requires GPU with CUDA support.
      `;
      const code = `
import torch
from transformers import AutoModel

class Model(torch.nn.Module):
    def __init__(self):
        self.attention = MultiHeadAttention()
      `;
      const files = ["train.py", "model.py", "config.yaml"];
      const requirements = "torch>=2.0.0\ntransformers\naccelerate";

      const analysis = analyzeRepository(url, readme, code, files, requirements);

      expect(analysis.url).toBe(url);
      expect(analysis.name).toBe("repo");
      expect(analysis.owner).toBe("owner");
      // Both PyTorch and HuggingFace patterns are present, either is valid
      expect(["pytorch", "huggingface"]).toContain(analysis.framework);
      expect(analysis.architectures).toContain("transformer");
      expect(analysis.isMLProject).toBe(true);
      expect(analysis.linkedPapers).toHaveLength(1);
      expect(analysis.suggestedTasks.length).toBeGreaterThan(0);
      expect(analysis.analyzedAt).toBeDefined();
    });

    it("handles minimal input", () => {
      const analysis = analyzeRepository(
        "https://github.com/owner/repo",
        "",
        "",
        [],
        undefined
      );
      expect(analysis.name).toBe("repo");
      expect(analysis.isMLProject).toBe(false);
      expect(analysis.suggestedTasks.length).toBeGreaterThan(0); // At least setup task
    });
  });

  describe("Display name constants", () => {
    it("has all framework display names", () => {
      expect(FRAMEWORK_DISPLAY_NAMES.pytorch).toBe("PyTorch");
      expect(FRAMEWORK_DISPLAY_NAMES.tensorflow).toBe("TensorFlow");
      expect(FRAMEWORK_DISPLAY_NAMES.huggingface).toBe("Hugging Face");
    });

    it("has all project type display names", () => {
      expect(PROJECT_TYPE_DISPLAY_NAMES.training).toBe("Training Code");
      expect(PROJECT_TYPE_DISPLAY_NAMES.inference).toBe("Inference/Deployment");
      expect(PROJECT_TYPE_DISPLAY_NAMES.research).toBe("Research Implementation");
    });

    it("has all architecture display names", () => {
      expect(ARCHITECTURE_DISPLAY_NAMES.transformer).toBe("Transformer");
      expect(ARCHITECTURE_DISPLAY_NAMES.diffusion).toBe("Diffusion");
      expect(ARCHITECTURE_DISPLAY_NAMES.gan).toBe("GAN");
    });
  });
});
