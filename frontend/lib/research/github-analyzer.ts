/**
 * Enhanced GitHub Repository Analyzer
 *
 * Provides deep analysis of ML/AI repositories:
 * - Detects project type and framework
 * - Extracts techniques and architectures
 * - Generates implementation tasks
 * - Links to related papers
 */

import type {
  PaperMetadata,
  ExtractedTechnique,
  ResourceRequirement,
  ComplexityLevel,
} from "@/lib/papers/types";

// ============================================================================
// Types
// ============================================================================

/**
 * ML Framework detection
 */
export type MLFramework =
  | "pytorch"
  | "tensorflow"
  | "jax"
  | "keras"
  | "huggingface"
  | "lightning"
  | "fastai"
  | "onnx"
  | "mlx"
  | "unknown";

/**
 * Project type classification
 */
export type ProjectType =
  | "model" // Pre-trained model
  | "training" // Training code/framework
  | "inference" // Inference/deployment
  | "dataset" // Dataset processing
  | "benchmark" // Benchmarking/evaluation
  | "library" // General ML library
  | "application" // End-user application
  | "research" // Research paper implementation
  | "unknown";

/**
 * Model architecture category
 */
export type ArchitectureType =
  | "transformer"
  | "cnn"
  | "rnn"
  | "gan"
  | "diffusion"
  | "vae"
  | "autoencoder"
  | "mlp"
  | "graph"
  | "hybrid"
  | "unknown";

/**
 * Extracted code pattern
 */
export interface CodePattern {
  /** Pattern name */
  name: string;
  /** Pattern type */
  type: "architecture" | "training" | "data" | "optimization" | "evaluation";
  /** Description */
  description: string;
  /** Files where found */
  files: string[];
  /** Confidence score (0-1) */
  confidence: number;
}

/**
 * Repository file analysis result
 */
export interface FileAnalysis {
  /** File path */
  path: string;
  /** Detected language */
  language: string;
  /** Detected patterns */
  patterns: string[];
  /** Key imports/dependencies */
  imports: string[];
  /** Whether file is significant for analysis */
  isSignificant: boolean;
}

/**
 * Repository analysis result
 */
export interface RepoAnalysis {
  /** Repository URL */
  url: string;
  /** Repository name */
  name: string;
  /** Owner/organization */
  owner: string;
  /** Detected ML framework */
  framework: MLFramework;
  /** Confidence in framework detection (0-1) */
  frameworkConfidence: number;
  /** Project type classification */
  projectType: ProjectType;
  /** Architecture types detected */
  architectures: ArchitectureType[];
  /** Is this an ML/AI project? */
  isMLProject: boolean;
  /** ML project confidence score (0-1) */
  mlConfidence: number;
  /** Extracted techniques */
  techniques: ExtractedTechnique[];
  /** Code patterns found */
  patterns: CodePattern[];
  /** Resource requirements */
  resources: ResourceRequirement[];
  /** Linked papers (arXiv IDs, DOIs) */
  linkedPapers: LinkedPaper[];
  /** Generated tasks */
  suggestedTasks: SuggestedTask[];
  /** Complexity assessment */
  complexity: ComplexityLevel;
  /** Complexity explanation */
  complexityReason: string;
  /** Key dependencies */
  dependencies: Dependency[];
  /** README summary */
  readmeSummary: string;
  /** Analysis timestamp */
  analyzedAt: string;
}

/**
 * Linked paper reference
 */
export interface LinkedPaper {
  /** Paper identifier (arXiv ID, DOI, etc.) */
  id: string;
  /** Type of identifier */
  type: "arxiv" | "doi" | "url" | "title";
  /** Where found in repo */
  source: "readme" | "citation" | "code" | "config";
  /** Paper title if extracted */
  title?: string;
}

/**
 * Suggested implementation task
 */
export interface SuggestedTask {
  /** Task title */
  title: string;
  /** Task description */
  description: string;
  /** Task category */
  category: "setup" | "understand" | "implement" | "train" | "evaluate" | "integrate";
  /** Priority (1-5, 1 = highest) */
  priority: number;
  /** Estimated hours */
  estimatedHours: number;
  /** Dependencies on other tasks (by title) */
  dependencies: string[];
  /** Related files in repo */
  relatedFiles: string[];
}

/**
 * Dependency information
 */
export interface Dependency {
  /** Package name */
  name: string;
  /** Version constraint */
  version?: string;
  /** Category */
  category: "ml" | "data" | "utility" | "system" | "unknown";
  /** Is it a key dependency? */
  isKey: boolean;
}

// ============================================================================
// Detection Patterns
// ============================================================================

/** Framework detection patterns */
const FRAMEWORK_PATTERNS: Record<MLFramework, RegExp[]> = {
  pytorch: [
    /import\s+torch/,
    /from\s+torch/,
    /torch\.(nn|optim|cuda)/,
    /nn\.Module/,
  ],
  tensorflow: [
    /import\s+tensorflow/,
    /from\s+tensorflow/,
    /tf\.(keras|nn|train)/,
    /tensorflow\.(keras|nn)/,
  ],
  jax: [
    /import\s+jax/,
    /from\s+jax/,
    /jax\.(numpy|random|grad)/,
    /import\s+flax/,
  ],
  keras: [
    /from\s+keras/,
    /import\s+keras/,
    /keras\.(layers|models)/,
  ],
  huggingface: [
    /from\s+transformers/,
    /import\s+transformers/,
    /from\s+diffusers/,
    /from\s+datasets/,
    /AutoModel\./,
    /AutoTokenizer\./,
  ],
  lightning: [
    /import\s+pytorch_lightning/,
    /from\s+pytorch_lightning/,
    /import\s+lightning/,
    /LightningModule/,
  ],
  fastai: [
    /from\s+fastai/,
    /import\s+fastai/,
    /Learner\(/,
  ],
  onnx: [
    /import\s+onnx/,
    /onnx\.(load|save)/,
    /onnxruntime/,
  ],
  mlx: [
    /import\s+mlx/,
    /from\s+mlx/,
    /mlx\.(core|nn)/,
  ],
  unknown: [],
};

/** Architecture detection patterns */
const ARCHITECTURE_PATTERNS: Record<ArchitectureType, RegExp[]> = {
  transformer: [
    /Transformer/i,
    /MultiHeadAttention/i,
    /self[\-_]?attention/i,
    /BERT|GPT|T5|ViT|CLIP/i,
    /AttentionLayer/i,
    /positional[\-_]?encoding/i,
  ],
  cnn: [
    /Conv2d|Conv1d|Conv3d/,
    /ConvNet|CNN/i,
    /ResNet|VGG|EfficientNet|MobileNet/i,
    /UNet|YOLO/i,
  ],
  rnn: [
    /LSTM|GRU|RNN/,
    /Recurrent/i,
    /BiLSTM|BiGRU/i,
    /sequence[\-_]?to[\-_]?sequence/i,
  ],
  gan: [
    /GAN|Generator|Discriminator/i,
    /StyleGAN|DCGAN|WGAN|CycleGAN/i,
    /adversarial/i,
  ],
  diffusion: [
    /Diffusion|DDPM|DDIM/i,
    /UNet2D|noise[\-_]?schedule/i,
    /StableDiffusion|LatentDiffusion/i,
    /denoise|denoising/i,
  ],
  vae: [
    /VAE|VariationalAutoencoder/i,
    /reparameterize|kl[\-_]?divergence/i,
    /encoder|decoder.*latent/i,
  ],
  autoencoder: [
    /Autoencoder|AE/i,
    /Encoder.*Decoder/i,
    /reconstruct/i,
  ],
  mlp: [
    /MLP|MultiLayerPerceptron/i,
    /FullyConnected|DenseNet/i,
  ],
  graph: [
    /GNN|GraphNN|GCN/i,
    /MessagePassing/i,
    /node[\-_]?embedding/i,
    /PyG|DGL/i,
  ],
  hybrid: [],
  unknown: [],
};

/** Project type indicators */
const PROJECT_TYPE_INDICATORS: Record<ProjectType, string[]> = {
  model: [
    "pretrained",
    "checkpoint",
    "weights",
    "model.safetensors",
    "model.pt",
    "model.bin",
  ],
  training: [
    "train.py",
    "trainer",
    "training_loop",
    "train_step",
    "epochs",
    "optimizer",
  ],
  inference: [
    "inference",
    "predict",
    "generate",
    "serve",
    "deploy",
    "api.py",
  ],
  dataset: [
    "dataloader",
    "dataset.py",
    "preprocess",
    "data_utils",
    "collate",
  ],
  benchmark: [
    "benchmark",
    "evaluate",
    "metrics",
    "leaderboard",
    "eval.py",
  ],
  library: [
    "pip install",
    "setup.py",
    "pyproject.toml",
    "__init__.py",
    "package",
  ],
  application: [
    "app.py",
    "streamlit",
    "gradio",
    "demo",
    "frontend",
    "backend",
  ],
  research: [
    "paper",
    "arxiv",
    "supplementary",
    "experiment",
    "ablation",
  ],
  unknown: [],
};

/** Key ML dependencies */
const KEY_ML_DEPENDENCIES: Record<string, "ml" | "data" | "utility"> = {
  torch: "ml",
  tensorflow: "ml",
  jax: "ml",
  flax: "ml",
  transformers: "ml",
  diffusers: "ml",
  accelerate: "ml",
  lightning: "ml",
  keras: "ml",
  numpy: "data",
  pandas: "data",
  scipy: "data",
  scikit: "data",
  datasets: "data",
  pillow: "data",
  opencv: "data",
  librosa: "data",
  torchaudio: "data",
  wandb: "utility",
  tensorboard: "utility",
  hydra: "utility",
  omegaconf: "utility",
  tqdm: "utility",
  einops: "utility",
};

// ============================================================================
// Analysis Functions
// ============================================================================

/**
 * Detect ML framework from code content
 */
export function detectFramework(
  content: string
): { framework: MLFramework; confidence: number } {
  const scores: Record<MLFramework, number> = {
    pytorch: 0,
    tensorflow: 0,
    jax: 0,
    keras: 0,
    huggingface: 0,
    lightning: 0,
    fastai: 0,
    onnx: 0,
    mlx: 0,
    unknown: 0,
  };

  for (const [framework, patterns] of Object.entries(FRAMEWORK_PATTERNS)) {
    for (const pattern of patterns) {
      const matches = content.match(new RegExp(pattern, "g"));
      if (matches) {
        scores[framework as MLFramework] += matches.length;
      }
    }
  }

  // Find highest scoring framework
  let maxScore = 0;
  let detectedFramework: MLFramework = "unknown";

  for (const [framework, score] of Object.entries(scores)) {
    if (score > maxScore) {
      maxScore = score;
      detectedFramework = framework as MLFramework;
    }
  }

  // Calculate confidence based on score
  const totalScore = Object.values(scores).reduce((a, b) => a + b, 0);
  const confidence = totalScore > 0 ? maxScore / totalScore : 0;

  return { framework: detectedFramework, confidence };
}

/**
 * Detect architectures from code content
 */
export function detectArchitectures(content: string): ArchitectureType[] {
  const detected: ArchitectureType[] = [];

  for (const [arch, patterns] of Object.entries(ARCHITECTURE_PATTERNS)) {
    if (arch === "unknown" || arch === "hybrid") continue;

    for (const pattern of patterns) {
      if (pattern.test(content)) {
        if (!detected.includes(arch as ArchitectureType)) {
          detected.push(arch as ArchitectureType);
        }
        break;
      }
    }
  }

  // If multiple architectures detected, might be hybrid
  if (detected.length > 2) {
    detected.push("hybrid");
  }

  return detected.length > 0 ? detected : ["unknown"];
}

/**
 * Detect project type from file structure and content
 */
export function detectProjectType(
  files: string[],
  content: string
): ProjectType {
  const scores: Record<ProjectType, number> = {
    model: 0,
    training: 0,
    inference: 0,
    dataset: 0,
    benchmark: 0,
    library: 0,
    application: 0,
    research: 0,
    unknown: 0,
  };

  const combinedText = files.join("\n").toLowerCase() + "\n" + content.toLowerCase();

  for (const [type, indicators] of Object.entries(PROJECT_TYPE_INDICATORS)) {
    for (const indicator of indicators) {
      if (combinedText.includes(indicator.toLowerCase())) {
        scores[type as ProjectType]++;
      }
    }
  }

  // Find highest scoring type
  let maxScore = 0;
  let detectedType: ProjectType = "unknown";

  for (const [type, score] of Object.entries(scores)) {
    if (score > maxScore) {
      maxScore = score;
      detectedType = type as ProjectType;
    }
  }

  return detectedType;
}

/**
 * Extract techniques from README and code
 */
export function extractTechniques(
  readme: string,
  codeContent: string
): ExtractedTechnique[] {
  const techniques: ExtractedTechnique[] = [];

  // Common technique patterns
  const techniquePatterns = [
    // Training techniques
    { pattern: /mixed[\-_]?precision|fp16|bf16/i, name: "Mixed Precision Training", type: "training" },
    { pattern: /gradient[\-_]?accumulation/i, name: "Gradient Accumulation", type: "training" },
    { pattern: /gradient[\-_]?checkpointing/i, name: "Gradient Checkpointing", type: "training" },
    { pattern: /LoRA|Low[\-_]?Rank/i, name: "LoRA Fine-tuning", type: "training" },
    { pattern: /QLoRA/i, name: "QLoRA", type: "training" },
    { pattern: /PEFT|parameter[\-_]?efficient/i, name: "Parameter-Efficient Fine-tuning", type: "training" },
    { pattern: /DDP|DistributedDataParallel/i, name: "Distributed Training", type: "training" },
    { pattern: /FSDP|FullyShardedDataParallel/i, name: "Fully Sharded Data Parallel", type: "training" },
    { pattern: /DeepSpeed|ZeRO/i, name: "DeepSpeed", type: "training" },
    { pattern: /flash[\-_]?attention/i, name: "Flash Attention", type: "optimization" },
    { pattern: /xformers/i, name: "xFormers", type: "optimization" },
    // Architecture techniques
    { pattern: /multi[\-_]?head[\-_]?attention/i, name: "Multi-Head Attention", type: "architecture" },
    { pattern: /cross[\-_]?attention/i, name: "Cross Attention", type: "architecture" },
    { pattern: /rotary[\-_]?embedding|RoPE/i, name: "Rotary Position Embedding", type: "architecture" },
    { pattern: /grouped[\-_]?query[\-_]?attention|GQA/i, name: "Grouped Query Attention", type: "architecture" },
    { pattern: /sliding[\-_]?window[\-_]?attention/i, name: "Sliding Window Attention", type: "architecture" },
    { pattern: /residual[\-_]?connection|skip[\-_]?connection/i, name: "Residual Connections", type: "architecture" },
    { pattern: /layer[\-_]?norm/i, name: "Layer Normalization", type: "architecture" },
    { pattern: /RMS[\-_]?Norm/i, name: "RMS Normalization", type: "architecture" },
    // Data techniques
    { pattern: /data[\-_]?augmentation/i, name: "Data Augmentation", type: "data" },
    { pattern: /tokeniz/i, name: "Tokenization", type: "data" },
    { pattern: /BPE|byte[\-_]?pair/i, name: "Byte-Pair Encoding", type: "data" },
    // Inference techniques
    { pattern: /quantiz/i, name: "Quantization", type: "inference" },
    { pattern: /GPTQ|AWQ/i, name: "Weight Quantization", type: "inference" },
    { pattern: /KV[\-_]?cache/i, name: "KV Cache", type: "inference" },
    { pattern: /speculative[\-_]?decoding/i, name: "Speculative Decoding", type: "inference" },
  ];

  const combinedContent = readme + "\n" + codeContent;

  for (const { pattern, name, type } of techniquePatterns) {
    if (pattern.test(combinedContent)) {
      // Extract context around the match
      const match = combinedContent.match(pattern);
      let description = `Uses ${name}`;

      // Try to extract more context
      if (match) {
        const idx = combinedContent.indexOf(match[0]);
        const context = combinedContent.slice(
          Math.max(0, idx - 100),
          Math.min(combinedContent.length, idx + 200)
        );
        // Clean up and extract relevant sentence
        const sentences = context.split(/[.!?\n]/).filter(s => s.includes(match[0]));
        if (sentences.length > 0) {
          description = sentences[0].trim();
        }
      }

      techniques.push({
        name,
        description,
        isMainContribution: false,
        relatedTo: [],
      });
    }
  }

  // Identify main contribution from README (usually in first heading or description)
  const mainContributionPatterns = [
    /(?:we\s+)?(?:propose|present|introduce)\s+([^.!?\n]+)/i,
    /(?:novel|new)\s+(?:method|approach|technique|framework)\s+(?:for\s+)?([^.!?\n]+)/i,
    /our\s+(?:main\s+)?contribution[s]?\s+(?:is|are|include)[s]?\s+([^.!?\n]+)/i,
  ];

  for (const pattern of mainContributionPatterns) {
    const match = readme.match(pattern);
    if (match) {
      techniques.unshift({
        name: "Main Contribution",
        description: match[0].trim(),
        isMainContribution: true,
      });
      break;
    }
  }

  return techniques;
}

/**
 * Extract linked papers from content
 */
export function extractLinkedPapers(
  readme: string,
  codeContent: string
): LinkedPaper[] {
  const papers: LinkedPaper[] = [];
  const seen = new Set<string>();

  const combinedContent = readme + "\n" + codeContent;

  // arXiv patterns
  const arxivPatterns = [
    /arxiv\.org\/abs\/(\d{4}\.\d{4,5})/gi,
    /arxiv\.org\/pdf\/(\d{4}\.\d{4,5})/gi,
    /arXiv[:\s]*(\d{4}\.\d{4,5})/gi,
  ];

  for (const pattern of arxivPatterns) {
    let match;
    while ((match = pattern.exec(combinedContent)) !== null) {
      const id = match[1];
      if (!seen.has(id)) {
        seen.add(id);
        papers.push({
          id,
          type: "arxiv",
          source: readme.includes(match[0]) ? "readme" : "code",
        });
      }
    }
  }

  // DOI patterns
  const doiPattern = /doi\.org\/(10\.\d{4,}\/[^\s\)\]"']+)/gi;
  let doiMatch;
  while ((doiMatch = doiPattern.exec(combinedContent)) !== null) {
    const doi = doiMatch[1];
    if (!seen.has(doi)) {
      seen.add(doi);
      papers.push({
        id: doi,
        type: "doi",
        source: readme.includes(doiMatch[0]) ? "readme" : "code",
      });
    }
  }

  return papers;
}

/**
 * Parse dependencies from requirements.txt or pyproject.toml
 */
export function parseDependencies(content: string): Dependency[] {
  const dependencies: Dependency[] = [];
  const lines = content.split("\n");

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#") || trimmed.startsWith("[")) {
      continue;
    }

    // Parse requirement line (e.g., "torch>=2.0.0", "transformers==4.30.0")
    const match = trimmed.match(/^([a-zA-Z0-9_-]+)(?:\[.*?\])?(?:(>=|<=|==|~=|!=)(.+))?/);
    if (match) {
      const name = match[1].toLowerCase();
      const version = match[3]?.trim();

      // Determine category
      let category: Dependency["category"] = "unknown";
      for (const [dep, cat] of Object.entries(KEY_ML_DEPENDENCIES)) {
        if (name.includes(dep)) {
          category = cat;
          break;
        }
      }

      // Determine if it's a key dependency
      const isKey = Object.keys(KEY_ML_DEPENDENCIES).some((dep) =>
        name.includes(dep)
      );

      dependencies.push({
        name,
        version,
        category,
        isKey,
      });
    }
  }

  return dependencies;
}

/**
 * Generate suggested tasks from repo analysis
 */
export function generateTasks(
  analysis: Partial<RepoAnalysis>,
  readme: string
): SuggestedTask[] {
  const tasks: SuggestedTask[] = [];

  // Task 1: Setup environment
  const setupTask: SuggestedTask = {
    title: "Set up development environment",
    description: `Clone repository and install dependencies. Framework: ${analysis.framework || "unknown"}.`,
    category: "setup",
    priority: 1,
    estimatedHours: 1,
    dependencies: [],
    relatedFiles: ["requirements.txt", "pyproject.toml", "setup.py"],
  };

  // Add framework-specific notes
  if (analysis.framework === "pytorch") {
    setupTask.description += " Ensure CUDA is properly configured for PyTorch.";
  } else if (analysis.framework === "tensorflow") {
    setupTask.description += " Verify TensorFlow GPU support.";
  }

  tasks.push(setupTask);

  // Task 2: Understand the codebase
  tasks.push({
    title: "Study repository structure and key components",
    description: `Review the codebase architecture. Focus on: ${analysis.architectures?.join(", ") || "main model components"}.`,
    category: "understand",
    priority: 2,
    estimatedHours: 2,
    dependencies: ["Set up development environment"],
    relatedFiles: ["README.md", "model.py", "train.py"],
  });

  // Task 3: Study linked papers
  if (analysis.linkedPapers && analysis.linkedPapers.length > 0) {
    tasks.push({
      title: "Read linked research papers",
      description: `Study ${analysis.linkedPapers.length} linked paper(s) to understand theoretical background: ${analysis.linkedPapers.slice(0, 3).map(p => p.id).join(", ")}`,
      category: "understand",
      priority: 2,
      estimatedHours: analysis.linkedPapers.length * 2,
      dependencies: ["Set up development environment"],
      relatedFiles: [],
    });
  }

  // Task 4: Implement/adapt based on project type
  if (analysis.projectType === "training" || analysis.projectType === "research") {
    tasks.push({
      title: "Run training pipeline",
      description: "Execute the training script with sample data to verify functionality.",
      category: "implement",
      priority: 3,
      estimatedHours: 4,
      dependencies: ["Study repository structure and key components"],
      relatedFiles: ["train.py", "config.yaml"],
    });

    tasks.push({
      title: "Adapt training for target task",
      description: "Modify training configuration and data pipeline for your specific use case.",
      category: "implement",
      priority: 4,
      estimatedHours: 8,
      dependencies: ["Run training pipeline"],
      relatedFiles: ["train.py", "config.yaml", "dataset.py"],
    });
  }

  if (analysis.projectType === "model" || analysis.projectType === "inference") {
    tasks.push({
      title: "Run inference pipeline",
      description: "Load pretrained model and run inference on sample inputs.",
      category: "implement",
      priority: 3,
      estimatedHours: 2,
      dependencies: ["Study repository structure and key components"],
      relatedFiles: ["inference.py", "generate.py", "predict.py"],
    });

    tasks.push({
      title: "Integrate model into project",
      description: "Adapt the inference code to work with your existing codebase.",
      category: "integrate",
      priority: 4,
      estimatedHours: 4,
      dependencies: ["Run inference pipeline"],
      relatedFiles: [],
    });
  }

  // Task 5: Evaluation
  tasks.push({
    title: "Evaluate results",
    description: "Run evaluation metrics to assess model performance on your data.",
    category: "evaluate",
    priority: 5,
    estimatedHours: 3,
    dependencies: tasks.filter(t => t.category === "implement").map(t => t.title),
    relatedFiles: ["evaluate.py", "metrics.py"],
  });

  return tasks;
}

/**
 * Assess complexity of implementing the repo
 */
export function assessComplexity(
  analysis: Partial<RepoAnalysis>,
  readme: string
): { complexity: ComplexityLevel; reason: string } {
  let score = 0;
  const factors: string[] = [];

  // Framework complexity
  if (analysis.framework === "unknown") {
    score += 2;
    factors.push("Unknown framework");
  }

  // Multiple architectures
  if (analysis.architectures && analysis.architectures.length > 2) {
    score += 2;
    factors.push("Multiple architecture types");
  }

  // Complex architecture types
  const complexArchs = ["diffusion", "hybrid", "graph"];
  if (analysis.architectures?.some(a => complexArchs.includes(a))) {
    score += 2;
    factors.push("Complex architecture type");
  }

  // Many techniques
  if (analysis.techniques && analysis.techniques.length > 5) {
    score += 1;
    factors.push("Many techniques to understand");
  }

  // Resource requirements
  if (analysis.resources?.some(r => r.type === "gpu" && r.required)) {
    score += 1;
    factors.push("GPU required");
  }

  // Large codebase indicators
  if (readme.length > 10000) {
    score += 1;
    factors.push("Extensive documentation");
  }

  // Many dependencies
  if (analysis.dependencies && analysis.dependencies.length > 20) {
    score += 1;
    factors.push("Many dependencies");
  }

  // Determine complexity level
  let complexity: ComplexityLevel;
  if (score <= 2) {
    complexity = "simple";
  } else if (score <= 4) {
    complexity = "moderate";
  } else if (score <= 6) {
    complexity = "complex";
  } else {
    complexity = "research";
  }

  const reason = factors.length > 0
    ? `Based on: ${factors.join(", ")}`
    : "Standard implementation";

  return { complexity, reason };
}

/**
 * Extract resource requirements from repo
 */
export function extractResources(
  readme: string,
  dependencies: Dependency[]
): ResourceRequirement[] {
  const resources: ResourceRequirement[] = [];

  // GPU requirements
  const gpuPatterns = [
    /(?:requires?|needs?)\s+(?:a\s+)?(?:GPU|CUDA|NVIDIA)/i,
    /GPU\s+(?:memory|VRAM)[:\s]*(\d+)\s*GB/i,
    /(?:tested|trained)\s+(?:on|with)\s+(?:A100|V100|RTX|A6000|H100)/i,
  ];

  for (const pattern of gpuPatterns) {
    if (pattern.test(readme)) {
      const match = readme.match(pattern);
      resources.push({
        type: "gpu",
        name: "NVIDIA GPU",
        required: true,
        estimate: match?.[1] ? `${match[1]}GB VRAM` : "GPU with CUDA support",
        notes: match?.[0],
      });
      break;
    }
  }

  // Memory requirements
  const memPattern = /(?:memory|RAM)[:\s]*(\d+)\s*GB/i;
  const memMatch = readme.match(memPattern);
  if (memMatch) {
    resources.push({
      type: "hardware",
      name: "System Memory",
      required: true,
      estimate: `${memMatch[1]}GB RAM`,
    });
  }

  // Dataset requirements
  const datasetPatterns = [
    /(?:dataset|data)[:\s]*([A-Za-z0-9_-]+(?:[\s,]+[A-Za-z0-9_-]+)*)/i,
    /(?:trained|evaluated)\s+on\s+([A-Za-z0-9_-]+)/i,
  ];

  for (const pattern of datasetPatterns) {
    const match = readme.match(pattern);
    if (match) {
      resources.push({
        type: "dataset",
        name: match[1].trim(),
        required: false,
        notes: "Required for training/evaluation",
      });
      break;
    }
  }

  // Key library requirements
  const keyDeps = dependencies.filter(d => d.isKey);
  for (const dep of keyDeps) {
    resources.push({
      type: "library",
      name: dep.name,
      required: true,
      estimate: dep.version ? `version ${dep.version}` : undefined,
    });
  }

  return resources;
}

/**
 * Summarize README content
 */
export function summarizeReadme(readme: string): string {
  // Extract first meaningful paragraph
  const lines = readme.split("\n");
  const nonEmptyLines: string[] = [];

  let foundContent = false;
  for (const line of lines) {
    const trimmed = line.trim();

    // Skip badges, images, and headings at the start
    if (!foundContent) {
      if (
        trimmed.startsWith("!") ||
        trimmed.startsWith("[") ||
        trimmed.startsWith("#") ||
        trimmed.length === 0
      ) {
        continue;
      }
      foundContent = true;
    }

    if (trimmed.length === 0 && nonEmptyLines.length > 0) {
      break; // End of first paragraph
    }

    if (trimmed.length > 0 && !trimmed.startsWith("#")) {
      nonEmptyLines.push(trimmed);
    }
  }

  const summary = nonEmptyLines.join(" ").replace(/\s+/g, " ").trim();

  // Truncate if too long
  if (summary.length > 500) {
    return summary.slice(0, 497) + "...";
  }

  return summary || "No description available.";
}

/**
 * Calculate ML project confidence score
 */
export function calculateMLConfidence(
  framework: MLFramework,
  architectures: ArchitectureType[],
  dependencies: Dependency[],
  readme: string
): number {
  let score = 0;

  // Framework detection
  if (framework !== "unknown") {
    score += 0.3;
  }

  // Architecture detection
  if (architectures.length > 0 && !architectures.includes("unknown")) {
    score += 0.2;
  }

  // Key ML dependencies
  const mlDeps = dependencies.filter(d => d.category === "ml").length;
  if (mlDeps >= 3) {
    score += 0.2;
  } else if (mlDeps >= 1) {
    score += 0.1;
  }

  // ML keywords in README
  const mlKeywords = [
    "model",
    "training",
    "inference",
    "neural",
    "machine learning",
    "deep learning",
    "transformer",
    "attention",
    "pretrained",
  ];

  const keywordMatches = mlKeywords.filter(kw =>
    readme.toLowerCase().includes(kw)
  ).length;

  if (keywordMatches >= 5) {
    score += 0.3;
  } else if (keywordMatches >= 3) {
    score += 0.2;
  } else if (keywordMatches >= 1) {
    score += 0.1;
  }

  return Math.min(1, score);
}

/**
 * Full repository analysis
 */
export function analyzeRepository(
  url: string,
  readme: string,
  codeContent: string,
  files: string[],
  requirementsContent?: string
): RepoAnalysis {
  // Parse URL
  const urlMatch = url.match(/github\.com\/([^/]+)\/([^/]+)/);
  const owner = urlMatch?.[1] || "unknown";
  const name = urlMatch?.[2]?.replace(/\.git$/, "") || "unknown";

  // Detect framework
  const { framework, confidence: frameworkConfidence } = detectFramework(
    codeContent + "\n" + (requirementsContent || "")
  );

  // Detect architectures
  const architectures = detectArchitectures(codeContent + "\n" + readme);

  // Detect project type
  const projectType = detectProjectType(files, codeContent + "\n" + readme);

  // Parse dependencies
  const dependencies = requirementsContent
    ? parseDependencies(requirementsContent)
    : [];

  // Extract techniques
  const techniques = extractTechniques(readme, codeContent);

  // Extract linked papers
  const linkedPapers = extractLinkedPapers(readme, codeContent);

  // Extract resources
  const resources = extractResources(readme, dependencies);

  // Calculate ML confidence
  const mlConfidence = calculateMLConfidence(
    framework,
    architectures,
    dependencies,
    readme
  );

  // Build partial analysis for task generation
  const partialAnalysis: Partial<RepoAnalysis> = {
    url,
    name,
    owner,
    framework,
    projectType,
    architectures,
    techniques,
    linkedPapers,
    dependencies,
    resources,
  };

  // Generate tasks
  const suggestedTasks = generateTasks(partialAnalysis, readme);

  // Assess complexity
  const { complexity, reason: complexityReason } = assessComplexity(
    partialAnalysis,
    readme
  );

  // Summarize README
  const readmeSummary = summarizeReadme(readme);

  return {
    url,
    name,
    owner,
    framework,
    frameworkConfidence,
    projectType,
    architectures,
    isMLProject: mlConfidence >= 0.5,
    mlConfidence,
    techniques,
    patterns: [], // Would require AST analysis for detailed patterns
    resources,
    linkedPapers,
    suggestedTasks,
    complexity,
    complexityReason,
    dependencies,
    readmeSummary,
    analyzedAt: new Date().toISOString(),
  };
}

// ============================================================================
// Utility Exports
// ============================================================================

export const FRAMEWORK_DISPLAY_NAMES: Record<MLFramework, string> = {
  pytorch: "PyTorch",
  tensorflow: "TensorFlow",
  jax: "JAX",
  keras: "Keras",
  huggingface: "Hugging Face",
  lightning: "PyTorch Lightning",
  fastai: "fast.ai",
  onnx: "ONNX",
  mlx: "MLX",
  unknown: "Unknown",
};

export const PROJECT_TYPE_DISPLAY_NAMES: Record<ProjectType, string> = {
  model: "Pre-trained Model",
  training: "Training Code",
  inference: "Inference/Deployment",
  dataset: "Dataset Processing",
  benchmark: "Benchmark/Evaluation",
  library: "ML Library",
  application: "Application",
  research: "Research Implementation",
  unknown: "Unknown",
};

export const ARCHITECTURE_DISPLAY_NAMES: Record<ArchitectureType, string> = {
  transformer: "Transformer",
  cnn: "CNN",
  rnn: "RNN/LSTM",
  gan: "GAN",
  diffusion: "Diffusion",
  vae: "VAE",
  autoencoder: "Autoencoder",
  mlp: "MLP",
  graph: "Graph Neural Network",
  hybrid: "Hybrid",
  unknown: "Unknown",
};
