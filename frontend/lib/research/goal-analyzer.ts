/**
 * Custom Research Goal Analyzer
 *
 * Analyzes natural language research goals and generates:
 * - Domain recommendations
 * - Relevant paper suggestions
 * - Multi-step research plans
 * - Task breakdowns
 * - Resource and timeline estimates
 */

import type { ExtractedTechnique, ComplexityLevel } from "@/lib/papers/types";

// ============================================================================
// Types
// ============================================================================

/**
 * Research domain
 */
export interface ResearchDomain {
  /** Domain slug */
  slug: string;
  /** Display name */
  name: string;
  /** Description */
  description: string;
  /** Related keywords */
  keywords: string[];
  /** Match score (0-1) */
  matchScore: number;
}

/**
 * Paper suggestion
 */
export interface PaperSuggestion {
  /** Paper title or query */
  title: string;
  /** Why this paper is relevant */
  reason: string;
  /** Search query to find the paper */
  searchQuery: string;
  /** Relevance score (0-1) */
  relevance: number;
  /** Paper category */
  category: "foundational" | "recent" | "technique" | "application";
}

/**
 * Research step in a plan
 */
export interface ResearchStep {
  /** Step number */
  order: number;
  /** Step title */
  title: string;
  /** Step description */
  description: string;
  /** Step type */
  type: "research" | "implementation" | "experiment" | "evaluation" | "documentation";
  /** Estimated hours */
  estimatedHours: number;
  /** Dependencies (step numbers) */
  dependencies: number[];
  /** Deliverables */
  deliverables: string[];
  /** Skills required */
  skills: string[];
}

/**
 * Research plan
 */
export interface ResearchPlan {
  /** Plan title */
  title: string;
  /** Plan summary */
  summary: string;
  /** Research steps */
  steps: ResearchStep[];
  /** Total estimated hours */
  totalHours: number;
  /** Timeline in weeks */
  timelineWeeks: number;
  /** Key milestones */
  milestones: ResearchMilestone[];
}

/**
 * Research milestone
 */
export interface ResearchMilestone {
  /** Milestone title */
  title: string;
  /** Week number */
  week: number;
  /** Related step numbers */
  steps: number[];
  /** Deliverable description */
  deliverable: string;
}

/**
 * Resource estimate
 */
export interface ResourceEstimate {
  /** Resource type */
  type: "compute" | "data" | "tooling" | "time" | "expertise";
  /** Resource name */
  name: string;
  /** Estimated requirement */
  estimate: string;
  /** Is it critical? */
  isCritical: boolean;
  /** Alternatives if available */
  alternatives?: string[];
}

/**
 * Goal analysis result
 */
export interface GoalAnalysis {
  /** Original goal text */
  originalGoal: string;
  /** Extracted key concepts */
  concepts: string[];
  /** Identified techniques */
  techniques: ExtractedTechnique[];
  /** Recommended domain */
  recommendedDomain: ResearchDomain;
  /** Alternative domains */
  alternativeDomains: ResearchDomain[];
  /** Paper suggestions */
  paperSuggestions: PaperSuggestion[];
  /** Research plan */
  plan: ResearchPlan;
  /** Resource estimates */
  resources: ResourceEstimate[];
  /** Complexity assessment */
  complexity: ComplexityLevel;
  /** Complexity explanation */
  complexityReason: string;
  /** Confidence score (0-1) */
  confidence: number;
  /** Analysis timestamp */
  analyzedAt: string;
}

/**
 * Goal parse result
 */
export interface GoalParseResult {
  /** Main objective */
  objective: string;
  /** Target domain area */
  domain: string;
  /** Specific techniques mentioned */
  techniques: string[];
  /** Desired outcomes */
  outcomes: string[];
  /** Constraints mentioned */
  constraints: string[];
  /** Keywords extracted */
  keywords: string[];
}

// ============================================================================
// Domain Database
// ============================================================================

/**
 * Available research domains
 */
export const RESEARCH_DOMAINS: Omit<ResearchDomain, "matchScore">[] = [
  {
    slug: "voice-clone",
    name: "Voice Cloning",
    description: "Zero-shot and few-shot voice cloning, speaker embedding, voice conversion",
    keywords: [
      "voice clone", "voice cloning", "speaker embedding", "speaker encoder",
      "voice conversion", "vc", "tts clone", "zero-shot tts", "few-shot tts",
      "speaker adaptation", "voice transfer", "timbre", "speaker similarity",
    ],
  },
  {
    slug: "prosody",
    name: "Prosody & Emotion",
    description: "Prosody modeling, emotion transfer, expressiveness in speech synthesis",
    keywords: [
      "prosody", "emotion", "expressiveness", "intonation", "pitch",
      "rhythm", "stress", "emphasis", "expressive tts", "emotion transfer",
      "emotional speech", "speaking style", "prosodic", "f0",
    ],
  },
  {
    slug: "tts",
    name: "Text-to-Speech",
    description: "Neural TTS systems, vocoder development, speech synthesis",
    keywords: [
      "tts", "text to speech", "speech synthesis", "vocoder", "acoustic model",
      "neural tts", "end-to-end tts", "fastspeech", "tacotron", "vits",
      "naturalspeech", "tortoise", "bark", "xtts",
    ],
  },
  {
    slug: "asr",
    name: "Speech Recognition",
    description: "Automatic speech recognition, transcription, speech-to-text",
    keywords: [
      "asr", "speech recognition", "transcription", "stt", "speech to text",
      "whisper", "wav2vec", "conformer", "transducer", "ctc",
    ],
  },
  {
    slug: "nlp",
    name: "Natural Language Processing",
    description: "Language models, text understanding, generation, translation",
    keywords: [
      "nlp", "language model", "llm", "transformer", "bert", "gpt",
      "text generation", "translation", "sentiment", "ner", "question answering",
      "summarization", "embeddings", "tokenization",
    ],
  },
  {
    slug: "vision",
    name: "Computer Vision",
    description: "Image recognition, object detection, segmentation, generation",
    keywords: [
      "vision", "image", "cnn", "object detection", "segmentation",
      "classification", "diffusion", "stable diffusion", "gan", "vit",
      "clip", "image generation", "inpainting",
    ],
  },
  {
    slug: "multimodal",
    name: "Multimodal AI",
    description: "Combining vision, language, audio, and other modalities",
    keywords: [
      "multimodal", "vision language", "audio visual", "cross-modal",
      "clip", "flamingo", "gpt-4v", "gemini", "llava",
    ],
  },
  {
    slug: "audio",
    name: "Audio Processing",
    description: "Music generation, audio classification, sound synthesis",
    keywords: [
      "audio", "music", "sound", "audio classification", "music generation",
      "audio synthesis", "audio enhancement", "noise reduction", "separation",
    ],
  },
  {
    slug: "rl",
    name: "Reinforcement Learning",
    description: "RL algorithms, policy learning, reward modeling",
    keywords: [
      "reinforcement learning", "rl", "policy", "reward", "ppo", "dpo",
      "rlhf", "agent", "environment", "q-learning", "actor-critic",
    ],
  },
  {
    slug: "efficiency",
    name: "ML Efficiency",
    description: "Model compression, quantization, distillation, efficient inference",
    keywords: [
      "efficiency", "quantization", "pruning", "distillation", "compression",
      "onnx", "tensorrt", "optimization", "inference", "latency", "throughput",
    ],
  },
];

// ============================================================================
// Concept Extraction Patterns
// ============================================================================

/**
 * Patterns for extracting research concepts
 */
const CONCEPT_PATTERNS = [
  // Techniques
  { pattern: /(?:using|with|via|through)\s+([a-z][a-z\s-]{2,30})/gi, type: "technique" },
  { pattern: /([a-z][a-z\s-]{2,30})\s+(?:based|approach|method|technique)/gi, type: "technique" },
  // Goals
  { pattern: /(?:improve|enhance|better|increase)\s+([a-z][a-z\s-]{2,30})/gi, type: "goal" },
  { pattern: /(?:reduce|decrease|minimize|lower)\s+([a-z][a-z\s-]{2,30})/gi, type: "goal" },
  // Objects
  { pattern: /(?:build|create|develop|implement)\s+(?:a\s+)?([a-z][a-z\s-]{2,30})/gi, type: "object" },
  { pattern: /([a-z][a-z\s-]{2,30})\s+(?:model|system|pipeline|framework)/gi, type: "object" },
];

/**
 * Stop words to filter from concepts
 */
const STOP_WORDS = new Set([
  "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
  "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
  "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
  "should", "may", "might", "must", "can", "this", "that", "these", "those",
  "my", "your", "their", "our", "its", "i", "we", "you", "they", "it",
  "what", "which", "who", "when", "where", "why", "how", "all", "each",
  "every", "both", "few", "more", "most", "other", "some", "such", "no",
  "not", "only", "same", "so", "than", "too", "very", "just", "also",
]);

// ============================================================================
// Analysis Functions
// ============================================================================

/**
 * Parse research goal to extract key components
 */
export function parseGoal(goal: string): GoalParseResult {
  const lowerGoal = goal.toLowerCase();
  const words = lowerGoal.split(/\s+/).filter(w => !STOP_WORDS.has(w) && w.length > 2);

  // Extract objective (main action + object)
  const objectivePatterns = [
    /(?:i want to|i'd like to|i need to|goal is to|aim is to)\s+(.+?)(?:\.|$)/i,
    /(?:build|create|develop|implement|train|improve)\s+(.+?)(?:that|which|for|to|\.|$)/i,
  ];

  let objective = goal.split(/[.!?]/)[0];
  for (const pattern of objectivePatterns) {
    const match = goal.match(pattern);
    if (match) {
      objective = match[1].trim();
      break;
    }
  }

  // Extract techniques mentioned
  const techniques: string[] = [];
  const techniquePatterns = [
    /(?:using|with|via)\s+([A-Za-z][A-Za-z0-9\s-]{2,30})/gi,
    /([A-Za-z][A-Za-z0-9-]+(?:Net|GAN|VAE|BERT|GPT|Transformer|RNN|LSTM|CNN))/gi,
  ];

  for (const pattern of techniquePatterns) {
    let match;
    while ((match = pattern.exec(goal)) !== null) {
      const tech = match[1].trim();
      if (tech.length > 2 && !STOP_WORDS.has(tech.toLowerCase())) {
        techniques.push(tech);
      }
    }
  }

  // Extract desired outcomes
  const outcomes: string[] = [];
  const outcomePatterns = [
    /(?:to achieve|to get|resulting in|leading to)\s+(.+?)(?:\.|,|$)/gi,
    /(?:improve|enhance|increase|boost)\s+([a-z][a-z\s-]{2,30})/gi,
    /(?:better|higher|lower|faster)\s+([a-z][a-z\s-]{2,30})/gi,
  ];

  for (const pattern of outcomePatterns) {
    let match;
    while ((match = pattern.exec(goal)) !== null) {
      outcomes.push(match[1].trim());
    }
  }

  // Extract constraints
  const constraints: string[] = [];
  const constraintPatterns = [
    /(?:without|limited to|must|need to|required to)\s+(.+?)(?:\.|,|$)/gi,
    /(?:within|under|less than)\s+(\d+\s*(?:hours|days|weeks|GB|samples))/gi,
  ];

  for (const pattern of constraintPatterns) {
    let match;
    while ((match = pattern.exec(goal)) !== null) {
      constraints.push(match[1].trim());
    }
  }

  // Identify domain area
  let domain = "";
  let maxScore = 0;
  for (const d of RESEARCH_DOMAINS) {
    let score = 0;
    for (const keyword of d.keywords) {
      if (lowerGoal.includes(keyword)) {
        score += keyword.split(" ").length; // Multi-word matches score higher
      }
    }
    if (score > maxScore) {
      maxScore = score;
      domain = d.slug;
    }
  }

  return {
    objective,
    domain,
    techniques,
    outcomes,
    constraints,
    keywords: words.slice(0, 20), // Top 20 keywords
  };
}

/**
 * Match goal to research domains
 */
export function matchDomains(goal: string): ResearchDomain[] {
  const lowerGoal = goal.toLowerCase();
  const results: ResearchDomain[] = [];

  for (const domain of RESEARCH_DOMAINS) {
    let score = 0;
    let matches = 0;

    for (const keyword of domain.keywords) {
      if (lowerGoal.includes(keyword)) {
        // Multi-word keywords score higher
        const wordCount = keyword.split(" ").length;
        score += wordCount * 0.2;
        matches++;
      }
    }

    // Normalize score
    const matchScore = Math.min(1, score);

    if (matchScore > 0) {
      results.push({
        ...domain,
        matchScore,
      });
    }
  }

  // Sort by score descending
  results.sort((a, b) => b.matchScore - a.matchScore);

  // If no matches, add top 3 domains with low scores
  if (results.length === 0) {
    return RESEARCH_DOMAINS.slice(0, 3).map(d => ({ ...d, matchScore: 0.1 }));
  }

  return results;
}

/**
 * Generate paper suggestions based on goal
 */
export function suggestPapers(
  parseResult: GoalParseResult,
  domains: ResearchDomain[]
): PaperSuggestion[] {
  const suggestions: PaperSuggestion[] = [];
  const primaryDomain = domains[0];

  // Foundational papers for the domain
  const foundationalQueries: Record<string, string[]> = {
    "voice-clone": [
      "speaker encoder neural network",
      "zero-shot voice cloning",
      "voice conversion GAN",
    ],
    "prosody": [
      "prosody modeling neural TTS",
      "emotion transfer speech synthesis",
      "expressive text to speech",
    ],
    "tts": [
      "end-to-end text to speech",
      "neural vocoder",
      "FastSpeech VITS",
    ],
    "asr": [
      "Whisper speech recognition",
      "wav2vec self-supervised",
      "Conformer ASR",
    ],
    "nlp": [
      "transformer attention mechanism",
      "BERT language understanding",
      "GPT language model",
    ],
    "vision": [
      "ResNet deep learning vision",
      "Vision Transformer ViT",
      "diffusion image generation",
    ],
    "multimodal": [
      "CLIP vision language",
      "multimodal learning",
      "audio visual representation",
    ],
  };

  // Add foundational paper suggestions
  const domainQueries = foundationalQueries[primaryDomain.slug] || [];
  for (const query of domainQueries) {
    suggestions.push({
      title: `Foundational: ${query}`,
      reason: `Core technique for ${primaryDomain.name}`,
      searchQuery: query,
      relevance: 0.9,
      category: "foundational",
    });
  }

  // Add technique-specific suggestions
  for (const technique of parseResult.techniques) {
    suggestions.push({
      title: `Technique: ${technique}`,
      reason: `Directly mentioned in your research goal`,
      searchQuery: `${technique} ${primaryDomain.name}`,
      relevance: 0.95,
      category: "technique",
    });
  }

  // Add recent papers query
  suggestions.push({
    title: `Recent advances in ${primaryDomain.name}`,
    reason: "Stay up-to-date with latest developments",
    searchQuery: `${primaryDomain.name} 2024`,
    relevance: 0.8,
    category: "recent",
  });

  // Add application-specific suggestions based on outcomes
  for (const outcome of parseResult.outcomes.slice(0, 2)) {
    suggestions.push({
      title: `Application: ${outcome}`,
      reason: "Aligned with your desired outcome",
      searchQuery: `${outcome} ${primaryDomain.name}`,
      relevance: 0.85,
      category: "application",
    });
  }

  return suggestions.slice(0, 10); // Limit to 10 suggestions
}

/**
 * Generate research plan from goal analysis
 */
export function generatePlan(
  parseResult: GoalParseResult,
  domain: ResearchDomain,
  techniques: ExtractedTechnique[]
): ResearchPlan {
  const steps: ResearchStep[] = [];
  let totalHours = 0;

  // Step 1: Literature Review
  steps.push({
    order: 1,
    title: "Literature Review",
    description: `Review foundational papers in ${domain.name}. Understand key techniques and state-of-the-art approaches.`,
    type: "research",
    estimatedHours: 8,
    dependencies: [],
    deliverables: ["Literature summary document", "Key papers list", "Technique comparison table"],
    skills: ["Research skills", "Technical reading"],
  });
  totalHours += 8;

  // Step 2: Problem Formulation
  steps.push({
    order: 2,
    title: "Problem Formulation",
    description: `Define the specific problem: ${parseResult.objective}. Identify metrics, baselines, and success criteria.`,
    type: "research",
    estimatedHours: 4,
    dependencies: [1],
    deliverables: ["Problem statement", "Evaluation metrics", "Baseline models"],
    skills: ["Problem analysis", "Metric design"],
  });
  totalHours += 4;

  // Step 3: Data Preparation
  steps.push({
    order: 3,
    title: "Data Preparation",
    description: "Collect, clean, and prepare datasets for training and evaluation.",
    type: "implementation",
    estimatedHours: 12,
    dependencies: [2],
    deliverables: ["Training dataset", "Evaluation dataset", "Data loading pipeline"],
    skills: ["Data engineering", "Python"],
  });
  totalHours += 12;

  // Step 4: Baseline Implementation
  steps.push({
    order: 4,
    title: "Baseline Implementation",
    description: "Implement or adapt existing baseline models for comparison.",
    type: "implementation",
    estimatedHours: 16,
    dependencies: [3],
    deliverables: ["Working baseline model", "Baseline evaluation results"],
    skills: ["PyTorch/TensorFlow", "Model implementation"],
  });
  totalHours += 16;

  // Step 5: Proposed Method Implementation
  const techniqueNames = techniques.map(t => t.name).slice(0, 3).join(", ");
  steps.push({
    order: 5,
    title: "Proposed Method Implementation",
    description: `Implement the proposed approach${techniqueNames ? ` using ${techniqueNames}` : ""}.`,
    type: "implementation",
    estimatedHours: 24,
    dependencies: [4],
    deliverables: ["Proposed model implementation", "Training script"],
    skills: ["Deep learning", "Model architecture design"],
  });
  totalHours += 24;

  // Step 6: Training & Tuning
  steps.push({
    order: 6,
    title: "Training & Hyperparameter Tuning",
    description: "Train the model and tune hyperparameters for optimal performance.",
    type: "experiment",
    estimatedHours: 16,
    dependencies: [5],
    deliverables: ["Trained model checkpoints", "Training logs", "Hyperparameter sweep results"],
    skills: ["ML training", "Experiment management"],
  });
  totalHours += 16;

  // Step 7: Evaluation
  steps.push({
    order: 7,
    title: "Comprehensive Evaluation",
    description: "Evaluate the model on all metrics. Compare with baselines and ablate components.",
    type: "evaluation",
    estimatedHours: 8,
    dependencies: [6],
    deliverables: ["Evaluation report", "Comparison tables", "Ablation study results"],
    skills: ["Statistical analysis", "Visualization"],
  });
  totalHours += 8;

  // Step 8: Documentation
  steps.push({
    order: 8,
    title: "Documentation & Writeup",
    description: "Document the approach, results, and learnings. Prepare for sharing.",
    type: "documentation",
    estimatedHours: 8,
    dependencies: [7],
    deliverables: ["Technical report", "Code documentation", "README"],
    skills: ["Technical writing", "Documentation"],
  });
  totalHours += 8;

  // Calculate timeline (assuming ~20 hours/week of research time)
  const timelineWeeks = Math.ceil(totalHours / 20);

  // Generate milestones
  const milestones: ResearchMilestone[] = [
    {
      title: "Research Phase Complete",
      week: Math.ceil(12 / 20),
      steps: [1, 2],
      deliverable: "Literature review and problem definition complete",
    },
    {
      title: "Baseline Ready",
      week: Math.ceil(40 / 20),
      steps: [3, 4],
      deliverable: "Data prepared and baseline model working",
    },
    {
      title: "Implementation Complete",
      week: Math.ceil(64 / 20),
      steps: [5],
      deliverable: "Proposed method implemented",
    },
    {
      title: "Results Ready",
      week: Math.ceil(88 / 20),
      steps: [6, 7],
      deliverable: "Model trained and evaluated",
    },
    {
      title: "Project Complete",
      week: timelineWeeks,
      steps: [8],
      deliverable: "Documentation and report finished",
    },
  ];

  return {
    title: `Research Plan: ${parseResult.objective.slice(0, 50)}...`,
    summary: `A structured plan to ${parseResult.objective} in the domain of ${domain.name}.`,
    steps,
    totalHours,
    timelineWeeks,
    milestones,
  };
}

/**
 * Estimate required resources
 */
export function estimateResources(
  parseResult: GoalParseResult,
  domain: ResearchDomain,
  plan: ResearchPlan
): ResourceEstimate[] {
  const resources: ResourceEstimate[] = [];

  // Compute resources
  const computeIntensive = [
    "voice-clone", "tts", "vision", "multimodal", "nlp",
  ].includes(domain.slug);

  if (computeIntensive) {
    resources.push({
      type: "compute",
      name: "GPU",
      estimate: "16-24GB VRAM recommended (RTX 4090, A5000, or cloud GPU)",
      isCritical: true,
      alternatives: ["Google Colab Pro", "RunPod", "Lambda Labs"],
    });
  } else {
    resources.push({
      type: "compute",
      name: "GPU",
      estimate: "8GB+ VRAM sufficient for most experiments",
      isCritical: false,
      alternatives: ["CPU-based alternatives available"],
    });
  }

  // Data resources
  resources.push({
    type: "data",
    name: "Training Data",
    estimate: `Domain-specific dataset for ${domain.name}`,
    isCritical: true,
    alternatives: ["Public datasets", "Synthetic data generation"],
  });

  // Time resource
  resources.push({
    type: "time",
    name: "Research Time",
    estimate: `${plan.totalHours} hours over ${plan.timelineWeeks} weeks`,
    isCritical: true,
  });

  // Tooling
  resources.push({
    type: "tooling",
    name: "Development Environment",
    estimate: "Python 3.9+, PyTorch/TensorFlow, experiment tracking (W&B)",
    isCritical: false,
  });

  // Expertise based on techniques
  const expertiseAreas: string[] = [domain.name];
  if (parseResult.techniques.length > 0) {
    expertiseAreas.push(...parseResult.techniques.slice(0, 2));
  }

  resources.push({
    type: "expertise",
    name: "Required Skills",
    estimate: expertiseAreas.join(", "),
    isCritical: true,
    alternatives: ["Online courses", "Paper tutorials", "Open-source implementations"],
  });

  return resources;
}

/**
 * Assess complexity of research goal
 */
export function assessGoalComplexity(
  parseResult: GoalParseResult,
  domain: ResearchDomain,
  plan: ResearchPlan
): { complexity: ComplexityLevel; reason: string } {
  let score = 0;
  const factors: string[] = [];

  // Many techniques mentioned
  if (parseResult.techniques.length > 3) {
    score += 2;
    factors.push("Multiple techniques involved");
  }

  // Ambitious outcomes
  if (parseResult.outcomes.length > 2) {
    score += 1;
    factors.push("Multiple desired outcomes");
  }

  // Constraints mentioned
  if (parseResult.constraints.length > 0) {
    score += 1;
    factors.push("Explicit constraints");
  }

  // Long plan
  if (plan.totalHours > 80) {
    score += 2;
    factors.push("Extensive implementation required");
  } else if (plan.totalHours > 50) {
    score += 1;
    factors.push("Moderate implementation effort");
  }

  // Complex domain
  const complexDomains = ["multimodal", "rl", "voice-clone"];
  if (complexDomains.includes(domain.slug)) {
    score += 1;
    factors.push("Complex domain");
  }

  // Determine complexity level
  let complexity: ComplexityLevel;
  if (score <= 1) {
    complexity = "simple";
  } else if (score <= 3) {
    complexity = "moderate";
  } else if (score <= 5) {
    complexity = "complex";
  } else {
    complexity = "research";
  }

  const reason = factors.length > 0
    ? `Based on: ${factors.join(", ")}`
    : "Standard research project";

  return { complexity, reason };
}

/**
 * Extract techniques from goal
 */
export function extractGoalTechniques(
  parseResult: GoalParseResult,
  domain: ResearchDomain
): ExtractedTechnique[] {
  const techniques: ExtractedTechnique[] = [];

  // Add techniques from parsed goal
  for (const tech of parseResult.techniques) {
    techniques.push({
      name: tech,
      description: `Technique mentioned in research goal`,
      isMainContribution: false,
    });
  }

  // Add domain-specific default techniques
  const domainTechniques: Record<string, string[]> = {
    "voice-clone": ["Speaker Embedding", "Voice Conversion"],
    "prosody": ["Prosody Modeling", "Emotion Embedding"],
    "tts": ["Neural TTS", "Neural Vocoder"],
    "asr": ["Acoustic Modeling", "Language Modeling"],
    "nlp": ["Transformer Architecture", "Attention Mechanism"],
    "vision": ["CNN Architecture", "Feature Extraction"],
  };

  const defaults = domainTechniques[domain.slug] || [];
  for (const tech of defaults) {
    if (!techniques.some(t => t.name.toLowerCase() === tech.toLowerCase())) {
      techniques.push({
        name: tech,
        description: `Common technique in ${domain.name}`,
        isMainContribution: false,
      });
    }
  }

  return techniques;
}

/**
 * Calculate confidence score for the analysis
 */
export function calculateConfidence(
  parseResult: GoalParseResult,
  domains: ResearchDomain[]
): number {
  let score = 0;

  // Clear objective
  if (parseResult.objective.length > 10) {
    score += 0.2;
  }

  // Domain match
  if (domains.length > 0 && domains[0].matchScore > 0.5) {
    score += 0.3;
  } else if (domains.length > 0 && domains[0].matchScore > 0.2) {
    score += 0.15;
  }

  // Techniques identified
  if (parseResult.techniques.length > 0) {
    score += 0.2;
  }

  // Outcomes clear
  if (parseResult.outcomes.length > 0) {
    score += 0.15;
  }

  // Keywords extracted
  if (parseResult.keywords.length > 5) {
    score += 0.15;
  }

  return Math.min(1, score);
}

/**
 * Full goal analysis
 */
export function analyzeGoal(goal: string): GoalAnalysis {
  // Parse the goal
  const parseResult = parseGoal(goal);

  // Match domains
  const domains = matchDomains(goal);
  const recommendedDomain = domains[0] || {
    ...RESEARCH_DOMAINS[0],
    matchScore: 0.1,
  };

  // Extract techniques
  const techniques = extractGoalTechniques(parseResult, recommendedDomain);

  // Suggest papers
  const paperSuggestions = suggestPapers(parseResult, domains);

  // Generate plan
  const plan = generatePlan(parseResult, recommendedDomain, techniques);

  // Estimate resources
  const resources = estimateResources(parseResult, recommendedDomain, plan);

  // Assess complexity
  const { complexity, reason: complexityReason } = assessGoalComplexity(
    parseResult,
    recommendedDomain,
    plan
  );

  // Calculate confidence
  const confidence = calculateConfidence(parseResult, domains);

  return {
    originalGoal: goal,
    concepts: parseResult.keywords,
    techniques,
    recommendedDomain,
    alternativeDomains: domains.slice(1, 4),
    paperSuggestions,
    plan,
    resources,
    complexity,
    complexityReason,
    confidence,
    analyzedAt: new Date().toISOString(),
  };
}

// ============================================================================
// Utility Exports
// ============================================================================

export const STEP_TYPE_LABELS: Record<ResearchStep["type"], string> = {
  research: "Research",
  implementation: "Implementation",
  experiment: "Experiment",
  evaluation: "Evaluation",
  documentation: "Documentation",
};

export const RESOURCE_TYPE_LABELS: Record<ResourceEstimate["type"], string> = {
  compute: "Compute",
  data: "Data",
  tooling: "Tooling",
  time: "Time",
  expertise: "Expertise",
};

export const PAPER_CATEGORY_LABELS: Record<PaperSuggestion["category"], string> = {
  foundational: "Foundational",
  recent: "Recent",
  technique: "Technique",
  application: "Application",
};
