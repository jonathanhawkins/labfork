/**
 * Lab Templates
 *
 * Pre-configured lab templates for quick setup across different research domains.
 * Each template provides sensible defaults for domain config, research goals, and hardware recommendations.
 */

import type { DomainConfig, DomainBranding, DomainScene, DomainResearch } from "@/lib/domain/types";
import type { LabConfig, ResearchGoal, InitialTask } from "./types";

/**
 * Template category for organizing templates
 */
export type TemplateCategory =
  | "ai-ml"
  | "science"
  | "engineering"
  | "creative"
  | "community";

/**
 * Template difficulty level
 */
export type TemplateDifficulty = "beginner" | "intermediate" | "advanced";

/**
 * Lab template definition
 */
export interface LabTemplate {
  /** Unique template ID */
  id: string;
  /** Display name */
  name: string;
  /** Short description */
  description: string;
  /** Longer description for template details view */
  longDescription?: string;
  /** Category for filtering */
  category: TemplateCategory;
  /** Difficulty level */
  difficulty: TemplateDifficulty;
  /** Icon name from lucide-react */
  icon: string;
  /** Whether this template is featured/recommended */
  featured?: boolean;
  /** Domain configuration */
  domain: Partial<DomainConfig>;
  /** Pre-filled research goals */
  researchGoal?: ResearchGoal;
  /** Suggested initial tasks */
  initialTasks?: InitialTask[];
  /** Minimum VRAM recommended (GB) */
  minVram?: number;
  /** Tags for search/discovery */
  tags?: string[];
}

/**
 * Get pre-configured LabConfig from template
 */
export function getLabConfigFromTemplate(template: LabTemplate): Partial<LabConfig> {
  return {
    createNewDomain: true,
    domain: template.domain,
    research: {
      path: "goal",
      goal: template.researchGoal,
    },
  };
}

/**
 * Category metadata
 */
export const TEMPLATE_CATEGORIES: Record<TemplateCategory, {
  name: string;
  description: string;
  icon: string;
}> = {
  "ai-ml": {
    name: "AI & Machine Learning",
    description: "Deep learning, NLP, computer vision, and more",
    icon: "Brain",
  },
  "science": {
    name: "Science",
    description: "Biology, chemistry, physics, and climate research",
    icon: "Flask",
  },
  "engineering": {
    name: "Engineering",
    description: "Robotics, hardware, and systems design",
    icon: "Cpu",
  },
  "creative": {
    name: "Creative",
    description: "Music, art, and generative content",
    icon: "Palette",
  },
  "community": {
    name: "Community",
    description: "Templates created by the LabFork community",
    icon: "Users",
  },
};

// ============================================================================
// TEMPLATE DEFINITIONS
// ============================================================================

/**
 * Voice Clone Lab Template
 * Research domain: Speech synthesis, TTS, prosody control
 */
const voiceCloneTemplate: LabTemplate = {
  id: "voice-clone",
  name: "Voice Cloning",
  description: "Speech synthesis and prosody control research",
  longDescription: "Build custom text-to-speech systems that capture voice characteristics, emotions, and speaking styles. Research areas include prosody analysis, neural TTS, and voice conversion.",
  category: "ai-ml",
  difficulty: "intermediate",
  icon: "Mic",
  featured: true,
  domain: {
    name: "Voice Cloning",
    slug: "voice-clone",
    description: "Speech synthesis and prosody control research",
    difficulty: "intermediate",
    branding: {
      primaryColor: "#3b82f6",
      accentColor: "#22c55e",
      backgroundStyle: "sky",
    },
    scene: {
      props: [
        { id: "mic", type: "microphone", position: [-2, 0, 0], scale: 0.8 },
        { id: "speaker", type: "speaker", position: [2, 0, 0], scale: 0.8 },
        { id: "wave", type: "waveform", position: [0, 1.5, -1], scale: 1.2 },
      ],
      decorations: { plants: true, floatingCubes: true, particles: true },
    },
    research: {
      arxivCategories: ["cs.SD", "eess.AS", "cs.CL"],
      keywords: ["TTS", "prosody", "emotion", "voice cloning", "speech synthesis"],
      additionalSources: ["papers-with-code", "github"],
    },
    hardware: {
      minGpuVram: 8,
      recommendedGpuVram: 24,
      gpuRequired: true,
    },
    tags: ["speech", "audio", "synthesis", "TTS", "voice"],
  },
  researchGoal: {
    description: "Develop a voice cloning system that captures prosody and emotion",
    keywords: ["prosody", "emotion", "TTS", "VITS", "voice conversion"],
    suggestedDomain: "voice-clone",
    suggestedCategories: ["cs.SD", "eess.AS"],
    suggestedKeywords: ["neural TTS", "prosody transfer", "emotion embedding"],
  },
  initialTasks: [
    {
      subject: "Review VITS and XTTS architectures",
      description: "Study the VITS and Coqui XTTS papers to understand modern TTS architectures",
      type: "research",
      priority: "high",
    },
    {
      subject: "Set up audio preprocessing pipeline",
      description: "Implement audio loading, resampling, and feature extraction",
      type: "setup",
      priority: "high",
    },
    {
      subject: "Implement prosody analyzer",
      description: "Build module to extract pitch, energy, and duration features",
      type: "implementation",
      priority: "medium",
    },
  ],
  minVram: 8,
  tags: ["speech", "audio", "TTS", "deep learning"],
};

/**
 * Quant Trading Lab Template
 */
const quantTradingTemplate: LabTemplate = {
  id: "quant-trading",
  name: "Quantitative Trading",
  description: "ML-powered trading strategies and market analysis",
  longDescription: "Research algorithmic trading strategies using machine learning. Covers time series prediction, portfolio optimization, risk management, and backtesting frameworks.",
  category: "ai-ml",
  difficulty: "advanced",
  icon: "LineChart",
  featured: true,
  domain: {
    name: "Quantitative Trading",
    slug: "quant-trading",
    description: "ML-powered trading strategies and market analysis",
    difficulty: "advanced",
    branding: {
      primaryColor: "#22c55e",
      accentColor: "#eab308",
      backgroundStyle: "grid",
    },
    scene: {
      props: [
        { id: "chart", type: "chart-wall", position: [0, 0, -2], scale: 1.5 },
        { id: "terminal1", type: "terminal", position: [-2, 0, 0], scale: 0.8 },
        { id: "terminal2", type: "terminal", position: [2, 0, 0], scale: 0.8 },
      ],
      decorations: { floatingCubes: true, particles: true },
    },
    research: {
      arxivCategories: ["q-fin.PM", "q-fin.TR", "cs.LG", "stat.ML"],
      keywords: ["algorithmic trading", "portfolio optimization", "time series", "reinforcement learning"],
      additionalSources: ["github", "papers-with-code"],
    },
    hardware: {
      minGpuVram: 4,
      recommendedGpuVram: 16,
      gpuRequired: false,
    },
    tags: ["finance", "trading", "ML", "time series"],
  },
  researchGoal: {
    description: "Build ML models for market prediction and portfolio optimization",
    keywords: ["time series", "LSTM", "transformer", "backtesting"],
    suggestedCategories: ["q-fin.PM", "cs.LG"],
  },
  initialTasks: [
    {
      subject: "Set up market data pipeline",
      description: "Implement data fetching from Yahoo Finance/Alpha Vantage APIs",
      type: "setup",
      priority: "high",
    },
    {
      subject: "Research transformer-based forecasting",
      description: "Study Temporal Fusion Transformer and Informer papers",
      type: "research",
      priority: "high",
    },
    {
      subject: "Implement backtesting framework",
      description: "Build vectorized backtester with transaction costs",
      type: "implementation",
      priority: "medium",
    },
  ],
  minVram: 4,
  tags: ["finance", "trading", "time series", "RL"],
};

/**
 * Robotics Lab Template
 */
const roboticsTemplate: LabTemplate = {
  id: "robotics",
  name: "Robotics & Control",
  description: "Robot learning, simulation, and control systems",
  longDescription: "Research robot learning using simulation and real hardware. Covers reinforcement learning for control, imitation learning, sim-to-real transfer, and manipulation skills.",
  category: "engineering",
  difficulty: "advanced",
  icon: "Bot",
  domain: {
    name: "Robotics & Control",
    slug: "robotics",
    description: "Robot learning, simulation, and control systems",
    difficulty: "advanced",
    branding: {
      primaryColor: "#8b5cf6",
      accentColor: "#f97316",
      backgroundStyle: "grid",
    },
    scene: {
      props: [
        { id: "arm", type: "robot-arm", position: [0, 0, 0], scale: 1.2 },
        { id: "server", type: "server", position: [-3, 0, 1], scale: 0.7 },
        { id: "camera", type: "camera", position: [2, 1, -1], scale: 0.6, rotation: -0.3 },
      ],
      decorations: { floatingCubes: true },
    },
    research: {
      arxivCategories: ["cs.RO", "cs.LG", "cs.AI"],
      keywords: ["robot learning", "reinforcement learning", "sim-to-real", "manipulation"],
      additionalSources: ["papers-with-code", "github"],
    },
    hardware: {
      minGpuVram: 8,
      recommendedGpuVram: 24,
      gpuRequired: true,
    },
    tags: ["robotics", "RL", "simulation", "control"],
  },
  researchGoal: {
    description: "Train robot manipulation skills using simulation and RL",
    keywords: ["manipulation", "MuJoCo", "Isaac Gym", "PPO", "SAC"],
  },
  initialTasks: [
    {
      subject: "Set up simulation environment",
      description: "Install Isaac Gym or MuJoCo and configure robot models",
      type: "setup",
      priority: "high",
    },
    {
      subject: "Research imitation learning approaches",
      description: "Study ACT, Diffusion Policy, and RT-2 papers",
      type: "research",
      priority: "high",
    },
  ],
  minVram: 8,
  tags: ["robotics", "simulation", "RL", "control"],
};

/**
 * Drug Discovery Lab Template
 */
const drugDiscoveryTemplate: LabTemplate = {
  id: "drug-discovery",
  name: "Drug Discovery",
  description: "ML for molecular design and drug development",
  longDescription: "Apply machine learning to accelerate drug discovery. Research areas include molecular property prediction, generative chemistry, protein-ligand binding, and ADMET prediction.",
  category: "science",
  difficulty: "advanced",
  icon: "Pill",
  featured: true,
  domain: {
    name: "Drug Discovery",
    slug: "drug-discovery",
    description: "ML for molecular design and drug development",
    difficulty: "advanced",
    branding: {
      primaryColor: "#ec4899",
      accentColor: "#06b6d4",
      backgroundStyle: "gradient",
      gradientColors: ["#1e1b4b", "#312e81"],
    },
    scene: {
      props: [
        { id: "mol", type: "molecule", position: [0, 1, 0], scale: 1.5 },
        { id: "server", type: "supercomputer", position: [-3, 0, -1], scale: 0.6 },
      ],
      decorations: { particles: true },
    },
    research: {
      arxivCategories: ["q-bio.BM", "cs.LG", "physics.chem-ph"],
      keywords: ["molecular generation", "drug design", "GNN", "SMILES", "AlphaFold"],
      additionalSources: ["papers-with-code", "semantic-scholar"],
    },
    hardware: {
      minGpuVram: 16,
      recommendedGpuVram: 48,
      gpuRequired: true,
    },
    tags: ["chemistry", "biology", "molecules", "GNN"],
  },
  researchGoal: {
    description: "Develop generative models for novel drug candidate design",
    keywords: ["molecular generation", "SMILES", "graph neural network", "property prediction"],
  },
  initialTasks: [
    {
      subject: "Set up molecular data pipeline",
      description: "Implement SMILES parsing and molecular featurization using RDKit",
      type: "setup",
      priority: "high",
    },
    {
      subject: "Research diffusion models for molecules",
      description: "Study EDM, GeoDiff, and DiffSBDD papers",
      type: "research",
      priority: "high",
    },
  ],
  minVram: 16,
  tags: ["chemistry", "drug discovery", "molecules", "healthcare"],
};

/**
 * Climate Science Lab Template
 */
const climateTemplate: LabTemplate = {
  id: "climate-science",
  name: "Climate Science",
  description: "ML for weather prediction and climate modeling",
  longDescription: "Apply deep learning to climate and weather challenges. Research areas include weather forecasting, climate downscaling, extreme event prediction, and carbon cycle modeling.",
  category: "science",
  difficulty: "intermediate",
  icon: "Cloud",
  domain: {
    name: "Climate Science",
    slug: "climate-science",
    description: "ML for weather prediction and climate modeling",
    difficulty: "intermediate",
    branding: {
      primaryColor: "#0ea5e9",
      accentColor: "#84cc16",
      backgroundStyle: "sky",
    },
    scene: {
      props: [
        { id: "server", type: "supercomputer", position: [0, 0, 0], scale: 1 },
      ],
      decorations: { plants: true, particles: true },
    },
    research: {
      arxivCategories: ["physics.ao-ph", "cs.LG", "stat.ML"],
      keywords: ["weather prediction", "climate modeling", "ERA5", "GraphCast", "Pangu-Weather"],
      additionalSources: ["papers-with-code"],
    },
    hardware: {
      minGpuVram: 24,
      recommendedGpuVram: 80,
      gpuRequired: true,
    },
    tags: ["climate", "weather", "earth science", "forecasting"],
  },
  researchGoal: {
    description: "Build ML models for weather forecasting using ERA5 data",
    keywords: ["weather", "ERA5", "transformer", "GNN", "forecasting"],
  },
  initialTasks: [
    {
      subject: "Download ERA5 sample dataset",
      description: "Get subset of ERA5 reanalysis data from CDS",
      type: "setup",
      priority: "high",
    },
    {
      subject: "Research neural weather models",
      description: "Study GraphCast, Pangu-Weather, and FourCastNet architectures",
      type: "research",
      priority: "high",
    },
  ],
  minVram: 24,
  tags: ["climate", "weather", "geoscience"],
};

/**
 * Computer Vision Lab Template
 */
const computerVisionTemplate: LabTemplate = {
  id: "computer-vision",
  name: "Computer Vision",
  description: "Image understanding, generation, and analysis",
  longDescription: "Research computer vision with modern deep learning. Covers image classification, object detection, segmentation, 3D vision, and generative models like diffusion.",
  category: "ai-ml",
  difficulty: "intermediate",
  icon: "Eye",
  domain: {
    name: "Computer Vision",
    slug: "computer-vision",
    description: "Image understanding, generation, and analysis",
    difficulty: "intermediate",
    branding: {
      primaryColor: "#f59e0b",
      accentColor: "#ef4444",
      backgroundStyle: "minimal",
    },
    scene: {
      props: [
        { id: "camera", type: "camera", position: [0, 0.5, 0], scale: 1.2 },
        { id: "terminal", type: "terminal", position: [2, 0, 1], scale: 0.7 },
      ],
      decorations: { floatingCubes: true },
    },
    research: {
      arxivCategories: ["cs.CV", "cs.LG"],
      keywords: ["image classification", "object detection", "segmentation", "diffusion models"],
      additionalSources: ["papers-with-code", "github"],
    },
    hardware: {
      minGpuVram: 8,
      recommendedGpuVram: 24,
      gpuRequired: true,
    },
    tags: ["vision", "images", "detection", "segmentation"],
  },
  researchGoal: {
    description: "Explore state-of-the-art vision models for image understanding",
    keywords: ["ViT", "CLIP", "SAM", "diffusion", "detection"],
  },
  initialTasks: [
    {
      subject: "Set up vision model inference",
      description: "Load pretrained models from HuggingFace for baseline evaluation",
      type: "setup",
      priority: "high",
    },
    {
      subject: "Research vision transformers",
      description: "Study ViT, DeiT, and Swin Transformer architectures",
      type: "research",
      priority: "medium",
    },
  ],
  minVram: 8,
  tags: ["vision", "images", "transformers"],
};

/**
 * NLP Lab Template
 */
const nlpTemplate: LabTemplate = {
  id: "nlp",
  name: "Natural Language Processing",
  description: "Language understanding, generation, and reasoning",
  longDescription: "Research NLP with large language models. Covers text classification, named entity recognition, question answering, summarization, and LLM fine-tuning.",
  category: "ai-ml",
  difficulty: "beginner",
  icon: "MessageSquare",
  domain: {
    name: "Natural Language Processing",
    slug: "nlp",
    description: "Language understanding, generation, and reasoning",
    difficulty: "beginner",
    branding: {
      primaryColor: "#6366f1",
      accentColor: "#a855f7",
      backgroundStyle: "gradient",
      gradientColors: ["#1e1b4b", "#4c1d95"],
    },
    scene: {
      props: [
        { id: "terminal", type: "terminal", position: [0, 0, 0], scale: 1.2 },
      ],
      decorations: { floatingCubes: true, particles: true },
    },
    research: {
      arxivCategories: ["cs.CL", "cs.LG"],
      keywords: ["LLM", "fine-tuning", "NER", "QA", "summarization", "transformers"],
      additionalSources: ["papers-with-code", "github"],
    },
    hardware: {
      minGpuVram: 8,
      recommendedGpuVram: 24,
      gpuRequired: false,
    },
    tags: ["NLP", "text", "LLM", "transformers"],
  },
  researchGoal: {
    description: "Fine-tune language models for custom NLP tasks",
    keywords: ["fine-tuning", "LoRA", "PEFT", "instruction tuning"],
  },
  initialTasks: [
    {
      subject: "Set up Hugging Face environment",
      description: "Install transformers, datasets, and PEFT libraries",
      type: "setup",
      priority: "high",
    },
    {
      subject: "Research parameter-efficient fine-tuning",
      description: "Study LoRA, QLoRA, and adapter methods",
      type: "research",
      priority: "high",
    },
  ],
  minVram: 8,
  tags: ["NLP", "LLM", "text", "language"],
};

/**
 * Game AI Lab Template
 */
const gameAITemplate: LabTemplate = {
  id: "game-ai",
  name: "Game AI",
  description: "Reinforcement learning for games and simulations",
  longDescription: "Research AI agents that learn to play games. Covers deep reinforcement learning, multi-agent systems, procedural content generation, and game-theoretic approaches.",
  category: "ai-ml",
  difficulty: "intermediate",
  icon: "Gamepad2",
  domain: {
    name: "Game AI",
    slug: "game-ai",
    description: "Reinforcement learning for games and simulations",
    difficulty: "intermediate",
    branding: {
      primaryColor: "#ef4444",
      accentColor: "#fbbf24",
      backgroundStyle: "grid",
    },
    scene: {
      props: [
        { id: "terminal1", type: "terminal", position: [-1.5, 0, 0], scale: 0.8 },
        { id: "terminal2", type: "terminal", position: [1.5, 0, 0], scale: 0.8 },
      ],
      decorations: { floatingCubes: true },
    },
    research: {
      arxivCategories: ["cs.LG", "cs.AI", "cs.GT"],
      keywords: ["reinforcement learning", "game playing", "AlphaZero", "multi-agent"],
      additionalSources: ["papers-with-code", "github"],
    },
    hardware: {
      minGpuVram: 8,
      recommendedGpuVram: 16,
      gpuRequired: true,
    },
    tags: ["games", "RL", "agents", "simulation"],
  },
  researchGoal: {
    description: "Train RL agents to master complex games",
    keywords: ["DQN", "PPO", "AlphaZero", "multi-agent RL"],
  },
  initialTasks: [
    {
      subject: "Set up game environment",
      description: "Install Gymnasium and configure Atari or custom game environments",
      type: "setup",
      priority: "high",
    },
    {
      subject: "Research modern RL algorithms",
      description: "Study PPO, SAC, and MuZero implementations",
      type: "research",
      priority: "high",
    },
  ],
  minVram: 8,
  tags: ["games", "RL", "agents"],
};

/**
 * Biotech NLP Lab Template
 */
const biotechNLPTemplate: LabTemplate = {
  id: "biotech-nlp",
  name: "Biotech NLP",
  description: "NLP for biomedical literature and clinical text",
  longDescription: "Apply NLP to biomedical domains. Research areas include named entity recognition for genes/proteins, relation extraction, clinical note processing, and literature mining.",
  category: "science",
  difficulty: "intermediate",
  icon: "Dna",
  domain: {
    name: "Biotech NLP",
    slug: "biotech-nlp",
    description: "NLP for biomedical literature and clinical text",
    difficulty: "intermediate",
    branding: {
      primaryColor: "#10b981",
      accentColor: "#3b82f6",
      backgroundStyle: "gradient",
      gradientColors: ["#064e3b", "#1e3a5f"],
    },
    scene: {
      props: [
        { id: "mol", type: "molecule", position: [-2, 1, 0], scale: 0.8 },
        { id: "terminal", type: "terminal", position: [1, 0, 0], scale: 1 },
      ],
      decorations: { particles: true },
    },
    research: {
      arxivCategories: ["cs.CL", "q-bio.QM", "cs.IR"],
      keywords: ["biomedical NLP", "PubMed", "clinical NER", "relation extraction", "BioGPT"],
      additionalSources: ["semantic-scholar", "papers-with-code"],
    },
    hardware: {
      minGpuVram: 8,
      recommendedGpuVram: 24,
      gpuRequired: false,
    },
    tags: ["biomedical", "NLP", "clinical", "PubMed"],
  },
  researchGoal: {
    description: "Extract structured knowledge from biomedical literature",
    keywords: ["NER", "relation extraction", "PubMed", "BioGPT", "entity linking"],
  },
  initialTasks: [
    {
      subject: "Set up PubMed data pipeline",
      description: "Configure access to PubMed API and download sample abstracts",
      type: "setup",
      priority: "high",
    },
    {
      subject: "Research biomedical language models",
      description: "Study PubMedBERT, BioGPT, and SciBERT architectures",
      type: "research",
      priority: "high",
    },
  ],
  minVram: 8,
  tags: ["biomedical", "NLP", "healthcare"],
};

/**
 * Firefly Network Lab Template
 * Community-focused collaborative research
 */
const fireflyNetworkTemplate: LabTemplate = {
  id: "firefly-network",
  name: "Firefly Network",
  description: "Distributed AI research and collaboration",
  longDescription: "Join the Firefly Network for distributed, collaborative AI research. Focus on federated learning, peer-to-peer model sharing, and collective intelligence.",
  category: "community",
  difficulty: "beginner",
  icon: "Sparkles",
  domain: {
    name: "Firefly Network",
    slug: "firefly-network",
    description: "Distributed AI research and collaboration",
    difficulty: "beginner",
    branding: {
      primaryColor: "#f59e0b",
      accentColor: "#fbbf24",
      backgroundStyle: "space",
    },
    scene: {
      props: [
        { id: "server1", type: "server", position: [-2, 0, 0], scale: 0.6 },
        { id: "server2", type: "server", position: [0, 0, -1], scale: 0.6 },
        { id: "server3", type: "server", position: [2, 0, 0], scale: 0.6 },
      ],
      decorations: { particles: true, floatingCubes: true },
    },
    research: {
      arxivCategories: ["cs.LG", "cs.DC", "cs.CR"],
      keywords: ["federated learning", "distributed training", "peer-to-peer", "collective intelligence"],
    },
    hardware: {
      minGpuVram: 4,
      gpuRequired: false,
    },
    tags: ["distributed", "collaborative", "network"],
  },
  researchGoal: {
    description: "Contribute to collaborative AI research through distributed computing",
    keywords: ["federated learning", "distributed", "collaboration"],
  },
  initialTasks: [
    {
      subject: "Connect to Firefly Network",
      description: "Set up node connection and verify network membership",
      type: "setup",
      priority: "high",
    },
    {
      subject: "Explore available research tasks",
      description: "Browse open tasks in the network and find ones matching your hardware",
      type: "research",
      priority: "medium",
    },
  ],
  minVram: 4,
  tags: ["distributed", "collaborative", "community"],
};

/**
 * Audio Generation Lab Template
 */
const audioGenerationTemplate: LabTemplate = {
  id: "audio-generation",
  name: "Audio Generation",
  description: "Music, sound effects, and audio synthesis",
  longDescription: "Research audio generation including music creation, sound effect synthesis, audio inpainting, and source separation. Covers diffusion models, GANs, and transformers for audio.",
  category: "creative",
  difficulty: "intermediate",
  icon: "Music",
  domain: {
    name: "Audio Generation",
    slug: "audio-generation",
    description: "Music, sound effects, and audio synthesis",
    difficulty: "intermediate",
    branding: {
      primaryColor: "#a855f7",
      accentColor: "#ec4899",
      backgroundStyle: "gradient",
      gradientColors: ["#2e1065", "#581c87"],
    },
    scene: {
      props: [
        { id: "wave", type: "waveform", position: [0, 1, 0], scale: 2 },
        { id: "speaker1", type: "speaker", position: [-2, 0, 0], scale: 0.7 },
        { id: "speaker2", type: "speaker", position: [2, 0, 0], scale: 0.7 },
      ],
      decorations: { particles: true },
    },
    research: {
      arxivCategories: ["cs.SD", "eess.AS", "cs.LG"],
      keywords: ["music generation", "audio synthesis", "diffusion", "source separation"],
      additionalSources: ["papers-with-code", "github"],
    },
    hardware: {
      minGpuVram: 12,
      recommendedGpuVram: 24,
      gpuRequired: true,
    },
    tags: ["audio", "music", "generation", "synthesis"],
  },
  researchGoal: {
    description: "Generate music and audio using diffusion models",
    keywords: ["MusicGen", "AudioLDM", "diffusion", "music generation"],
  },
  initialTasks: [
    {
      subject: "Set up audio processing pipeline",
      description: "Install librosa, torchaudio, and audiocraft libraries",
      type: "setup",
      priority: "high",
    },
    {
      subject: "Research audio diffusion models",
      description: "Study AudioLDM, Riffusion, and MusicGen architectures",
      type: "research",
      priority: "high",
    },
  ],
  minVram: 12,
  tags: ["audio", "music", "creative"],
};

// ============================================================================
// EXPORTED DATA
// ============================================================================

/**
 * All available lab templates
 */
export const LAB_TEMPLATES: LabTemplate[] = [
  voiceCloneTemplate,
  quantTradingTemplate,
  roboticsTemplate,
  drugDiscoveryTemplate,
  climateTemplate,
  computerVisionTemplate,
  nlpTemplate,
  gameAITemplate,
  biotechNLPTemplate,
  fireflyNetworkTemplate,
  audioGenerationTemplate,
];

/**
 * Featured templates (shown prominently in UI)
 */
export const FEATURED_TEMPLATES = LAB_TEMPLATES.filter((t) => t.featured);

/**
 * Get templates by category
 */
export function getTemplatesByCategory(category: TemplateCategory): LabTemplate[] {
  return LAB_TEMPLATES.filter((t) => t.category === category);
}

/**
 * Get template by ID
 */
export function getTemplateById(id: string): LabTemplate | undefined {
  return LAB_TEMPLATES.find((t) => t.id === id);
}

/**
 * Search templates by query
 */
export function searchTemplates(query: string): LabTemplate[] {
  const q = query.toLowerCase();
  return LAB_TEMPLATES.filter(
    (t) =>
      t.name.toLowerCase().includes(q) ||
      t.description.toLowerCase().includes(q) ||
      t.tags?.some((tag) => tag.toLowerCase().includes(q))
  );
}

/**
 * Filter templates by difficulty
 */
export function getTemplatesByDifficulty(difficulty: TemplateDifficulty): LabTemplate[] {
  return LAB_TEMPLATES.filter((t) => t.difficulty === difficulty);
}

/**
 * Filter templates by minimum VRAM
 */
export function getTemplatesForVram(availableVram: number): LabTemplate[] {
  return LAB_TEMPLATES.filter((t) => !t.minVram || t.minVram <= availableVram);
}
