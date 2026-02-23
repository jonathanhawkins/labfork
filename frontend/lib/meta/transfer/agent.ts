/**
 * Cross-Domain Transfer Agent
 *
 * Analyzes techniques for transfer between domains, extracts abstract principles,
 * maps concepts, assesses feasibility, and generates implementation guides.
 */

import {
  AbstractPrinciple,
  PrincipleComponent,
  DomainMapping,
  ConceptMapping,
  ConceptMappingType,
  MappingQuality,
  DomainAnalogy,
  MappingChallenge,
  TransferProposal,
  TransferFeasibility,
  FeasibilityLevel,
  FeasibilityComponents,
  RiskFactor,
  RequiredAdaptation,
  EffortEstimate,
  ImplementationGuide,
  ImplementationStep,
  Prerequisite,
  CodeTemplate,
  TestingStrategy,
  Pitfall,
  SuccessPrediction,
  PredictionBasis,
  ComparableTransfer,
  ExpectedOutcome,
  ResearchDomain,
  DomainConcept,
  DomainCharacteristics,
  DataModality,
  createTransferProposalId,
  createDomainMappingId,
  createPrincipleId,
} from "./types";
import { KnowledgeGraph, TechniqueNode, isTechniqueNode } from "../knowledge-graph";

// ============================================================================
// Agent Configuration
// ============================================================================

export interface TransferAgentConfig {
  /** Minimum similarity for concept mapping */
  minConceptSimilarity: number;
  /** Minimum feasibility for proposal */
  minFeasibilityScore: number;
  /** Maximum risks to include */
  maxRisks: number;
  /** Include code templates */
  includeCodeTemplates: boolean;
  /** Prediction confidence threshold */
  predictionConfidenceThreshold: number;
}

export const DEFAULT_TRANSFER_AGENT_CONFIG: TransferAgentConfig = {
  minConceptSimilarity: 0.3,
  minFeasibilityScore: 0.3,
  maxRisks: 10,
  includeCodeTemplates: true,
  predictionConfidenceThreshold: 0.6,
};

// ============================================================================
// Standard Domains
// ============================================================================

export const STANDARD_DOMAINS: ResearchDomain[] = [
  {
    id: "nlp",
    name: "Natural Language Processing",
    description: "Processing and understanding human language",
    concepts: [
      { id: "tokenization", name: "Tokenization", description: "Breaking text into tokens", abstractionLevel: 2, examples: ["BPE", "WordPiece"], relatedConceptIds: [] },
      { id: "attention", name: "Attention", description: "Focusing on relevant parts", abstractionLevel: 3, examples: ["Self-attention", "Cross-attention"], relatedConceptIds: [] },
      { id: "embedding", name: "Embedding", description: "Dense vector representations", abstractionLevel: 3, examples: ["Word2Vec", "BERT embeddings"], relatedConceptIds: [] },
      { id: "sequence_modeling", name: "Sequence Modeling", description: "Processing sequential data", abstractionLevel: 4, examples: ["RNN", "Transformer"], relatedConceptIds: [] },
    ],
    techniqueIds: [],
    characteristics: {
      dataModality: "text",
      taskTypes: ["generation", "classification", "translation"],
      scaleRequirements: { datasetSize: "large", modelSize: "large", trainingDuration: "days" },
      evaluationMetrics: ["BLEU", "ROUGE", "Perplexity"],
      hardwareRequirements: { gpuMemory: 16, multiGpu: true, tpuBeneficial: true },
    },
    relatedDomainIds: ["speech", "vision"],
  },
  {
    id: "speech",
    name: "Speech Processing",
    description: "Processing and synthesizing speech audio",
    concepts: [
      { id: "mel_spectrogram", name: "Mel Spectrogram", description: "Audio frequency representation", abstractionLevel: 2, examples: ["80-band mel"], relatedConceptIds: [] },
      { id: "prosody", name: "Prosody", description: "Speech rhythm and intonation", abstractionLevel: 3, examples: ["Pitch", "Duration", "Energy"], relatedConceptIds: [] },
      { id: "vocoder", name: "Vocoder", description: "Audio synthesis from features", abstractionLevel: 2, examples: ["HiFi-GAN", "WaveGlow"], relatedConceptIds: [] },
      { id: "speaker_embedding", name: "Speaker Embedding", description: "Voice identity representation", abstractionLevel: 3, examples: ["x-vector", "d-vector"], relatedConceptIds: ["embedding"] },
    ],
    techniqueIds: [],
    characteristics: {
      dataModality: "audio",
      taskTypes: ["synthesis", "translation", "enhancement"],
      scaleRequirements: { datasetSize: "medium", modelSize: "medium", trainingDuration: "days" },
      evaluationMetrics: ["MOS", "WER", "MCD"],
      hardwareRequirements: { gpuMemory: 12, multiGpu: false, tpuBeneficial: false },
    },
    relatedDomainIds: ["nlp", "music"],
  },
  {
    id: "vision",
    name: "Computer Vision",
    description: "Processing and understanding visual data",
    concepts: [
      { id: "convolution", name: "Convolution", description: "Local feature extraction", abstractionLevel: 2, examples: ["3x3 conv", "Depthwise conv"], relatedConceptIds: [] },
      { id: "feature_map", name: "Feature Map", description: "Spatial feature representation", abstractionLevel: 2, examples: ["ResNet features"], relatedConceptIds: [] },
      { id: "patch_embedding", name: "Patch Embedding", description: "Image region encoding", abstractionLevel: 3, examples: ["ViT patches"], relatedConceptIds: ["embedding"] },
      { id: "spatial_attention", name: "Spatial Attention", description: "Attention over image regions", abstractionLevel: 3, examples: ["SENet", "CBAM"], relatedConceptIds: ["attention"] },
    ],
    techniqueIds: [],
    characteristics: {
      dataModality: "image",
      taskTypes: ["classification", "detection", "segmentation", "generation"],
      scaleRequirements: { datasetSize: "large", modelSize: "large", trainingDuration: "days" },
      evaluationMetrics: ["Accuracy", "mAP", "IoU", "FID"],
      hardwareRequirements: { gpuMemory: 24, multiGpu: true, tpuBeneficial: true },
    },
    relatedDomainIds: ["nlp", "video"],
  },
  {
    id: "music",
    name: "Music Generation",
    description: "Generating and processing musical audio",
    concepts: [
      { id: "midi", name: "MIDI Representation", description: "Symbolic music encoding", abstractionLevel: 2, examples: ["Piano roll"], relatedConceptIds: [] },
      { id: "harmony", name: "Harmony", description: "Chord progressions and tonality", abstractionLevel: 3, examples: ["Chord embeddings"], relatedConceptIds: [] },
      { id: "rhythm", name: "Rhythm", description: "Temporal structure of music", abstractionLevel: 3, examples: ["Beat tracking"], relatedConceptIds: ["prosody"] },
      { id: "timbre", name: "Timbre", description: "Sound quality/color", abstractionLevel: 3, examples: ["Instrument embeddings"], relatedConceptIds: ["speaker_embedding"] },
    ],
    techniqueIds: [],
    characteristics: {
      dataModality: "audio",
      taskTypes: ["generation", "synthesis", "translation"],
      scaleRequirements: { datasetSize: "medium", modelSize: "medium", trainingDuration: "days" },
      evaluationMetrics: ["FAD", "Human preference"],
      hardwareRequirements: { gpuMemory: 16, multiGpu: false, tpuBeneficial: false },
    },
    relatedDomainIds: ["speech", "audio"],
  },
  {
    id: "reinforcement_learning",
    name: "Reinforcement Learning",
    description: "Learning from environment interactions",
    concepts: [
      { id: "reward", name: "Reward Signal", description: "Feedback for actions", abstractionLevel: 2, examples: ["Sparse reward", "Dense reward"], relatedConceptIds: [] },
      { id: "policy", name: "Policy", description: "Action selection strategy", abstractionLevel: 3, examples: ["PPO", "SAC"], relatedConceptIds: [] },
      { id: "value_function", name: "Value Function", description: "State/action value estimation", abstractionLevel: 3, examples: ["Q-learning", "V-function"], relatedConceptIds: [] },
      { id: "exploration", name: "Exploration", description: "Discovering new states/actions", abstractionLevel: 4, examples: ["Epsilon-greedy", "UCB"], relatedConceptIds: [] },
    ],
    techniqueIds: [],
    characteristics: {
      dataModality: "structured",
      taskTypes: ["reasoning", "generation"],
      scaleRequirements: { datasetSize: "small", modelSize: "medium", trainingDuration: "hours" },
      evaluationMetrics: ["Return", "Success rate"],
      hardwareRequirements: { gpuMemory: 8, multiGpu: false, tpuBeneficial: false },
    },
    relatedDomainIds: ["nlp", "robotics"],
  },
];

// ============================================================================
// Cross-Domain Transfer Agent
// ============================================================================

export class CrossDomainTransferAgent {
  private config: TransferAgentConfig;
  private graph: KnowledgeGraph;
  private domains: Map<string, ResearchDomain>;
  private proposals: Map<string, TransferProposal>;
  private mappings: Map<string, DomainMapping>;
  private principles: Map<string, AbstractPrinciple>;

  constructor(
    graph: KnowledgeGraph,
    config: Partial<TransferAgentConfig> = {}
  ) {
    this.config = { ...DEFAULT_TRANSFER_AGENT_CONFIG, ...config };
    this.graph = graph;
    this.domains = new Map();
    this.proposals = new Map();
    this.mappings = new Map();
    this.principles = new Map();

    // Initialize standard domains
    for (const domain of STANDARD_DOMAINS) {
      this.domains.set(domain.id, domain);
    }
  }

  // ============================================================================
  // Abstract Principle Extraction
  // ============================================================================

  /**
   * Extract abstract principles from a technique
   */
  extractAbstractPrinciples(techniqueId: string): AbstractPrinciple[] {
    const node = this.graph.getNode(techniqueId);
    if (!node || !isTechniqueNode(node)) {
      return [];
    }

    const technique = node as TechniqueNode;
    const principles: AbstractPrinciple[] = [];

    // Extract architecture principle
    if (technique.architecture) {
      const archPrinciple = this.extractArchitecturePrinciple(technique);
      if (archPrinciple) {
        principles.push(archPrinciple);
        this.principles.set(archPrinciple.id, archPrinciple);
      }
    }

    // Extract training principle
    const trainingPrinciple = this.extractTrainingPrinciple(technique);
    if (trainingPrinciple) {
      principles.push(trainingPrinciple);
      this.principles.set(trainingPrinciple.id, trainingPrinciple);
    }

    // Extract conditioning principle
    const conditioningPrinciple = this.extractConditioningPrinciple(technique);
    if (conditioningPrinciple) {
      principles.push(conditioningPrinciple);
      this.principles.set(conditioningPrinciple.id, conditioningPrinciple);
    }

    return principles;
  }

  private extractArchitecturePrinciple(technique: TechniqueNode): AbstractPrinciple | null {
    const arch = technique.architecture?.toLowerCase() || "";
    const tags = technique.tags.map((t) => t.toLowerCase());

    const components: PrincipleComponent[] = [];
    let coreInsight = "";

    if (arch.includes("transformer") || tags.includes("transformer")) {
      coreInsight = "Self-attention enables capturing long-range dependencies without sequential processing";
      components.push({
        name: "Self-Attention",
        role: "Compute attention weights between all positions",
        required: true,
        alternatives: ["Cross-attention", "Linear attention"],
      });
      components.push({
        name: "Positional Encoding",
        role: "Inject position information",
        required: true,
        alternatives: ["Sinusoidal", "Learned", "Rotary"],
      });
    } else if (arch.includes("diffusion") || tags.includes("diffusion")) {
      coreInsight = "Iterative denoising from noise to data enables high-quality generation with stable training";
      components.push({
        name: "Noise Schedule",
        role: "Define noise levels across timesteps",
        required: true,
        alternatives: ["Linear", "Cosine", "Learned"],
      });
      components.push({
        name: "Denoising Network",
        role: "Predict noise or clean data",
        required: true,
        alternatives: ["U-Net", "Transformer", "DiT"],
      });
    } else if (arch.includes("vae") || tags.includes("vae")) {
      coreInsight = "Variational inference enables learning smooth latent spaces for generation";
      components.push({
        name: "Encoder",
        role: "Map input to latent distribution",
        required: true,
        alternatives: ["MLP", "CNN", "Transformer"],
      });
      components.push({
        name: "Decoder",
        role: "Reconstruct from latent",
        required: true,
        alternatives: ["MLP", "CNN", "Transformer"],
      });
    } else {
      return null;
    }

    return {
      id: createPrincipleId(),
      name: `${technique.name} Architecture Principle`,
      description: `Architectural principle from ${technique.name}`,
      level: 4,
      coreInsight,
      sourceTechniqueIds: [technique.id],
      sourceDomains: technique.tags.filter((t) =>
        STANDARD_DOMAINS.some((d) => d.name.toLowerCase().includes(t.toLowerCase()))
      ),
      components,
      applicabilityConditions: [
        "Sufficient computational resources",
        "Data format compatible with architecture",
      ],
      counterIndications: [
        "Real-time inference with strict latency requirements",
        "Extremely limited compute budget",
      ],
      confidence: 0.8,
    };
  }

  private extractTrainingPrinciple(technique: TechniqueNode): AbstractPrinciple | null {
    const tags = technique.tags.map((t) => t.toLowerCase());
    const components: PrincipleComponent[] = [];
    let coreInsight = "";
    let name = "";

    if (tags.includes("self-supervised") || tags.includes("contrastive")) {
      name = "Self-Supervised Learning";
      coreInsight = "Learn representations from unlabeled data through pretext tasks or contrastive objectives";
      components.push({
        name: "Pretext Task",
        role: "Define self-supervision signal",
        required: true,
        alternatives: ["Masking", "Contrastive", "Predictive"],
      });
    } else if (tags.includes("reinforcement") || tags.includes("rl")) {
      name = "Reinforcement Learning";
      coreInsight = "Optimize behavior through reward signals and environment interaction";
      components.push({
        name: "Reward Function",
        role: "Define optimization objective",
        required: true,
        alternatives: ["Sparse", "Dense", "Shaped"],
      });
    } else if (tags.includes("adversarial") || tags.includes("gan")) {
      name = "Adversarial Training";
      coreInsight = "Train generator and discriminator in competition for realistic outputs";
      components.push({
        name: "Discriminator",
        role: "Distinguish real from generated",
        required: true,
        alternatives: ["PatchGAN", "StyleGAN discriminator"],
      });
    } else {
      return null;
    }

    return {
      id: createPrincipleId(),
      name,
      description: `Training principle from ${technique.name}`,
      level: 4,
      coreInsight,
      sourceTechniqueIds: [technique.id],
      sourceDomains: [],
      components,
      applicabilityConditions: ["Appropriate data available", "Compute budget sufficient"],
      counterIndications: ["Labeled data abundant", "Simple task"],
      confidence: 0.75,
    };
  }

  private extractConditioningPrinciple(technique: TechniqueNode): AbstractPrinciple | null {
    const tags = technique.tags.map((t) => t.toLowerCase());

    if (!tags.some((t) => t.includes("condition") || t.includes("control"))) {
      return null;
    }

    return {
      id: createPrincipleId(),
      name: "Conditional Generation",
      description: `Conditioning principle from ${technique.name}`,
      level: 3,
      coreInsight: "Guide generation with external signals to control output characteristics",
      sourceTechniqueIds: [technique.id],
      sourceDomains: [],
      components: [
        {
          name: "Condition Encoder",
          role: "Encode conditioning signal",
          required: true,
          alternatives: ["Embedding", "Encoder network", "Cross-attention"],
        },
        {
          name: "Injection Method",
          role: "Inject condition into generator",
          required: true,
          alternatives: ["Concatenation", "Cross-attention", "AdaIN", "FiLM"],
        },
      ],
      applicabilityConditions: ["Conditioning signal available", "Clear relationship to output"],
      counterIndications: ["Unconditional generation needed"],
      confidence: 0.7,
    };
  }

  // ============================================================================
  // Domain Mapping
  // ============================================================================

  /**
   * Find analogies between two domains
   */
  findDomainAnalogies(
    sourceDomainId: string,
    targetDomainId: string
  ): DomainMapping {
    const existingKey = `${sourceDomainId}-${targetDomainId}`;
    const existing = this.mappings.get(existingKey);
    if (existing) return existing;

    const sourceDomain = this.domains.get(sourceDomainId);
    const targetDomain = this.domains.get(targetDomainId);

    if (!sourceDomain || !targetDomain) {
      throw new Error(`Domain not found: ${sourceDomainId} or ${targetDomainId}`);
    }

    // Map concepts
    const conceptMappings = this.mapConcepts(sourceDomain, targetDomain);

    // Find analogies
    const analogies = this.findAnalogies(sourceDomain, targetDomain);

    // Identify challenges
    const challenges = this.identifyChallenges(sourceDomain, targetDomain);

    // Calculate similarities
    const structuralSimilarity = this.calculateStructuralSimilarity(
      sourceDomain,
      targetDomain
    );
    const functionalSimilarity = this.calculateFunctionalSimilarity(
      sourceDomain,
      targetDomain
    );
    const dataCompatibility = this.calculateDataCompatibility(
      sourceDomain.characteristics,
      targetDomain.characteristics
    );

    const mappingStrength =
      (structuralSimilarity + functionalSimilarity + dataCompatibility) / 3;

    const quality = this.assessMappingQuality(mappingStrength);

    const mapping: DomainMapping = {
      id: createDomainMappingId(),
      sourceDomainId,
      targetDomainId,
      conceptMappings,
      structuralSimilarity,
      functionalSimilarity,
      dataCompatibility,
      mappingStrength,
      quality,
      analogies,
      challenges,
      createdAt: new Date(),
    };

    this.mappings.set(existingKey, mapping);
    return mapping;
  }

  /**
   * Map concepts between domains
   */
  mapConcepts(
    sourceDomain: ResearchDomain,
    targetDomain: ResearchDomain
  ): ConceptMapping[] {
    const mappings: ConceptMapping[] = [];

    for (const sourceConcept of sourceDomain.concepts) {
      let bestMatch: ConceptMapping | null = null;
      let bestSimilarity = 0;

      for (const targetConcept of targetDomain.concepts) {
        const similarity = this.calculateConceptSimilarity(
          sourceConcept,
          targetConcept
        );

        if (similarity > bestSimilarity && similarity >= this.config.minConceptSimilarity) {
          bestSimilarity = similarity;
          bestMatch = {
            sourceConceptId: sourceConcept.id,
            sourceConceptName: sourceConcept.name,
            targetConceptId: targetConcept.id,
            targetConceptName: targetConcept.name,
            mappingType: this.determineMappingType(sourceConcept, targetConcept, similarity),
            similarity,
            justification: this.generateMappingJustification(sourceConcept, targetConcept),
            transformation: this.determineTransformation(sourceConcept, targetConcept),
            confidence: similarity,
          };
        }
      }

      if (bestMatch) {
        mappings.push(bestMatch);
      }
    }

    return mappings;
  }

  private calculateConceptSimilarity(
    source: DomainConcept,
    target: DomainConcept
  ): number {
    let similarity = 0;

    // Name similarity
    const sourceName = source.name.toLowerCase();
    const targetName = target.name.toLowerCase();
    if (sourceName === targetName) {
      similarity += 0.4;
    } else if (sourceName.includes(targetName) || targetName.includes(sourceName)) {
      similarity += 0.25;
    }

    // Abstraction level similarity
    const levelDiff = Math.abs(source.abstractionLevel - target.abstractionLevel);
    similarity += 0.3 * (1 - levelDiff / 4);

    // Related concepts
    if (source.relatedConceptIds.some((id) => target.relatedConceptIds.includes(id))) {
      similarity += 0.2;
    }

    // Description overlap (simple word overlap)
    const sourceWords = new Set(source.description.toLowerCase().split(/\s+/));
    const targetWords = new Set(target.description.toLowerCase().split(/\s+/));
    const overlap = Array.from(sourceWords).filter((w) => targetWords.has(w)).length;
    const totalWords = Math.max(sourceWords.size, targetWords.size);
    similarity += 0.1 * (overlap / totalWords);

    return Math.min(1, similarity);
  }

  private determineMappingType(
    source: DomainConcept,
    target: DomainConcept,
    similarity: number
  ): ConceptMappingType {
    if (similarity > 0.9) return "equivalent";
    if (source.abstractionLevel > target.abstractionLevel) return "generalization";
    if (source.abstractionLevel < target.abstractionLevel) return "specialization";
    if (similarity > 0.6) return "analogous";
    return "partial";
  }

  private generateMappingJustification(
    source: DomainConcept,
    target: DomainConcept
  ): string {
    return `${source.name} in the source domain serves a similar role to ${target.name} in the target domain. Both operate at abstraction level ${source.abstractionLevel}/${target.abstractionLevel} and involve ${source.description.split(" ").slice(0, 5).join(" ")}...`;
  }

  private determineTransformation(
    source: DomainConcept,
    target: DomainConcept
  ): string | undefined {
    if (source.abstractionLevel !== target.abstractionLevel) {
      return `Adjust abstraction level from ${source.abstractionLevel} to ${target.abstractionLevel}`;
    }
    return undefined;
  }

  private findAnalogies(
    sourceDomain: ResearchDomain,
    targetDomain: ResearchDomain
  ): DomainAnalogy[] {
    const analogies: DomainAnalogy[] = [];

    // Data representation analogy
    if (sourceDomain.characteristics.dataModality !== targetDomain.characteristics.dataModality) {
      analogies.push({
        id: `analogy-${Date.now()}-1`,
        sourcePattern: `${sourceDomain.characteristics.dataModality} representation`,
        targetPattern: `${targetDomain.characteristics.dataModality} representation`,
        description: `Data representation analogy between ${sourceDomain.name} and ${targetDomain.name}`,
        strength: 0.6,
        examples: [
          {
            source: `${sourceDomain.name} uses ${sourceDomain.characteristics.dataModality} data`,
            target: `${targetDomain.name} uses ${targetDomain.characteristics.dataModality} data`,
            explanation: "Different modalities may have analogous processing pipelines",
          },
        ],
      });
    }

    // Task type analogy
    const sharedTasks = sourceDomain.characteristics.taskTypes.filter((t) =>
      targetDomain.characteristics.taskTypes.includes(t)
    );
    if (sharedTasks.length > 0) {
      analogies.push({
        id: `analogy-${Date.now()}-2`,
        sourcePattern: `${sharedTasks[0]} in ${sourceDomain.name}`,
        targetPattern: `${sharedTasks[0]} in ${targetDomain.name}`,
        description: `Shared ${sharedTasks[0]} task pattern`,
        strength: 0.8,
        examples: [
          {
            source: `${sourceDomain.name} ${sharedTasks[0]}`,
            target: `${targetDomain.name} ${sharedTasks[0]}`,
            explanation: "Similar task types suggest transferable solutions",
          },
        ],
      });
    }

    return analogies;
  }

  private identifyChallenges(
    sourceDomain: ResearchDomain,
    targetDomain: ResearchDomain
  ): MappingChallenge[] {
    const challenges: MappingChallenge[] = [];

    // Data format challenge
    if (sourceDomain.characteristics.dataModality !== targetDomain.characteristics.dataModality) {
      challenges.push({
        type: "data_format",
        description: `Converting from ${sourceDomain.characteristics.dataModality} to ${targetDomain.characteristics.dataModality}`,
        severity: 3,
        mitigations: [
          "Use modality-agnostic intermediate representations",
          "Design custom preprocessing pipeline",
        ],
      });
    }

    // Scale challenge
    const sourceScale = sourceDomain.characteristics.scaleRequirements;
    const targetScale = targetDomain.characteristics.scaleRequirements;
    if (sourceScale.modelSize !== targetScale.modelSize) {
      challenges.push({
        type: "scale_mismatch",
        description: `Model size differs: ${sourceScale.modelSize} vs ${targetScale.modelSize}`,
        severity: 2,
        mitigations: ["Scale architecture appropriately", "Use knowledge distillation"],
      });
    }

    // Evaluation challenge
    const sourceMetrics = sourceDomain.characteristics.evaluationMetrics;
    const targetMetrics = targetDomain.characteristics.evaluationMetrics;
    if (!sourceMetrics.some((m) => targetMetrics.includes(m))) {
      challenges.push({
        type: "evaluation_mismatch",
        description: "No shared evaluation metrics between domains",
        severity: 2,
        mitigations: ["Define custom metrics", "Use proxy metrics"],
      });
    }

    return challenges;
  }

  private calculateStructuralSimilarity(
    source: ResearchDomain,
    target: ResearchDomain
  ): number {
    // Compare concept hierarchies
    const sourceConceptCount = source.concepts.length;
    const targetConceptCount = target.concepts.length;
    const sizeSimilarity = 1 - Math.abs(sourceConceptCount - targetConceptCount) / Math.max(sourceConceptCount, targetConceptCount);

    // Compare abstraction levels
    const sourceAvgLevel = source.concepts.reduce((a, c) => a + c.abstractionLevel, 0) / sourceConceptCount;
    const targetAvgLevel = target.concepts.reduce((a, c) => a + c.abstractionLevel, 0) / targetConceptCount;
    const levelSimilarity = 1 - Math.abs(sourceAvgLevel - targetAvgLevel) / 5;

    return (sizeSimilarity + levelSimilarity) / 2;
  }

  private calculateFunctionalSimilarity(
    source: ResearchDomain,
    target: ResearchDomain
  ): number {
    const sharedTasks = source.characteristics.taskTypes.filter((t) =>
      target.characteristics.taskTypes.includes(t)
    );
    const totalTasks = new Set([
      ...source.characteristics.taskTypes,
      ...target.characteristics.taskTypes,
    ]).size;

    return sharedTasks.length / totalTasks;
  }

  private calculateDataCompatibility(
    source: DomainCharacteristics,
    target: DomainCharacteristics
  ): number {
    // Same modality = high compatibility
    if (source.dataModality === target.dataModality) return 0.9;

    // Related modalities
    const modalityPairs: Record<string, string[]> = {
      text: ["audio", "multimodal"],
      audio: ["text", "music", "multimodal"],
      image: ["video", "multimodal"],
      video: ["image", "multimodal"],
      multimodal: ["text", "audio", "image", "video"],
    };

    if (modalityPairs[source.dataModality]?.includes(target.dataModality)) {
      return 0.6;
    }

    return 0.3;
  }

  private assessMappingQuality(strength: number): MappingQuality {
    if (strength >= 0.8) return "excellent";
    if (strength >= 0.6) return "good";
    if (strength >= 0.4) return "moderate";
    if (strength >= 0.2) return "poor";
    return "none";
  }

  // ============================================================================
  // Feasibility Assessment
  // ============================================================================

  /**
   * Assess feasibility of a transfer
   */
  assessFeasibility(
    techniqueId: string,
    targetDomainId: string
  ): TransferFeasibility {
    const node = this.graph.getNode(techniqueId);
    if (!node || !isTechniqueNode(node)) {
      throw new Error(`Technique not found: ${techniqueId}`);
    }

    const technique = node as TechniqueNode;
    const targetDomain = this.domains.get(targetDomainId);
    if (!targetDomain) {
      throw new Error(`Domain not found: ${targetDomainId}`);
    }

    // Identify source domain
    const sourceDomainId = this.inferSourceDomain(technique);
    const sourceDomain = this.domains.get(sourceDomainId);

    // Get domain mapping
    const mapping = sourceDomain
      ? this.findDomainAnalogies(sourceDomainId, targetDomainId)
      : null;

    // Calculate component scores
    const components: FeasibilityComponents = {
      technical: this.assessTechnicalFeasibility(technique, targetDomain),
      data: this.assessDataFeasibility(technique, targetDomain),
      computational: this.assessComputationalFeasibility(technique, targetDomain),
      knowledge: mapping ? mapping.mappingStrength : 0.5,
      resources: this.assessResourceFeasibility(technique, targetDomain),
    };

    const overallScore =
      components.technical * 0.25 +
      components.data * 0.2 +
      components.computational * 0.2 +
      components.knowledge * 0.2 +
      components.resources * 0.15;

    const level = this.determineFeasibilityLevel(overallScore);

    // Identify risks
    const risks = this.identifyRisks(technique, targetDomain, mapping);

    // Identify required adaptations
    const adaptations = this.identifyAdaptations(technique, targetDomain, mapping);

    // Estimate effort
    const effort = this.estimateEffort(adaptations, level);

    return {
      overallScore,
      level,
      components,
      risks: risks.slice(0, this.config.maxRisks),
      enablers: this.identifyEnablers(technique, targetDomain),
      adaptations,
      effort,
      recommendations: this.generateRecommendations(level, risks, adaptations),
    };
  }

  private inferSourceDomain(technique: TechniqueNode): string {
    const tags = technique.tags.map((t) => t.toLowerCase());

    for (const domain of STANDARD_DOMAINS) {
      if (tags.includes(domain.id) || tags.includes(domain.name.toLowerCase())) {
        return domain.id;
      }
    }

    // Infer from architecture
    if (technique.architecture) {
      const arch = technique.architecture.toLowerCase();
      if (arch.includes("tts") || arch.includes("speech") || arch.includes("audio")) {
        return "speech";
      }
      if (arch.includes("nlp") || arch.includes("language") || arch.includes("text")) {
        return "nlp";
      }
      if (arch.includes("vision") || arch.includes("image") || arch.includes("cnn")) {
        return "vision";
      }
    }

    return "nlp"; // Default
  }

  private assessTechnicalFeasibility(
    technique: TechniqueNode,
    targetDomain: ResearchDomain
  ): number {
    let score = 0.5;

    // Architecture compatibility
    const arch = technique.architecture?.toLowerCase() || "";
    if (arch.includes("transformer")) {
      score += 0.2; // Transformers are broadly applicable
    }

    // Complexity consideration
    if (technique.complexity === "simple") score += 0.15;
    if (technique.complexity === "complex") score -= 0.1;

    return Math.max(0, Math.min(1, score));
  }

  private assessDataFeasibility(
    technique: TechniqueNode,
    targetDomain: ResearchDomain
  ): number {
    const sourceDomainId = this.inferSourceDomain(technique);
    const sourceDomain = this.domains.get(sourceDomainId);

    if (!sourceDomain) return 0.5;

    return this.calculateDataCompatibility(
      sourceDomain.characteristics,
      targetDomain.characteristics
    );
  }

  private assessComputationalFeasibility(
    technique: TechniqueNode,
    targetDomain: ResearchDomain
  ): number {
    let score = 0.7;

    // Complexity affects computational needs
    if (technique.complexity === "simple") score += 0.2;
    if (technique.complexity === "complex") score -= 0.2;

    // Target domain requirements
    if (targetDomain.characteristics.hardwareRequirements.gpuMemory > 16) {
      score -= 0.1;
    }
    if (targetDomain.characteristics.hardwareRequirements.multiGpu) {
      score -= 0.15;
    }

    return Math.max(0, Math.min(1, score));
  }

  private assessResourceFeasibility(
    technique: TechniqueNode,
    targetDomain: ResearchDomain
  ): number {
    let score = 0.6;

    // Simple techniques need fewer resources
    if (technique.complexity === "simple") score += 0.2;
    if (technique.complexity === "moderate") score += 0.1;

    // Small datasets are easier
    if (targetDomain.characteristics.scaleRequirements.datasetSize === "small") {
      score += 0.15;
    }

    return Math.max(0, Math.min(1, score));
  }

  private determineFeasibilityLevel(score: number): FeasibilityLevel {
    if (score >= 0.8) return "trivial";
    if (score >= 0.65) return "straightforward";
    if (score >= 0.45) return "moderate";
    if (score >= 0.25) return "challenging";
    return "infeasible";
  }

  private identifyRisks(
    technique: TechniqueNode,
    targetDomain: ResearchDomain,
    mapping: DomainMapping | null
  ): RiskFactor[] {
    const risks: RiskFactor[] = [];

    // Domain gap risk
    if (mapping && mapping.mappingStrength < 0.5) {
      risks.push({
        name: "Domain Gap",
        description: "Significant conceptual differences between domains",
        probability: 0.7,
        impact: 4,
        score: 2.8,
        mitigation: "Use domain adaptation techniques",
      });
    }

    // Complexity risk
    if (technique.complexity === "complex") {
      risks.push({
        name: "Implementation Complexity",
        description: "Complex techniques are harder to adapt correctly",
        probability: 0.6,
        impact: 3,
        score: 1.8,
        mitigation: "Start with simplified version, iterate",
      });
    }

    // Data format risk
    if (mapping && mapping.challenges.some((c) => c.type === "data_format")) {
      risks.push({
        name: "Data Format Incompatibility",
        description: "Data formats differ significantly between domains",
        probability: 0.8,
        impact: 3,
        score: 2.4,
        mitigation: "Design robust preprocessing pipeline",
      });
    }

    // Sort by score
    risks.sort((a, b) => b.score - a.score);
    return risks;
  }

  private identifyAdaptations(
    technique: TechniqueNode,
    targetDomain: ResearchDomain,
    mapping: DomainMapping | null
  ): RequiredAdaptation[] {
    const adaptations: RequiredAdaptation[] = [];

    // Data preprocessing adaptation
    adaptations.push({
      type: "data_preprocessing",
      description: `Adapt input pipeline for ${targetDomain.characteristics.dataModality} data`,
      effort: "medium",
      priority: 1,
      dependencies: [],
    });

    // Architecture modification
    if (technique.architecture) {
      adaptations.push({
        type: "architecture_modification",
        description: `Modify ${technique.architecture} for ${targetDomain.name} requirements`,
        effort: technique.complexity === "complex" ? "high" : "medium",
        priority: 2,
        dependencies: ["data_preprocessing"],
      });
    }

    // Evaluation adaptation
    adaptations.push({
      type: "evaluation_metrics",
      description: `Define evaluation using ${targetDomain.characteristics.evaluationMetrics.join(", ")}`,
      effort: "low",
      priority: 3,
      dependencies: [],
    });

    // Hyperparameter tuning
    adaptations.push({
      type: "hyperparameter_tuning",
      description: "Tune hyperparameters for target domain",
      effort: "medium",
      priority: 4,
      dependencies: ["architecture_modification"],
    });

    return adaptations;
  }

  private estimateEffort(
    adaptations: RequiredAdaptation[],
    level: FeasibilityLevel
  ): EffortEstimate {
    const effortMap: Record<string, number> = {
      low: 1,
      medium: 3,
      high: 7,
    };

    let totalDays = 0;
    const breakdown: { activity: string; days: number; notes?: string }[] = [];

    for (const adaptation of adaptations) {
      const days = effortMap[adaptation.effort];
      totalDays += days;
      breakdown.push({
        activity: adaptation.description,
        days,
      });
    }

    // Adjust by feasibility level
    const multiplier: Record<FeasibilityLevel, number> = {
      trivial: 0.5,
      straightforward: 0.8,
      moderate: 1.0,
      challenging: 1.5,
      infeasible: 2.0,
    };

    totalDays *= multiplier[level];

    return {
      personDays: Math.round(totalDays),
      range: {
        min: Math.round(totalDays * 0.7),
        max: Math.round(totalDays * 1.5),
      },
      breakdown,
    };
  }

  private identifyEnablers(
    technique: TechniqueNode,
    targetDomain: ResearchDomain
  ): string[] {
    const enablers: string[] = [];

    // Architecture enablers
    if (technique.architecture?.toLowerCase().includes("transformer")) {
      enablers.push("Transformer architecture is broadly applicable across domains");
    }

    // Training enablers
    if (technique.tags.some((t) => t.toLowerCase().includes("self-supervised"))) {
      enablers.push("Self-supervised learning reduces labeled data requirements");
    }

    // Domain enablers
    if (targetDomain.relatedDomainIds.includes(this.inferSourceDomain(technique))) {
      enablers.push("Source and target domains are related");
    }

    return enablers;
  }

  private generateRecommendations(
    level: FeasibilityLevel,
    risks: RiskFactor[],
    adaptations: RequiredAdaptation[]
  ): string[] {
    const recommendations: string[] = [];

    if (level === "infeasible") {
      recommendations.push("Consider alternative techniques better suited for target domain");
      return recommendations;
    }

    if (level === "challenging") {
      recommendations.push("Start with a proof-of-concept on simplified version");
      recommendations.push("Plan for multiple iteration cycles");
    }

    // Address top risks
    for (const risk of risks.slice(0, 2)) {
      recommendations.push(`Address ${risk.name}: ${risk.mitigation}`);
    }

    // Priority adaptations
    const highPriority = adaptations.filter((a) => a.priority <= 2);
    if (highPriority.length > 0) {
      recommendations.push(
        `Focus first on: ${highPriority.map((a) => a.type).join(", ")}`
      );
    }

    return recommendations;
  }

  // ============================================================================
  // Implementation Guide Generation
  // ============================================================================

  /**
   * Generate implementation guide for a transfer
   */
  generateImplementationGuide(
    techniqueId: string,
    targetDomainId: string,
    feasibility: TransferFeasibility
  ): ImplementationGuide {
    const node = this.graph.getNode(techniqueId);
    if (!node || !isTechniqueNode(node)) {
      throw new Error(`Technique not found: ${techniqueId}`);
    }

    const technique = node as TechniqueNode;
    const targetDomain = this.domains.get(targetDomainId);
    if (!targetDomain) {
      throw new Error(`Domain not found: ${targetDomainId}`);
    }

    const prerequisites = this.generatePrerequisites(technique, targetDomain);
    const steps = this.generateSteps(technique, targetDomain, feasibility);
    const codeTemplates = this.config.includeCodeTemplates
      ? this.generateCodeTemplates(technique, targetDomain)
      : [];
    const testingStrategy = this.generateTestingStrategy(technique, targetDomain);
    const pitfalls = this.generatePitfalls(technique, targetDomain, feasibility);

    return {
      overview: `Transfer ${technique.name} from ${this.inferSourceDomain(technique)} to ${targetDomain.name}. ${feasibility.level} difficulty level.`,
      prerequisites,
      steps,
      codeTemplates,
      testingStrategy,
      pitfalls,
      successCriteria: this.generateSuccessCriteria(technique, targetDomain),
    };
  }

  private generatePrerequisites(
    technique: TechniqueNode,
    targetDomain: ResearchDomain
  ): Prerequisite[] {
    const prerequisites: Prerequisite[] = [];

    prerequisites.push({
      name: "Understanding of source technique",
      description: `Familiarity with ${technique.name} and its core concepts`,
      type: "knowledge",
      required: true,
    });

    prerequisites.push({
      name: "Target domain expertise",
      description: `Knowledge of ${targetDomain.name} conventions and practices`,
      type: "knowledge",
      required: true,
    });

    prerequisites.push({
      name: "Target domain dataset",
      description: `${targetDomain.characteristics.dataModality} data for training and evaluation`,
      type: "data",
      required: true,
    });

    prerequisites.push({
      name: "GPU resources",
      description: `GPU with ${targetDomain.characteristics.hardwareRequirements.gpuMemory}GB+ memory`,
      type: "infrastructure",
      required: true,
    });

    return prerequisites;
  }

  private generateSteps(
    technique: TechniqueNode,
    targetDomain: ResearchDomain,
    feasibility: TransferFeasibility
  ): ImplementationStep[] {
    const steps: ImplementationStep[] = [];
    let stepNum = 1;

    // Step 1: Data preparation
    steps.push({
      step: stepNum++,
      title: "Prepare Target Domain Data",
      description: `Collect and preprocess ${targetDomain.characteristics.dataModality} data`,
      details: [
        "Gather representative dataset",
        "Define preprocessing pipeline",
        "Create train/val/test splits",
      ],
      estimatedHours: 8,
      dependencies: [],
      validation: "Data loads correctly and matches expected format",
    });

    // Step 2: Adapt architecture
    steps.push({
      step: stepNum++,
      title: "Adapt Architecture",
      description: `Modify ${technique.architecture || "model"} for ${targetDomain.name}`,
      details: [
        "Adjust input/output dimensions",
        "Modify domain-specific layers",
        "Update conditioning mechanisms",
      ],
      estimatedHours: 16,
      dependencies: [1],
      validation: "Model compiles and processes sample input",
    });

    // Step 3: Training setup
    steps.push({
      step: stepNum++,
      title: "Configure Training",
      description: "Set up training loop and objectives",
      details: [
        "Define loss functions",
        "Configure optimizer and scheduler",
        "Set hyperparameters",
      ],
      estimatedHours: 8,
      dependencies: [2],
      validation: "Training loop runs without errors",
    });

    // Step 4: Initial training
    steps.push({
      step: stepNum++,
      title: "Initial Training",
      description: "Train adapted model on target domain",
      details: [
        "Run training with monitoring",
        "Track key metrics",
        "Save checkpoints",
      ],
      estimatedHours: 24,
      dependencies: [3],
      validation: "Loss decreases, metrics improve",
    });

    // Step 5: Evaluation
    steps.push({
      step: stepNum++,
      title: "Evaluate and Iterate",
      description: `Evaluate using ${targetDomain.characteristics.evaluationMetrics.join(", ")}`,
      details: [
        "Run evaluation on test set",
        "Compare to baselines",
        "Analyze failure cases",
      ],
      estimatedHours: 8,
      dependencies: [4],
      validation: "Meets minimum performance thresholds",
    });

    return steps;
  }

  private generateCodeTemplates(
    technique: TechniqueNode,
    targetDomain: ResearchDomain
  ): CodeTemplate[] {
    return [
      {
        name: "Data Loader",
        language: "python",
        code: `class ${targetDomain.name.replace(/\s+/g, "")}Dataset(Dataset):
    def __init__(self, data_path: str):
        # TODO: Load ${targetDomain.characteristics.dataModality} data
        self.data = self._load_data(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO: Return preprocessed sample
        return self.data[idx]`,
        description: "Template for target domain data loading",
        placeholders: ["data_path", "preprocessing_logic"],
      },
      {
        name: "Model Adapter",
        language: "python",
        code: `class Adapted${technique.name.replace(/\s+/g, "")}(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO: Adapt architecture for ${targetDomain.name}
        self.encoder = self._build_encoder(config)
        self.decoder = self._build_decoder(config)

    def forward(self, x):
        # TODO: Implement forward pass
        return self.decoder(self.encoder(x))`,
        description: "Template for adapted model architecture",
        placeholders: ["config", "encoder_layers", "decoder_layers"],
      },
    ];
  }

  private generateTestingStrategy(
    technique: TechniqueNode,
    targetDomain: ResearchDomain
  ): TestingStrategy {
    return {
      unitTests: [
        "Test data loading and preprocessing",
        "Test model forward pass with sample input",
        "Test loss computation",
      ],
      integrationTests: [
        "Test full training loop for one batch",
        "Test checkpoint save/load",
        "Test inference pipeline",
      ],
      benchmarks: targetDomain.characteristics.evaluationMetrics.map(
        (m) => `Evaluate ${m} on test set`
      ),
      acceptanceCriteria: [
        "Model trains without errors",
        "Metrics improve over random baseline",
        "Inference time within acceptable range",
      ],
    };
  }

  private generatePitfalls(
    technique: TechniqueNode,
    targetDomain: ResearchDomain,
    feasibility: TransferFeasibility
  ): Pitfall[] {
    const pitfalls: Pitfall[] = [];

    pitfalls.push({
      name: "Inadequate Domain Adaptation",
      description: "Directly applying source technique without proper adaptation",
      avoidance: "Carefully analyze domain differences and adapt accordingly",
      symptoms: ["Poor performance", "Training instability"],
    });

    pitfalls.push({
      name: "Hyperparameter Transfer",
      description: "Using source domain hyperparameters without tuning",
      avoidance: "Perform hyperparameter search for target domain",
      symptoms: ["Suboptimal convergence", "Overfitting"],
    });

    if (feasibility.risks.length > 0) {
      for (const risk of feasibility.risks.slice(0, 2)) {
        pitfalls.push({
          name: risk.name,
          description: risk.description,
          avoidance: risk.mitigation,
          symptoms: ["Performance degradation"],
        });
      }
    }

    return pitfalls;
  }

  private generateSuccessCriteria(
    technique: TechniqueNode,
    targetDomain: ResearchDomain
  ): string[] {
    return [
      `Model successfully processes ${targetDomain.characteristics.dataModality} data`,
      `Achieves competitive ${targetDomain.characteristics.evaluationMetrics[0]} scores`,
      "Training converges stably",
      "Inference meets latency requirements",
      "Resource usage within budget",
    ];
  }

  // ============================================================================
  // Success Prediction
  // ============================================================================

  /**
   * Predict success probability for a transfer
   */
  predictSuccess(
    techniqueId: string,
    targetDomainId: string,
    feasibility: TransferFeasibility
  ): SuccessPrediction {
    const node = this.graph.getNode(techniqueId);
    const technique = node && isTechniqueNode(node) ? (node as TechniqueNode) : null;
    const targetDomain = this.domains.get(targetDomainId);

    const basis: PredictionBasis[] = [];

    // Feasibility-based prediction
    const feasibilityContribution = feasibility.overallScore * 0.4;
    basis.push({
      factor: "Feasibility Assessment",
      contribution: feasibilityContribution,
      evidence: `Overall feasibility score: ${feasibility.overallScore.toFixed(2)}`,
    });

    // Risk-based adjustment
    const avgRiskScore =
      feasibility.risks.length > 0
        ? feasibility.risks.reduce((a, r) => a + r.score, 0) / feasibility.risks.length
        : 0;
    const riskAdjustment = -avgRiskScore * 0.1;
    basis.push({
      factor: "Risk Assessment",
      contribution: riskAdjustment,
      evidence: `Average risk score: ${avgRiskScore.toFixed(2)}`,
    });

    // Domain relatedness
    let domainContribution = 0.2;
    if (technique) {
      const sourceDomainId = this.inferSourceDomain(technique);
      if (targetDomain?.relatedDomainIds.includes(sourceDomainId)) {
        domainContribution = 0.3;
      }
    }
    basis.push({
      factor: "Domain Relatedness",
      contribution: domainContribution,
      evidence: "Based on domain relationship analysis",
    });

    // Technique maturity
    const maturityContribution = technique?.complexity === "simple" ? 0.15 : 0.1;
    basis.push({
      factor: "Technique Maturity",
      contribution: maturityContribution,
      evidence: technique ? `Complexity: ${technique.complexity}` : "Unknown",
    });

    const probability = Math.max(
      0,
      Math.min(1, basis.reduce((a, b) => a + b.contribution, 0))
    );

    // Confidence interval
    const uncertainty = 0.15;
    const confidenceInterval = {
      low: Math.max(0, probability - uncertainty),
      high: Math.min(1, probability + uncertainty),
    };

    // Find comparable transfers
    const comparableTransfers = this.findComparableTransfers(
      techniqueId,
      targetDomainId
    );

    // Expected outcomes
    const expectedOutcomes: ExpectedOutcome[] = [];
    if (targetDomain) {
      expectedOutcomes.push({
        type: "performance",
        expectedValue: 0.7 + probability * 0.2,
        unit: "relative to baseline",
        baselineComparison: "Competitive with domain-specific methods",
        confidence: probability,
      });
    }

    return {
      probability,
      confidenceInterval,
      confidence: this.config.predictionConfidenceThreshold,
      basis,
      comparableTransfers,
      expectedOutcomes,
      successFactors: feasibility.enablers,
      failureRisks: feasibility.risks.slice(0, 3).map((r) => r.name),
    };
  }

  private findComparableTransfers(
    techniqueId: string,
    targetDomainId: string
  ): ComparableTransfer[] {
    // In a real implementation, this would query a database of past transfers
    // For now, return synthetic examples
    return [
      {
        sourceTechnique: "Transformer",
        sourceDomain: "NLP",
        targetDomain: targetDomainId,
        outcome: "success",
        similarity: 0.7,
        lessons: ["Attention mechanism transfers well", "Need domain-specific tokenization"],
      },
    ];
  }

  // ============================================================================
  // Transfer Proposal Creation
  // ============================================================================

  /**
   * Create a complete transfer proposal
   */
  createTransferProposal(
    techniqueId: string,
    targetDomainId: string
  ): TransferProposal {
    const principles = this.extractAbstractPrinciples(techniqueId);
    if (principles.length === 0) {
      throw new Error("Could not extract principles from technique");
    }

    const sourceDomainId = this.inferSourceDomain(
      this.graph.getNode(techniqueId) as TechniqueNode
    );
    const domainMapping = this.findDomainAnalogies(sourceDomainId, targetDomainId);
    const feasibility = this.assessFeasibility(techniqueId, targetDomainId);

    if (feasibility.overallScore < this.config.minFeasibilityScore) {
      // Still create proposal but mark as unlikely
    }

    const implementationGuide = this.generateImplementationGuide(
      techniqueId,
      targetDomainId,
      feasibility
    );
    const successPrediction = this.predictSuccess(
      techniqueId,
      targetDomainId,
      feasibility
    );

    const node = this.graph.getNode(techniqueId) as TechniqueNode;

    const proposal: TransferProposal = {
      id: createTransferProposalId(),
      sourceTechniqueId: techniqueId,
      sourceTechniqueName: node.name,
      sourceDomain: sourceDomainId,
      targetDomain: targetDomainId,
      principle: principles[0],
      domainMapping,
      feasibility,
      implementationGuide,
      successPrediction,
      status: "proposed",
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    this.proposals.set(proposal.id, proposal);
    return proposal;
  }

  // ============================================================================
  // Query Methods
  // ============================================================================

  /**
   * Get all proposals
   */
  getProposals(): TransferProposal[] {
    return Array.from(this.proposals.values());
  }

  /**
   * Get proposals by target domain
   */
  getProposalsByTargetDomain(domainId: string): TransferProposal[] {
    return Array.from(this.proposals.values()).filter(
      (p) => p.targetDomain === domainId
    );
  }

  /**
   * Get all domain mappings
   */
  getMappings(): DomainMapping[] {
    return Array.from(this.mappings.values());
  }

  /**
   * Get mapping between specific domains
   */
  getMapping(sourceDomainId: string, targetDomainId: string): DomainMapping | undefined {
    return this.mappings.get(`${sourceDomainId}-${targetDomainId}`);
  }

  /**
   * Get all domains
   */
  getDomains(): ResearchDomain[] {
    return Array.from(this.domains.values());
  }

  /**
   * Get domain by ID
   */
  getDomain(domainId: string): ResearchDomain | undefined {
    return this.domains.get(domainId);
  }

  /**
   * Add a custom domain
   */
  addDomain(domain: ResearchDomain): void {
    this.domains.set(domain.id, domain);
  }
}

// ============================================================================
// Factory
// ============================================================================

export function createCrossDomainTransferAgent(
  graph: KnowledgeGraph,
  config?: Partial<TransferAgentConfig>
): CrossDomainTransferAgent {
  return new CrossDomainTransferAgent(graph, config);
}
