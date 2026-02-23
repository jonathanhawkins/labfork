/**
 * Cross-Domain Transfer Types
 *
 * Types for transferring research techniques between domains,
 * mapping concepts, and predicting transfer success.
 */

// ============================================================================
// Core Transfer Types
// ============================================================================

/**
 * Research domain
 */
export interface ResearchDomain {
  /** Domain identifier */
  id: string;
  /** Domain name */
  name: string;
  /** Domain description */
  description: string;
  /** Parent domain (for hierarchy) */
  parentDomainId?: string;
  /** Key concepts */
  concepts: DomainConcept[];
  /** Common techniques in this domain */
  techniqueIds: string[];
  /** Domain characteristics */
  characteristics: DomainCharacteristics;
  /** Related domains */
  relatedDomainIds: string[];
}

/**
 * Domain concept
 */
export interface DomainConcept {
  /** Concept ID */
  id: string;
  /** Concept name */
  name: string;
  /** Description */
  description: string;
  /** Abstraction level (1-5, higher = more abstract) */
  abstractionLevel: number;
  /** Example implementations in this domain */
  examples: string[];
  /** Related concepts */
  relatedConceptIds: string[];
}

/**
 * Domain characteristics
 */
export interface DomainCharacteristics {
  /** Data modality (text, audio, image, video, etc.) */
  dataModality: DataModality;
  /** Typical task types */
  taskTypes: TaskType[];
  /** Scale requirements */
  scaleRequirements: ScaleRequirements;
  /** Evaluation metrics */
  evaluationMetrics: string[];
  /** Hardware requirements */
  hardwareRequirements: HardwareRequirements;
}

export type DataModality =
  | "text"
  | "audio"
  | "image"
  | "video"
  | "multimodal"
  | "structured"
  | "time_series"
  | "graph";

export type TaskType =
  | "generation"
  | "classification"
  | "regression"
  | "translation"
  | "detection"
  | "segmentation"
  | "synthesis"
  | "enhancement"
  | "compression"
  | "reasoning";

export interface ScaleRequirements {
  /** Typical dataset size */
  datasetSize: "small" | "medium" | "large" | "massive";
  /** Typical model size */
  modelSize: "tiny" | "small" | "medium" | "large" | "xlarge";
  /** Training duration */
  trainingDuration: "minutes" | "hours" | "days" | "weeks";
}

export interface HardwareRequirements {
  /** GPU memory required (GB) */
  gpuMemory: number;
  /** Multi-GPU required */
  multiGpu: boolean;
  /** TPU beneficial */
  tpuBeneficial: boolean;
  /** Special hardware needs */
  specialHardware?: string[];
}

// ============================================================================
// Abstract Principles
// ============================================================================

/**
 * Abstract principle extracted from technique
 */
export interface AbstractPrinciple {
  /** Principle ID */
  id: string;
  /** Principle name */
  name: string;
  /** Description */
  description: string;
  /** Abstraction level (1-5) */
  level: number;
  /** Core insight */
  coreInsight: string;
  /** Source technique IDs */
  sourceTechniqueIds: string[];
  /** Source domains */
  sourceDomains: string[];
  /** Key components */
  components: PrincipleComponent[];
  /** Applicability conditions */
  applicabilityConditions: string[];
  /** Counter-indications */
  counterIndications: string[];
  /** Confidence score (0-1) */
  confidence: number;
}

/**
 * Component of an abstract principle
 */
export interface PrincipleComponent {
  /** Component name */
  name: string;
  /** Role in the principle */
  role: string;
  /** Is required */
  required: boolean;
  /** Alternatives */
  alternatives: string[];
}

// ============================================================================
// Domain Mapping
// ============================================================================

/**
 * Mapping between two domains
 */
export interface DomainMapping {
  /** Mapping ID */
  id: string;
  /** Source domain */
  sourceDomainId: string;
  /** Target domain */
  targetDomainId: string;
  /** Concept mappings */
  conceptMappings: ConceptMapping[];
  /** Structural similarity (0-1) */
  structuralSimilarity: number;
  /** Functional similarity (0-1) */
  functionalSimilarity: number;
  /** Data compatibility (0-1) */
  dataCompatibility: number;
  /** Overall mapping strength (0-1) */
  mappingStrength: number;
  /** Mapping quality */
  quality: MappingQuality;
  /** Analogies found */
  analogies: DomainAnalogy[];
  /** Challenges in mapping */
  challenges: MappingChallenge[];
  /** Created at */
  createdAt: Date;
}

/**
 * Mapping between concepts
 */
export interface ConceptMapping {
  /** Source concept ID */
  sourceConceptId: string;
  /** Source concept name */
  sourceConceptName: string;
  /** Target concept ID */
  targetConceptId: string;
  /** Target concept name */
  targetConceptName: string;
  /** Mapping type */
  mappingType: ConceptMappingType;
  /** Similarity score (0-1) */
  similarity: number;
  /** Mapping justification */
  justification: string;
  /** Transformation needed */
  transformation?: string;
  /** Confidence (0-1) */
  confidence: number;
}

export type ConceptMappingType =
  | "equivalent" // Concepts are equivalent
  | "analogous" // Concepts serve similar roles
  | "partial" // Partial overlap
  | "generalization" // Source is more general
  | "specialization" // Source is more specific
  | "composition"; // Source maps to multiple targets

export type MappingQuality = "excellent" | "good" | "moderate" | "poor" | "none";

/**
 * Analogy between domains
 */
export interface DomainAnalogy {
  /** Analogy ID */
  id: string;
  /** Source pattern */
  sourcePattern: string;
  /** Target pattern */
  targetPattern: string;
  /** Description */
  description: string;
  /** Strength (0-1) */
  strength: number;
  /** Examples */
  examples: AnalogySample[];
}

/**
 * Sample illustrating an analogy
 */
export interface AnalogySample {
  /** Source example */
  source: string;
  /** Target equivalent */
  target: string;
  /** Explanation */
  explanation: string;
}

/**
 * Challenge in domain mapping
 */
export interface MappingChallenge {
  /** Challenge type */
  type: ChallengeType;
  /** Description */
  description: string;
  /** Severity (1-5) */
  severity: number;
  /** Mitigation strategies */
  mitigations: string[];
}

export type ChallengeType =
  | "data_format"
  | "scale_mismatch"
  | "semantic_gap"
  | "evaluation_mismatch"
  | "resource_constraint"
  | "domain_specific";

// ============================================================================
// Transfer Proposal
// ============================================================================

/**
 * Proposal for transferring a technique
 */
export interface TransferProposal {
  /** Proposal ID */
  id: string;
  /** Source technique ID */
  sourceTechniqueId: string;
  /** Source technique name */
  sourceTechniqueName: string;
  /** Source domain */
  sourceDomain: string;
  /** Target domain */
  targetDomain: string;
  /** Extracted principle */
  principle: AbstractPrinciple;
  /** Domain mapping used */
  domainMapping: DomainMapping;
  /** Feasibility assessment */
  feasibility: TransferFeasibility;
  /** Implementation guide */
  implementationGuide: ImplementationGuide;
  /** Success prediction */
  successPrediction: SuccessPrediction;
  /** Status */
  status: TransferStatus;
  /** Created at */
  createdAt: Date;
  /** Updated at */
  updatedAt: Date;
}

export type TransferStatus =
  | "proposed"
  | "analyzing"
  | "validated"
  | "in_progress"
  | "completed"
  | "failed"
  | "abandoned";

// ============================================================================
// Feasibility Assessment
// ============================================================================

/**
 * Assessment of transfer feasibility
 */
export interface TransferFeasibility {
  /** Overall feasibility score (0-1) */
  overallScore: number;
  /** Feasibility level */
  level: FeasibilityLevel;
  /** Component scores */
  components: FeasibilityComponents;
  /** Risk factors */
  risks: RiskFactor[];
  /** Enabling factors */
  enablers: string[];
  /** Required adaptations */
  adaptations: RequiredAdaptation[];
  /** Estimated effort */
  effort: EffortEstimate;
  /** Recommendations */
  recommendations: string[];
}

export type FeasibilityLevel =
  | "trivial" // Direct transfer possible
  | "straightforward" // Minor adaptations needed
  | "moderate" // Significant adaptations needed
  | "challenging" // Major adaptations, uncertain outcome
  | "infeasible"; // Not recommended

/**
 * Component feasibility scores
 */
export interface FeasibilityComponents {
  /** Technical compatibility (0-1) */
  technical: number;
  /** Data compatibility (0-1) */
  data: number;
  /** Computational feasibility (0-1) */
  computational: number;
  /** Knowledge transferability (0-1) */
  knowledge: number;
  /** Resource availability (0-1) */
  resources: number;
}

/**
 * Risk factor
 */
export interface RiskFactor {
  /** Risk name */
  name: string;
  /** Description */
  description: string;
  /** Probability (0-1) */
  probability: number;
  /** Impact (1-5) */
  impact: number;
  /** Risk score */
  score: number;
  /** Mitigation */
  mitigation: string;
}

/**
 * Required adaptation
 */
export interface RequiredAdaptation {
  /** Adaptation type */
  type: AdaptationType;
  /** Description */
  description: string;
  /** Effort required */
  effort: "low" | "medium" | "high";
  /** Priority */
  priority: number;
  /** Dependencies */
  dependencies: string[];
}

export type AdaptationType =
  | "data_preprocessing"
  | "architecture_modification"
  | "loss_function"
  | "training_procedure"
  | "evaluation_metrics"
  | "hyperparameter_tuning"
  | "infrastructure";

/**
 * Effort estimate
 */
export interface EffortEstimate {
  /** Person-days */
  personDays: number;
  /** Confidence range */
  range: { min: number; max: number };
  /** Breakdown */
  breakdown: EffortBreakdown[];
}

export interface EffortBreakdown {
  /** Activity */
  activity: string;
  /** Days */
  days: number;
  /** Notes */
  notes?: string;
}

// ============================================================================
// Implementation Guide
// ============================================================================

/**
 * Guide for implementing a transfer
 */
export interface ImplementationGuide {
  /** Overview */
  overview: string;
  /** Prerequisites */
  prerequisites: Prerequisite[];
  /** Implementation steps */
  steps: ImplementationStep[];
  /** Code templates */
  codeTemplates: CodeTemplate[];
  /** Testing strategy */
  testingStrategy: TestingStrategy;
  /** Common pitfalls */
  pitfalls: Pitfall[];
  /** Success criteria */
  successCriteria: string[];
}

/**
 * Prerequisite for implementation
 */
export interface Prerequisite {
  /** Name */
  name: string;
  /** Description */
  description: string;
  /** Type */
  type: "knowledge" | "data" | "infrastructure" | "library";
  /** Required */
  required: boolean;
}

/**
 * Implementation step
 */
export interface ImplementationStep {
  /** Step number */
  step: number;
  /** Title */
  title: string;
  /** Description */
  description: string;
  /** Details */
  details: string[];
  /** Estimated hours */
  estimatedHours: number;
  /** Dependencies (step numbers) */
  dependencies: number[];
  /** Validation */
  validation: string;
}

/**
 * Code template
 */
export interface CodeTemplate {
  /** Template name */
  name: string;
  /** Language */
  language: string;
  /** Code */
  code: string;
  /** Description */
  description: string;
  /** Placeholders to fill */
  placeholders: string[];
}

/**
 * Testing strategy
 */
export interface TestingStrategy {
  /** Unit tests */
  unitTests: string[];
  /** Integration tests */
  integrationTests: string[];
  /** Validation benchmarks */
  benchmarks: string[];
  /** Acceptance criteria */
  acceptanceCriteria: string[];
}

/**
 * Common pitfall
 */
export interface Pitfall {
  /** Name */
  name: string;
  /** Description */
  description: string;
  /** How to avoid */
  avoidance: string;
  /** Symptoms */
  symptoms: string[];
}

// ============================================================================
// Success Prediction
// ============================================================================

/**
 * Prediction of transfer success
 */
export interface SuccessPrediction {
  /** Predicted success probability (0-1) */
  probability: number;
  /** Confidence interval */
  confidenceInterval: { low: number; high: number };
  /** Prediction confidence */
  confidence: number;
  /** Basis for prediction */
  basis: PredictionBasis[];
  /** Comparable transfers */
  comparableTransfers: ComparableTransfer[];
  /** Expected outcomes */
  expectedOutcomes: ExpectedOutcome[];
  /** Key success factors */
  successFactors: string[];
  /** Key failure risks */
  failureRisks: string[];
}

/**
 * Basis for prediction
 */
export interface PredictionBasis {
  /** Factor */
  factor: string;
  /** Contribution to prediction */
  contribution: number;
  /** Evidence */
  evidence: string;
}

/**
 * Comparable transfer
 */
export interface ComparableTransfer {
  /** Source technique */
  sourceTechnique: string;
  /** Source domain */
  sourceDomain: string;
  /** Target domain */
  targetDomain: string;
  /** Outcome */
  outcome: "success" | "partial" | "failure";
  /** Similarity to current proposal */
  similarity: number;
  /** Key lessons */
  lessons: string[];
}

/**
 * Expected outcome
 */
export interface ExpectedOutcome {
  /** Outcome type */
  type: "performance" | "efficiency" | "capability";
  /** Expected value */
  expectedValue: number;
  /** Unit */
  unit: string;
  /** Baseline comparison */
  baselineComparison: string;
  /** Confidence */
  confidence: number;
}

// ============================================================================
// Type Guards
// ============================================================================

export function isTransferProposal(obj: unknown): obj is TransferProposal {
  return (
    typeof obj === "object" &&
    obj !== null &&
    "id" in obj &&
    "sourceTechniqueId" in obj &&
    "targetDomain" in obj &&
    "feasibility" in obj
  );
}

export function isDomainMapping(obj: unknown): obj is DomainMapping {
  return (
    typeof obj === "object" &&
    obj !== null &&
    "id" in obj &&
    "sourceDomainId" in obj &&
    "targetDomainId" in obj &&
    "conceptMappings" in obj
  );
}

export function isFeasibilityLevel(value: string): value is FeasibilityLevel {
  return ["trivial", "straightforward", "moderate", "challenging", "infeasible"].includes(value);
}

export function isTransferStatus(value: string): value is TransferStatus {
  return ["proposed", "analyzing", "validated", "in_progress", "completed", "failed", "abandoned"].includes(value);
}

// ============================================================================
// Factory Functions
// ============================================================================

export function createTransferProposalId(): string {
  return `transfer-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createDomainMappingId(): string {
  return `mapping-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createPrincipleId(): string {
  return `principle-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

// ============================================================================
// Display Helpers
// ============================================================================

export function getFeasibilityLevelLabel(level: FeasibilityLevel): string {
  const labels: Record<FeasibilityLevel, string> = {
    trivial: "Trivial",
    straightforward: "Straightforward",
    moderate: "Moderate",
    challenging: "Challenging",
    infeasible: "Infeasible",
  };
  return labels[level];
}

export function getFeasibilityLevelColor(level: FeasibilityLevel): string {
  const colors: Record<FeasibilityLevel, string> = {
    trivial: "#22c55e",
    straightforward: "#84cc16",
    moderate: "#f59e0b",
    challenging: "#ef4444",
    infeasible: "#6b7280",
  };
  return colors[level];
}

export function getTransferStatusLabel(status: TransferStatus): string {
  const labels: Record<TransferStatus, string> = {
    proposed: "Proposed",
    analyzing: "Analyzing",
    validated: "Validated",
    in_progress: "In Progress",
    completed: "Completed",
    failed: "Failed",
    abandoned: "Abandoned",
  };
  return labels[status];
}

export function getTransferStatusColor(status: TransferStatus): string {
  const colors: Record<TransferStatus, string> = {
    proposed: "#6b7280",
    analyzing: "#3b82f6",
    validated: "#8b5cf6",
    in_progress: "#f59e0b",
    completed: "#22c55e",
    failed: "#ef4444",
    abandoned: "#9ca3af",
  };
  return colors[status];
}

export function getMappingQualityColor(quality: MappingQuality): string {
  const colors: Record<MappingQuality, string> = {
    excellent: "#22c55e",
    good: "#84cc16",
    moderate: "#f59e0b",
    poor: "#ef4444",
    none: "#6b7280",
  };
  return colors[quality];
}
