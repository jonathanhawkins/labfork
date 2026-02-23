/**
 * Cross-Domain Transfer Module
 *
 * Exports for analyzing and executing cross-domain technique transfers.
 */

// Types
export type {
  ResearchDomain,
  DomainConcept,
  DomainCharacteristics,
  DataModality,
  TaskType,
  ScaleRequirements,
  HardwareRequirements,
  AbstractPrinciple,
  PrincipleComponent,
  DomainMapping,
  ConceptMapping,
  ConceptMappingType,
  MappingQuality,
  DomainAnalogy,
  AnalogySample,
  MappingChallenge,
  ChallengeType,
  TransferProposal,
  TransferStatus,
  TransferFeasibility,
  FeasibilityLevel,
  FeasibilityComponents,
  RiskFactor,
  RequiredAdaptation,
  AdaptationType,
  EffortEstimate,
  EffortBreakdown,
  ImplementationGuide,
  Prerequisite,
  ImplementationStep,
  CodeTemplate,
  TestingStrategy,
  Pitfall,
  SuccessPrediction,
  PredictionBasis,
  ComparableTransfer,
  ExpectedOutcome,
} from "./types";

// Type guards and factories
export {
  isTransferProposal,
  isDomainMapping,
  isFeasibilityLevel,
  isTransferStatus,
  createTransferProposalId,
  createDomainMappingId,
  createPrincipleId,
  getFeasibilityLevelLabel,
  getFeasibilityLevelColor,
  getTransferStatusLabel,
  getTransferStatusColor,
  getMappingQualityColor,
} from "./types";

// Agent
export {
  CrossDomainTransferAgent,
  createCrossDomainTransferAgent,
  STANDARD_DOMAINS,
  DEFAULT_TRANSFER_AGENT_CONFIG,
} from "./agent";

export type { TransferAgentConfig } from "./agent";
