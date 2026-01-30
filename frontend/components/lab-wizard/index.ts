/**
 * Lab Wizard Components
 *
 * Multi-step wizard for creating a new research lab with:
 * - Domain selection/creation
 * - Hardware configuration (local/SSH/cloud)
 * - AI-assisted research goal analysis
 * - Lab scaffolding and task generation
 */

// Main wizard container
export { LabWizard } from "./LabWizard";
export type { LabWizardProps } from "./LabWizard";

// Individual step components
export { WizardStepWelcome } from "./WizardStepWelcome";
export type { WizardStepWelcomeProps } from "./WizardStepWelcome";

export { WizardStepDomain } from "./WizardStepDomain";
export type { WizardStepDomainProps } from "./WizardStepDomain";

export { WizardStepHardware } from "./WizardStepHardware";
export type { WizardStepHardwareProps } from "./WizardStepHardware";

export { WizardStepResearch } from "./WizardStepResearch";
export type { WizardStepResearchProps, GoalAnalysisResult } from "./WizardStepResearch";

export { WizardStepReview } from "./WizardStepReview";
export type { WizardStepReviewProps } from "./WizardStepReview";

// Re-export types from lib for convenience
export type {
  LabWizardStep,
  LabWizardState,
  LabConfig,
  HardwareType,
  HardwareConfig,
  LocalConfig,
  SystemInfo,
  SSHConfig,
  CloudConfig,
  CloudProvider,
  CloudProviderInfo,
  GpuInfo,
  ResearchGoal,
  InitialTask,
  RecommendedPaper,
} from "@/lib/lab-wizard/types";

export {
  WIZARD_STEPS,
  CLOUD_PROVIDERS,
  CLOUD_PROVIDERS_MAP,
  getStepIndex,
  getNextStep,
  getPrevStep,
  isStepCompleted,
} from "@/lib/lab-wizard/types";
