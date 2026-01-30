/**
 * Domain Wizard Components
 *
 * Step-by-step wizard for creating new research domains.
 */

export { WizardStepTemplate, TEMPLATES } from "./WizardStepTemplate";
export type { DomainTemplate, WizardStepTemplateProps } from "./WizardStepTemplate";

export { WizardStepBranding } from "./WizardStepBranding";
export type { DomainBranding, WizardStepBrandingProps } from "./WizardStepBranding";

export { WizardStepResearch, ARXIV_CATEGORIES, SUGGESTED_TAGS } from "./WizardStepResearch";
export type { ResearchFocus, WizardStepResearchProps } from "./WizardStepResearch";

export { WizardStepScene, AVAILABLE_PROPS, BACKGROUND_STYLES, CAMERA_ANGLES } from "./WizardStepScene";
export type { SceneConfig, WizardStepSceneProps } from "./WizardStepScene";

export { DomainWizard } from "./DomainWizard";
export type { DomainWizardProps, WizardDomainConfig } from "./DomainWizard";
