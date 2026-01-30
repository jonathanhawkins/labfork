/**
 * Domain Components
 *
 * Components for managing domain context and theming in the AI Research Lab Platform.
 */

export {
  DomainProvider,
  useDomain,
  useDomainSafe,
  useDomainConfig,
} from "./DomainProvider";

export {
  DomainBranding,
  useDomainColors,
  DomainColoredText,
  DomainColoredBg,
} from "./DomainBranding";

export { DomainCard } from "./DomainCard";
export type { DomainCardProps } from "./DomainCard";

export { DomainBrowser } from "./DomainBrowser";
export type { DomainBrowserProps, DomainSummary } from "./DomainBrowser";

export { DomainSwitcher } from "./DomainSwitcher";
export type { DomainSwitcherProps } from "./DomainSwitcher";

export { DomainPreview } from "./DomainPreview";
export type { DomainPreviewProps } from "./DomainPreview";

// Wizard components
export {
  DomainWizard,
  WizardStepTemplate,
  WizardStepBranding,
  WizardStepResearch,
  WizardStepScene,
  TEMPLATES,
  ARXIV_CATEGORIES,
  SUGGESTED_TAGS,
  AVAILABLE_PROPS,
  BACKGROUND_STYLES,
  CAMERA_ANGLES,
} from "./wizard";
export type {
  DomainWizardProps,
  WizardDomainConfig,
  DomainTemplate,
  WizardStepTemplateProps,
  DomainBranding as WizardDomainBranding,
  WizardStepBrandingProps,
  ResearchFocus,
  WizardStepResearchProps,
  SceneConfig,
  WizardStepSceneProps,
} from "./wizard";
