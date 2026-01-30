"use client";

import { useState, useCallback, useMemo } from "react";
import { cn } from "@/lib/utils";
import { ArrowLeft, ArrowRight, Check, Loader2, X } from "lucide-react";
import { WizardStepTemplate, TEMPLATES, DomainTemplate } from "./WizardStepTemplate";
import { WizardStepBranding, DomainBranding } from "./WizardStepBranding";
import { WizardStepResearch, ResearchFocus } from "./WizardStepResearch";
import { WizardStepScene, SceneConfig } from "./WizardStepScene";

/**
 * Complete domain configuration from wizard
 */
export interface WizardDomainConfig {
  templateId: string;
  branding: DomainBranding;
  research: ResearchFocus;
  scene: SceneConfig;
}

/**
 * Wizard step definition
 */
interface WizardStep {
  id: string;
  title: string;
  description: string;
}

const WIZARD_STEPS: WizardStep[] = [
  { id: "template", title: "Template", description: "Choose a starting point" },
  { id: "branding", title: "Branding", description: "Name and colors" },
  { id: "research", title: "Research", description: "Focus areas" },
  { id: "scene", title: "Scene", description: "3D environment" },
];

export interface DomainWizardProps {
  /** Called when wizard completes with final config */
  onComplete: (config: WizardDomainConfig) => void | Promise<void>;
  /** Called when wizard is cancelled */
  onCancel?: () => void;
  /** Initial template to pre-select */
  initialTemplate?: string;
  /** Whether the wizard is in a saving state */
  isSaving?: boolean;
  /** Custom class name */
  className?: string;
}

/**
 * Get default branding from template
 */
function getDefaultBranding(template: DomainTemplate | null): DomainBranding {
  if (!template) {
    return {
      name: "",
      slug: "",
      description: "",
      primaryColor: "#6b7280",
      accentColor: "#9ca3af",
      difficulty: "intermediate",
    };
  }
  return {
    name: template.name === "Blank Slate" ? "" : template.name,
    slug: template.id === "blank" ? "" : template.id,
    description: template.description,
    primaryColor: template.primaryColor,
    accentColor: template.accentColor,
    difficulty: template.difficulty,
  };
}

/**
 * Get default research from template
 */
function getDefaultResearch(template: DomainTemplate | null): ResearchFocus {
  if (!template || template.id === "blank") {
    return {
      arxivCategories: [],
      keywords: [],
      tags: [],
    };
  }

  // Template-specific defaults
  switch (template.id) {
    case "voice-research":
      return {
        arxivCategories: ["cs.SD", "eess.AS", "cs.CL"],
        keywords: ["text-to-speech", "prosody", "voice cloning"],
        tags: template.tags,
      };
    case "quant-trading":
      return {
        arxivCategories: ["q-fin.ST", "cs.LG", "stat.ML"],
        keywords: ["algorithmic trading", "market prediction"],
        tags: template.tags,
      };
    case "robotics":
      return {
        arxivCategories: ["cs.RO", "cs.AI", "cs.LG"],
        keywords: ["robot learning", "embodied agent"],
        tags: template.tags,
      };
    case "biotech":
      return {
        arxivCategories: ["q-bio.BM", "cs.CL", "cs.LG"],
        keywords: ["drug discovery", "molecular", "biomedical nlp"],
        tags: template.tags,
      };
    default:
      return {
        arxivCategories: ["cs.LG"],
        keywords: [],
        tags: template.tags,
      };
  }
}

/**
 * Get default scene from template
 */
function getDefaultScene(template: DomainTemplate | null): SceneConfig {
  if (!template || template.id === "blank") {
    return {
      backgroundStyle: "sky",
      props: [],
      cameraAngle: "isometric",
    };
  }

  // Template-specific defaults
  switch (template.id) {
    case "voice-research":
      return {
        backgroundStyle: "sky",
        props: ["microphone", "speaker", "waveform", "gpu", "monitor"],
        cameraAngle: "isometric",
      };
    case "quant-trading":
      return {
        backgroundStyle: "grid",
        props: ["trading-terminal", "server-rack", "monitor", "data-hub"],
        cameraAngle: "front",
      };
    case "robotics":
      return {
        backgroundStyle: "grid",
        props: ["robot-arm", "gpu", "neural-network", "monitor", "server-rack"],
        cameraAngle: "isometric",
      };
    case "biotech":
      return {
        backgroundStyle: "gradient",
        props: ["lab-equipment", "monitor", "neural-network", "data-hub"],
        cameraAngle: "isometric",
      };
    default:
      return {
        backgroundStyle: "sky",
        props: ["gpu", "monitor"],
        cameraAngle: "isometric",
      };
  }
}

/**
 * DomainWizard - Multi-step wizard for creating new domains
 */
export function DomainWizard({
  onComplete,
  onCancel,
  initialTemplate,
  isSaving = false,
  className,
}: DomainWizardProps) {
  // Current step (0-indexed)
  const [currentStep, setCurrentStep] = useState(0);

  // Wizard state
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(
    initialTemplate || null
  );
  const [branding, setBranding] = useState<DomainBranding>(() =>
    getDefaultBranding(TEMPLATES.find((t) => t.id === initialTemplate) || null)
  );
  const [research, setResearch] = useState<ResearchFocus>(() =>
    getDefaultResearch(TEMPLATES.find((t) => t.id === initialTemplate) || null)
  );
  const [scene, setScene] = useState<SceneConfig>(() =>
    getDefaultScene(TEMPLATES.find((t) => t.id === initialTemplate) || null)
  );

  // Validation errors
  const [errors, setErrors] = useState<
    Partial<Record<keyof DomainBranding, string>>
  >({});

  // Handle template selection
  const handleSelectTemplate = useCallback((templateId: string) => {
    setSelectedTemplate(templateId);
    const template = TEMPLATES.find((t) => t.id === templateId);
    if (template) {
      setBranding(getDefaultBranding(template));
      setResearch(getDefaultResearch(template));
      setScene(getDefaultScene(template));
    }
  }, []);

  // Validate current step
  const validateStep = useCallback((): boolean => {
    if (currentStep === 0) {
      return selectedTemplate !== null;
    }
    if (currentStep === 1) {
      const newErrors: Partial<Record<keyof DomainBranding, string>> = {};
      if (!branding.name.trim()) {
        newErrors.name = "Name is required";
      }
      if (!branding.slug.trim()) {
        newErrors.slug = "Slug is required";
      } else if (!/^[a-z0-9-]+$/.test(branding.slug)) {
        newErrors.slug = "Slug must be lowercase with hyphens only";
      }
      if (!branding.description.trim()) {
        newErrors.description = "Description is required";
      }
      setErrors(newErrors);
      return Object.keys(newErrors).length === 0;
    }
    return true;
  }, [currentStep, selectedTemplate, branding]);

  // Navigation
  const goNext = useCallback(() => {
    if (!validateStep()) return;
    if (currentStep < WIZARD_STEPS.length - 1) {
      setCurrentStep((prev) => prev + 1);
    }
  }, [currentStep, validateStep]);

  const goBack = useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep((prev) => prev - 1);
    }
  }, [currentStep]);

  // Submit
  const handleSubmit = useCallback(async () => {
    if (!validateStep()) return;
    if (!selectedTemplate) return;

    const config: WizardDomainConfig = {
      templateId: selectedTemplate,
      branding,
      research,
      scene,
    };

    await onComplete(config);
  }, [selectedTemplate, branding, research, scene, validateStep, onComplete]);

  // Current step component
  const currentStepData = WIZARD_STEPS[currentStep];
  const isLastStep = currentStep === WIZARD_STEPS.length - 1;

  // Can proceed check
  const canProceed = useMemo(() => {
    if (currentStep === 0) return selectedTemplate !== null;
    if (currentStep === 1) {
      return (
        branding.name.trim() &&
        branding.slug.trim() &&
        branding.description.trim()
      );
    }
    return true;
  }, [currentStep, selectedTemplate, branding]);

  return (
    <div
      className={cn(
        "flex flex-col bg-background-card rounded-xl border border-border overflow-hidden",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-border">
        <div>
          <h2 className="text-lg font-normal text-foreground-bright">
            Create New Domain
          </h2>
          <p className="text-sm text-foreground-muted">
            {currentStepData.description}
          </p>
        </div>
        {onCancel && (
          <button
            onClick={onCancel}
            className="p-2 text-foreground-muted hover:text-foreground rounded-lg hover:bg-foreground-muted/10 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Progress steps */}
      <div className="flex items-center gap-1 px-6 py-3 border-b border-border bg-background/50">
        {WIZARD_STEPS.map((step, index) => (
          <div key={step.id} className="flex items-center">
            {/* Step indicator */}
            <button
              onClick={() => index < currentStep && setCurrentStep(index)}
              disabled={index > currentStep}
              className={cn(
                "flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs transition-colors",
                index === currentStep
                  ? "bg-foreground-bright/10 text-foreground-bright"
                  : index < currentStep
                  ? "text-foreground-muted hover:text-foreground cursor-pointer"
                  : "text-foreground-subtle cursor-not-allowed"
              )}
            >
              <span
                className={cn(
                  "w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-medium",
                  index === currentStep
                    ? "bg-foreground-bright text-background"
                    : index < currentStep
                    ? "bg-foreground-muted/30 text-foreground"
                    : "bg-foreground-muted/10 text-foreground-subtle"
                )}
              >
                {index < currentStep ? (
                  <Check className="w-3 h-3" />
                ) : (
                  index + 1
                )}
              </span>
              <span className="hidden sm:inline">{step.title}</span>
            </button>

            {/* Connector */}
            {index < WIZARD_STEPS.length - 1 && (
              <div
                className={cn(
                  "w-8 h-px mx-1",
                  index < currentStep
                    ? "bg-foreground-muted/50"
                    : "bg-foreground-muted/20"
                )}
              />
            )}
          </div>
        ))}
      </div>

      {/* Step content */}
      <div className="flex-1 overflow-y-auto p-6 max-h-[60vh]">
        {currentStep === 0 && (
          <WizardStepTemplate
            selectedTemplate={selectedTemplate}
            onSelectTemplate={handleSelectTemplate}
          />
        )}
        {currentStep === 1 && (
          <WizardStepBranding
            branding={branding}
            onBrandingChange={setBranding}
            errors={errors}
          />
        )}
        {currentStep === 2 && (
          <WizardStepResearch
            research={research}
            onResearchChange={setResearch}
          />
        )}
        {currentStep === 3 && (
          <WizardStepScene
            scene={scene}
            onSceneChange={setScene}
            primaryColor={branding.primaryColor}
          />
        )}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between px-6 py-4 border-t border-border bg-background/50">
        <button
          onClick={currentStep === 0 ? onCancel : goBack}
          disabled={isSaving}
          className={cn(
            "flex items-center gap-2 px-4 py-2 text-sm rounded-lg transition-colors",
            "text-foreground-muted hover:text-foreground",
            "disabled:opacity-50 disabled:cursor-not-allowed"
          )}
        >
          <ArrowLeft className="w-4 h-4" />
          {currentStep === 0 ? "Cancel" : "Back"}
        </button>

        <button
          onClick={isLastStep ? handleSubmit : goNext}
          disabled={!canProceed || isSaving}
          className={cn(
            "flex items-center gap-2 px-4 py-2 text-sm rounded-lg transition-colors",
            "bg-foreground-bright text-background hover:bg-white",
            "disabled:opacity-50 disabled:cursor-not-allowed"
          )}
        >
          {isSaving ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Saving...
            </>
          ) : isLastStep ? (
            <>
              <Check className="w-4 h-4" />
              Create Domain
            </>
          ) : (
            <>
              Next
              <ArrowRight className="w-4 h-4" />
            </>
          )}
        </button>
      </div>
    </div>
  );
}

export default DomainWizard;
