"use client";

import { useState, useCallback, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  ChevronLeft,
  ChevronRight,
  X,
  Check,
  Circle,
} from "lucide-react";
import type {
  LabWizardStep,
  LabWizardState,
  LabConfig,
  HardwareConfig,
  ResearchGoal,
} from "@/lib/lab-wizard/types";
import {
  WIZARD_STEPS,
  getStepIndex,
  getNextStep,
  getPrevStep,
  isStepCompleted,
} from "@/lib/lab-wizard/types";
import type { DomainConfig } from "@/lib/domain/types";
import { WizardStepWelcome } from "./WizardStepWelcome";
import { WizardStepDomain } from "./WizardStepDomain";
import { WizardStepHardware } from "./WizardStepHardware";
import { WizardStepResearch, GoalAnalysisResult } from "./WizardStepResearch";
import { WizardStepReview } from "./WizardStepReview";

export interface LabWizardProps {
  /** Existing domains to select from */
  existingDomains?: DomainConfig[];
  /** Pre-selected domain slug */
  initialDomain?: string;
  /** Called when wizard is closed */
  onClose?: () => void;
  /** Custom class name */
  className?: string;
}

/**
 * Initial lab config state
 */
const getInitialConfig = (initialDomain?: string): LabConfig => ({
  createNewDomain: !initialDomain,
  existingDomainSlug: initialDomain,
  hardware: {
    type: "local",
  },
  research: {
    path: "goal",
    goal: {
      description: "",
      keywords: [],
    },
  },
});

/**
 * LabWizard - Multi-step wizard for creating a new research lab
 */
export function LabWizard({
  existingDomains = [],
  initialDomain,
  onClose,
  className,
}: LabWizardProps) {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Get URL parameters
  const urlStep = searchParams.get("step") as LabWizardStep | null;
  const urlDomain = searchParams.get("domain");
  const urlQuick = searchParams.get("quick") === "true";

  // Determine initial step based on URL params
  const getInitialStep = (): LabWizardStep => {
    // Quick mode skips to review (requires domain)
    if (urlQuick && urlDomain) return "review";
    // If domain is specified, skip welcome
    if (urlDomain) return "domain";
    // Use URL step if valid
    if (urlStep && WIZARD_STEPS.some((s) => s.id === urlStep)) return urlStep;
    // Default to welcome
    return "welcome";
  };
  const initialStep = getInitialStep();

  // Pre-select domain from URL or prop
  const getInitialDomain = () => {
    if (urlDomain) {
      return urlDomain;
    }
    return initialDomain;
  };

  // Wizard state
  const [currentStep, setCurrentStep] = useState<LabWizardStep>(initialStep);
  const [config, setConfig] = useState<LabConfig>(() => {
    const baseConfig = getInitialConfig(initialDomain);
    const domainSlug = getInitialDomain();
    if (domainSlug) {
      return {
        ...baseConfig,
        existingDomainSlug: domainSlug,
        createNewDomain: false,
      };
    }
    return baseConfig;
  });
  const [analysis, setAnalysis] = useState<GoalAnalysisResult | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  // Update URL when step changes
  useEffect(() => {
    const params = new URLSearchParams(searchParams.toString());
    params.set("step", currentStep);
    router.replace(`/lab/new?${params.toString()}`, { scroll: false });
  }, [currentStep, router, searchParams]);

  // Validate current step
  const validateStep = useCallback(
    (step: LabWizardStep): string[] => {
      const errors: string[] = [];

      switch (step) {
        case "domain":
          if (!config.createNewDomain && !config.existingDomainSlug) {
            errors.push("Please select a domain or create a new one");
          }
          if (config.createNewDomain) {
            if (!config.domain?.name?.trim()) {
              errors.push("Domain name is required");
            }
            if (!config.domain?.slug?.trim()) {
              errors.push("Domain slug is required");
            }
          }
          break;

        case "hardware":
          if (config.hardware.type === "remote-ssh") {
            if (!config.hardware.ssh?.host) {
              errors.push("SSH host is required");
            }
            if (!config.hardware.ssh?.user) {
              errors.push("SSH user is required");
            }
          }
          if (config.hardware.type === "cloud") {
            if (!config.hardware.cloud?.provider) {
              errors.push("Please select a cloud provider");
            }
          }
          break;

        case "research":
          // Research step is optional
          break;

        case "review":
          // Aggregate all validation
          errors.push(...validateStep("domain"));
          errors.push(...validateStep("hardware"));
          break;
      }

      return errors;
    },
    [config]
  );

  // Navigate to step
  const goToStep = useCallback(
    (step: LabWizardStep) => {
      setCurrentStep(step);
      setValidationErrors([]);
    },
    []
  );

  // Navigate to next step
  const goNext = useCallback(() => {
    const errors = validateStep(currentStep);
    if (errors.length > 0) {
      setValidationErrors(errors);
      return;
    }

    const next = getNextStep(currentStep);
    if (next) {
      goToStep(next);
    }
  }, [currentStep, validateStep, goToStep]);

  // Navigate to previous step
  const goPrev = useCallback(() => {
    const prev = getPrevStep(currentStep);
    if (prev) {
      goToStep(prev);
    }
  }, [currentStep, goToStep]);

  // Handle domain selection
  const handleSelectDomain = useCallback((slug: string) => {
    setConfig((prev) => ({
      ...prev,
      existingDomainSlug: slug,
      domain: undefined,
    }));
  }, []);

  // Handle toggle create new
  const handleToggleCreateNew = useCallback((createNew: boolean) => {
    setConfig((prev) => ({
      ...prev,
      createNewDomain: createNew,
    }));
  }, []);

  // Handle new domain config change
  const handleNewDomainChange = useCallback((domainConfig: Partial<DomainConfig>) => {
    setConfig((prev) => ({
      ...prev,
      domain: {
        ...prev.domain,
        ...domainConfig,
      },
    }));
  }, []);

  // Handle hardware config change
  const handleHardwareChange = useCallback((hardware: HardwareConfig) => {
    setConfig((prev) => ({
      ...prev,
      hardware,
    }));
  }, []);

  // Handle research goal change
  const handleGoalChange = useCallback((goal: ResearchGoal) => {
    setConfig((prev) => ({
      ...prev,
      research: {
        ...prev.research,
        goal,
      },
    }));
  }, []);

  // Handle analysis change
  const handleAnalysisChange = useCallback((newAnalysis: GoalAnalysisResult | null) => {
    setAnalysis(newAnalysis);
  }, []);

  // Handle edit step from review
  const handleEditStep = useCallback((step: LabWizardStep) => {
    goToStep(step);
  }, [goToStep]);

  // Create lab
  const handleCreateLab = useCallback(async () => {
    // Validate all steps
    const errors = validateStep("review");
    if (errors.length > 0) {
      setValidationErrors(errors);
      return;
    }

    setIsCreating(true);
    setCreateError(null);

    try {
      // Add analysis tasks to config if not already
      const finalConfig = { ...config };
      if (analysis?.suggestedTasks && !config.research.goal?.initialTasks?.length) {
        finalConfig.research = {
          ...finalConfig.research,
          goal: {
            ...finalConfig.research.goal,
            initialTasks: analysis.suggestedTasks,
          },
        };
      }

      const response = await fetch("/api/lab/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(finalConfig),
      });

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || "Failed to create lab");
      }

      // Redirect to new lab
      router.push(data.redirectUrl || `/lab?domain=${data.domainSlug}`);
    } catch (error) {
      setCreateError(error instanceof Error ? error.message : "Creation failed");
    } finally {
      setIsCreating(false);
    }
  }, [config, analysis, validateStep, router]);

  // Get step completion status
  const getStepStatus = (stepId: LabWizardStep) => {
    const stepIndex = getStepIndex(stepId);
    const currentIndex = getStepIndex(currentStep);

    if (stepIndex < currentIndex) {
      return "completed";
    }
    if (stepIndex === currentIndex) {
      return "current";
    }
    return "upcoming";
  };

  // Get selected domain for context
  const selectedDomain = config.createNewDomain
    ? config.domain?.slug
    : config.existingDomainSlug;

  // Get hardware VRAM for context
  const hardwareVram =
    config.hardware.type === "local"
      ? config.hardware.local?.gpu?.vram
      : config.hardware.ssh?.remoteGpu?.vram;

  return (
    <div className={cn("min-h-screen bg-background", className)}>
      {/* Header */}
      <header className="sticky top-0 z-50 bg-background/80 backdrop-blur-sm border-b border-border">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-lg font-medium text-foreground">
              Create New Lab
            </h1>
            {onClose && (
              <button
                onClick={onClose}
                className="p-2 rounded-lg hover:bg-foreground-muted/10 transition-colors"
              >
                <X className="w-5 h-5 text-foreground-muted" />
              </button>
            )}
          </div>

          {/* Progress steps */}
          <div className="flex items-center gap-2 mt-4 overflow-x-auto pb-2">
            {WIZARD_STEPS.map((step, index) => {
              const status = getStepStatus(step.id);
              const isClickable = status === "completed";

              return (
                <button
                  key={step.id}
                  onClick={() => isClickable && goToStep(step.id)}
                  disabled={!isClickable}
                  className={cn(
                    "flex items-center gap-2 px-3 py-1.5 rounded-full text-xs whitespace-nowrap transition-colors",
                    status === "current"
                      ? "bg-foreground-bright text-background"
                      : status === "completed"
                      ? "bg-green-500/20 text-green-400 hover:bg-green-500/30 cursor-pointer"
                      : "bg-foreground-muted/10 text-foreground-subtle cursor-not-allowed"
                  )}
                >
                  {status === "completed" ? (
                    <Check className="w-3 h-3" />
                  ) : (
                    <Circle
                      className={cn(
                        "w-3 h-3",
                        status === "current" ? "fill-current" : ""
                      )}
                    />
                  )}
                  {step.label}
                </button>
              );
            })}
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-2xl mx-auto px-4 py-8">
        {/* Welcome step */}
        {currentStep === "welcome" && (
          <WizardStepWelcome
            onGetStarted={goNext}
            onQuickSetup={() => {
              // Quick setup: jump to review with defaults
              setConfig({
                ...config,
                createNewDomain: true,
                domain: {
                  name: "Voice Cloning",
                  slug: "voice-clone",
                  description: "Speech synthesis and prosody control research",
                  research: {
                    arxivCategories: ["cs.SD", "eess.AS", "cs.CL"],
                    keywords: ["TTS", "prosody", "emotion"],
                  },
                  branding: {
                    primaryColor: "#3b82f6",
                    accentColor: "#22c55e",
                    backgroundStyle: "sky",
                  },
                },
              });
              goToStep("review");
            }}
          />
        )}

        {/* Domain step */}
        {currentStep === "domain" && (
          <WizardStepDomain
            selectedDomain={selectedDomain}
            createNew={config.createNewDomain}
            newDomainConfig={config.domain}
            existingDomains={existingDomains}
            onSelectDomain={handleSelectDomain}
            onToggleCreateNew={handleToggleCreateNew}
            onNewDomainChange={handleNewDomainChange}
          />
        )}

        {/* Hardware step */}
        {currentStep === "hardware" && (
          <WizardStepHardware
            config={config.hardware}
            onConfigChange={handleHardwareChange}
            selectedDomain={selectedDomain}
          />
        )}

        {/* Research step */}
        {currentStep === "research" && (
          <WizardStepResearch
            goal={config.research.goal || { description: "", keywords: [] }}
            onGoalChange={handleGoalChange}
            analysis={analysis}
            onAnalysisChange={handleAnalysisChange}
            selectedDomain={selectedDomain}
            hardwareVram={hardwareVram}
          />
        )}

        {/* Review step */}
        {currentStep === "review" && (
          <WizardStepReview
            config={config}
            analysis={analysis}
            onEditStep={handleEditStep}
            onCreateLab={handleCreateLab}
            isCreating={isCreating}
            createError={createError}
            validationErrors={validationErrors}
          />
        )}

        {/* Validation errors */}
        {validationErrors.length > 0 && currentStep !== "review" && (
          <div className="mt-6 p-4 rounded-lg bg-red-500/10 border border-red-500/20">
            <ul className="space-y-1">
              {validationErrors.map((error, i) => (
                <li key={i} className="text-sm text-red-400">
                  {error}
                </li>
              ))}
            </ul>
          </div>
        )}
      </main>

      {/* Footer navigation */}
      {currentStep !== "welcome" && currentStep !== "review" && (
        <footer className="fixed bottom-0 left-0 right-0 bg-background/80 backdrop-blur-sm border-t border-border">
          <div className="max-w-2xl mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <button
                onClick={goPrev}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 text-sm rounded-lg",
                  "border border-border text-foreground-muted",
                  "hover:bg-foreground-muted/10 transition-colors"
                )}
              >
                <ChevronLeft className="w-4 h-4" />
                Back
              </button>

              <button
                onClick={goNext}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 text-sm rounded-lg",
                  "bg-foreground-bright text-background hover:bg-white",
                  "transition-colors"
                )}
              >
                Continue
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </footer>
      )}
    </div>
  );
}

export default LabWizard;
