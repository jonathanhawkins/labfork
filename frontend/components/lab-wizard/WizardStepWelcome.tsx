"use client";

import { cn } from "@/lib/utils";
import {
  Sparkles,
  Cpu,
  FileText,
  Bot,
  CheckCircle2,
  Clock,
  ArrowRight,
  Zap,
} from "lucide-react";
import { getEstimatedTime } from "@/lib/lab-wizard/types";

export interface WizardStepWelcomeProps {
  /** Called when user clicks "Get Started" */
  onGetStarted: () => void;
  /** Called when user clicks "Quick Setup" */
  onQuickSetup?: () => void;
  /** Custom class name */
  className?: string;
}

/**
 * Features to highlight
 */
const FEATURES = [
  {
    icon: Cpu,
    title: "Configure Hardware",
    description: "Local GPU, remote SSH, or cloud computing",
    color: "text-blue-400",
    bgColor: "bg-blue-500/10",
  },
  {
    icon: FileText,
    title: "Choose Your Domain",
    description: "Voice cloning, trading, robotics, and more",
    color: "text-purple-400",
    bgColor: "bg-purple-500/10",
  },
  {
    icon: Sparkles,
    title: "AI-Assisted Setup",
    description: "Describe your goal, get personalized recommendations",
    color: "text-yellow-400",
    bgColor: "bg-yellow-500/10",
  },
  {
    icon: Bot,
    title: "Launch Your Lab",
    description: "Automated task creation and agent orchestration",
    color: "text-green-400",
    bgColor: "bg-green-500/10",
  },
];

/**
 * What gets created
 */
const OUTPUTS = [
  "Domain configuration file",
  "Custom prompt templates",
  "Initial research tasks",
  "Hardware connection",
];

/**
 * WizardStepWelcome - First step of the lab creation wizard
 */
export function WizardStepWelcome({
  onGetStarted,
  onQuickSetup,
  className,
}: WizardStepWelcomeProps) {
  const estimatedMinutes = getEstimatedTime();

  return (
    <div className={cn("space-y-8", className)}>
      {/* Hero section */}
      <div className="text-center space-y-4">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-foreground-bright/10 text-foreground-bright text-sm">
          <Zap className="w-4 h-4" />
          New Lab Setup
        </div>

        <h1 className="text-3xl font-normal text-foreground-bright">
          Create Your Research Lab
        </h1>

        <p className="text-foreground-muted max-w-lg mx-auto">
          Set up a personalized AI research environment with the right hardware,
          domain focus, and initial goals. We will guide you through each step.
        </p>

        <div className="flex items-center justify-center gap-2 text-sm text-foreground-subtle">
          <Clock className="w-4 h-4" />
          <span>About {estimatedMinutes} minutes to complete</span>
        </div>
      </div>

      {/* Features grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {FEATURES.map((feature) => {
          const Icon = feature.icon;
          return (
            <div
              key={feature.title}
              className="flex items-start gap-4 p-4 rounded-lg bg-background-card border border-border"
            >
              <div
                className={cn(
                  "w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0",
                  feature.bgColor
                )}
              >
                <Icon className={cn("w-5 h-5", feature.color)} />
              </div>
              <div>
                <h3 className="text-sm font-medium text-foreground">
                  {feature.title}
                </h3>
                <p className="text-xs text-foreground-muted mt-1">
                  {feature.description}
                </p>
              </div>
            </div>
          );
        })}
      </div>

      {/* What you'll create */}
      <div className="p-4 rounded-lg bg-background border border-border">
        <h3 className="text-sm text-foreground-muted mb-3">
          What we will create for you:
        </h3>
        <div className="grid grid-cols-2 gap-2">
          {OUTPUTS.map((output) => (
            <div key={output} className="flex items-center gap-2 text-sm">
              <CheckCircle2 className="w-4 h-4 text-green-400" />
              <span className="text-foreground">{output}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
        <button
          onClick={onGetStarted}
          className={cn(
            "flex items-center gap-2 px-6 py-3 text-sm rounded-lg",
            "bg-foreground-bright text-background hover:bg-white",
            "transition-colors"
          )}
        >
          Get Started
          <ArrowRight className="w-4 h-4" />
        </button>

        {onQuickSetup && (
          <button
            onClick={onQuickSetup}
            className={cn(
              "flex items-center gap-2 px-6 py-3 text-sm rounded-lg",
              "border border-border text-foreground-muted hover:text-foreground",
              "hover:bg-foreground-muted/10 transition-colors"
            )}
          >
            <Zap className="w-4 h-4" />
            Use Template
          </button>
        )}
      </div>

      {/* Help text */}
      <p className="text-center text-xs text-foreground-subtle">
        You can save your progress and continue later at any point.
      </p>
    </div>
  );
}

export default WizardStepWelcome;
