"use client";

import { cn } from "@/lib/utils";
import { DomainCard } from "../DomainCard";
import { Check, FileCode, Layers, Sparkles } from "lucide-react";

/**
 * Template options for starting a new domain
 */
export interface DomainTemplate {
  id: string;
  name: string;
  description: string;
  icon: "blank" | "voice" | "trading" | "robotics" | "biotech" | "custom";
  tags: string[];
  primaryColor: string;
  accentColor: string;
  difficulty: "beginner" | "intermediate" | "advanced";
  propsCount: number;
}

const TEMPLATES: DomainTemplate[] = [
  {
    id: "blank",
    name: "Blank Slate",
    description: "Start from scratch with minimal configuration",
    icon: "blank",
    tags: ["custom"],
    primaryColor: "#6b7280",
    accentColor: "#9ca3af",
    difficulty: "beginner",
    propsCount: 0,
  },
  {
    id: "voice-research",
    name: "Voice Research",
    description: "TTS, speech synthesis, and prosody research",
    icon: "voice",
    tags: ["tts", "speech", "prosody"],
    primaryColor: "#4ecdc4",
    accentColor: "#66ffaa",
    difficulty: "advanced",
    propsCount: 5,
  },
  {
    id: "quant-trading",
    name: "Quantitative Trading",
    description: "Financial ML and algorithmic trading",
    icon: "trading",
    tags: ["finance", "trading", "ml"],
    primaryColor: "#10b981",
    accentColor: "#f59e0b",
    difficulty: "advanced",
    propsCount: 4,
  },
  {
    id: "robotics",
    name: "Robotics & Embodied AI",
    description: "Robot learning and embodied agents",
    icon: "robotics",
    tags: ["robotics", "embodied", "rl"],
    primaryColor: "#f97316",
    accentColor: "#fbbf24",
    difficulty: "advanced",
    propsCount: 5,
  },
  {
    id: "biotech",
    name: "Biotech NLP",
    description: "Drug discovery and biomedical NLP",
    icon: "biotech",
    tags: ["nlp", "biotech", "drug-discovery"],
    primaryColor: "#8b5cf6",
    accentColor: "#22c55e",
    difficulty: "intermediate",
    propsCount: 4,
  },
];

export interface WizardStepTemplateProps {
  /** Currently selected template ID */
  selectedTemplate: string | null;
  /** Called when a template is selected */
  onSelectTemplate: (templateId: string) => void;
  /** Custom class name */
  className?: string;
}

/**
 * WizardStepTemplate - Step 1: Choose a template or start from scratch
 */
export function WizardStepTemplate({
  selectedTemplate,
  onSelectTemplate,
  className,
}: WizardStepTemplateProps) {
  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="text-center">
        <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-foreground-muted/10 mb-4">
          <Layers className="w-6 h-6 text-foreground-muted" />
        </div>
        <h2 className="text-lg font-normal text-foreground-bright">
          Choose a Template
        </h2>
        <p className="text-sm text-foreground-muted mt-1">
          Start with a template or create from scratch
        </p>
      </div>

      {/* Template grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {TEMPLATES.map((template) => (
          <button
            key={template.id}
            onClick={() => onSelectTemplate(template.id)}
            className={cn(
              "relative text-left p-4 rounded-lg border transition-all duration-200",
              "bg-background-card hover:bg-background-elevated",
              selectedTemplate === template.id
                ? "border-foreground-bright ring-1 ring-foreground-bright/50"
                : "border-border hover:border-foreground-muted"
            )}
          >
            {/* Color accent */}
            <div
              className="absolute top-0 left-0 right-0 h-1 rounded-t-lg"
              style={{ backgroundColor: template.primaryColor }}
            />

            {/* Selected indicator */}
            {selectedTemplate === template.id && (
              <div className="absolute top-3 right-3">
                <div
                  className="w-5 h-5 rounded-full flex items-center justify-center"
                  style={{ backgroundColor: template.primaryColor }}
                >
                  <Check className="w-3 h-3 text-white" />
                </div>
              </div>
            )}

            {/* Content */}
            <div className="mt-2">
              <div className="flex items-center gap-2">
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center"
                  style={{ backgroundColor: `${template.primaryColor}20` }}
                >
                  {template.id === "blank" ? (
                    <FileCode className="w-4 h-4" style={{ color: template.primaryColor }} />
                  ) : (
                    <Sparkles className="w-4 h-4" style={{ color: template.primaryColor }} />
                  )}
                </div>
                <h3 className="text-sm font-normal text-foreground-bright">
                  {template.name}
                </h3>
              </div>
              <p className="text-xs text-foreground-muted mt-2 line-clamp-2">
                {template.description}
              </p>

              {/* Tags */}
              {template.tags.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-3">
                  {template.tags.slice(0, 3).map((tag) => (
                    <span
                      key={tag}
                      className="text-[10px] px-1.5 py-0.5 rounded bg-background border border-border text-foreground-muted"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </button>
        ))}
      </div>

      {/* Info text */}
      <p className="text-xs text-foreground-subtle text-center">
        Templates provide pre-configured settings that you can customize in the next steps
      </p>
    </div>
  );
}

export { TEMPLATES };
export default WizardStepTemplate;
