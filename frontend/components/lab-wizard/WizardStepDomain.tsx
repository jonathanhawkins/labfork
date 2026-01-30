"use client";

import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import {
  Mic,
  TrendingUp,
  Bot as RobotIcon,
  FlaskConical,
  Layers,
  Plus,
  Check,
  Info,
} from "lucide-react";
import type { DomainConfig } from "@/lib/domain/types";

/**
 * Domain template for quick selection
 */
interface DomainTemplate {
  id: string;
  name: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  bgColor: string;
  arxivCategories: string[];
  tags: string[];
}

const DOMAIN_TEMPLATES: DomainTemplate[] = [
  {
    id: "voice-clone",
    name: "Voice Cloning",
    description: "Speech synthesis, prosody control, and voice conversion",
    icon: Mic,
    color: "text-blue-400",
    bgColor: "bg-blue-500/10",
    arxivCategories: ["cs.SD", "eess.AS", "cs.CL"],
    tags: ["TTS", "prosody", "emotion"],
  },
  {
    id: "quant-trading",
    name: "Quantitative Trading",
    description: "Algorithmic trading, market prediction, and finance ML",
    icon: TrendingUp,
    color: "text-green-400",
    bgColor: "bg-green-500/10",
    arxivCategories: ["q-fin.ST", "cs.LG", "stat.ML"],
    tags: ["finance", "prediction", "time-series"],
  },
  {
    id: "robotics",
    name: "Robotics",
    description: "Robot learning, embodied AI, and motion control",
    icon: RobotIcon,
    color: "text-purple-400",
    bgColor: "bg-purple-500/10",
    arxivCategories: ["cs.RO", "cs.AI", "cs.LG"],
    tags: ["control", "simulation", "embodied"],
  },
  {
    id: "biotech",
    name: "Biotech & Drug Discovery",
    description: "Molecular modeling, drug design, and biomedical NLP",
    icon: FlaskConical,
    color: "text-yellow-400",
    bgColor: "bg-yellow-500/10",
    arxivCategories: ["q-bio.BM", "cs.CL", "cs.LG"],
    tags: ["molecules", "proteins", "biomedical"],
  },
];

export interface WizardStepDomainProps {
  /** Currently selected domain slug */
  selectedDomain?: string;
  /** Whether creating new domain */
  createNew: boolean;
  /** New domain config (if creating new) */
  newDomainConfig?: Partial<DomainConfig>;
  /** Existing domains available */
  existingDomains?: DomainConfig[];
  /** Called when domain is selected */
  onSelectDomain: (slug: string) => void;
  /** Called when create new is toggled */
  onToggleCreateNew: (createNew: boolean) => void;
  /** Called when new domain config changes */
  onNewDomainChange?: (config: Partial<DomainConfig>) => void;
  /** Custom class name */
  className?: string;
}

/**
 * WizardStepDomain - Domain selection step
 */
export function WizardStepDomain({
  selectedDomain,
  createNew,
  newDomainConfig,
  existingDomains = [],
  onSelectDomain,
  onToggleCreateNew,
  onNewDomainChange,
  className,
}: WizardStepDomainProps) {
  const [showCustomForm, setShowCustomForm] = useState(false);

  // When a template is selected, populate the new domain config
  const handleSelectTemplate = (template: DomainTemplate) => {
    onSelectDomain(template.id);
    onToggleCreateNew(true);

    if (onNewDomainChange) {
      onNewDomainChange({
        name: template.name,
        slug: template.id,
        description: template.description,
        research: {
          arxivCategories: template.arxivCategories,
          keywords: template.tags,
        },
        branding: {
          primaryColor: getTemplateColor(template.id),
          accentColor: "#22c55e",
          backgroundStyle: "sky",
        },
      });
    }
  };

  // Handle existing domain selection
  const handleSelectExisting = (slug: string) => {
    onSelectDomain(slug);
    onToggleCreateNew(false);
  };

  return (
    <div className={cn("space-y-6", className)}>
      {/* Info banner */}
      <div className="flex items-start gap-3 p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
        <Info className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
        <div>
          <p className="text-sm text-foreground">
            Domains organize your research by topic. Each domain has its own
            configuration, arXiv categories, and evaluation metrics.
          </p>
        </div>
      </div>

      {/* Existing domains */}
      {existingDomains.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm text-foreground-muted">Your Existing Domains</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {existingDomains.map((domain) => (
              <button
                key={domain.slug}
                onClick={() => handleSelectExisting(domain.slug)}
                className={cn(
                  "flex items-center gap-3 p-4 rounded-lg border text-left transition-colors",
                  selectedDomain === domain.slug && !createNew
                    ? "border-foreground-bright bg-foreground-bright/5"
                    : "border-border hover:border-foreground-muted bg-background-card"
                )}
              >
                <div
                  className="w-10 h-10 rounded-lg flex items-center justify-center"
                  style={{ backgroundColor: `${domain.branding.primaryColor}20` }}
                >
                  <Layers
                    className="w-5 h-5"
                    style={{ color: domain.branding.primaryColor }}
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <h4 className="text-sm font-medium text-foreground truncate">
                    {domain.name}
                  </h4>
                  <p className="text-xs text-foreground-muted truncate">
                    {domain.description}
                  </p>
                </div>
                {selectedDomain === domain.slug && !createNew && (
                  <Check className="w-5 h-5 text-foreground-bright" />
                )}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Domain templates */}
      <div className="space-y-3">
        <h3 className="text-sm text-foreground-muted">
          {existingDomains.length > 0
            ? "Or Start with a Template"
            : "Choose a Domain Template"}
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {DOMAIN_TEMPLATES.map((template) => {
            const Icon = template.icon;
            const isSelected = selectedDomain === template.id && createNew;

            return (
              <button
                key={template.id}
                onClick={() => handleSelectTemplate(template)}
                className={cn(
                  "flex items-start gap-3 p-4 rounded-lg border text-left transition-colors",
                  isSelected
                    ? "border-foreground-bright bg-foreground-bright/5"
                    : "border-border hover:border-foreground-muted bg-background-card"
                )}
              >
                <div
                  className={cn(
                    "w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0",
                    template.bgColor
                  )}
                >
                  <Icon className={cn("w-5 h-5", template.color)} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <h4 className="text-sm font-medium text-foreground">
                      {template.name}
                    </h4>
                    {isSelected && (
                      <Check className="w-4 h-4 text-foreground-bright" />
                    )}
                  </div>
                  <p className="text-xs text-foreground-muted mt-1">
                    {template.description}
                  </p>
                  <div className="flex flex-wrap gap-1 mt-2">
                    {template.tags.slice(0, 3).map((tag) => (
                      <span
                        key={tag}
                        className="px-1.5 py-0.5 rounded text-[10px] bg-foreground-muted/10 text-foreground-subtle"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Custom domain option */}
      <div className="space-y-3">
        <button
          onClick={() => {
            setShowCustomForm(!showCustomForm);
            if (!showCustomForm) {
              onToggleCreateNew(true);
              onSelectDomain("custom");
            }
          }}
          className={cn(
            "flex items-center gap-3 w-full p-4 rounded-lg border text-left transition-colors",
            showCustomForm
              ? "border-foreground-bright bg-foreground-bright/5"
              : "border-dashed border-border hover:border-foreground-muted"
          )}
        >
          <div className="w-10 h-10 rounded-lg bg-foreground-muted/10 flex items-center justify-center">
            <Plus className="w-5 h-5 text-foreground-muted" />
          </div>
          <div>
            <h4 className="text-sm font-medium text-foreground">
              Create Custom Domain
            </h4>
            <p className="text-xs text-foreground-muted">
              Define your own research focus and categories
            </p>
          </div>
        </button>

        {/* Custom domain form */}
        {showCustomForm && (
          <div className="p-4 rounded-lg bg-background border border-border space-y-4">
            <div>
              <label className="block text-xs text-foreground-muted mb-1">
                Domain Name
              </label>
              <input
                type="text"
                value={newDomainConfig?.name || ""}
                onChange={(e) =>
                  onNewDomainChange?.({
                    ...newDomainConfig,
                    name: e.target.value,
                    slug: e.target.value
                      .toLowerCase()
                      .replace(/[^a-z0-9]+/g, "-")
                      .replace(/^-|-$/g, ""),
                  })
                }
                placeholder="My Research Domain"
                className={cn(
                  "w-full px-3 py-2 rounded-lg text-sm",
                  "bg-background-card border border-border",
                  "text-foreground placeholder-foreground-subtle",
                  "focus:outline-none focus:ring-2 focus:ring-foreground-bright/50"
                )}
              />
            </div>

            <div>
              <label className="block text-xs text-foreground-muted mb-1">
                Description
              </label>
              <textarea
                value={newDomainConfig?.description || ""}
                onChange={(e) =>
                  onNewDomainChange?.({
                    ...newDomainConfig,
                    description: e.target.value,
                  })
                }
                placeholder="What is this research about?"
                rows={2}
                className={cn(
                  "w-full px-3 py-2 rounded-lg text-sm resize-none",
                  "bg-background-card border border-border",
                  "text-foreground placeholder-foreground-subtle",
                  "focus:outline-none focus:ring-2 focus:ring-foreground-bright/50"
                )}
              />
            </div>

            <div>
              <label className="block text-xs text-foreground-muted mb-1">
                arXiv Categories (comma-separated)
              </label>
              <input
                type="text"
                value={newDomainConfig?.research?.arxivCategories?.join(", ") || ""}
                onChange={(e) =>
                  onNewDomainChange?.({
                    ...newDomainConfig,
                    research: {
                      ...newDomainConfig?.research,
                      arxivCategories: e.target.value
                        .split(",")
                        .map((s) => s.trim())
                        .filter(Boolean),
                      keywords: newDomainConfig?.research?.keywords || [],
                    },
                  })
                }
                placeholder="cs.LG, cs.AI"
                className={cn(
                  "w-full px-3 py-2 rounded-lg text-sm",
                  "bg-background-card border border-border",
                  "text-foreground placeholder-foreground-subtle",
                  "focus:outline-none focus:ring-2 focus:ring-foreground-bright/50"
                )}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Get primary color for template
 */
function getTemplateColor(templateId: string): string {
  switch (templateId) {
    case "voice-clone":
      return "#3b82f6";
    case "quant-trading":
      return "#22c55e";
    case "robotics":
      return "#a855f7";
    case "biotech":
      return "#eab308";
    default:
      return "#6b7280";
  }
}

export default WizardStepDomain;
