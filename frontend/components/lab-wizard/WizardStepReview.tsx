"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import {
  Check,
  Layers,
  Cpu,
  Server,
  Cloud,
  Target,
  ListTodo,
  Folder,
  FileText,
  AlertTriangle,
  Loader2,
  Rocket,
  Edit2,
  ChevronDown,
  ChevronUp,
  Clock,
  Zap,
} from "lucide-react";
import type { LabConfig, LabWizardStep, InitialTask } from "@/lib/lab-wizard/types";
import type { GoalAnalysisResult } from "./WizardStepResearch";

export interface WizardStepReviewProps {
  /** Complete lab configuration */
  config: LabConfig;
  /** AI analysis result */
  analysis?: GoalAnalysisResult | null;
  /** Called when user wants to edit a step */
  onEditStep: (step: LabWizardStep) => void;
  /** Called when lab creation is confirmed */
  onCreateLab: () => Promise<void>;
  /** Whether creation is in progress */
  isCreating?: boolean;
  /** Creation error message */
  createError?: string | null;
  /** Validation errors */
  validationErrors?: string[];
  /** Custom class name */
  className?: string;
}

/**
 * WizardStepReview - Final review and launch step
 */
export function WizardStepReview({
  config,
  analysis,
  onEditStep,
  onCreateLab,
  isCreating = false,
  createError,
  validationErrors = [],
  className,
}: WizardStepReviewProps) {
  const [showTasks, setShowTasks] = useState(true);
  const [showFiles, setShowFiles] = useState(false);

  // Get hardware display info
  const getHardwareInfo = () => {
    switch (config.hardware.type) {
      case "local":
        const gpu = config.hardware.local?.gpu;
        return {
          icon: Cpu,
          label: "Local Machine",
          detail: gpu?.name
            ? `${gpu.name} (${gpu.vram}GB)`
            : "No GPU detected",
          color: "text-blue-400",
          bgColor: "bg-blue-500/10",
        };
      case "remote-ssh":
        const ssh = config.hardware.ssh;
        return {
          icon: Server,
          label: "Remote SSH",
          detail: ssh?.host
            ? `${ssh.user}@${ssh.host}:${ssh.port || 22}`
            : "Not configured",
          color: "text-purple-400",
          bgColor: "bg-purple-500/10",
        };
      case "cloud":
        const cloud = config.hardware.cloud;
        return {
          icon: Cloud,
          label: `Cloud (${cloud?.provider || "Not selected"})`,
          detail: cloud?.apiKey ? "API key configured" : "No API key",
          color: "text-green-400",
          bgColor: "bg-green-500/10",
        };
      default:
        return {
          icon: Cpu,
          label: "Unknown",
          detail: "Not configured",
          color: "text-foreground-muted",
          bgColor: "bg-foreground-muted/10",
        };
    }
  };

  // Get domain display info
  const getDomainInfo = () => {
    if (config.createNewDomain && config.domain) {
      return {
        name: config.domain.name || "New Domain",
        slug: config.domain.slug || "custom",
        isNew: true,
      };
    }
    return {
      name: config.existingDomainSlug || "No domain",
      slug: config.existingDomainSlug,
      isNew: false,
    };
  };

  // Get files that will be created
  const getFilesToCreate = () => {
    const domain = getDomainInfo();
    if (!domain.isNew) return [];

    return [
      `.domains/${domain.slug}/domain.yaml`,
      `.domains/${domain.slug}/prompts/research.md`,
      `.domains/${domain.slug}/prompts/implementation.md`,
      `.domains/${domain.slug}/prompts/evaluation.md`,
    ];
  };

  // Get tasks to create
  const getTasksToCreate = (): InitialTask[] => {
    return config.research.goal?.initialTasks || analysis?.suggestedTasks || [];
  };

  const hardwareInfo = getHardwareInfo();
  const domainInfo = getDomainInfo();
  const filesToCreate = getFilesToCreate();
  const tasksToCreate = getTasksToCreate();
  const HardwareIcon = hardwareInfo.icon;

  // Check if ready to create
  const isValid = validationErrors.length === 0;

  return (
    <div className={cn("space-y-6", className)}>
      {/* Summary header */}
      <div className="text-center space-y-2">
        <h2 className="text-xl font-medium text-foreground">
          Review Your Lab Configuration
        </h2>
        <p className="text-sm text-foreground-muted">
          Verify everything looks correct before creating your lab
        </p>
      </div>

      {/* Validation errors */}
      {validationErrors.length > 0 && (
        <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/20">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-5 h-5 text-red-400" />
            <h4 className="text-sm font-medium text-red-400">
              Please fix the following issues:
            </h4>
          </div>
          <ul className="space-y-1">
            {validationErrors.map((error, i) => (
              <li key={i} className="text-sm text-red-300 flex items-start gap-2">
                <span className="text-red-400">-</span>
                {error}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Configuration sections */}
      <div className="space-y-4">
        {/* Domain section */}
        <div className="p-4 rounded-lg bg-background border border-border">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-foreground-bright/10 flex items-center justify-center">
                <Layers className="w-5 h-5 text-foreground-bright" />
              </div>
              <div>
                <h3 className="text-sm font-medium text-foreground">
                  {domainInfo.name}
                </h3>
                <p className="text-xs text-foreground-muted">
                  {domainInfo.isNew ? "New domain" : "Existing domain"} -{" "}
                  {domainInfo.slug}
                </p>
              </div>
            </div>
            <button
              onClick={() => onEditStep("domain")}
              className="flex items-center gap-1 px-2 py-1 text-xs text-foreground-muted hover:text-foreground transition-colors"
            >
              <Edit2 className="w-3 h-3" />
              Edit
            </button>
          </div>

          {config.domain?.description && (
            <p className="text-xs text-foreground-subtle mt-3 pl-13">
              {config.domain.description}
            </p>
          )}

          {config.domain?.research?.arxivCategories?.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2 pl-13">
              {config.domain.research.arxivCategories.map((cat) => (
                <span
                  key={cat}
                  className="px-1.5 py-0.5 rounded text-[10px] bg-blue-500/20 text-blue-400"
                >
                  {cat}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Hardware section */}
        <div className="p-4 rounded-lg bg-background border border-border">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div
                className={cn(
                  "w-10 h-10 rounded-lg flex items-center justify-center",
                  hardwareInfo.bgColor
                )}
              >
                <HardwareIcon className={cn("w-5 h-5", hardwareInfo.color)} />
              </div>
              <div>
                <h3 className="text-sm font-medium text-foreground">
                  {hardwareInfo.label}
                </h3>
                <p className="text-xs text-foreground-muted">
                  {hardwareInfo.detail}
                </p>
              </div>
            </div>
            <button
              onClick={() => onEditStep("hardware")}
              className="flex items-center gap-1 px-2 py-1 text-xs text-foreground-muted hover:text-foreground transition-colors"
            >
              <Edit2 className="w-3 h-3" />
              Edit
            </button>
          </div>

          {config.hardware.type === "remote-ssh" && config.hardware.ssh?.verified && (
            <div className="flex items-center gap-2 mt-3 pl-13 text-green-400 text-xs">
              <Check className="w-3 h-3" />
              Connection verified
            </div>
          )}
        </div>

        {/* Research goal section */}
        <div className="p-4 rounded-lg bg-background border border-border">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-yellow-500/10 flex items-center justify-center">
                <Target className="w-5 h-5 text-yellow-400" />
              </div>
              <div>
                <h3 className="text-sm font-medium text-foreground">
                  Research Goal
                </h3>
                <p className="text-xs text-foreground-muted">
                  {config.research.goal?.description
                    ? `${config.research.goal.description.slice(0, 60)}...`
                    : "No goal defined"}
                </p>
              </div>
            </div>
            <button
              onClick={() => onEditStep("research")}
              className="flex items-center gap-1 px-2 py-1 text-xs text-foreground-muted hover:text-foreground transition-colors"
            >
              <Edit2 className="w-3 h-3" />
              Edit
            </button>
          </div>

          {config.research.goal?.keywords?.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-3 pl-13">
              {config.research.goal.keywords.slice(0, 5).map((kw) => (
                <span
                  key={kw}
                  className="px-1.5 py-0.5 rounded text-[10px] bg-yellow-500/20 text-yellow-400"
                >
                  {kw}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Tasks to create */}
        {tasksToCreate.length > 0 && (
          <div className="p-4 rounded-lg bg-background border border-border">
            <button
              onClick={() => setShowTasks(!showTasks)}
              className="flex items-center gap-3 w-full"
            >
              <div className="w-10 h-10 rounded-lg bg-purple-500/10 flex items-center justify-center">
                <ListTodo className="w-5 h-5 text-purple-400" />
              </div>
              <div className="flex-1 text-left">
                <h3 className="text-sm font-medium text-foreground">
                  Initial Tasks
                </h3>
                <p className="text-xs text-foreground-muted">
                  {tasksToCreate.length} tasks will be created
                </p>
              </div>
              {showTasks ? (
                <ChevronUp className="w-4 h-4 text-foreground-muted" />
              ) : (
                <ChevronDown className="w-4 h-4 text-foreground-muted" />
              )}
            </button>

            {showTasks && (
              <div className="mt-3 pl-13 space-y-2">
                {tasksToCreate.map((task, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-2 text-xs"
                  >
                    <Check className="w-3 h-3 text-green-400 mt-0.5 flex-shrink-0" />
                    <span className="text-foreground">{task.subject}</span>
                    {task.estimatedHours && (
                      <span className="text-foreground-subtle ml-auto">
                        {task.estimatedHours}h
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Files to create */}
        {filesToCreate.length > 0 && (
          <div className="p-4 rounded-lg bg-background border border-border">
            <button
              onClick={() => setShowFiles(!showFiles)}
              className="flex items-center gap-3 w-full"
            >
              <div className="w-10 h-10 rounded-lg bg-green-500/10 flex items-center justify-center">
                <Folder className="w-5 h-5 text-green-400" />
              </div>
              <div className="flex-1 text-left">
                <h3 className="text-sm font-medium text-foreground">
                  Files to Create
                </h3>
                <p className="text-xs text-foreground-muted">
                  {filesToCreate.length} files in .domains/{domainInfo.slug}/
                </p>
              </div>
              {showFiles ? (
                <ChevronUp className="w-4 h-4 text-foreground-muted" />
              ) : (
                <ChevronDown className="w-4 h-4 text-foreground-muted" />
              )}
            </button>

            {showFiles && (
              <div className="mt-3 pl-13 space-y-1">
                {filesToCreate.map((file, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-2 text-xs text-foreground-muted"
                  >
                    <FileText className="w-3 h-3" />
                    {file}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Estimated time */}
      {analysis?.estimatedTimeline && (
        <div className="flex items-center justify-center gap-2 text-sm text-foreground-muted">
          <Clock className="w-4 h-4" />
          <span>Estimated research timeline: {analysis.estimatedTimeline}</span>
        </div>
      )}

      {/* Creation error */}
      {createError && (
        <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/20">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-red-400" />
            <p className="text-sm text-red-400">{createError}</p>
          </div>
        </div>
      )}

      {/* Create button */}
      <div className="flex flex-col items-center gap-4 pt-4">
        <button
          onClick={onCreateLab}
          disabled={!isValid || isCreating}
          className={cn(
            "flex items-center gap-3 px-8 py-4 text-base rounded-xl",
            "bg-gradient-to-r from-purple-500 via-blue-500 to-green-500",
            "text-white font-medium shadow-lg shadow-purple-500/20",
            "hover:shadow-xl hover:shadow-purple-500/30",
            "transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          )}
        >
          {isCreating ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Creating Lab...
            </>
          ) : (
            <>
              <Rocket className="w-5 h-5" />
              Launch Lab
              <Zap className="w-5 h-5" />
            </>
          )}
        </button>

        <p className="text-xs text-foreground-subtle text-center max-w-md">
          This will create your domain configuration, initialize research tasks,
          and redirect you to your new lab dashboard.
        </p>
      </div>
    </div>
  );
}

export default WizardStepReview;
