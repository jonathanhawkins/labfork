"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import {
  X,
  Target,
  Loader2,
  AlertCircle,
  Check,
  BookOpen,
  Calendar,
  Layers,
  Lightbulb,
  Clock,
} from "lucide-react";
import type { Paper } from "@/lib/papers/types";
import type { GoalAnalysis } from "@/lib/research/goal-analyzer";

export interface AddCustomGoalProps {
  /** Whether dialog is open */
  isOpen: boolean;
  /** Close handler */
  onClose: () => void;
  /** Analysis complete handler */
  onAnalysisComplete?: (analysis: GoalAnalysis, papers?: Paper[]) => void;
  /** Custom class name */
  className?: string;
}

type DialogStep = "input" | "analyzing" | "result";

/**
 * AddCustomGoal - Dialog for defining custom research goals
 */
export function AddCustomGoal({
  isOpen,
  onClose,
  onAnalysisComplete,
  className,
}: AddCustomGoalProps) {
  // State
  const [step, setStep] = useState<DialogStep>("input");
  const [goal, setGoal] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<GoalAnalysis | null>(null);
  const [papers, setPapers] = useState<Paper[]>([]);

  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Focus input when dialog opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  // Reset state when dialog closes
  useEffect(() => {
    if (!isOpen) {
      setStep("input");
      setGoal("");
      setError(null);
      setAnalysis(null);
      setPapers([]);
    }
  }, [isOpen]);

  // Validate goal
  const validateGoal = useCallback((text: string): boolean => {
    const trimmed = text.trim();
    // Need at least 20 characters for a meaningful goal
    return trimmed.length >= 20 && trimmed.split(" ").length >= 4;
  }, []);

  // Analyze goal
  const handleAnalyze = useCallback(async () => {
    if (!goal.trim()) return;

    setIsLoading(true);
    setError(null);
    setStep("analyzing");

    try {
      const response = await fetch("/api/research/goal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal }),
      });

      const data = await response.json();

      if (!data.success) {
        setError(data.error || "Failed to analyze goal");
        setStep("input");
        return;
      }

      setAnalysis(data.analysis);
      if (data.papers) {
        setPapers(data.papers);
      }
      setStep("result");

      if (onAnalysisComplete) {
        onAnalysisComplete(data.analysis, data.papers);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to analyze goal");
      setStep("input");
    } finally {
      setIsLoading(false);
    }
  }, [goal, onAnalysisComplete]);

  // Handle keyboard
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && e.metaKey) {
        e.preventDefault();
        if (step === "input" && validateGoal(goal) && !isLoading) {
          handleAnalyze();
        }
      }
      if (e.key === "Escape") {
        onClose();
      }
    },
    [step, goal, isLoading, validateGoal, handleAnalyze, onClose]
  );

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        className={cn(
          "relative w-full max-w-lg mx-4 bg-background-card border border-border rounded-xl shadow-2xl",
          className
        )}
        onKeyDown={handleKeyDown}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-foreground-muted/10 flex items-center justify-center">
              <Target className="w-5 h-5 text-foreground-muted" />
            </div>
            <div>
              <h2 className="text-lg font-normal text-foreground-bright">
                Define Research Goal
              </h2>
              <p className="text-xs text-foreground-muted">
                {step === "input" && "Describe what you want to achieve"}
                {step === "analyzing" && "Analyzing your goal..."}
                {step === "result" && "Research plan ready"}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-foreground-muted hover:text-foreground p-2 rounded-lg hover:bg-foreground-muted/10"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {/* Input step */}
          {step === "input" && (
            <div className="space-y-4">
              {/* Input field */}
              <div className="relative">
                <textarea
                  ref={inputRef}
                  value={goal}
                  onChange={(e) => setGoal(e.target.value)}
                  placeholder="e.g., I want to improve prosody control in voice cloning by using emotional embeddings and style transfer techniques..."
                  rows={4}
                  className={cn(
                    "w-full px-4 py-3 text-sm rounded-lg resize-none",
                    "bg-background border",
                    "text-foreground placeholder:text-foreground-subtle",
                    "focus:outline-none focus:border-foreground-muted",
                    error ? "border-red-500/50" : "border-border"
                  )}
                />
                <div className="absolute right-3 bottom-3 text-xs text-foreground-subtle">
                  {goal.length} chars
                </div>
              </div>

              {/* Error message */}
              {error && (
                <div className="flex items-center gap-2 text-sm text-red-400">
                  <AlertCircle className="w-4 h-4" />
                  {error}
                </div>
              )}

              {/* Validation hint */}
              {goal.length > 0 && !validateGoal(goal) && (
                <div className="text-xs text-foreground-subtle">
                  Please provide more detail (at least 20 characters)
                </div>
              )}

              {/* What we generate */}
              <div className="text-xs text-foreground-subtle">
                <p className="mb-2">We will generate:</p>
                <ul className="space-y-1 ml-4">
                  <li className="flex items-center gap-2">
                    <Layers className="w-3 h-3" />
                    Research domain identification
                  </li>
                  <li className="flex items-center gap-2">
                    <BookOpen className="w-3 h-3" />
                    Suggested papers to read
                  </li>
                  <li className="flex items-center gap-2">
                    <Calendar className="w-3 h-3" />
                    Research plan with milestones
                  </li>
                  <li className="flex items-center gap-2">
                    <Clock className="w-3 h-3" />
                    Time and resource estimates
                  </li>
                </ul>
              </div>

              {/* Example goals */}
              <div className="text-xs text-foreground-subtle">
                <p className="mb-1">Example goals:</p>
                <ul className="space-y-0.5 ml-4 list-disc">
                  <li>Improve emotional expressiveness in TTS</li>
                  <li>Reduce training time for voice cloning models</li>
                  <li>Add real-time prosody control during inference</li>
                </ul>
              </div>
            </div>
          )}

          {/* Analyzing step */}
          {step === "analyzing" && (
            <div className="flex flex-col items-center py-8 space-y-4">
              <div className="relative">
                <div className="w-16 h-16 rounded-full bg-foreground-bright/10 flex items-center justify-center">
                  <Target className="w-8 h-8 text-foreground-bright animate-pulse" />
                </div>
                <div className="absolute inset-0 rounded-full border-2 border-foreground-bright/30 border-t-foreground-bright animate-spin" />
              </div>
              <div className="text-center">
                <p className="text-foreground-bright">
                  Analyzing your goal...
                </p>
                <p className="text-sm text-foreground-muted mt-1">
                  Matching domains and finding relevant papers
                </p>
              </div>
            </div>
          )}

          {/* Result step */}
          {step === "result" && analysis && (
            <div className="space-y-4 max-h-[50vh] overflow-y-auto">
              {/* Success message */}
              <div className="flex items-center gap-3 p-4 rounded-lg bg-green-500/10 border border-green-500/20">
                <Check className="w-5 h-5 text-green-400" />
                <div>
                  <p className="text-sm text-green-400">
                    Research plan generated
                  </p>
                  <p className="text-xs text-foreground-muted mt-0.5">
                    Complexity: {analysis.complexity}
                  </p>
                </div>
              </div>

              {/* Recommended domain */}
              {analysis.recommendedDomain && (
                <div>
                  <h4 className="text-sm text-foreground-muted mb-2 flex items-center gap-2">
                    <Layers className="w-4 h-4" />
                    Research Domain
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    <span className="px-3 py-1 text-xs bg-foreground-bright/10 text-foreground-bright rounded-full">
                      {analysis.recommendedDomain.name}
                    </span>
                    {analysis.alternativeDomains.slice(0, 2).map((domain) => (
                      <span
                        key={domain.slug}
                        className="px-3 py-1 text-xs bg-foreground-muted/10 rounded-full"
                      >
                        {domain.name}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Suggested papers */}
              {analysis.paperSuggestions.length > 0 && (
                <div>
                  <h4 className="text-sm text-foreground-muted mb-2 flex items-center gap-2">
                    <BookOpen className="w-4 h-4" />
                    Suggested Papers
                  </h4>
                  <ul className="space-y-2">
                    {analysis.paperSuggestions.slice(0, 4).map((paper, idx) => (
                      <li
                        key={idx}
                        className="text-sm text-foreground bg-background/50 p-2 rounded border border-border"
                      >
                        <p className="font-medium">{paper.title}</p>
                        <p className="text-xs text-foreground-muted mt-0.5">
                          {paper.reason}
                        </p>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Research plan */}
              {analysis.plan && (
                <div>
                  <h4 className="text-sm text-foreground-muted mb-2 flex items-center gap-2">
                    <Calendar className="w-4 h-4" />
                    Research Plan
                  </h4>
                  <div className="space-y-2">
                    {analysis.plan.milestones.map((milestone, idx) => (
                      <div
                        key={idx}
                        className="flex items-start gap-3 text-sm"
                      >
                        <div className="w-6 h-6 rounded-full bg-foreground-muted/10 flex items-center justify-center flex-shrink-0 text-xs">
                          {idx + 1}
                        </div>
                        <div>
                          <p className="text-foreground">{milestone.title}</p>
                          <p className="text-xs text-foreground-muted">
                            Week {milestone.week}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Resource estimates */}
              {analysis.resources.length > 0 && (
                <div className="p-3 bg-background/50 rounded border border-border">
                  <h4 className="text-xs text-foreground-muted mb-2">Resources</h4>
                  <div className="flex flex-wrap gap-2 text-xs">
                    {analysis.resources.slice(0, 3).map((resource, idx) => (
                      <span
                        key={idx}
                        className={cn(
                          "px-2 py-1 rounded",
                          resource.isCritical
                            ? "bg-yellow-500/10 text-yellow-400"
                            : "bg-foreground-muted/10 text-foreground-muted"
                        )}
                      >
                        {resource.name}: {resource.estimate}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-border bg-background/50">
          {step === "input" && (
            <>
              <button
                onClick={onClose}
                className="px-4 py-2 text-sm text-foreground-muted hover:text-foreground"
              >
                Cancel
              </button>
              <button
                onClick={handleAnalyze}
                disabled={!validateGoal(goal) || isLoading}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 text-sm rounded-lg",
                  "bg-foreground-bright text-background hover:bg-white",
                  "disabled:opacity-50 disabled:cursor-not-allowed"
                )}
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Target className="w-4 h-4" />
                )}
                Generate Plan
              </button>
            </>
          )}

          {step === "analyzing" && (
            <div className="w-full text-center text-sm text-foreground-muted">
              This may take a few moments...
            </div>
          )}

          {step === "result" && (
            <>
              <button
                onClick={() => {
                  setStep("input");
                  setGoal("");
                  setAnalysis(null);
                  setPapers([]);
                }}
                className="px-4 py-2 text-sm text-foreground-muted hover:text-foreground"
              >
                New Goal
              </button>
              <button
                onClick={onClose}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 text-sm rounded-lg",
                  "bg-foreground-bright text-background hover:bg-white"
                )}
              >
                Done
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default AddCustomGoal;
