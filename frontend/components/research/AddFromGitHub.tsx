"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import {
  X,
  Github,
  Loader2,
  AlertCircle,
  Check,
  GitBranch,
  Code,
  Lightbulb,
  Layers,
} from "lucide-react";
import type { Paper } from "@/lib/papers/types";
import type { RepoAnalysis } from "@/lib/research/github-analyzer";

export interface AddFromGitHubProps {
  /** Whether dialog is open */
  isOpen: boolean;
  /** Close handler */
  onClose: () => void;
  /** Analysis complete handler */
  onAnalysisComplete?: (analysis: RepoAnalysis, paper?: Paper) => void;
  /** Custom class name */
  className?: string;
}

type DialogStep = "input" | "analyzing" | "result";

/**
 * AddFromGitHub - Dialog for analyzing GitHub repositories
 */
export function AddFromGitHub({
  isOpen,
  onClose,
  onAnalysisComplete,
  className,
}: AddFromGitHubProps) {
  // State
  const [step, setStep] = useState<DialogStep>("input");
  const [repoUrl, setRepoUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<RepoAnalysis | null>(null);
  const [paper, setPaper] = useState<Paper | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);

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
      setRepoUrl("");
      setError(null);
      setAnalysis(null);
      setPaper(null);
    }
  }, [isOpen]);

  // Validate repo URL
  const validateRepoUrl = useCallback((url: string): boolean => {
    const trimmed = url.trim();
    // GitHub URL pattern
    if (/github\.com\/[^/]+\/[^/]+/i.test(trimmed)) {
      return true;
    }
    // owner/repo pattern
    if (/^[^/]+\/[^/]+$/.test(trimmed)) {
      return true;
    }
    return false;
  }, []);

  // Analyze repository
  const handleAnalyze = useCallback(async () => {
    if (!repoUrl.trim()) return;

    setIsLoading(true);
    setError(null);
    setStep("analyzing");

    try {
      const response = await fetch("/api/research/github", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repoUrl }),
      });

      const data = await response.json();

      if (!data.success) {
        setError(data.error || "Failed to analyze repository");
        setStep("input");
        return;
      }

      setAnalysis(data.analysis);
      if (data.paper) {
        setPaper(data.paper);
      }
      setStep("result");

      if (onAnalysisComplete) {
        onAnalysisComplete(data.analysis, data.paper);
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to analyze repository"
      );
      setStep("input");
    } finally {
      setIsLoading(false);
    }
  }, [repoUrl, onAnalysisComplete]);

  // Handle keyboard
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (step === "input" && validateRepoUrl(repoUrl) && !isLoading) {
          handleAnalyze();
        }
      }
      if (e.key === "Escape") {
        onClose();
      }
    },
    [step, repoUrl, isLoading, validateRepoUrl, handleAnalyze, onClose]
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
              <Github className="w-5 h-5 text-foreground-muted" />
            </div>
            <div>
              <h2 className="text-lg font-normal text-foreground-bright">
                Analyze GitHub Repository
              </h2>
              <p className="text-xs text-foreground-muted">
                {step === "input" && "Enter repository URL or path"}
                {step === "analyzing" && "Analyzing repository..."}
                {step === "result" && "Analysis complete"}
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
                <input
                  ref={inputRef}
                  type="text"
                  value={repoUrl}
                  onChange={(e) => setRepoUrl(e.target.value)}
                  placeholder="e.g., owner/repo or https://github.com/owner/repo"
                  className={cn(
                    "w-full px-4 py-3 pr-12 text-sm rounded-lg",
                    "bg-background border",
                    "text-foreground placeholder:text-foreground-subtle",
                    "focus:outline-none focus:border-foreground-muted",
                    error ? "border-red-500/50" : "border-border"
                  )}
                />
                {validateRepoUrl(repoUrl) && (
                  <div className="absolute right-3 top-1/2 -translate-y-1/2">
                    <Check className="w-4 h-4 text-green-400" />
                  </div>
                )}
              </div>

              {/* Error message */}
              {error && (
                <div className="flex items-center gap-2 text-sm text-red-400">
                  <AlertCircle className="w-4 h-4" />
                  {error}
                </div>
              )}

              {/* What we analyze */}
              <div className="text-xs text-foreground-subtle">
                <p className="mb-2">We will analyze:</p>
                <ul className="space-y-1 ml-4">
                  <li className="flex items-center gap-2">
                    <Code className="w-3 h-3" />
                    ML framework detection (PyTorch, TensorFlow, etc.)
                  </li>
                  <li className="flex items-center gap-2">
                    <Layers className="w-3 h-3" />
                    Architecture patterns (Transformer, CNN, etc.)
                  </li>
                  <li className="flex items-center gap-2">
                    <GitBranch className="w-3 h-3" />
                    Training techniques and methods
                  </li>
                  <li className="flex items-center gap-2">
                    <Lightbulb className="w-3 h-3" />
                    Generate research tasks
                  </li>
                </ul>
              </div>
            </div>
          )}

          {/* Analyzing step */}
          {step === "analyzing" && (
            <div className="flex flex-col items-center py-8 space-y-4">
              <div className="relative">
                <div className="w-16 h-16 rounded-full bg-foreground-bright/10 flex items-center justify-center">
                  <Github className="w-8 h-8 text-foreground-bright animate-pulse" />
                </div>
                <div className="absolute inset-0 rounded-full border-2 border-foreground-bright/30 border-t-foreground-bright animate-spin" />
              </div>
              <div className="text-center">
                <p className="text-foreground-bright">
                  Analyzing repository...
                </p>
                <p className="text-sm text-foreground-muted mt-1">
                  Reading README and detecting patterns
                </p>
              </div>
            </div>
          )}

          {/* Result step */}
          {step === "result" && analysis && (
            <div className="space-y-4">
              {/* Success message */}
              <div className="flex items-center gap-3 p-4 rounded-lg bg-green-500/10 border border-green-500/20">
                <Check className="w-5 h-5 text-green-400" />
                <div>
                  <p className="text-sm text-green-400">Analysis complete</p>
                  <p className="text-xs text-foreground-muted mt-0.5">
                    {analysis.name}
                  </p>
                </div>
              </div>

              {/* Analysis details */}
              <div className="space-y-3">
                {/* Framework */}
                {analysis.framework && (
                  <div className="flex items-center gap-2">
                    <Code className="w-4 h-4 text-foreground-muted" />
                    <span className="text-sm text-foreground-muted">
                      Framework:
                    </span>
                    <span className="text-sm text-foreground-bright capitalize">
                      {analysis.framework}
                    </span>
                  </div>
                )}

                {/* Architectures */}
                {analysis.architectures.length > 0 && (
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-foreground-muted" />
                    <span className="text-sm text-foreground-muted">
                      Architectures:
                    </span>
                    <div className="flex flex-wrap gap-1">
                      {analysis.architectures.map((arch) => (
                        <span
                          key={arch}
                          className="px-2 py-0.5 text-xs bg-foreground-muted/10 rounded"
                        >
                          {arch}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Techniques */}
                {analysis.techniques.length > 0 && (
                  <div className="flex items-center gap-2">
                    <GitBranch className="w-4 h-4 text-foreground-muted" />
                    <span className="text-sm text-foreground-muted">
                      Techniques:
                    </span>
                    <div className="flex flex-wrap gap-1">
                      {analysis.techniques.slice(0, 5).map((tech) => (
                        <span
                          key={tech.name}
                          className="px-2 py-0.5 text-xs bg-foreground-muted/10 rounded"
                        >
                          {tech.name}
                        </span>
                      ))}
                      {analysis.techniques.length > 5 && (
                        <span className="text-xs text-foreground-muted">
                          +{analysis.techniques.length - 5} more
                        </span>
                      )}
                    </div>
                  </div>
                )}

                {/* Generated tasks */}
                {analysis.suggestedTasks.length > 0 && (
                  <div className="mt-4">
                    <h4 className="text-sm text-foreground-muted mb-2">
                      Suggested Tasks
                    </h4>
                    <ul className="space-y-1 text-sm text-foreground">
                      {analysis.suggestedTasks.slice(0, 3).map((task, idx) => (
                        <li
                          key={idx}
                          className="flex items-start gap-2"
                        >
                          <Lightbulb className="w-3 h-3 mt-1 text-yellow-400" />
                          {task.description}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
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
                disabled={!validateRepoUrl(repoUrl) || isLoading}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 text-sm rounded-lg",
                  "bg-foreground-bright text-background hover:bg-white",
                  "disabled:opacity-50 disabled:cursor-not-allowed"
                )}
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Github className="w-4 h-4" />
                )}
                Analyze
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
                  setRepoUrl("");
                  setAnalysis(null);
                  setPaper(null);
                }}
                className="px-4 py-2 text-sm text-foreground-muted hover:text-foreground"
              >
                Analyze Another
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

export default AddFromGitHub;
