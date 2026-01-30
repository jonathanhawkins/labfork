"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import {
  X,
  Plus,
  Loader2,
  AlertCircle,
  Check,
  FileText,
  Link,
  Sparkles,
  ArrowRight,
} from "lucide-react";
import type { Paper, PaperInputDetection } from "@/lib/papers/types";
import { getSourceDisplayInfo } from "@/lib/papers/types";

export interface AddPaperDialogProps {
  /** Whether dialog is open */
  isOpen: boolean;
  /** Close handler */
  onClose: () => void;
  /** Paper added handler */
  onPaperAdded?: (paper: Paper) => void;
  /** Current domain slug for context */
  domainSlug?: string;
  /** Custom class name */
  className?: string;
}

type DialogStep = "input" | "preview" | "analyzing" | "result";

/**
 * AddPaperDialog - Dialog for adding papers to the queue
 */
export function AddPaperDialog({
  isOpen,
  onClose,
  onPaperAdded,
  domainSlug,
  className,
}: AddPaperDialogProps) {
  // State
  const [step, setStep] = useState<DialogStep>("input");
  const [input, setInput] = useState("");
  const [detection, setDetection] = useState<PaperInputDetection | null>(null);
  const [paper, setPaper] = useState<Paper | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
      setInput("");
      setDetection(null);
      setPaper(null);
      setError(null);
    }
  }, [isOpen]);

  // Detect input type with debouncing
  useEffect(() => {
    if (!input.trim()) {
      setDetection(null);
      return;
    }

    const timer = setTimeout(async () => {
      try {
        const response = await fetch(
          `/api/papers/fetch?input=${encodeURIComponent(input)}`
        );
        const data = await response.json();

        if (data.detection) {
          setDetection(data.detection);
          setError(null);
        } else if (data.error) {
          setDetection(null);
        }
      } catch {
        // Ignore detection errors
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [input]);

  // Fetch paper metadata
  const handleFetch = useCallback(async () => {
    if (!input.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/papers/fetch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input }),
      });

      const data = await response.json();

      if (!data.success) {
        setError(data.error || "Failed to fetch paper");
        return;
      }

      setPaper(data.paper);
      setStep("preview");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch paper");
    } finally {
      setIsLoading(false);
    }
  }, [input]);

  // Add paper to queue
  const handleAddToQueue = useCallback(async () => {
    if (!input.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/papers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input, domainSlug }),
      });

      const data = await response.json();

      if (!data.success) {
        setError(data.error || "Failed to add paper");
        return;
      }

      setPaper(data.paper);

      if (data.fromCache) {
        setStep("result");
      } else {
        setStep("preview");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add paper");
    } finally {
      setIsLoading(false);
    }
  }, [input, domainSlug]);

  // Analyze paper
  const handleAnalyze = useCallback(async () => {
    if (!paper) return;

    setStep("analyzing");
    setError(null);

    try {
      const response = await fetch("/api/papers/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paperId: paper.id, domainSlug }),
      });

      const data = await response.json();

      if (!data.success) {
        setError(data.error || "Analysis failed");
        setStep("preview");
        return;
      }

      setPaper(data.paper);
      setStep("result");

      if (onPaperAdded) {
        onPaperAdded(data.paper);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
      setStep("preview");
    }
  }, [paper, domainSlug, onPaperAdded]);

  // Handle keyboard
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (step === "input" && input.trim() && !isLoading) {
          handleAddToQueue();
        }
      }
      if (e.key === "Escape") {
        onClose();
      }
    },
    [step, input, isLoading, handleAddToQueue, onClose]
  );

  if (!isOpen) return null;

  const sourceInfo = detection
    ? getSourceDisplayInfo(detection.source)
    : null;

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
              <Plus className="w-5 h-5 text-foreground-muted" />
            </div>
            <div>
              <h2 className="text-lg font-normal text-foreground-bright">
                Add Paper
              </h2>
              <p className="text-xs text-foreground-muted">
                {step === "input" && "Enter arXiv ID, DOI, or URL"}
                {step === "preview" && "Review paper details"}
                {step === "analyzing" && "AI analyzing paper..."}
                {step === "result" && "Paper added to queue"}
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
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="e.g., 2401.12345 or https://arxiv.org/abs/2401.12345"
                  className={cn(
                    "w-full px-4 py-3 pr-12 text-sm rounded-lg",
                    "bg-background border",
                    "text-foreground placeholder:text-foreground-subtle",
                    "focus:outline-none focus:border-foreground-muted",
                    error ? "border-red-500/50" : "border-border"
                  )}
                />
                {detection && (
                  <div className="absolute right-3 top-1/2 -translate-y-1/2">
                    <span
                      className={cn(
                        "px-2 py-0.5 rounded text-xs",
                        sourceInfo?.bgColor,
                        sourceInfo?.color
                      )}
                    >
                      {sourceInfo?.label}
                    </span>
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

              {/* Detection info */}
              {detection && !error && (
                <div className="flex items-center gap-2 text-sm text-foreground-muted">
                  <Check className="w-4 h-4 text-green-400" />
                  Detected as {sourceInfo?.label} paper
                  {detection.confidence < 1 && (
                    <span className="text-foreground-subtle">
                      ({Math.round(detection.confidence * 100)}% confidence)
                    </span>
                  )}
                </div>
              )}

              {/* Examples */}
              <div className="text-xs text-foreground-subtle">
                <p className="mb-2">Supported formats:</p>
                <ul className="space-y-1 ml-4">
                  <li className="flex items-center gap-2">
                    <FileText className="w-3 h-3" />
                    arXiv ID: 2401.12345
                  </li>
                  <li className="flex items-center gap-2">
                    <Link className="w-3 h-3" />
                    arXiv URL: https://arxiv.org/abs/...
                  </li>
                  <li className="flex items-center gap-2">
                    <Link className="w-3 h-3" />
                    DOI: 10.1234/example
                  </li>
                </ul>
              </div>
            </div>
          )}

          {/* Preview step */}
          {step === "preview" && paper && (
            <div className="space-y-4">
              {/* Paper title */}
              <div>
                <h3 className="text-base font-normal text-foreground-bright">
                  {paper.metadata.title}
                </h3>
                <p className="text-sm text-foreground-muted mt-1">
                  {paper.metadata.authors
                    .slice(0, 3)
                    .map((a) => a.name)
                    .join(", ")}
                  {paper.metadata.authors.length > 3 && " et al."}
                </p>
              </div>

              {/* Abstract preview */}
              <div className="text-sm text-foreground line-clamp-4 bg-background/50 p-3 rounded-lg border border-border">
                {paper.metadata.abstract}
              </div>

              {/* Error message */}
              {error && (
                <div className="flex items-center gap-2 text-sm text-red-400">
                  <AlertCircle className="w-4 h-4" />
                  {error}
                </div>
              )}

              {/* Info */}
              <p className="text-xs text-foreground-muted">
                Click "Analyze" to assess relevance and generate task
                recommendations.
              </p>
            </div>
          )}

          {/* Analyzing step */}
          {step === "analyzing" && (
            <div className="flex flex-col items-center py-8 space-y-4">
              <div className="relative">
                <div className="w-16 h-16 rounded-full bg-foreground-bright/10 flex items-center justify-center">
                  <Sparkles className="w-8 h-8 text-foreground-bright animate-pulse" />
                </div>
                <div className="absolute inset-0 rounded-full border-2 border-foreground-bright/30 border-t-foreground-bright animate-spin" />
              </div>
              <div className="text-center">
                <p className="text-foreground-bright">Analyzing paper...</p>
                <p className="text-sm text-foreground-muted mt-1">
                  Evaluating relevance and generating task breakdown
                </p>
              </div>
            </div>
          )}

          {/* Result step */}
          {step === "result" && paper && (
            <div className="space-y-4">
              {/* Success message */}
              <div className="flex items-center gap-3 p-4 rounded-lg bg-green-500/10 border border-green-500/20">
                <Check className="w-5 h-5 text-green-400" />
                <div>
                  <p className="text-sm text-green-400">
                    Paper added to queue
                  </p>
                  <p className="text-xs text-foreground-muted mt-0.5">
                    {paper.analysis
                      ? `Relevance score: ${paper.analysis.relevanceScore}/100`
                      : "Ready for analysis"}
                  </p>
                </div>
              </div>

              {/* Analysis summary */}
              {paper.analysis && (
                <div className="space-y-2">
                  <h4 className="text-sm text-foreground-muted">
                    Analysis Summary
                  </h4>
                  <p className="text-sm text-foreground">
                    {paper.analysis.relevanceReason}
                  </p>
                  <div className="flex items-center gap-2 text-xs text-foreground-muted">
                    <span>Complexity: {paper.analysis.complexity}</span>
                    <span>|</span>
                    <span>
                      {paper.analysis.techniques.length} techniques identified
                    </span>
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
                onClick={handleAddToQueue}
                disabled={!input.trim() || isLoading}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 text-sm rounded-lg",
                  "bg-foreground-bright text-background hover:bg-white",
                  "disabled:opacity-50 disabled:cursor-not-allowed"
                )}
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Plus className="w-4 h-4" />
                )}
                Add Paper
              </button>
            </>
          )}

          {step === "preview" && (
            <>
              <button
                onClick={() => setStep("input")}
                className="px-4 py-2 text-sm text-foreground-muted hover:text-foreground"
              >
                Back
              </button>
              <button
                onClick={handleAnalyze}
                disabled={isLoading}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 text-sm rounded-lg",
                  "bg-foreground-bright text-background hover:bg-white",
                  "disabled:opacity-50 disabled:cursor-not-allowed"
                )}
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Sparkles className="w-4 h-4" />
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
                  setInput("");
                  setPaper(null);
                }}
                className="px-4 py-2 text-sm text-foreground-muted hover:text-foreground"
              >
                Add Another
              </button>
              <button
                onClick={onClose}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 text-sm rounded-lg",
                  "bg-foreground-bright text-background hover:bg-white"
                )}
              >
                Done
                <ArrowRight className="w-4 h-4" />
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default AddPaperDialog;
