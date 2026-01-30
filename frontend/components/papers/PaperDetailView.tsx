"use client";

import { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  FileText,
  Users,
  Calendar,
  ExternalLink,
  Check,
  X,
  Loader2,
  AlertCircle,
  Sparkles,
  Clock,
  Quote,
  Copy,
  CheckCircle,
  ChevronRight,
  Cpu,
  Database,
  Zap,
  BookOpen,
  Code,
  TestTube,
  ArrowRight,
  Link as LinkIcon,
} from "lucide-react";
import type { Paper } from "@/lib/papers/types";
import {
  getStatusDisplayInfo,
  getComplexityDisplayInfo,
  getRelevanceDisplayInfo,
  getSourceDisplayInfo,
  formatAuthors,
} from "@/lib/papers/types";
import {
  generateTasksFromPaper,
  estimateTotalEffort,
  type GeneratedTasks,
} from "@/lib/papers/task-generator";

export interface PaperDetailViewProps {
  /** Paper data */
  paper: Paper;
  /** Accept handler */
  onAccept?: (paperId: string) => Promise<void>;
  /** Reject handler */
  onReject?: (paperId: string) => Promise<void>;
  /** Analyze handler */
  onAnalyze?: (paperId: string) => Promise<void>;
  /** Close handler */
  onClose?: () => void;
  /** View tasks handler */
  onViewTasks?: (taskIds: string[]) => void;
  /** Domain slug for task generation */
  domainSlug?: string;
  /** Custom class name */
  className?: string;
}

/**
 * PaperDetailView - Full detail view for a paper with analysis
 */
export function PaperDetailView({
  paper,
  onAccept,
  onReject,
  onAnalyze,
  onClose,
  onViewTasks,
  domainSlug,
  className,
}: PaperDetailViewProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [actionType, setActionType] = useState<
    "accept" | "reject" | "analyze" | null
  >(null);
  const [showConfirm, setShowConfirm] = useState<"accept" | "reject" | null>(
    null
  );
  const [copiedCitation, setCopiedCitation] = useState(false);

  const statusInfo = getStatusDisplayInfo(paper.status);
  const sourceInfo = getSourceDisplayInfo(paper.metadata.source);
  const analysis = paper.analysis;
  const complexityInfo = analysis
    ? getComplexityDisplayInfo(analysis.complexity)
    : null;
  const relevanceInfo = analysis
    ? getRelevanceDisplayInfo(analysis.relevanceScore)
    : null;

  // Generate task preview
  const generatedTasks = analysis
    ? generateTasksFromPaper(paper, domainSlug)
    : null;
  const effort = analysis ? estimateTotalEffort(analysis) : null;

  // Format date
  const formatDate = (dateStr?: string) => {
    if (!dateStr) return null;
    try {
      return new Date(dateStr).toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
      });
    } catch {
      return null;
    }
  };

  // Generate citation
  const generateCitation = (): string => {
    const authors = paper.metadata.authors;
    const firstAuthor = authors.length > 0 ? authors[0].name : "Unknown";
    const year = paper.metadata.publishedDate
      ? new Date(paper.metadata.publishedDate).getFullYear()
      : "n.d.";
    const etAl = authors.length > 1 ? " et al." : "";

    return `${firstAuthor}${etAl} (${year}). ${paper.metadata.title}. ${paper.metadata.url}`;
  };

  // Copy citation to clipboard
  const handleCopyCitation = useCallback(async () => {
    const citation = generateCitation();
    try {
      await navigator.clipboard.writeText(citation);
      setCopiedCitation(true);
      setTimeout(() => setCopiedCitation(false), 2000);
    } catch (err) {
      console.error("Failed to copy citation:", err);
    }
  }, [paper]);

  // Handle accept
  const handleAccept = useCallback(async () => {
    if (!onAccept || isLoading) return;

    setIsLoading(true);
    setActionType("accept");

    try {
      await onAccept(paper.id);
      setShowConfirm(null);
    } catch (err) {
      console.error("Accept failed:", err);
    } finally {
      setIsLoading(false);
      setActionType(null);
    }
  }, [paper.id, onAccept, isLoading]);

  // Handle reject
  const handleReject = useCallback(async () => {
    if (!onReject || isLoading) return;

    setIsLoading(true);
    setActionType("reject");

    try {
      await onReject(paper.id);
      setShowConfirm(null);
    } catch (err) {
      console.error("Reject failed:", err);
    } finally {
      setIsLoading(false);
      setActionType(null);
    }
  }, [paper.id, onReject, isLoading]);

  // Handle analyze
  const handleAnalyze = useCallback(async () => {
    if (!onAnalyze || isLoading) return;

    setIsLoading(true);
    setActionType("analyze");

    try {
      await onAnalyze(paper.id);
    } catch (err) {
      console.error("Analyze failed:", err);
    } finally {
      setIsLoading(false);
      setActionType(null);
    }
  }, [paper.id, onAnalyze, isLoading]);

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="space-y-4">
        {/* Title and close */}
        <div className="flex items-start justify-between gap-4">
          <h1 className="text-xl font-normal text-foreground-bright leading-tight">
            {paper.metadata.title}
          </h1>
          {onClose && (
            <button
              onClick={onClose}
              className="flex-shrink-0 p-2 text-foreground-muted hover:text-foreground rounded-lg hover:bg-foreground-muted/10"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* Meta badges */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Source */}
          <span
            className={cn(
              "px-2 py-1 rounded text-xs",
              sourceInfo.bgColor,
              sourceInfo.color
            )}
          >
            {sourceInfo.label}
          </span>

          {/* Status */}
          <span
            className={cn(
              "px-2 py-1 rounded text-xs",
              statusInfo.bgColor,
              statusInfo.color
            )}
          >
            {statusInfo.label}
          </span>

          {/* Relevance */}
          {relevanceInfo && (
            <span
              className={cn(
                "px-2 py-1 rounded text-xs flex items-center gap-1",
                relevanceInfo.bgColor,
                relevanceInfo.color
              )}
            >
              <Sparkles className="w-3 h-3" />
              {analysis!.relevanceScore}/100
            </span>
          )}

          {/* Complexity */}
          {complexityInfo && (
            <span
              className={cn(
                "px-2 py-1 rounded text-xs",
                complexityInfo.bgColor,
                complexityInfo.color
              )}
            >
              {complexityInfo.label}
            </span>
          )}
        </div>

        {/* Authors and date */}
        <div className="flex flex-wrap items-center gap-4 text-sm text-foreground-muted">
          <span className="flex items-center gap-2">
            <Users className="w-4 h-4" />
            {formatAuthors(paper.metadata.authors, 5)}
          </span>

          {paper.metadata.publishedDate && (
            <span className="flex items-center gap-2">
              <Calendar className="w-4 h-4" />
              {formatDate(paper.metadata.publishedDate)}
            </span>
          )}

          {paper.metadata.citationCount !== undefined &&
            paper.metadata.citationCount > 0 && (
              <span className="flex items-center gap-2">
                <Quote className="w-4 h-4" />
                {paper.metadata.citationCount} citations
              </span>
            )}
        </div>

        {/* Links */}
        <div className="flex flex-wrap items-center gap-3">
          <a
            href={paper.metadata.url}
            target="_blank"
            rel="noopener noreferrer"
            className={cn(
              "flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg",
              "bg-foreground-muted/10 text-foreground-muted hover:text-foreground"
            )}
          >
            <ExternalLink className="w-4 h-4" />
            View Paper
          </a>

          {paper.metadata.pdfUrl && (
            <a
              href={paper.metadata.pdfUrl}
              target="_blank"
              rel="noopener noreferrer"
              className={cn(
                "flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg",
                "bg-foreground-muted/10 text-foreground-muted hover:text-foreground"
              )}
            >
              <FileText className="w-4 h-4" />
              PDF
            </a>
          )}

          <button
            onClick={handleCopyCitation}
            className={cn(
              "flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg",
              "bg-foreground-muted/10 text-foreground-muted hover:text-foreground"
            )}
          >
            {copiedCitation ? (
              <>
                <CheckCircle className="w-4 h-4 text-green-400" />
                Copied!
              </>
            ) : (
              <>
                <Copy className="w-4 h-4" />
                Copy Citation
              </>
            )}
          </button>
        </div>
      </div>

      {/* Abstract */}
      <div className="p-4 rounded-lg bg-background-card border border-border">
        <h2 className="text-sm text-foreground-muted mb-2 flex items-center gap-2">
          <BookOpen className="w-4 h-4" />
          Abstract
        </h2>
        <p className="text-sm text-foreground leading-relaxed">
          {paper.metadata.abstract}
        </p>
      </div>

      {/* Analysis section (if analyzed) */}
      {analysis && (
        <>
          {/* Relevance Analysis */}
          <div className="p-4 rounded-lg bg-background-card border border-border">
            <h2 className="text-sm text-foreground-muted mb-2 flex items-center gap-2">
              <Sparkles className="w-4 h-4" />
              Relevance Analysis
            </h2>
            <p className="text-sm text-foreground">{analysis.relevanceReason}</p>
          </div>

          {/* Key Contribution */}
          <div className="p-4 rounded-lg bg-background-card border border-border">
            <h2 className="text-sm text-foreground-muted mb-2 flex items-center gap-2">
              <Zap className="w-4 h-4" />
              Key Contribution
            </h2>
            <p className="text-sm text-foreground">{analysis.novelty}</p>
          </div>

          {/* Techniques */}
          {analysis.techniques.length > 0 && (
            <div className="p-4 rounded-lg bg-background-card border border-border">
              <h2 className="text-sm text-foreground-muted mb-3 flex items-center gap-2">
                <Cpu className="w-4 h-4" />
                Techniques ({analysis.techniques.length})
              </h2>
              <div className="space-y-2">
                {analysis.techniques.map((tech, i) => (
                  <div
                    key={i}
                    className={cn(
                      "p-3 rounded-lg border",
                      tech.isMainContribution
                        ? "bg-foreground-bright/5 border-foreground-bright/20"
                        : "bg-background border-border"
                    )}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span
                        className={cn(
                          "text-sm font-medium",
                          tech.isMainContribution
                            ? "text-foreground-bright"
                            : "text-foreground"
                        )}
                      >
                        {tech.name}
                      </span>
                      {tech.isMainContribution && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] bg-foreground-bright/20 text-foreground-bright">
                          Main
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-foreground-muted">
                      {tech.description}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Resources */}
          {analysis.resources.length > 0 && (
            <div className="p-4 rounded-lg bg-background-card border border-border">
              <h2 className="text-sm text-foreground-muted mb-3 flex items-center gap-2">
                <Database className="w-4 h-4" />
                Required Resources
              </h2>
              <div className="grid gap-2">
                {analysis.resources.map((resource, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between p-2 rounded bg-background border border-border"
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className={cn(
                          "w-2 h-2 rounded-full",
                          resource.required ? "bg-yellow-400" : "bg-gray-400"
                        )}
                      />
                      <span className="text-sm text-foreground">
                        {resource.name}
                      </span>
                    </div>
                    <span className="text-xs text-foreground-muted">
                      {resource.type}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Implementation Complexity */}
          <div className="p-4 rounded-lg bg-background-card border border-border">
            <h2 className="text-sm text-foreground-muted mb-2 flex items-center gap-2">
              <Clock className="w-4 h-4" />
              Implementation Complexity
            </h2>
            <div className="flex items-center gap-3 mb-2">
              {complexityInfo && (
                <span
                  className={cn(
                    "px-2 py-1 rounded text-xs",
                    complexityInfo.bgColor,
                    complexityInfo.color
                  )}
                >
                  {complexityInfo.label}
                </span>
              )}
              {effort && (
                <span className="text-sm text-foreground-muted">
                  ~{effort.hours} hours ({effort.days} days)
                </span>
              )}
            </div>
            <p className="text-sm text-foreground">{analysis.complexityReason}</p>
          </div>

          {/* Task Preview */}
          {generatedTasks && (
            <div className="p-4 rounded-lg bg-background-card border border-border">
              <h2 className="text-sm text-foreground-muted mb-3 flex items-center gap-2">
                <ChevronRight className="w-4 h-4" />
                Proposed Tasks (3)
              </h2>

              <div className="space-y-3">
                {/* Research Task */}
                <div className="flex items-start gap-3 p-3 rounded-lg bg-background border border-border">
                  <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center flex-shrink-0">
                    <BookOpen className="w-4 h-4 text-blue-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm text-foreground">
                      {generatedTasks.research.subject}
                    </h3>
                    <p className="text-xs text-foreground-muted mt-1 line-clamp-2">
                      {analysis.taskBreakdown.research.description
                        .split("\n")[0]
                        .substring(0, 100)}
                      ...
                    </p>
                    <span className="text-[10px] text-foreground-subtle mt-1 inline-block">
                      ~{analysis.taskBreakdown.research.estimatedHours || 4}h
                    </span>
                  </div>
                </div>

                {/* Arrow */}
                <div className="flex items-center justify-center">
                  <ArrowRight className="w-4 h-4 text-foreground-subtle" />
                </div>

                {/* Implementation Task */}
                <div className="flex items-start gap-3 p-3 rounded-lg bg-background border border-border">
                  <div className="w-8 h-8 rounded-lg bg-purple-500/10 flex items-center justify-center flex-shrink-0">
                    <Code className="w-4 h-4 text-purple-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm text-foreground">
                      {generatedTasks.implementation.subject}
                    </h3>
                    <p className="text-xs text-foreground-muted mt-1 line-clamp-2">
                      {analysis.taskBreakdown.implementation.description
                        .split("\n")[0]
                        .substring(0, 100)}
                      ...
                    </p>
                    <span className="text-[10px] text-foreground-subtle mt-1 inline-block">
                      ~{analysis.taskBreakdown.implementation.estimatedHours || 16}h
                    </span>
                  </div>
                </div>

                {/* Arrow */}
                <div className="flex items-center justify-center">
                  <ArrowRight className="w-4 h-4 text-foreground-subtle" />
                </div>

                {/* Evaluation Task */}
                <div className="flex items-start gap-3 p-3 rounded-lg bg-background border border-border">
                  <div className="w-8 h-8 rounded-lg bg-green-500/10 flex items-center justify-center flex-shrink-0">
                    <TestTube className="w-4 h-4 text-green-400" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm text-foreground">
                      {generatedTasks.evaluation.subject}
                    </h3>
                    <p className="text-xs text-foreground-muted mt-1 line-clamp-2">
                      {analysis.taskBreakdown.evaluation.description
                        .split("\n")[0]
                        .substring(0, 100)}
                      ...
                    </p>
                    <span className="text-[10px] text-foreground-subtle mt-1 inline-block">
                      ~{analysis.taskBreakdown.evaluation.estimatedHours || 8}h
                    </span>
                  </div>
                </div>
              </div>

              {/* Effort summary */}
              {effort && (
                <div className="mt-4 pt-3 border-t border-border">
                  <p className="text-xs text-foreground-muted">
                    Total estimated effort: <strong>{effort.hours} hours</strong>{" "}
                    (~{effort.days} days) - {effort.description}
                  </p>
                </div>
              )}
            </div>
          )}
        </>
      )}

      {/* Error message */}
      {paper.status === "error" && paper.error && (
        <div className="flex items-center gap-3 p-4 rounded-lg bg-red-500/10 border border-red-500/20">
          <AlertCircle className="w-5 h-5 text-red-400" />
          <div>
            <p className="text-sm text-red-400">Error</p>
            <p className="text-xs text-foreground-muted mt-0.5">{paper.error}</p>
          </div>
        </div>
      )}

      {/* Implemented status with task links */}
      {paper.status === "implemented" && paper.taskIds && paper.taskIds.length > 0 && (
        <div className="flex items-center gap-3 p-4 rounded-lg bg-green-500/10 border border-green-500/20">
          <CheckCircle className="w-5 h-5 text-green-400" />
          <div className="flex-1">
            <p className="text-sm text-green-400">
              Tasks Created ({paper.taskIds.length})
            </p>
            <p className="text-xs text-foreground-muted mt-0.5">
              Paper has been accepted and tasks have been generated.
            </p>
          </div>
          {onViewTasks && (
            <button
              onClick={() => onViewTasks(paper.taskIds!)}
              className={cn(
                "flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg",
                "bg-green-500/10 text-green-400 hover:bg-green-500/20"
              )}
            >
              <LinkIcon className="w-4 h-4" />
              View Tasks
            </button>
          )}
        </div>
      )}

      {/* Confirmation dialogs */}
      {showConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="w-full max-w-sm mx-4 p-6 bg-background-card border border-border rounded-xl">
            <h3 className="text-lg font-normal text-foreground-bright mb-2">
              {showConfirm === "accept" ? "Accept Paper?" : "Reject Paper?"}
            </h3>
            <p className="text-sm text-foreground-muted mb-4">
              {showConfirm === "accept"
                ? "This will create 3 tasks from the paper analysis (research, implementation, evaluation)."
                : "This paper will be marked as rejected and no tasks will be created."}
            </p>
            <div className="flex items-center justify-end gap-3">
              <button
                onClick={() => setShowConfirm(null)}
                disabled={isLoading}
                className="px-4 py-2 text-sm text-foreground-muted hover:text-foreground disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={showConfirm === "accept" ? handleAccept : handleReject}
                disabled={isLoading}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 text-sm rounded-lg",
                  showConfirm === "accept"
                    ? "bg-green-500/10 text-green-400 hover:bg-green-500/20"
                    : "bg-red-500/10 text-red-400 hover:bg-red-500/20",
                  "disabled:opacity-50 disabled:cursor-not-allowed"
                )}
              >
                {isLoading && actionType === showConfirm && (
                  <Loader2 className="w-4 h-4 animate-spin" />
                )}
                {showConfirm === "accept" ? "Accept & Create Tasks" : "Reject"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex items-center justify-between pt-4 border-t border-border">
        {/* Left side - back/close */}
        {onClose && (
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-foreground-muted hover:text-foreground"
          >
            Back to Queue
          </button>
        )}

        {/* Right side - actions */}
        <div className="flex items-center gap-3">
          {/* Analyze button (for fetched papers) */}
          {(paper.status === "fetched" || paper.status === "error") &&
            onAnalyze && (
              <button
                onClick={handleAnalyze}
                disabled={isLoading}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 text-sm rounded-lg",
                  "bg-foreground-bright/10 text-foreground-bright hover:bg-foreground-bright/20",
                  "disabled:opacity-50 disabled:cursor-not-allowed"
                )}
              >
                {isLoading && actionType === "analyze" ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Sparkles className="w-4 h-4" />
                )}
                {paper.status === "error" ? "Retry Analysis" : "Analyze"}
              </button>
            )}

          {/* Accept/Reject buttons (for analyzed papers) */}
          {paper.status === "analyzed" && (
            <>
              {onReject && (
                <button
                  onClick={() => setShowConfirm("reject")}
                  disabled={isLoading}
                  className={cn(
                    "flex items-center gap-2 px-4 py-2 text-sm rounded-lg",
                    "bg-red-500/10 text-red-400 hover:bg-red-500/20",
                    "disabled:opacity-50 disabled:cursor-not-allowed"
                  )}
                >
                  <X className="w-4 h-4" />
                  Reject
                </button>
              )}

              {onAccept && (
                <button
                  onClick={() => setShowConfirm("accept")}
                  disabled={isLoading}
                  className={cn(
                    "flex items-center gap-2 px-4 py-2 text-sm rounded-lg",
                    "bg-green-500/10 text-green-400 hover:bg-green-500/20",
                    "disabled:opacity-50 disabled:cursor-not-allowed"
                  )}
                >
                  <Check className="w-4 h-4" />
                  Accept
                </button>
              )}
            </>
          )}

          {/* Analyzing status */}
          {paper.status === "analyzing" && (
            <span className="flex items-center gap-2 text-sm text-yellow-400">
              <Loader2 className="w-4 h-4 animate-spin" />
              Analyzing...
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

export default PaperDetailView;
