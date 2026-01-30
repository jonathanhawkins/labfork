"use client";

import { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  Sparkles,
  Loader2,
  Target,
  Tag,
  Clock,
  FileText,
  BookOpen,
  ChevronDown,
  ChevronUp,
  Check,
  AlertTriangle,
  Info,
  Lightbulb,
  ListTodo,
} from "lucide-react";
import type { ResearchGoal, InitialTask, RecommendedPaper } from "@/lib/lab-wizard/types";

export interface GoalAnalysisResult {
  suggestedDomain: string;
  domainName: string;
  domainReason?: string;
  arxivCategories: string[];
  keywords: string[];
  reasoning?: string;
  suggestedTasks?: InitialTask[];
  initialTasks?: InitialTask[];
  suggestedPapers?: RecommendedPaper[];
  recommendedPapers?: RecommendedPaper[];
  estimatedTimeline?: string;
  estimatedHours?: number;
  confidenceScore?: number;
  confidence?: number;
}

export interface WizardStepResearchProps {
  /** Current research goal */
  goal: ResearchGoal;
  /** Called when research goal changes */
  onGoalChange: (goal: ResearchGoal) => void;
  /** Analysis result from AI */
  analysis?: GoalAnalysisResult | null;
  /** Called when analysis is updated */
  onAnalysisChange?: (analysis: GoalAnalysisResult | null) => void;
  /** Selected domain slug (for context) */
  selectedDomain?: string;
  /** Hardware VRAM (for context) */
  hardwareVram?: number;
  /** Custom class name */
  className?: string;
}

/**
 * WizardStepResearch - Research goal and AI analysis step
 */
export function WizardStepResearch({
  goal,
  onGoalChange,
  analysis,
  onAnalysisChange,
  selectedDomain,
  hardwareVram,
  className,
}: WizardStepResearchProps) {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [showTasks, setShowTasks] = useState(true);
  const [showPapers, setShowPapers] = useState(false);
  const [selectedTasks, setSelectedTasks] = useState<Set<number>>(new Set());

  // Helper to get tasks from analysis (handles both field names)
  const getTasks = (a: GoalAnalysisResult | null | undefined): InitialTask[] => {
    if (!a) return [];
    return a.suggestedTasks || a.initialTasks || [];
  };

  // Helper to get papers from analysis (handles both field names)
  const getPapers = (a: GoalAnalysisResult | null | undefined): RecommendedPaper[] => {
    if (!a) return [];
    return a.suggestedPapers || a.recommendedPapers || [];
  };

  // Helper to get reasoning from analysis
  const getReasoning = (a: GoalAnalysisResult | null | undefined): string => {
    if (!a) return "";
    return a.reasoning || a.domainReason || "";
  };

  // Helper to get estimated timeline
  const getTimeline = (a: GoalAnalysisResult | null | undefined): string => {
    if (!a) return "";
    if (a.estimatedTimeline) return a.estimatedTimeline;
    if (a.estimatedHours) {
      const hours = a.estimatedHours;
      if (hours <= 8) return "About 1 day";
      if (hours <= 40) return `About ${Math.ceil(hours / 8)} days`;
      return `About ${Math.ceil(hours / 40)} weeks`;
    }
    return "";
  };

  // Analyze goal with AI
  const analyzeGoal = useCallback(async () => {
    if (!goal.description?.trim()) {
      setAnalyzeError("Please describe your research goal first");
      return;
    }

    setIsAnalyzing(true);
    setAnalyzeError(null);

    try {
      const response = await fetch("/api/lab/analyze-goal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          goalText: goal.description,
          preferredDomain: selectedDomain,
          hardwareVram,
          generateTasks: true,
        }),
      });

      const data = await response.json();

      if (!data.success) {
        setAnalyzeError(data.error || "Analysis failed");
        return;
      }

      const result = data.analysis as GoalAnalysisResult;
      onAnalysisChange?.(result);

      // Select all suggested tasks by default
      const tasks = getTasks(result);
      if (tasks.length > 0) {
        setSelectedTasks(new Set(tasks.map((_, i) => i)));
      }

      // Update goal with suggested keywords
      if (result.keywords?.length) {
        onGoalChange({
          ...goal,
          keywords: result.keywords,
        });
      }
    } catch (error) {
      setAnalyzeError(error instanceof Error ? error.message : "Analysis failed");
    } finally {
      setIsAnalyzing(false);
    }
  }, [goal, selectedDomain, hardwareVram, onGoalChange, onAnalysisChange]);

  // Toggle task selection
  const toggleTask = (index: number) => {
    const newSelected = new Set(selectedTasks);
    if (newSelected.has(index)) {
      newSelected.delete(index);
    } else {
      newSelected.add(index);
    }
    setSelectedTasks(newSelected);

    // Update goal with selected tasks
    const allTasks = getTasks(analysis);
    if (allTasks.length > 0) {
      const tasks = allTasks.filter((_, i) => newSelected.has(i));
      onGoalChange({
        ...goal,
        initialTasks: tasks,
      });
    }
  };

  // Get priority color
  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "high":
        return "text-red-400 bg-red-500/10";
      case "medium":
        return "text-yellow-400 bg-yellow-500/10";
      case "low":
        return "text-green-400 bg-green-500/10";
      default:
        return "text-foreground-muted bg-foreground-muted/10";
    }
  };

  // Get task type color
  const getTypeColor = (type: string) => {
    switch (type) {
      case "research":
        return "text-blue-400";
      case "implementation":
        return "text-purple-400";
      case "evaluation":
        return "text-green-400";
      case "documentation":
        return "text-yellow-400";
      default:
        return "text-foreground-muted";
    }
  };

  return (
    <div className={cn("space-y-6", className)}>
      {/* Info banner */}
      <div className="flex items-start gap-3 p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
        <Info className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
        <div>
          <p className="text-sm text-foreground">
            Describe your research goal and let AI suggest initial tasks, keywords, and relevant papers.
          </p>
        </div>
      </div>

      {/* Goal input */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <label className="text-sm text-foreground-muted">
            What do you want to research?
          </label>
          <span className="text-xs text-foreground-subtle">
            {goal.description?.length || 0} / 500
          </span>
        </div>
        <textarea
          value={goal.description || ""}
          onChange={(e) =>
            onGoalChange({
              ...goal,
              description: e.target.value.slice(0, 500),
            })
          }
          placeholder="Example: I want to create an expressive voice cloning system that can control emotion intensity and speaking style. The goal is to generate natural-sounding speech with precise prosody control for audiobook narration."
          rows={4}
          className={cn(
            "w-full px-4 py-3 rounded-lg text-sm resize-none",
            "bg-background-card border border-border",
            "text-foreground placeholder-foreground-subtle",
            "focus:outline-none focus:ring-2 focus:ring-foreground-bright/50"
          )}
        />
      </div>

      {/* Keywords input */}
      <div className="space-y-2">
        <label className="text-sm text-foreground-muted">
          Keywords (comma-separated)
        </label>
        <input
          type="text"
          value={goal.keywords?.join(", ") || ""}
          onChange={(e) =>
            onGoalChange({
              ...goal,
              keywords: e.target.value
                .split(",")
                .map((k) => k.trim())
                .filter(Boolean),
            })
          }
          placeholder="TTS, prosody, emotion, voice cloning"
          className={cn(
            "w-full px-4 py-2 rounded-lg text-sm",
            "bg-background-card border border-border",
            "text-foreground placeholder-foreground-subtle",
            "focus:outline-none focus:ring-2 focus:ring-foreground-bright/50"
          )}
        />
      </div>

      {/* Analyze button */}
      <div className="flex items-center gap-4">
        <button
          onClick={analyzeGoal}
          disabled={isAnalyzing || !goal.description?.trim()}
          className={cn(
            "flex items-center gap-2 px-5 py-2.5 text-sm rounded-lg",
            "bg-gradient-to-r from-purple-500 to-blue-500 text-white",
            "hover:from-purple-600 hover:to-blue-600",
            "transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          )}
        >
          {isAnalyzing ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Sparkles className="w-4 h-4" />
          )}
          {isAnalyzing ? "Analyzing..." : "Analyze with AI"}
        </button>

        {analyzeError && (
          <div className="flex items-center gap-2 text-red-400 text-sm">
            <AlertTriangle className="w-4 h-4" />
            {analyzeError}
          </div>
        )}
      </div>

      {/* Analysis results */}
      {analysis && (
        <div className="space-y-4 pt-4 border-t border-border">
          {/* Analysis summary */}
          <div className="p-4 rounded-lg bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/20">
            <div className="flex items-start gap-3">
              <Lightbulb className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-foreground">
                  AI Analysis
                </h4>
                <p className="text-sm text-foreground-muted">
                  {getReasoning(analysis)}
                </p>
                <div className="flex flex-wrap gap-2 mt-2">
                  {analysis.arxivCategories?.map((cat) => (
                    <span
                      key={cat}
                      className="px-2 py-0.5 rounded text-xs bg-blue-500/20 text-blue-400"
                    >
                      {cat}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Timeline estimate */}
          {getTimeline(analysis) && (
            <div className="flex items-center gap-3 p-3 rounded-lg bg-background-card">
              <Clock className="w-5 h-5 text-foreground-muted" />
              <div>
                <h4 className="text-sm text-foreground">Estimated Timeline</h4>
                <p className="text-xs text-foreground-muted">
                  {getTimeline(analysis)}
                </p>
              </div>
            </div>
          )}

          {/* Suggested tasks */}
          {getTasks(analysis).length > 0 && (
            <div className="space-y-3">
              <button
                onClick={() => setShowTasks(!showTasks)}
                className="flex items-center gap-2 w-full text-left"
              >
                <ListTodo className="w-5 h-5 text-foreground-muted" />
                <h4 className="text-sm font-medium text-foreground flex-1">
                  Suggested Tasks ({getTasks(analysis).length})
                </h4>
                {showTasks ? (
                  <ChevronUp className="w-4 h-4 text-foreground-muted" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-foreground-muted" />
                )}
              </button>

              {showTasks && (
                <div className="space-y-2">
                  {getTasks(analysis).map((task, index) => (
                    <div
                      key={index}
                      onClick={() => toggleTask(index)}
                      className={cn(
                        "flex items-start gap-3 p-3 rounded-lg cursor-pointer transition-colors",
                        selectedTasks.has(index)
                          ? "bg-foreground-bright/5 border border-foreground-bright/20"
                          : "bg-background-card border border-border hover:border-foreground-muted"
                      )}
                    >
                      <div
                        className={cn(
                          "w-5 h-5 rounded border flex items-center justify-center flex-shrink-0 mt-0.5",
                          selectedTasks.has(index)
                            ? "bg-foreground-bright border-foreground-bright"
                            : "border-foreground-muted"
                        )}
                      >
                        {selectedTasks.has(index) && (
                          <Check className="w-3 h-3 text-background" />
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <h5 className="text-sm text-foreground">
                            {task.subject}
                          </h5>
                          <span
                            className={cn(
                              "px-1.5 py-0.5 rounded text-[10px]",
                              getPriorityColor(task.priority || "medium")
                            )}
                          >
                            {task.priority || "medium"}
                          </span>
                          <span
                            className={cn(
                              "text-[10px]",
                              getTypeColor(task.type)
                            )}
                          >
                            {task.type}
                          </span>
                        </div>
                        <p className="text-xs text-foreground-muted mt-1 line-clamp-2">
                          {task.description}
                        </p>
                        {task.estimatedHours && (
                          <p className="text-xs text-foreground-subtle mt-1">
                            Est. {task.estimatedHours}h
                          </p>
                        )}
                      </div>
                    </div>
                  ))}

                  <p className="text-xs text-foreground-subtle">
                    {selectedTasks.size} of {getTasks(analysis).length} tasks
                    selected
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Suggested papers */}
          {getPapers(analysis).length > 0 && (
            <div className="space-y-3">
              <button
                onClick={() => setShowPapers(!showPapers)}
                className="flex items-center gap-2 w-full text-left"
              >
                <BookOpen className="w-5 h-5 text-foreground-muted" />
                <h4 className="text-sm font-medium text-foreground flex-1">
                  Suggested Papers ({getPapers(analysis).length})
                </h4>
                {showPapers ? (
                  <ChevronUp className="w-4 h-4 text-foreground-muted" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-foreground-muted" />
                )}
              </button>

              {showPapers && (
                <div className="space-y-2">
                  {getPapers(analysis).map((paper, index) => (
                    <div
                      key={index}
                      className="flex items-start gap-3 p-3 rounded-lg bg-background-card border border-border"
                    >
                      <FileText className="w-5 h-5 text-foreground-muted flex-shrink-0 mt-0.5" />
                      <div className="flex-1 min-w-0">
                        <h5 className="text-sm text-foreground line-clamp-1">
                          {paper.title}
                        </h5>
                        {paper.authors && (
                          <p className="text-xs text-foreground-muted mt-0.5">
                            {paper.authors}
                          </p>
                        )}
                        {paper.reason && (
                          <p className="text-xs text-foreground-subtle mt-1">
                            {paper.reason}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Skip analysis option */}
      {!analysis && (
        <div className="text-center pt-4 border-t border-border">
          <p className="text-xs text-foreground-subtle">
            You can skip AI analysis and add tasks manually later
          </p>
        </div>
      )}
    </div>
  );
}

export default WizardStepResearch;
