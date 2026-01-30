"use client";

import { useState, useCallback, useEffect } from "react";
import { cn } from "@/lib/utils";
import {
  Plus,
  FileText,
  ArrowLeft,
  Loader2,
  RefreshCw,
} from "lucide-react";
import type { Paper } from "@/lib/papers/types";
import {
  PaperQueue,
  AddPaperDialog,
  PaperDetailView,
} from "@/components/papers";
import { createTasksViaAPI, generateTasksFromPaper } from "@/lib/papers/task-generator";

/**
 * Papers Page - Paper ingestion and management interface
 */
export default function PapersPage() {
  // State
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);
  const [isAddDialogOpen, setIsAddDialogOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Domain context (can be selected in future)
  const domainSlug = "voice-cloning";

  // Fetch papers
  const fetchPapers = useCallback(async () => {
    try {
      const response = await fetch("/api/papers");
      const data = await response.json();

      if (data.papers) {
        setPapers(data.papers);
        setError(null);

        // Update selected paper if it changed
        if (selectedPaper) {
          const updated = data.papers.find(
            (p: Paper) => p.id === selectedPaper.id
          );
          if (updated) {
            setSelectedPaper(updated);
          }
        }
      } else {
        setError("Failed to fetch papers");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch papers");
    } finally {
      setIsLoading(false);
    }
  }, [selectedPaper]);

  // Initial fetch
  useEffect(() => {
    fetchPapers();
  }, []);

  // Handle paper click
  const handlePaperClick = useCallback((paper: Paper) => {
    setSelectedPaper(paper);
  }, []);

  // Handle paper added from dialog
  const handlePaperAdded = useCallback(
    (paper: Paper) => {
      setPapers((prev) => {
        const exists = prev.find((p) => p.id === paper.id);
        if (exists) {
          return prev.map((p) => (p.id === paper.id ? paper : p));
        }
        return [paper, ...prev];
      });
    },
    []
  );

  // Handle analyze
  const handleAnalyze = useCallback(
    async (paperId: string) => {
      try {
        const response = await fetch("/api/papers/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ paperId, domainSlug }),
        });

        const data = await response.json();

        if (data.success && data.paper) {
          setPapers((prev) =>
            prev.map((p) => (p.id === paperId ? data.paper : p))
          );
          if (selectedPaper?.id === paperId) {
            setSelectedPaper(data.paper);
          }
        }
      } catch (err) {
        console.error("Analyze failed:", err);
      }
    },
    [domainSlug, selectedPaper]
  );

  // Handle accept
  const handleAccept = useCallback(
    async (paperId: string) => {
      const paper = papers.find((p) => p.id === paperId);
      if (!paper?.analysis) {
        console.error("Cannot accept: paper has no analysis");
        return;
      }

      try {
        // Generate tasks
        const tasks = generateTasksFromPaper(paper, domainSlug);
        if (!tasks) {
          console.error("Failed to generate tasks");
          return;
        }

        // Create tasks via API
        const taskResult = await createTasksViaAPI(tasks);
        if (!taskResult.success) {
          console.error("Failed to create tasks:", taskResult.error);
        }

        // Update paper status
        const response = await fetch("/api/papers", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            id: paperId,
            status: "implemented",
            taskIds: taskResult.taskIds,
          }),
        });

        const data = await response.json();

        if (data.success && data.paper) {
          setPapers((prev) =>
            prev.map((p) => (p.id === paperId ? data.paper : p))
          );
          if (selectedPaper?.id === paperId) {
            setSelectedPaper(data.paper);
          }
        }
      } catch (err) {
        console.error("Accept failed:", err);
      }
    },
    [papers, domainSlug, selectedPaper]
  );

  // Handle reject
  const handleReject = useCallback(
    async (paperId: string) => {
      try {
        const response = await fetch("/api/papers", {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: paperId, status: "rejected" }),
        });

        const data = await response.json();

        if (data.success && data.paper) {
          setPapers((prev) =>
            prev.map((p) => (p.id === paperId ? data.paper : p))
          );
          if (selectedPaper?.id === paperId) {
            setSelectedPaper(data.paper);
          }
        }
      } catch (err) {
        console.error("Reject failed:", err);
      }
    },
    [selectedPaper]
  );

  // Handle view tasks
  const handleViewTasks = useCallback((taskIds: string[]) => {
    // Navigate to lab with task filter
    // For now, just log - can be enhanced to navigate or open panel
    console.log("View tasks:", taskIds);
    window.location.href = `/lab?tasks=${taskIds.join(",")}`;
  }, []);

  // Stats
  const stats = {
    total: papers.length,
    pending: papers.filter(
      (p) => p.status === "fetched" || p.status === "pending"
    ).length,
    analyzing: papers.filter((p) => p.status === "analyzing").length,
    analyzed: papers.filter((p) => p.status === "analyzed").length,
    accepted: papers.filter((p) => p.status === "accepted" || p.status === "implemented").length,
    rejected: papers.filter((p) => p.status === "rejected").length,
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-background-elevated">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              {selectedPaper ? (
                <button
                  onClick={() => setSelectedPaper(null)}
                  className="flex items-center gap-2 text-foreground-muted hover:text-foreground"
                >
                  <ArrowLeft className="w-4 h-4" />
                  Back to Queue
                </button>
              ) : (
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-foreground-muted/10 flex items-center justify-center">
                    <FileText className="w-5 h-5 text-foreground-muted" />
                  </div>
                  <div>
                    <h1 className="text-lg font-normal text-foreground-bright">
                      Papers
                    </h1>
                    <p className="text-xs text-foreground-muted">
                      Research paper ingestion and analysis
                    </p>
                  </div>
                </div>
              )}
            </div>

            <div className="flex items-center gap-3">
              {/* Refresh */}
              <button
                onClick={() => fetchPapers()}
                disabled={isLoading}
                className={cn(
                  "p-2 rounded-lg border border-border",
                  "bg-background-card text-foreground-muted hover:text-foreground",
                  "disabled:opacity-50"
                )}
              >
                <RefreshCw
                  className={cn("w-4 h-4", isLoading && "animate-spin")}
                />
              </button>

              {/* Add Paper */}
              {!selectedPaper && (
                <button
                  onClick={() => setIsAddDialogOpen(true)}
                  className={cn(
                    "flex items-center gap-2 px-4 py-2 text-sm rounded-lg",
                    "bg-foreground-bright text-background hover:bg-white"
                  )}
                >
                  <Plus className="w-4 h-4" />
                  Add Paper
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {isLoading && papers.length === 0 ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-8 h-8 text-foreground-muted animate-spin" />
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center py-20">
            <p className="text-foreground-muted">{error}</p>
            <button
              onClick={() => fetchPapers()}
              className="mt-4 text-sm text-foreground-bright hover:underline"
            >
              Try again
            </button>
          </div>
        ) : selectedPaper ? (
          /* Detail view */
          <div className="max-w-3xl">
            <PaperDetailView
              paper={selectedPaper}
              domainSlug={domainSlug}
              onAccept={handleAccept}
              onReject={handleReject}
              onAnalyze={handleAnalyze}
              onClose={() => setSelectedPaper(null)}
              onViewTasks={handleViewTasks}
            />
          </div>
        ) : (
          /* Queue view */
          <div className="space-y-6">
            {/* Quick stats */}
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-4">
              <div className="p-4 rounded-lg bg-background-card border border-border">
                <div className="text-2xl font-bold text-foreground-bright">
                  {stats.total}
                </div>
                <div className="text-xs text-foreground-muted">Total</div>
              </div>
              <div className="p-4 rounded-lg bg-background-card border border-border">
                <div className="text-2xl font-bold text-yellow-400">
                  {stats.pending}
                </div>
                <div className="text-xs text-foreground-muted">Pending</div>
              </div>
              <div className="p-4 rounded-lg bg-background-card border border-border">
                <div className="text-2xl font-bold text-blue-400">
                  {stats.analyzing}
                </div>
                <div className="text-xs text-foreground-muted">Analyzing</div>
              </div>
              <div className="p-4 rounded-lg bg-background-card border border-border">
                <div className="text-2xl font-bold text-purple-400">
                  {stats.analyzed}
                </div>
                <div className="text-xs text-foreground-muted">Analyzed</div>
              </div>
              <div className="p-4 rounded-lg bg-background-card border border-border">
                <div className="text-2xl font-bold text-green-400">
                  {stats.accepted}
                </div>
                <div className="text-xs text-foreground-muted">Accepted</div>
              </div>
              <div className="p-4 rounded-lg bg-background-card border border-border">
                <div className="text-2xl font-bold text-red-400">
                  {stats.rejected}
                </div>
                <div className="text-xs text-foreground-muted">Rejected</div>
              </div>
            </div>

            {/* Paper queue */}
            <PaperQueue
              initialPapers={papers}
              domainSlug={domainSlug}
              onPaperClick={handlePaperClick}
              onPaperAccepted={(paper) => {
                setPapers((prev) =>
                  prev.map((p) => (p.id === paper.id ? paper : p))
                );
              }}
              onPaperRejected={(paper) => {
                setPapers((prev) =>
                  prev.map((p) => (p.id === paper.id ? paper : p))
                );
              }}
              autoRefresh={false}
            />

            {/* Empty state */}
            {papers.length === 0 && (
              <div className="flex flex-col items-center justify-center py-20 text-center">
                <div className="w-16 h-16 rounded-full bg-foreground-muted/10 flex items-center justify-center mb-4">
                  <FileText className="w-8 h-8 text-foreground-muted/50" />
                </div>
                <h3 className="text-lg font-normal text-foreground-bright mb-2">
                  No papers yet
                </h3>
                <p className="text-sm text-foreground-muted mb-6 max-w-md">
                  Add research papers to analyze and convert into actionable
                  tasks. Paste an arXiv ID, DOI, or URL to get started.
                </p>
                <button
                  onClick={() => setIsAddDialogOpen(true)}
                  className={cn(
                    "flex items-center gap-2 px-4 py-2 text-sm rounded-lg",
                    "bg-foreground-bright text-background hover:bg-white"
                  )}
                >
                  <Plus className="w-4 h-4" />
                  Add Your First Paper
                </button>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Add Paper Dialog */}
      <AddPaperDialog
        isOpen={isAddDialogOpen}
        onClose={() => setIsAddDialogOpen(false)}
        onPaperAdded={handlePaperAdded}
        domainSlug={domainSlug}
      />
    </div>
  );
}
