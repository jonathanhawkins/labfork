"use client";

/**
 * Labs Overview Page
 *
 * Displays all public research labs.
 * Shows DB-backed labs (works on production) with fallback to research manager labs.
 */

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import {
  Activity,
  Bot,
  Clock,
  FileText,
  Loader2,
  Star,
  GitFork,
  Eye,
  Plus,
} from "lucide-react";
import type { Lab } from "@/lib/labs/types";

export default function LabsOverviewPage() {
  const [labs, setLabs] = useState<Lab[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [isSeeding, setIsSeeding] = useState(false);
  const [seedError, setSeedError] = useState<string | null>(null);
  const [seedSuccess, setSeedSuccess] = useState(false);

  const fetchLabs = useCallback(async () => {
    try {
      // Fetch from DB-backed API (works on production)
      const response = await fetch("/api/labs?visibility=public&sortBy=stars&limit=50");
      const data = await response.json();

      if (data.success && data.labs) {
        setLabs(data.labs);
        setError(null);
        setLastUpdated(new Date());
      } else {
        setError(data.error || "Failed to load labs");
      }
    } catch (err) {
      setError("Failed to connect to server");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const seedLabs = async () => {
    setIsSeeding(true);
    setSeedError(null);
    setSeedSuccess(false);
    try {
      const response = await fetch("/api/labs/seed", { method: "POST" });
      const data = await response.json();
      if (data.success) {
        await fetchLabs();
        setSeedSuccess(true);
        setTimeout(() => setSeedSuccess(false), 3000);
      } else {
        setSeedError(data.error || "Failed to seed demo labs. Please try again.");
      }
    } catch (err) {
      console.error("Failed to seed labs:", err);
      setSeedError("Failed to seed demo labs. Please try again.");
    } finally {
      setIsSeeding(false);
    }
  };

  useEffect(() => {
    fetchLabs();
  }, [fetchLabs]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-foreground-muted mx-auto mb-4" />
          <p className="text-foreground-muted">Loading research labs...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-foreground-bright">
                Research Labs
              </h1>
              <p className="text-foreground-muted mt-2">
                Live status of AI research labs and their agents
              </p>
            </div>

            {lastUpdated && (
              <div className="flex items-center gap-2 text-sm text-foreground-subtle">
                <Clock className="w-4 h-4" />
                <span>
                  Updated {lastUpdated.toLocaleTimeString()}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        {error ? (
          <div className="text-center py-12">
            <p className="text-red-400 mb-4">{error}</p>
            <button
              onClick={fetchLabs}
              className="px-4 py-2 rounded-lg bg-foreground-bright text-background hover:bg-white transition-colors"
            >
              Retry
            </button>
          </div>
        ) : labs.length === 0 ? (
          <div className="text-center py-12">
            <FileText className="w-12 h-12 mx-auto text-foreground-subtle mb-4" />
            <h2 className="text-xl font-medium text-foreground-bright mb-2">
              No Research Labs Found
            </h2>
            <p className="text-foreground-muted mb-6">
              Create your first lab or seed demo labs to get started.
            </p>
            <div className="flex flex-col items-center gap-3">
              <div className="flex gap-4">
                <Link
                  href="/lab/new"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
                >
                  <Plus className="w-4 h-4" />
                  Create Lab
                </Link>
                <button
                  onClick={seedLabs}
                  disabled={isSeeding}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-border hover:bg-muted transition-colors disabled:opacity-50"
                >
                  {isSeeding ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Activity className="w-4 h-4" />
                  )}
                  {isSeeding ? "Seeding..." : "Seed Demo Labs"}
                </button>
              </div>
              {seedError && (
                <p className="text-sm text-red-500">{seedError}</p>
              )}
              {seedSuccess && (
                <p className="text-sm text-green-500">Demo labs created successfully!</p>
              )}
            </div>
          </div>
        ) : (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {labs.map((lab) => (
              <LabCard key={lab.id} lab={lab} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function LabCard({ lab }: { lab: Lab }) {
  return (
    <Link
      href={`/labs/${lab.owner.username}/${lab.slug}`}
      className="block p-6 rounded-lg border border-border transition-colors hover:border-foreground-muted hover:bg-muted/50"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1 min-w-0">
          <h2 className="text-lg font-semibold text-foreground-bright truncate">
            {lab.name}
          </h2>
          <p className="text-sm text-foreground-muted mt-1 line-clamp-2">
            {lab.description}
          </p>
        </div>
        {lab.status === "active" && (
          <span className="ml-2 flex items-center gap-1 px-2 py-1 rounded text-xs bg-green-500/20 text-green-400 shrink-0">
            <span className="w-2 h-2 rounded-full bg-green-400" />
            Active
          </span>
        )}
      </div>

      {/* Tags */}
      {lab.tags && lab.tags.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-4">
          {lab.tags.slice(0, 4).map((tag) => (
            <span
              key={tag}
              className="px-2 py-0.5 text-xs rounded bg-muted text-foreground-muted"
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      {/* Stats */}
      <div className="flex items-center gap-4 text-sm text-foreground-muted">
        <div className="flex items-center gap-1">
          <Star className="w-4 h-4" />
          <span>{lab.stats?.stars || 0}</span>
        </div>
        <div className="flex items-center gap-1">
          <GitFork className="w-4 h-4" />
          <span>{lab.stats?.forks || 0}</span>
        </div>
        <div className="flex items-center gap-1">
          <FileText className="w-4 h-4" />
          <span>{lab.stats?.papers || 0} papers</span>
        </div>
        {(lab.stats?.viewers ?? 0) > 0 && (
          <div className="flex items-center gap-1 text-green-400">
            <Eye className="w-4 h-4" />
            <span>{lab.stats?.viewers} watching</span>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="mt-4 pt-4 border-t border-border flex items-center justify-between text-xs text-foreground-subtle">
        <span className="flex items-center gap-1">
          <span
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: lab.primaryColor || "#6366f1" }}
          />
          {lab.domainName || lab.domainSlug}
        </span>
        <span>
          by {lab.owner.displayName || lab.owner.username}
        </span>
      </div>
    </Link>
  );
}
