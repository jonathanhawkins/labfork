"use client";

/**
 * ForkDialog Component
 *
 * Dialog for forking a lab with custom name/slug.
 */

import { useState, useCallback, useEffect } from "react";
import { cn } from "@/lib/utils";
import {
  GitFork,
  X,
  Loader2,
  Check,
  AlertCircle,
  ArrowRight,
  FileText,
  Settings,
  ListTodo,
} from "lucide-react";
import type { Lab } from "@/lib/labs/types";
import { slugify, getLabPath } from "@/lib/labs/types";

export interface ForkDialogProps {
  /** Lab to fork */
  lab: Lab;
  /** Is dialog open */
  isOpen: boolean;
  /** Close handler */
  onClose: () => void;
  /** Success handler */
  onSuccess?: (forkedLab: Lab) => void;
  /** Custom class name */
  className?: string;
}

export function ForkDialog({
  lab,
  isOpen,
  onClose,
  onSuccess,
  className,
}: ForkDialogProps) {
  const [name, setName] = useState(lab.name);
  const [slug, setSlug] = useState(`${lab.slug}-fork`);
  const [isAutoSlug, setIsAutoSlug] = useState(true);
  const [isForking, setIsForking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<Lab | null>(null);

  // Reset state when dialog opens
  useEffect(() => {
    if (isOpen) {
      setName(lab.name);
      setSlug(`${lab.slug}-fork`);
      setIsAutoSlug(true);
      setError(null);
      setSuccess(null);
    }
  }, [isOpen, lab]);

  // Auto-generate slug from name
  useEffect(() => {
    if (isAutoSlug) {
      setSlug(slugify(name) || `${lab.slug}-fork`);
    }
  }, [name, isAutoSlug, lab.slug]);

  const handleFork = useCallback(async () => {
    if (isForking) return;

    setIsForking(true);
    setError(null);

    try {
      const response = await fetch(`/api/labs/${lab.id}/fork`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slug }),
      });

      const data = await response.json();

      if (data.success) {
        setSuccess(data.lab);
        onSuccess?.(data.lab);
      } else {
        setError(data.error || "Failed to fork lab");
      }
    } catch (err) {
      console.error("Fork error:", err);
      setError("Failed to fork lab. Please try again.");
    } finally {
      setIsForking(false);
    }
  }, [lab.id, slug, isForking, onSuccess]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-background/80 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Dialog */}
      <div
        className={cn(
          "relative w-full max-w-md mx-4 bg-background-elevated rounded-lg border border-border shadow-xl",
          className
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div className="flex items-center gap-2">
            <GitFork className="w-5 h-5 text-foreground-bright" />
            <h2 className="text-lg font-medium text-foreground-bright">
              Fork Lab
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-foreground-muted/10 transition-colors"
          >
            <X className="w-5 h-5 text-foreground-muted" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* Success state */}
          {success ? (
            <div className="space-y-4">
              <div className="flex items-center gap-3 p-4 rounded-lg bg-green-500/10 border border-green-500/20">
                <Check className="w-6 h-6 text-green-400" />
                <div>
                  <p className="text-sm font-medium text-foreground">
                    Lab forked successfully!
                  </p>
                  <p className="text-xs text-foreground-muted mt-0.5">
                    Your fork is ready at {success.owner.username}/{success.slug}
                  </p>
                </div>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={onClose}
                  className="flex-1 px-4 py-2 text-sm border border-border rounded-lg hover:bg-foreground-muted/10 transition-colors"
                >
                  Close
                </button>
                <a
                  href={getLabPath(success.owner.username, success.slug)}
                  className="flex-1 flex items-center justify-center gap-2 px-4 py-2 text-sm bg-foreground-bright text-background rounded-lg hover:bg-white transition-colors"
                >
                  Go to Fork
                  <ArrowRight className="w-4 h-4" />
                </a>
              </div>
            </div>
          ) : (
            <>
              {/* Source info */}
              <div className="p-3 rounded-lg bg-foreground-muted/5 border border-border">
                <p className="text-xs text-foreground-muted mb-1">Forking from</p>
                <p className="text-sm font-medium text-foreground">
                  {lab.owner.username}/{lab.slug}
                </p>
              </div>

              {/* Form */}
              <div className="space-y-3">
                <div>
                  <label className="block text-xs text-foreground-muted mb-1">
                    Lab Name
                  </label>
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="My Lab"
                    className={cn(
                      "w-full px-3 py-2 rounded-lg text-sm",
                      "bg-background border border-border",
                      "text-foreground placeholder-foreground-subtle",
                      "focus:outline-none focus:ring-2 focus:ring-foreground-bright/50"
                    )}
                  />
                </div>

                <div>
                  <label className="block text-xs text-foreground-muted mb-1">
                    URL Slug
                  </label>
                  <div className="flex items-center gap-2">
                    <input
                      type="text"
                      value={slug}
                      onChange={(e) => {
                        setIsAutoSlug(false);
                        setSlug(e.target.value);
                      }}
                      placeholder="my-lab"
                      className={cn(
                        "flex-1 px-3 py-2 rounded-lg text-sm",
                        "bg-background border border-border",
                        "text-foreground placeholder-foreground-subtle",
                        "focus:outline-none focus:ring-2 focus:ring-foreground-bright/50"
                      )}
                    />
                  </div>
                  <p className="text-xs text-foreground-subtle mt-1">
                    labs/your-username/{slug}
                  </p>
                </div>
              </div>

              {/* What gets copied */}
              <div className="space-y-2">
                <p className="text-xs text-foreground-muted">What gets copied:</p>
                <div className="grid grid-cols-3 gap-2">
                  <div className="flex items-center gap-2 p-2 rounded bg-foreground-muted/5">
                    <Settings className="w-3.5 h-3.5 text-foreground-subtle" />
                    <span className="text-xs text-foreground-muted">Config</span>
                  </div>
                  <div className="flex items-center gap-2 p-2 rounded bg-foreground-muted/5">
                    <FileText className="w-3.5 h-3.5 text-foreground-subtle" />
                    <span className="text-xs text-foreground-muted">Prompts</span>
                  </div>
                  <div className="flex items-center gap-2 p-2 rounded bg-foreground-muted/5">
                    <ListTodo className="w-3.5 h-3.5 text-foreground-subtle" />
                    <span className="text-xs text-foreground-muted">Tasks</span>
                  </div>
                </div>
              </div>

              {/* Error */}
              {error && (
                <div className="flex items-start gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/20">
                  <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                  <p className="text-sm text-red-400">{error}</p>
                </div>
              )}

              {/* Actions */}
              <div className="flex gap-2 pt-2">
                <button
                  onClick={onClose}
                  disabled={isForking}
                  className="flex-1 px-4 py-2 text-sm border border-border rounded-lg hover:bg-foreground-muted/10 transition-colors disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  onClick={handleFork}
                  disabled={isForking || !name.trim() || !slug.trim()}
                  className={cn(
                    "flex-1 flex items-center justify-center gap-2 px-4 py-2 text-sm rounded-lg transition-colors",
                    "bg-foreground-bright text-background hover:bg-white",
                    "disabled:opacity-50 disabled:cursor-not-allowed"
                  )}
                >
                  {isForking ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Forking...
                    </>
                  ) : (
                    <>
                      <GitFork className="w-4 h-4" />
                      Fork Lab
                    </>
                  )}
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default ForkDialog;
