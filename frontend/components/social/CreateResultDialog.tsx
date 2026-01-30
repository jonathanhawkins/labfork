"use client";

/**
 * CreateResultDialog Component
 *
 * Modal dialog for creating new research results.
 * Supports different result types with type-specific fields.
 */

import { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  X,
  Box,
  Play,
  Lightbulb,
  GitCompare,
  Database,
  FileText,
  Plus,
  Upload,
  Trash2,
  Loader2,
} from "lucide-react";
import type {
  ResultType,
  CreateResultInput,
  ResultAuthor,
  ModelMetadata,
  DemoMetadata,
  FindingMetadata,
  ComparisonMetadata,
} from "@/lib/social/results/types";
import { RESULT_TYPE_LABELS } from "@/lib/social/results/types";

export interface CreateResultDialogProps {
  /** Whether the dialog is open */
  isOpen: boolean;
  /** Close handler */
  onClose: () => void;
  /** Submit handler */
  onSubmit: (input: CreateResultInput, author: ResultAuthor) => Promise<void>;
  /** Current user (author) */
  author: ResultAuthor;
  /** Lab ID */
  labId: string;
  /** Task ID (optional) */
  taskId?: string;
  /** Default type */
  defaultType?: ResultType;
}

/**
 * Get result type icon
 */
function getTypeIcon(type: ResultType) {
  switch (type) {
    case "model":
      return Box;
    case "demo":
      return Play;
    case "finding":
      return Lightbulb;
    case "comparison":
      return GitCompare;
    case "dataset":
      return Database;
    case "paper":
      return FileText;
    default:
      return Box;
  }
}

/**
 * Result type options
 */
const RESULT_TYPES: ResultType[] = [
  "finding",
  "model",
  "demo",
  "comparison",
  "dataset",
  "paper",
];

export function CreateResultDialog({
  isOpen,
  onClose,
  onSubmit,
  author,
  labId,
  taskId,
  defaultType = "finding",
}: CreateResultDialogProps) {
  const [type, setType] = useState<ResultType>(defaultType);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [content, setContent] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [tagInput, setTagInput] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Type-specific metadata
  const [modelMetadata, setModelMetadata] = useState<Partial<ModelMetadata>>({});
  const [demoMetadata, setDemoMetadata] = useState<Partial<DemoMetadata>>({
    isInteractive: true,
  });
  const [findingMetadata, setFindingMetadata] = useState<Partial<FindingMetadata>>({});
  const [comparisonMetadata, setComparisonMetadata] = useState<Partial<ComparisonMetadata>>({
    modelsCompared: [],
    criteria: [],
  });

  const handleClose = useCallback(() => {
    if (!isSubmitting) {
      setTitle("");
      setDescription("");
      setContent("");
      setTags([]);
      setTagInput("");
      setError(null);
      setModelMetadata({});
      setDemoMetadata({ isInteractive: true });
      setFindingMetadata({});
      setComparisonMetadata({ modelsCompared: [], criteria: [] });
      onClose();
    }
  }, [isSubmitting, onClose]);

  const handleAddTag = useCallback(() => {
    const trimmed = tagInput.trim().toLowerCase();
    if (trimmed && !tags.includes(trimmed) && tags.length < 10) {
      setTags([...tags, trimmed]);
      setTagInput("");
    }
  }, [tagInput, tags]);

  const handleRemoveTag = useCallback((tag: string) => {
    setTags(tags.filter((t) => t !== tag));
  }, [tags]);

  const handleSubmit = useCallback(async () => {
    if (!title.trim()) {
      setError("Title is required");
      return;
    }
    if (!description.trim()) {
      setError("Description is required");
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      // Build metadata based on type
      let metadata: CreateResultInput["metadata"];

      switch (type) {
        case "model":
          if (modelMetadata.modelName) {
            metadata = modelMetadata as ModelMetadata;
          }
          break;
        case "demo":
          if (demoMetadata.demoUrl) {
            metadata = {
              demoUrl: demoMetadata.demoUrl,
              isInteractive: demoMetadata.isInteractive || false,
              technologies: demoMetadata.technologies,
              sourceUrl: demoMetadata.sourceUrl,
            } as DemoMetadata;
          }
          break;
        case "finding":
          if (findingMetadata.keyInsight) {
            metadata = findingMetadata as FindingMetadata;
          }
          break;
        case "comparison":
          if (
            comparisonMetadata.modelsCompared &&
            comparisonMetadata.modelsCompared.length > 0 &&
            comparisonMetadata.summary
          ) {
            metadata = comparisonMetadata as ComparisonMetadata;
          }
          break;
      }

      const input: CreateResultInput = {
        type,
        title: title.trim(),
        description: description.trim(),
        content: content.trim() || undefined,
        labId,
        taskId,
        visibility: "public",
        tags,
        metadata,
      };

      await onSubmit(input, author);
      handleClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create result");
    } finally {
      setIsSubmitting(false);
    }
  }, [
    type,
    title,
    description,
    content,
    labId,
    taskId,
    tags,
    modelMetadata,
    demoMetadata,
    findingMetadata,
    comparisonMetadata,
    author,
    onSubmit,
    handleClose,
  ]);

  if (!isOpen) return null;

  const TypeIcon = getTypeIcon(type);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={handleClose}
      />

      {/* Dialog */}
      <div className="relative w-full max-w-2xl max-h-[90vh] overflow-auto bg-background rounded-lg border border-border shadow-xl">
        {/* Header */}
        <div className="sticky top-0 bg-background border-b border-border px-6 py-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-foreground">
            Share Result
          </h2>
          <button
            onClick={handleClose}
            className="p-1.5 rounded-lg text-foreground-muted hover:text-foreground hover:bg-foreground-muted/10 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Error message */}
          {error && (
            <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
              {error}
            </div>
          )}

          {/* Result Type Selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">
              Result Type
            </label>
            <div className="grid grid-cols-3 gap-2">
              {RESULT_TYPES.map((t) => {
                const Icon = getTypeIcon(t);
                return (
                  <button
                    key={t}
                    onClick={() => setType(t)}
                    className={cn(
                      "flex items-center gap-2 px-3 py-2 rounded-lg border text-sm transition-colors",
                      type === t
                        ? "border-foreground-bright bg-foreground-bright/10 text-foreground"
                        : "border-border hover:border-foreground-muted text-foreground-muted"
                    )}
                  >
                    <Icon className="w-4 h-4" />
                    {RESULT_TYPE_LABELS[t]}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Title */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">
              Title <span className="text-red-400">*</span>
            </label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g., Improved prosody control with hierarchical attention"
              className="w-full px-3 py-2 rounded-lg bg-background-darker border border-border text-foreground placeholder:text-foreground-subtle focus:outline-none focus:border-foreground-muted"
              maxLength={200}
            />
          </div>

          {/* Description */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">
              Short Description <span className="text-red-400">*</span>
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Brief summary of the result (1-2 sentences)"
              rows={2}
              className="w-full px-3 py-2 rounded-lg bg-background-darker border border-border text-foreground placeholder:text-foreground-subtle focus:outline-none focus:border-foreground-muted resize-none"
              maxLength={500}
            />
          </div>

          {/* Type-specific fields */}
          {type === "model" && (
            <div className="space-y-4 p-4 rounded-lg bg-blue-500/5 border border-blue-500/20">
              <h3 className="text-sm font-medium text-foreground flex items-center gap-2">
                <Box className="w-4 h-4 text-blue-400" />
                Model Details
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-xs text-foreground-muted">Model Name</label>
                  <input
                    type="text"
                    value={modelMetadata.modelName || ""}
                    onChange={(e) =>
                      setModelMetadata({ ...modelMetadata, modelName: e.target.value })
                    }
                    placeholder="e.g., csm-1b-prosody"
                    className="w-full px-3 py-2 rounded-lg bg-background-darker border border-border text-sm text-foreground focus:outline-none focus:border-foreground-muted"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-xs text-foreground-muted">Version</label>
                  <input
                    type="text"
                    value={modelMetadata.version || ""}
                    onChange={(e) =>
                      setModelMetadata({ ...modelMetadata, version: e.target.value })
                    }
                    placeholder="e.g., 1.0.0"
                    className="w-full px-3 py-2 rounded-lg bg-background-darker border border-border text-sm text-foreground focus:outline-none focus:border-foreground-muted"
                  />
                </div>
                <div className="space-y-2 col-span-2">
                  <label className="text-xs text-foreground-muted">Checkpoint URL</label>
                  <input
                    type="url"
                    value={modelMetadata.checkpointUrl || ""}
                    onChange={(e) =>
                      setModelMetadata({ ...modelMetadata, checkpointUrl: e.target.value })
                    }
                    placeholder="https://huggingface.co/..."
                    className="w-full px-3 py-2 rounded-lg bg-background-darker border border-border text-sm text-foreground focus:outline-none focus:border-foreground-muted"
                  />
                </div>
              </div>
            </div>
          )}

          {type === "demo" && (
            <div className="space-y-4 p-4 rounded-lg bg-green-500/5 border border-green-500/20">
              <h3 className="text-sm font-medium text-foreground flex items-center gap-2">
                <Play className="w-4 h-4 text-green-400" />
                Demo Details
              </h3>
              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="text-xs text-foreground-muted">Demo URL</label>
                  <input
                    type="url"
                    value={demoMetadata.demoUrl || ""}
                    onChange={(e) =>
                      setDemoMetadata({ ...demoMetadata, demoUrl: e.target.value })
                    }
                    placeholder="https://..."
                    className="w-full px-3 py-2 rounded-lg bg-background-darker border border-border text-sm text-foreground focus:outline-none focus:border-foreground-muted"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="isInteractive"
                    checked={demoMetadata.isInteractive}
                    onChange={(e) =>
                      setDemoMetadata({ ...demoMetadata, isInteractive: e.target.checked })
                    }
                    className="rounded border-border"
                  />
                  <label htmlFor="isInteractive" className="text-sm text-foreground-muted">
                    Interactive demo (users can try it)
                  </label>
                </div>
              </div>
            </div>
          )}

          {type === "finding" && (
            <div className="space-y-4 p-4 rounded-lg bg-yellow-500/5 border border-yellow-500/20">
              <h3 className="text-sm font-medium text-foreground flex items-center gap-2">
                <Lightbulb className="w-4 h-4 text-yellow-400" />
                Finding Details
              </h3>
              <div className="space-y-2">
                <label className="text-xs text-foreground-muted">Key Insight</label>
                <textarea
                  value={findingMetadata.keyInsight || ""}
                  onChange={(e) =>
                    setFindingMetadata({ ...findingMetadata, keyInsight: e.target.value })
                  }
                  placeholder="What is the main takeaway?"
                  rows={2}
                  className="w-full px-3 py-2 rounded-lg bg-background-darker border border-border text-sm text-foreground focus:outline-none focus:border-foreground-muted resize-none"
                />
              </div>
            </div>
          )}

          {type === "comparison" && (
            <div className="space-y-4 p-4 rounded-lg bg-purple-500/5 border border-purple-500/20">
              <h3 className="text-sm font-medium text-foreground flex items-center gap-2">
                <GitCompare className="w-4 h-4 text-purple-400" />
                Comparison Details
              </h3>
              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="text-xs text-foreground-muted">
                    Models Compared (comma-separated)
                  </label>
                  <input
                    type="text"
                    value={comparisonMetadata.modelsCompared?.join(", ") || ""}
                    onChange={(e) =>
                      setComparisonMetadata({
                        ...comparisonMetadata,
                        modelsCompared: e.target.value.split(",").map((s) => s.trim()).filter(Boolean),
                      })
                    }
                    placeholder="e.g., CSM-1B, XTTS, Bark"
                    className="w-full px-3 py-2 rounded-lg bg-background-darker border border-border text-sm text-foreground focus:outline-none focus:border-foreground-muted"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-xs text-foreground-muted">Summary</label>
                  <textarea
                    value={comparisonMetadata.summary || ""}
                    onChange={(e) =>
                      setComparisonMetadata({ ...comparisonMetadata, summary: e.target.value })
                    }
                    placeholder="Brief summary of comparison results"
                    rows={2}
                    className="w-full px-3 py-2 rounded-lg bg-background-darker border border-border text-sm text-foreground focus:outline-none focus:border-foreground-muted resize-none"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Full Content */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">
              Full Content (Markdown)
            </label>
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="Detailed description, methodology, results... (supports Markdown)"
              rows={6}
              className="w-full px-3 py-2 rounded-lg bg-background-darker border border-border text-foreground placeholder:text-foreground-subtle focus:outline-none focus:border-foreground-muted resize-none font-mono text-sm"
            />
          </div>

          {/* Tags */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Tags</label>
            <div className="flex flex-wrap gap-2 mb-2">
              {tags.map((tag) => (
                <span
                  key={tag}
                  className="flex items-center gap-1 px-2 py-1 rounded-lg bg-foreground-muted/10 text-sm text-foreground-muted"
                >
                  {tag}
                  <button
                    onClick={() => handleRemoveTag(tag)}
                    className="p-0.5 hover:text-foreground"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </span>
              ))}
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    handleAddTag();
                  }
                }}
                placeholder="Add a tag..."
                className="flex-1 px-3 py-2 rounded-lg bg-background-darker border border-border text-foreground placeholder:text-foreground-subtle focus:outline-none focus:border-foreground-muted text-sm"
                maxLength={30}
              />
              <button
                onClick={handleAddTag}
                disabled={!tagInput.trim()}
                className="px-3 py-2 rounded-lg bg-foreground-muted/10 text-foreground-muted hover:text-foreground disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Plus className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="sticky bottom-0 bg-background border-t border-border px-6 py-4 flex items-center justify-end gap-3">
          <button
            onClick={handleClose}
            disabled={isSubmitting}
            className="px-4 py-2 rounded-lg text-foreground-muted hover:text-foreground hover:bg-foreground-muted/10 transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={isSubmitting || !title.trim() || !description.trim()}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-foreground-bright text-background font-medium hover:bg-foreground-bright/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <Plus className="w-4 h-4" />
                Create Result
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

export default CreateResultDialog;
