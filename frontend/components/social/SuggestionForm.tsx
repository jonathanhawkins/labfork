"use client";

/**
 * SuggestionForm Component
 *
 * Form for creating new suggestions.
 */

import { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  X,
  Plus,
  Loader2,
  Compass,
  TrendingUp,
  Bug,
  Lightbulb,
  HelpCircle,
  Users,
} from "lucide-react";
import type {
  SuggestionCategory,
  SuggestionPriority,
  CreateSuggestionInput,
  SuggestionAuthor,
} from "@/lib/social/suggestions/types";
import { CATEGORY_LABELS, PRIORITY_LABELS } from "@/lib/social/suggestions/types";

export interface SuggestionFormProps {
  /** Whether the form is open */
  isOpen: boolean;
  /** Close handler */
  onClose: () => void;
  /** Submit handler */
  onSubmit: (input: CreateSuggestionInput, author: SuggestionAuthor) => Promise<void>;
  /** Current user (author) */
  author: SuggestionAuthor;
  /** Lab ID */
  labId: string;
  /** Task ID (optional) */
  taskId?: string;
  /** Result ID (optional) */
  resultId?: string;
}

const CATEGORIES: { value: SuggestionCategory; icon: typeof Compass }[] = [
  { value: "research_direction", icon: Compass },
  { value: "improvement", icon: TrendingUp },
  { value: "feature_request", icon: Lightbulb },
  { value: "bug_report", icon: Bug },
  { value: "question", icon: HelpCircle },
  { value: "collaboration", icon: Users },
];

const PRIORITIES: SuggestionPriority[] = ["low", "medium", "high", "critical"];

export function SuggestionForm({
  isOpen,
  onClose,
  onSubmit,
  author,
  labId,
  taskId,
  resultId,
}: SuggestionFormProps) {
  const [category, setCategory] = useState<SuggestionCategory>("feature_request");
  const [priority, setPriority] = useState<SuggestionPriority>("medium");
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [tagInput, setTagInput] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleClose = useCallback(() => {
    if (!isSubmitting) {
      setTitle("");
      setDescription("");
      setTags([]);
      setTagInput("");
      setError(null);
      onClose();
    }
  }, [isSubmitting, onClose]);

  const handleAddTag = useCallback(() => {
    const trimmed = tagInput.trim().toLowerCase();
    if (trimmed && !tags.includes(trimmed) && tags.length < 5) {
      setTags([...tags, trimmed]);
      setTagInput("");
    }
  }, [tagInput, tags]);

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
      const input: CreateSuggestionInput = {
        labId,
        taskId,
        resultId,
        title: title.trim(),
        description: description.trim(),
        category,
        priority,
        tags,
      };

      await onSubmit(input, author);
      handleClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create suggestion");
    } finally {
      setIsSubmitting(false);
    }
  }, [
    labId,
    taskId,
    resultId,
    title,
    description,
    category,
    priority,
    tags,
    author,
    onSubmit,
    handleClose,
  ]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={handleClose}
      />

      {/* Dialog */}
      <div className="relative w-full max-w-lg max-h-[90vh] overflow-auto bg-background rounded-lg border border-border shadow-xl">
        {/* Header */}
        <div className="sticky top-0 bg-background border-b border-border px-6 py-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-foreground">
            New Suggestion
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
          {/* Error */}
          {error && (
            <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
              {error}
            </div>
          )}

          {/* Category */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">
              Category
            </label>
            <div className="grid grid-cols-3 gap-2">
              {CATEGORIES.map(({ value, icon: Icon }) => (
                <button
                  key={value}
                  onClick={() => setCategory(value)}
                  className={cn(
                    "flex items-center gap-2 px-3 py-2 rounded-lg border text-sm transition-colors",
                    category === value
                      ? "border-foreground-bright bg-foreground-bright/10 text-foreground"
                      : "border-border hover:border-foreground-muted text-foreground-muted"
                  )}
                >
                  <Icon className="w-4 h-4" />
                  <span className="truncate">{CATEGORY_LABELS[value]}</span>
                </button>
              ))}
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
              placeholder="Brief summary of your suggestion"
              className="w-full px-3 py-2 rounded-lg bg-background-darker border border-border text-foreground placeholder:text-foreground-subtle focus:outline-none focus:border-foreground-muted"
              maxLength={200}
            />
          </div>

          {/* Description */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">
              Description <span className="text-red-400">*</span>
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe your suggestion in detail..."
              rows={4}
              className="w-full px-3 py-2 rounded-lg bg-background-darker border border-border text-foreground placeholder:text-foreground-subtle focus:outline-none focus:border-foreground-muted resize-none"
              maxLength={2000}
            />
          </div>

          {/* Priority */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">
              Priority
            </label>
            <div className="flex gap-2">
              {PRIORITIES.map((p) => (
                <button
                  key={p}
                  onClick={() => setPriority(p)}
                  className={cn(
                    "px-3 py-1.5 rounded-lg border text-sm transition-colors",
                    priority === p
                      ? "border-foreground-bright bg-foreground-bright/10 text-foreground"
                      : "border-border hover:border-foreground-muted text-foreground-muted"
                  )}
                >
                  {PRIORITY_LABELS[p]}
                </button>
              ))}
            </div>
          </div>

          {/* Tags */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">
              Tags (optional)
            </label>
            <div className="flex flex-wrap gap-2 mb-2">
              {tags.map((tag) => (
                <span
                  key={tag}
                  className="flex items-center gap-1 px-2 py-1 rounded-lg bg-foreground-muted/10 text-sm text-foreground-muted"
                >
                  {tag}
                  <button
                    onClick={() => setTags(tags.filter((t) => t !== tag))}
                    className="p-0.5 hover:text-foreground"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </span>
              ))}
            </div>
            {tags.length < 5 && (
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
                  maxLength={20}
                />
                <button
                  onClick={handleAddTag}
                  disabled={!tagInput.trim()}
                  className="px-3 py-2 rounded-lg bg-foreground-muted/10 text-foreground-muted hover:text-foreground disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <Plus className="w-4 h-4" />
                </button>
              </div>
            )}
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
                Submitting...
              </>
            ) : (
              <>
                <Plus className="w-4 h-4" />
                Submit Suggestion
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

export default SuggestionForm;
