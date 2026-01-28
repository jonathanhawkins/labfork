"use client";

import { useState, useCallback } from "react";
import { Send, AlertCircle, CheckCircle2 } from "lucide-react";

interface SuggestionFormProps {
  onSubmitted?: () => void;
}

type Category = "feature" | "improvement" | "bug";

const CATEGORIES: { value: Category; label: string; description: string }[] = [
  {
    value: "feature",
    label: "Feature",
    description: "New functionality or capability",
  },
  {
    value: "improvement",
    label: "Improvement",
    description: "Enhance existing features",
  },
  {
    value: "bug",
    label: "Bug",
    description: "Something isn't working correctly",
  },
];

export function SuggestionForm({ onSubmitted }: SuggestionFormProps) {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [category, setCategory] = useState<Category>("feature");

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [remaining, setRemaining] = useState<number | null>(null);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();

      // Validation
      if (title.trim().length < 5) {
        setError("Title must be at least 5 characters");
        return;
      }

      if (description.trim().length < 10) {
        setError("Description must be at least 10 characters");
        return;
      }

      setIsSubmitting(true);
      setError(null);
      setSuccess(false);

      try {
        const response = await fetch("/api/suggestions", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            title: title.trim(),
            description: description.trim(),
            category,
          }),
        });

        const data = await response.json();

        if (!response.ok) {
          if (response.status === 429) {
            setError(data.message || "Rate limit exceeded. Please try again later.");
          } else {
            setError(data.error || "Failed to submit suggestion");
          }
          return;
        }

        // Success
        setSuccess(true);
        setTitle("");
        setDescription("");
        setCategory("feature");
        setRemaining(data.remaining);

        // Notify parent
        onSubmitted?.();

        // Reset success message after 3 seconds
        setTimeout(() => setSuccess(false), 3000);
      } catch (err) {
        setError("Failed to connect to server. Please try again.");
      } finally {
        setIsSubmitting(false);
      }
    },
    [title, description, category, onSubmitted]
  );

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Error Message */}
      {error && (
        <div className="flex items-start gap-2 p-3 bg-red-500/10 border border-red-500/30 rounded">
          <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
          <span className="text-sm text-red-400">{error}</span>
        </div>
      )}

      {/* Success Message */}
      {success && (
        <div className="flex items-start gap-2 p-3 bg-green-500/10 border border-green-500/30 rounded">
          <CheckCircle2 className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
          <div>
            <span className="text-sm text-green-400">
              Suggestion submitted successfully!
            </span>
            {remaining !== null && (
              <span className="text-xs text-green-400/70 block mt-0.5">
                {remaining} submissions remaining this hour
              </span>
            )}
          </div>
        </div>
      )}

      {/* Title */}
      <div>
        <label htmlFor="title" className="text-sm text-muted-foreground block mb-1.5">
          Title
        </label>
        <input
          id="title"
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Brief summary of your suggestion"
          maxLength={100}
          className="w-full px-3 py-2 bg-background border border-border rounded text-sm text-foreground placeholder-muted-foreground focus:outline-none focus:border-foreground-muted transition-colors"
        />
        <div className="text-xs text-foreground-subtle text-right mt-1">
          {title.length}/100
        </div>
      </div>

      {/* Category */}
      <div>
        <label className="text-sm text-muted-foreground block mb-1.5">
          Category
        </label>
        <div className="grid grid-cols-3 gap-2">
          {CATEGORIES.map((cat) => (
            <button
              key={cat.value}
              type="button"
              onClick={() => setCategory(cat.value)}
              className={`p-2 rounded border text-left transition-colors ${
                category === cat.value
                  ? "border-foreground bg-foreground/10 text-foreground"
                  : "border-border text-muted-foreground hover:border-foreground-muted"
              }`}
            >
              <div className="text-sm">{cat.label}</div>
              <div className="text-xs text-foreground-subtle">{cat.description}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Description */}
      <div>
        <label htmlFor="description" className="text-sm text-muted-foreground block mb-1.5">
          Description
        </label>
        <textarea
          id="description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Describe your suggestion in detail. What problem does it solve? How should it work?"
          rows={4}
          maxLength={1000}
          className="w-full px-3 py-2 bg-background border border-border rounded text-sm text-foreground placeholder-muted-foreground focus:outline-none focus:border-foreground-muted resize-none transition-colors"
        />
        <div className="text-xs text-foreground-subtle text-right mt-1">
          {description.length}/1000
        </div>
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={isSubmitting || title.trim().length < 5 || description.trim().length < 10}
        className="w-full flex items-center justify-center gap-2 py-2.5 px-4 bg-foreground text-background rounded text-sm font-medium hover:bg-foreground-bright disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {isSubmitting ? (
          "Submitting..."
        ) : (
          <>
            <Send className="w-4 h-4" />
            Submit Suggestion
          </>
        )}
      </button>

      {/* Rate limit info */}
      <p className="text-xs text-foreground-subtle text-center">
        Submissions are limited to 5 per hour to prevent spam.
      </p>
    </form>
  );
}

export default SuggestionForm;
