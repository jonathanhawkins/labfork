"use client";

/**
 * CommentForm Component
 *
 * Form for creating and editing comments with mention support.
 */

import { useState, useCallback, useRef, useEffect } from "react";
import { cn } from "@/lib/utils";
import { Send, X, Loader2, AtSign } from "lucide-react";
import type { CommentAuthor } from "@/lib/social/comments/types";
import { MAX_COMMENT_LENGTH } from "@/lib/social/comments/types";

export interface CommentFormProps {
  /** Current user (author) */
  author: CommentAuthor;
  /** Parent comment ID (for replies) */
  parentId?: string;
  /** Initial content (for editing) */
  initialContent?: string;
  /** Placeholder text */
  placeholder?: string;
  /** Auto focus on mount */
  autoFocus?: boolean;
  /** Is in compact mode */
  compact?: boolean;
  /** Submit handler */
  onSubmit: (content: string) => Promise<void>;
  /** Cancel handler (for editing/replying) */
  onCancel?: () => void;
  /** Custom class name */
  className?: string;
}

/**
 * Mention suggestion item
 */
interface MentionSuggestion {
  id: string;
  username: string;
  displayName: string;
  avatar?: string;
}

export function CommentForm({
  author,
  parentId,
  initialContent = "",
  placeholder = "Write a comment...",
  autoFocus = false,
  compact = false,
  onSubmit,
  onCancel,
  className,
}: CommentFormProps) {
  const [content, setContent] = useState(initialContent);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isFocused, setIsFocused] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [content]);

  // Auto focus
  useEffect(() => {
    if (autoFocus && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [autoFocus]);

  const handleSubmit = useCallback(async () => {
    const trimmed = content.trim();

    if (!trimmed) {
      setError("Comment cannot be empty");
      return;
    }

    if (trimmed.length > MAX_COMMENT_LENGTH) {
      setError(`Comment exceeds maximum length of ${MAX_COMMENT_LENGTH} characters`);
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      await onSubmit(trimmed);
      setContent("");
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to post comment");
    } finally {
      setIsSubmitting(false);
    }
  }, [content, onSubmit]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      // Cmd/Ctrl + Enter to submit
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        handleSubmit();
      }
      // Escape to cancel (if editing/replying)
      if (e.key === "Escape" && onCancel) {
        e.preventDefault();
        onCancel();
      }
    },
    [handleSubmit, onCancel]
  );

  const charactersRemaining = MAX_COMMENT_LENGTH - content.length;
  const isOverLimit = charactersRemaining < 0;
  const showCharacterCount = content.length > MAX_COMMENT_LENGTH * 0.8;

  return (
    <div className={cn("flex gap-3", className)}>
      {/* Avatar */}
      {!compact && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-foreground-muted/20 flex items-center justify-center text-sm font-medium text-foreground-muted overflow-hidden">
          {author.avatar ? (
            <img
              src={author.avatar}
              alt={author.displayName}
              className="w-full h-full object-cover"
            />
          ) : (
            author.displayName.charAt(0).toUpperCase()
          )}
        </div>
      )}

      {/* Form */}
      <div className="flex-1 min-w-0">
        <div
          className={cn(
            "rounded-lg border transition-colors",
            isFocused
              ? "border-foreground-muted bg-background"
              : "border-border bg-background-darker",
            error && "border-red-500/50"
          )}
        >
          {/* Textarea */}
          <textarea
            ref={textareaRef}
            value={content}
            onChange={(e) => setContent(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder={placeholder}
            disabled={isSubmitting}
            rows={1}
            className={cn(
              "w-full px-3 py-2 bg-transparent text-sm text-foreground placeholder:text-foreground-subtle resize-none focus:outline-none",
              compact ? "min-h-[36px]" : "min-h-[44px]"
            )}
            style={{ maxHeight: "200px" }}
          />

          {/* Actions bar */}
          {(isFocused || content.length > 0) && (
            <div className="flex items-center justify-between px-3 py-2 border-t border-border">
              <div className="flex items-center gap-2">
                {/* Mention button */}
                <button
                  type="button"
                  className="p-1 text-foreground-subtle hover:text-foreground-muted transition-colors"
                  title="Mention someone (@)"
                  onClick={() => {
                    setContent(content + "@");
                    textareaRef.current?.focus();
                  }}
                >
                  <AtSign className="w-4 h-4" />
                </button>

                {/* Character count */}
                {showCharacterCount && (
                  <span
                    className={cn(
                      "text-xs",
                      isOverLimit ? "text-red-400" : "text-foreground-subtle"
                    )}
                  >
                    {charactersRemaining}
                  </span>
                )}
              </div>

              <div className="flex items-center gap-2">
                {/* Cancel button */}
                {onCancel && (
                  <button
                    type="button"
                    onClick={onCancel}
                    disabled={isSubmitting}
                    className="px-2 py-1 text-xs text-foreground-muted hover:text-foreground transition-colors disabled:opacity-50"
                  >
                    Cancel
                  </button>
                )}

                {/* Submit button */}
                <button
                  type="button"
                  onClick={handleSubmit}
                  disabled={isSubmitting || !content.trim() || isOverLimit}
                  className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-foreground-bright text-background text-xs font-medium hover:bg-foreground-bright/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      Posting...
                    </>
                  ) : (
                    <>
                      <Send className="w-3.5 h-3.5" />
                      {parentId ? "Reply" : "Post"}
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Error message */}
        {error && (
          <p className="mt-1 text-xs text-red-400">{error}</p>
        )}

        {/* Keyboard shortcut hint */}
        {isFocused && !compact && (
          <p className="mt-1 text-xs text-foreground-subtle">
            Press <kbd className="px-1 py-0.5 rounded bg-foreground-muted/10">Cmd</kbd>+<kbd className="px-1 py-0.5 rounded bg-foreground-muted/10">Enter</kbd> to {parentId ? "reply" : "post"}
          </p>
        )}
      </div>
    </div>
  );
}

export default CommentForm;
