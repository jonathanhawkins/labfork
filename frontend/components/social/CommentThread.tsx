"use client";

/**
 * CommentThread Component
 *
 * Displays a threaded comment with nested replies and reactions.
 */

import { useState, useCallback } from "react";
import Image from "next/image";
import { cn } from "@/lib/utils";
import {
  MessageCircle,
  MoreHorizontal,
  Pencil,
  Trash2,
  Flag,
  ChevronDown,
  ChevronUp,
  Reply,
} from "lucide-react";
import type {
  CommentWithReplies,
  ReactionType,
  ReactionCounts,
} from "@/lib/social/comments/types";
import {
  REACTION_EMOJIS,
  getTotalReactions,
} from "@/lib/social/comments/types";

export interface CommentThreadProps {
  /** Comment data with replies */
  comment: CommentWithReplies;
  /** Current user ID (for permissions) */
  currentUserId?: string;
  /** User's reaction on this comment */
  userReaction?: ReactionType | null;
  /** Depth in thread (0 = top-level) */
  depth?: number;
  /** Max visible depth before collapsing */
  maxVisibleDepth?: number;
  /** Reaction click handler */
  onReact?: (commentId: string, type: ReactionType) => void;
  /** Reply click handler */
  onReply?: (commentId: string) => void;
  /** Edit click handler */
  onEdit?: (commentId: string) => void;
  /** Delete click handler */
  onDelete?: (commentId: string) => void;
  /** Report click handler */
  onReport?: (commentId: string) => void;
  /** Custom class name */
  className?: string;
}

/**
 * Format relative time
 */
function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMinutes = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffMinutes < 1) return "just now";
  if (diffMinutes < 60) return `${diffMinutes}m`;
  if (diffHours < 24) return `${diffHours}h`;
  if (diffDays < 7) return `${diffDays}d`;
  if (diffDays < 30) return `${Math.floor(diffDays / 7)}w`;
  return date.toLocaleDateString();
}

/**
 * Reaction picker component
 */
function ReactionPicker({
  onSelect,
  onClose,
}: {
  onSelect: (type: ReactionType) => void;
  onClose: () => void;
}) {
  const reactions: ReactionType[] = [
    "like",
    "love",
    "insightful",
    "celebrate",
    "curious",
    "disagree",
  ];

  return (
    <div className="absolute bottom-full left-0 mb-1 p-1 rounded-lg bg-background-darker border border-border shadow-lg flex gap-1 z-10">
      {reactions.map((type) => (
        <button
          key={type}
          onClick={() => {
            onSelect(type);
            onClose();
          }}
          className="p-1.5 hover:bg-foreground-muted/10 rounded transition-colors text-lg"
          title={type}
        >
          {REACTION_EMOJIS[type]}
        </button>
      ))}
    </div>
  );
}

/**
 * Reaction display component
 */
function ReactionDisplay({
  counts,
  userReaction,
  onReact,
}: {
  counts: ReactionCounts;
  userReaction?: ReactionType | null;
  onReact?: (type: ReactionType) => void;
}) {
  const [showPicker, setShowPicker] = useState(false);
  const total = getTotalReactions(counts);

  // Get top reactions (non-zero counts)
  const topReactions = (Object.entries(counts) as [ReactionType, number][])
    .filter(([, count]) => count > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);

  return (
    <div className="relative flex items-center gap-2">
      <button
        onClick={() => setShowPicker(!showPicker)}
        className={cn(
          "flex items-center gap-1 text-xs transition-colors",
          userReaction
            ? "text-foreground"
            : "text-foreground-subtle hover:text-foreground-muted"
        )}
      >
        {userReaction ? (
          <span className="text-base">{REACTION_EMOJIS[userReaction]}</span>
        ) : (
          <span className="text-foreground-subtle">+</span>
        )}
      </button>

      {showPicker && (
        <ReactionPicker
          onSelect={(type) => onReact?.(type)}
          onClose={() => setShowPicker(false)}
        />
      )}

      {total > 0 && (
        <div className="flex items-center -space-x-1">
          {topReactions.map(([type]) => (
            <span key={type} className="text-sm">
              {REACTION_EMOJIS[type]}
            </span>
          ))}
          <span className="ml-1.5 text-xs text-foreground-subtle">{total}</span>
        </div>
      )}
    </div>
  );
}

/**
 * More menu component
 */
function MoreMenu({
  isOwner,
  onEdit,
  onDelete,
  onReport,
  onClose,
}: {
  isOwner: boolean;
  onEdit?: () => void;
  onDelete?: () => void;
  onReport?: () => void;
  onClose: () => void;
}) {
  return (
    <div className="absolute top-full right-0 mt-1 py-1 min-w-[120px] rounded-lg bg-background-darker border border-border shadow-lg z-10">
      {isOwner ? (
        <>
          <button
            onClick={() => {
              onEdit?.();
              onClose();
            }}
            className="w-full flex items-center gap-2 px-3 py-1.5 text-sm text-foreground-muted hover:text-foreground hover:bg-foreground-muted/10"
          >
            <Pencil className="w-3.5 h-3.5" />
            Edit
          </button>
          <button
            onClick={() => {
              onDelete?.();
              onClose();
            }}
            className="w-full flex items-center gap-2 px-3 py-1.5 text-sm text-red-400 hover:text-red-300 hover:bg-red-500/10"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Delete
          </button>
        </>
      ) : (
        <button
          onClick={() => {
            onReport?.();
            onClose();
          }}
          className="w-full flex items-center gap-2 px-3 py-1.5 text-sm text-foreground-muted hover:text-foreground hover:bg-foreground-muted/10"
        >
          <Flag className="w-3.5 h-3.5" />
          Report
        </button>
      )}
    </div>
  );
}

export function CommentThread({
  comment,
  currentUserId,
  userReaction,
  depth = 0,
  maxVisibleDepth = 3,
  onReact,
  onReply,
  onEdit,
  onDelete,
  onReport,
  className,
}: CommentThreadProps) {
  const [isCollapsed, setIsCollapsed] = useState(depth >= maxVisibleDepth);
  const [showMenu, setShowMenu] = useState(false);
  const [showReplies, setShowReplies] = useState(true);

  const isOwner = currentUserId === comment.author.id;
  const hasReplies = comment.replies && comment.replies.length > 0;

  const handleReact = useCallback(
    (type: ReactionType) => {
      onReact?.(comment.id, type);
    },
    [comment.id, onReact]
  );

  if (isCollapsed) {
    return (
      <button
        onClick={() => setIsCollapsed(false)}
        className={cn(
          "flex items-center gap-2 py-2 text-sm text-foreground-muted hover:text-foreground transition-colors",
          className
        )}
      >
        <ChevronDown className="w-4 h-4" />
        <span>Show reply from {comment.author.displayName}</span>
        {hasReplies && (
          <span className="text-foreground-subtle">
            (+{comment.replyCount} more)
          </span>
        )}
      </button>
    );
  }

  return (
    <div className={cn("group", className)}>
      {/* Comment header */}
      <div className="flex items-start gap-3">
        {/* Avatar */}
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-foreground-muted/20 flex items-center justify-center text-sm font-medium text-foreground-muted overflow-hidden">
          {comment.author.avatar ? (
            <Image
              src={comment.author.avatar}
              alt={comment.author.displayName}
              width={32}
              height={32}
              className="w-full h-full object-cover"
              unoptimized
            />
          ) : (
            comment.author.displayName.charAt(0).toUpperCase()
          )}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Author line */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium text-sm text-foreground">
              {comment.author.displayName}
            </span>
            <span className="text-xs text-foreground-subtle">
              @{comment.author.username}
            </span>
            <span className="text-xs text-foreground-subtle">
              {formatRelativeTime(comment.createdAt)}
            </span>
            {comment.isEdited && (
              <span className="text-xs text-foreground-subtle">(edited)</span>
            )}
          </div>

          {/* Comment body */}
          <div className="mt-1 text-sm text-foreground whitespace-pre-wrap break-words">
            {comment.content}
          </div>

          {/* Actions */}
          <div className="mt-2 flex items-center gap-4">
            {/* Reactions */}
            <ReactionDisplay
              counts={comment.reactionCounts}
              userReaction={userReaction}
              onReact={handleReact}
            />

            {/* Reply */}
            <button
              onClick={() => onReply?.(comment.id)}
              className="flex items-center gap-1 text-xs text-foreground-subtle hover:text-foreground-muted transition-colors"
            >
              <Reply className="w-3.5 h-3.5" />
              Reply
            </button>

            {/* Reply count */}
            {hasReplies && (
              <button
                onClick={() => setShowReplies(!showReplies)}
                className="flex items-center gap-1 text-xs text-foreground-subtle hover:text-foreground-muted transition-colors"
              >
                <MessageCircle className="w-3.5 h-3.5" />
                {comment.replyCount} {comment.replyCount === 1 ? "reply" : "replies"}
                {showReplies ? (
                  <ChevronUp className="w-3 h-3" />
                ) : (
                  <ChevronDown className="w-3 h-3" />
                )}
              </button>
            )}

            {/* More menu */}
            <div className="relative ml-auto opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                onClick={() => setShowMenu(!showMenu)}
                className="p-1 text-foreground-subtle hover:text-foreground-muted transition-colors"
              >
                <MoreHorizontal className="w-4 h-4" />
              </button>

              {showMenu && (
                <MoreMenu
                  isOwner={isOwner}
                  onEdit={() => onEdit?.(comment.id)}
                  onDelete={() => onDelete?.(comment.id)}
                  onReport={() => onReport?.(comment.id)}
                  onClose={() => setShowMenu(false)}
                />
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Nested replies */}
      {hasReplies && showReplies && (
        <div className="ml-11 mt-3 pl-4 border-l border-border space-y-3">
          {comment.replies.map((reply) => (
            <CommentThread
              key={reply.id}
              comment={reply}
              currentUserId={currentUserId}
              depth={depth + 1}
              maxVisibleDepth={maxVisibleDepth}
              onReact={onReact}
              onReply={onReply}
              onEdit={onEdit}
              onDelete={onDelete}
              onReport={onReport}
            />
          ))}
        </div>
      )}

      {/* Collapse button for deep threads */}
      {depth > 0 && (
        <button
          onClick={() => setIsCollapsed(true)}
          className="mt-2 ml-11 flex items-center gap-1 text-xs text-foreground-subtle hover:text-foreground-muted transition-colors"
        >
          <ChevronUp className="w-3 h-3" />
          Collapse
        </button>
      )}
    </div>
  );
}

export default CommentThread;
