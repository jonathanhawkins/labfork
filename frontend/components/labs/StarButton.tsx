"use client";

/**
 * StarButton Component
 *
 * Star/unstar button with count display and optimistic updates.
 */

import { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { Star, Loader2 } from "lucide-react";

export interface StarButtonProps {
  /** Lab ID to star */
  labId: string;
  /** Initial star count */
  initialCount: number;
  /** Initial starred state */
  initialStarred: boolean;
  /** Size variant */
  size?: "sm" | "md" | "lg";
  /** Show count */
  showCount?: boolean;
  /** Custom class name */
  className?: string;
  /** Callback after star/unstar */
  onToggle?: (starred: boolean, count: number) => void;
}

export function StarButton({
  labId,
  initialCount,
  initialStarred,
  size = "md",
  showCount = true,
  className,
  onToggle,
}: StarButtonProps) {
  const [isStarred, setIsStarred] = useState(initialStarred);
  const [count, setCount] = useState(initialCount);
  const [isLoading, setIsLoading] = useState(false);

  const handleClick = useCallback(async () => {
    if (isLoading) return;

    // Optimistic update
    const newStarred = !isStarred;
    const newCount = newStarred ? count + 1 : count - 1;
    setIsStarred(newStarred);
    setCount(newCount);

    setIsLoading(true);

    try {
      const response = await fetch(`/api/labs/${labId}/star?toggle=true`, {
        method: "POST",
      });

      const data = await response.json();

      if (data.success) {
        // Update with server response
        setIsStarred(data.starred);
        setCount(data.count);
        onToggle?.(data.starred, data.count);
      } else {
        // Revert on error
        setIsStarred(isStarred);
        setCount(count);
      }
    } catch (error) {
      console.error("Failed to toggle star:", error);
      // Revert on error
      setIsStarred(isStarred);
      setCount(count);
    } finally {
      setIsLoading(false);
    }
  }, [labId, isStarred, count, isLoading, onToggle]);

  const sizeClasses = {
    sm: {
      button: "px-3 py-2 min-h-[44px] text-xs gap-1",
      icon: "w-4 h-4",
    },
    md: {
      button: "px-4 py-2.5 min-h-[44px] text-sm gap-1.5",
      icon: "w-5 h-5",
    },
    lg: {
      button: "px-5 py-3 min-h-[48px] text-base gap-2",
      icon: "w-5 h-5",
    },
  };

  const { button: buttonClass, icon: iconClass } = sizeClasses[size];

  return (
    <button
      onClick={handleClick}
      disabled={isLoading}
      className={cn(
        "flex items-center rounded-lg border transition-all duration-200 active:scale-[0.98]",
        isStarred
          ? "border-yellow-500/30 bg-yellow-500/10 text-yellow-400 hover:bg-yellow-500/20 active:bg-yellow-500/25"
          : "border-border bg-background hover:bg-foreground-muted/10 text-foreground-muted hover:text-foreground active:bg-foreground-muted/15",
        isLoading && "opacity-70 cursor-not-allowed",
        buttonClass,
        className
      )}
      title={isStarred ? "Unstar this lab" : "Star this lab"}
    >
      {isLoading ? (
        <Loader2 className={cn(iconClass, "animate-spin")} />
      ) : (
        <Star
          className={cn(iconClass, isStarred && "fill-current")}
        />
      )}
      {showCount && (
        <span className="font-medium">
          {count > 0 ? count : "Star"}
        </span>
      )}
    </button>
  );
}

/**
 * Simple star icon button (no count)
 */
export function StarIconButton({
  labId,
  initialStarred,
  size = "md",
  className,
  onToggle,
}: Omit<StarButtonProps, "initialCount" | "showCount">) {
  const [isStarred, setIsStarred] = useState(initialStarred);
  const [isLoading, setIsLoading] = useState(false);

  const handleClick = useCallback(async () => {
    if (isLoading) return;

    const newStarred = !isStarred;
    setIsStarred(newStarred);
    setIsLoading(true);

    try {
      const response = await fetch(`/api/labs/${labId}/star?toggle=true`, {
        method: "POST",
      });

      const data = await response.json();

      if (data.success) {
        setIsStarred(data.starred);
        onToggle?.(data.starred, data.count);
      } else {
        setIsStarred(isStarred);
      }
    } catch (error) {
      console.error("Failed to toggle star:", error);
      setIsStarred(isStarred);
    } finally {
      setIsLoading(false);
    }
  }, [labId, isStarred, isLoading, onToggle]);

  const sizeClasses = {
    sm: "p-2.5 min-w-[44px] min-h-[44px]",
    md: "p-3 min-w-[44px] min-h-[44px]",
    lg: "p-3.5 min-w-[48px] min-h-[48px]",
  };

  const iconSizes = {
    sm: "w-4 h-4",
    md: "w-5 h-5",
    lg: "w-6 h-6",
  };

  return (
    <button
      onClick={handleClick}
      disabled={isLoading}
      className={cn(
        "flex items-center justify-center rounded-lg transition-all duration-200 active:scale-[0.95]",
        isStarred
          ? "text-yellow-400 hover:text-yellow-300 active:text-yellow-200"
          : "text-foreground-subtle hover:text-foreground-muted active:text-foreground",
        isLoading && "opacity-70 cursor-not-allowed",
        sizeClasses[size],
        className
      )}
      title={isStarred ? "Unstar" : "Star"}
    >
      {isLoading ? (
        <Loader2 className={cn(iconSizes[size], "animate-spin")} />
      ) : (
        <Star
          className={cn(iconSizes[size], isStarred && "fill-current")}
        />
      )}
    </button>
  );
}

export default StarButton;
