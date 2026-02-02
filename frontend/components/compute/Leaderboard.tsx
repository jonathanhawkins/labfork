/**
 * Leaderboard Component
 *
 * Display top contributors ranked by credits earned.
 * Shows credits, tasks completed, and compute time.
 */

"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useLeaderboard } from "@/lib/compute/useContributor";
import type { ContributorRank } from "@/lib/compute/user-types";
import { cn } from "@/lib/utils";
import { Trophy, Clock, Zap, Award, Users } from "lucide-react";

interface LeaderboardProps {
  /** Maximum number of contributors to show */
  limit?: number;
  /** Additional CSS classes */
  className?: string;
}

const RANK_COLORS: Record<ContributorRank, string> = {
  novice: "text-foreground-muted",
  contributor: "text-blue-400",
  expert: "text-purple-400",
  legend: "text-yellow-400",
};

const RANK_BG: Record<ContributorRank, string> = {
  novice: "bg-foreground-muted/10",
  contributor: "bg-blue-500/10",
  expert: "bg-purple-500/10",
  legend: "bg-yellow-500/10",
};

/**
 * Format compute time from seconds to human-readable string
 */
function formatComputeTime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h`;
  return `${(seconds / 86400).toFixed(1)}d`;
}

export default function Leaderboard({ limit = 10, className }: LeaderboardProps) {
  const { leaders, isLoading, error } = useLeaderboard(limit);

  if (isLoading) {
    return (
      <div className={cn("space-y-2", className)}>
        {Array.from({ length: 5 }).map((_, i) => (
          <div
            key={i}
            className="animate-pulse h-20 bg-foreground-muted/10 rounded-lg"
          />
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className={cn("p-4 border border-red-500/20 bg-red-500/10 rounded-lg", className)}>
        <p className="text-sm text-red-400">
          Failed to load leaderboard: {error.message}
        </p>
      </div>
    );
  }

  if (leaders.length === 0) {
    return (
      <div className={cn("p-8 text-center border border-border rounded-lg bg-background-card", className)}>
        <Users className="w-10 h-10 text-foreground-muted mx-auto mb-3" />
        <p className="text-sm font-medium text-foreground mb-1">
          No contributors yet
        </p>
        <p className="text-xs text-foreground-muted">
          Be the first to contribute compute power and appear on the leaderboard!
        </p>
      </div>
    );
  }

  return (
    <div className={cn("space-y-2", className)}>
      <AnimatePresence mode="popLayout">
        {leaders.map((contributor, index) => {
          const rankColor = RANK_COLORS[contributor.rank];
          const rankBg = RANK_BG[contributor.rank];
          const isTopThree = index < 3;

          return (
            <motion.div
              key={contributor.userId}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.2, delay: index * 0.05 }}
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
              className={cn(
                "flex items-center gap-3 sm:gap-4 p-3 sm:p-4 rounded-lg transition-colors active:bg-foreground/5",
                isTopThree
                  ? "bg-gradient-to-r from-yellow-500/10 to-background-card border-2 border-yellow-500/20"
                  : "bg-background-card border border-border hover:border-foreground-muted/30"
              )}
            >
              {/* Rank Position */}
              <div className="text-center w-10 sm:w-12 flex-shrink-0">
                {index === 0 ? (
                  <Trophy className="w-6 h-6 sm:w-7 sm:h-7 text-yellow-400 mx-auto" />
                ) : index === 1 ? (
                  <Trophy className="w-5 h-5 sm:w-6 sm:h-6 text-gray-300 mx-auto" />
                ) : index === 2 ? (
                  <Trophy className="w-5 h-5 sm:w-6 sm:h-6 text-amber-600 mx-auto" />
                ) : (
                  <span className="text-lg sm:text-xl font-bold text-foreground-muted">
                    #{index + 1}
                  </span>
                )}
              </div>

              {/* Avatar */}
              <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-sm sm:text-base flex-shrink-0">
                {contributor.displayName.charAt(0).toUpperCase()}
              </div>

              {/* Name and Stats */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <h3 className="font-medium text-foreground text-sm sm:text-base truncate">
                    {contributor.displayName}
                  </h3>
                  <span
                    className={cn(
                      "text-[10px] sm:text-xs font-medium px-1.5 sm:px-2 py-0.5 rounded-full capitalize",
                      rankColor,
                      rankBg
                    )}
                  >
                    {contributor.rank}
                  </span>
                </div>

                {/* Stats Row - Mobile Optimized */}
                <div className="flex items-center gap-2 sm:gap-4 mt-1 text-[10px] sm:text-xs text-foreground-muted">
                  <div className="flex items-center gap-1">
                    <Zap className="w-3 h-3" />
                    <span>{contributor.totalTasksCompleted.toLocaleString()} tasks</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    <span>{formatComputeTime(contributor.totalComputeTime)}</span>
                  </div>
                  {contributor.badges.length > 0 && (
                    <div className="hidden sm:flex items-center gap-1">
                      <span>{contributor.badges.map((b) => b.icon).join(" ")}</span>
                    </div>
                  )}
                </div>

                {/* Mobile Badges */}
                {contributor.badges.length > 0 && (
                  <div className="flex sm:hidden items-center gap-1 mt-1">
                    <span className="text-xs">
                      {contributor.badges.map((b) => b.icon).join(" ")}
                    </span>
                  </div>
                )}
              </div>

              {/* Credits */}
              <div className="text-right flex-shrink-0">
                <div className="flex items-center gap-1 justify-end">
                  <Award className="w-4 h-4 text-blue-400" />
                  <span className="text-lg sm:text-xl font-bold text-blue-400">
                    {contributor.totalCreditsEarned.toLocaleString()}
                  </span>
                </div>
                <div className="text-[10px] sm:text-xs text-foreground-muted">credits</div>
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
