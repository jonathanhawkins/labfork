/**
 * User Profile and Contribution Types
 *
 * Type definitions for contributor profiles, badges, and ranks.
 */

/**
 * Contributor rank based on tasks completed
 */
export type ContributorRank = "novice" | "contributor" | "expert" | "legend";

/**
 * Badge earned by contributors
 */
export interface Badge {
  /** Unique badge ID */
  id: string;
  /** Badge name */
  name: string;
  /** Badge description */
  description: string;
  /** Badge icon (emoji or URL) */
  icon: string;
  /** When the badge was earned */
  earnedAt: string;
}

/**
 * Contributor profile
 */
export interface ContributorProfile {
  /** User ID */
  userId: string;
  /** Display name */
  displayName: string;
  /** Avatar URL (optional) */
  avatarUrl?: string;
  /** Bio (optional) */
  bio?: string;
  /** Contributor rank */
  rank: ContributorRank;
  /** Total credits earned */
  totalCreditsEarned: number;
  /** Total tasks completed */
  totalTasksCompleted: number;
  /** Total compute time in seconds */
  totalComputeTime: number;
  /** Device IDs owned by this user */
  devices: string[];
  /** Badges earned */
  badges: Badge[];
  /** When the user joined */
  joinedAt: string;
}

/**
 * Badge definitions
 */
export const BADGE_DEFINITIONS = {
  FIRST_TASK: {
    id: "first_task",
    name: "First Contribution",
    description: "Completed your first task",
    icon: "🌟",
  },
  HUNDRED_TASKS: {
    id: "hundred_tasks",
    name: "Century",
    description: "Completed 100 tasks",
    icon: "💯",
  },
  THOUSAND_CREDITS: {
    id: "thousand_credits",
    name: "Millionaire",
    description: "Earned 1,000 credits",
    icon: "💰",
  },
  WEEK_STREAK: {
    id: "week_streak",
    name: "Week Warrior",
    description: "Contributed every day for a week",
    icon: "🔥",
  },
  POWER_CONTRIBUTOR: {
    id: "power_contributor",
    name: "Power Contributor",
    description: "Registered a power-tier device",
    icon: "⚡",
  },
  THOUSAND_TASKS: {
    id: "thousand_tasks",
    name: "Legend",
    description: "Completed 1,000 tasks",
    icon: "🏆",
  },
} as const;

/**
 * Calculate contributor rank based on tasks completed
 */
export function calculateRank(tasksCompleted: number): ContributorRank {
  if (tasksCompleted >= 1000) return "legend";
  if (tasksCompleted >= 100) return "expert";
  if (tasksCompleted >= 10) return "contributor";
  return "novice";
}

/**
 * Check which badges should be awarded based on stats
 */
export function checkBadgeEligibility(
  tasksCompleted: number,
  creditsEarned: number,
  hasPowerDevice: boolean
): string[] {
  const eligibleBadges: string[] = [];

  if (tasksCompleted >= 1) {
    eligibleBadges.push(BADGE_DEFINITIONS.FIRST_TASK.id);
  }
  if (tasksCompleted >= 100) {
    eligibleBadges.push(BADGE_DEFINITIONS.HUNDRED_TASKS.id);
  }
  if (tasksCompleted >= 1000) {
    eligibleBadges.push(BADGE_DEFINITIONS.THOUSAND_TASKS.id);
  }
  if (creditsEarned >= 1000) {
    eligibleBadges.push(BADGE_DEFINITIONS.THOUSAND_CREDITS.id);
  }
  if (hasPowerDevice) {
    eligibleBadges.push(BADGE_DEFINITIONS.POWER_CONTRIBUTOR.id);
  }

  return eligibleBadges;
}

/**
 * Generate a default display name from user ID
 */
export function generateDisplayName(userId: string): string {
  // Take last 6 characters of user ID for anonymity
  const shortId = userId.slice(-6);
  return `Contributor ${shortId}`;
}
