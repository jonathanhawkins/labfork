/**
 * Contributor Profile Database Helpers
 *
 * Helper functions for managing contributor profiles in Supabase.
 */

import { supabase, createServerClient } from "./client";
import { getUserCredits } from "./credits";
import { getUserDevices } from "./devices";
import type {
  ContributorProfile,
  Badge,
} from "@/lib/compute/user-types";
import {
  calculateRank,
  checkBadgeEligibility,
  generateDisplayName,
  BADGE_DEFINITIONS,
} from "@/lib/compute/user-types";

/**
 * Get contributor profile by user ID
 */
export async function getContributorProfile(
  userId: string
): Promise<ContributorProfile | null> {
  if (!supabase) {
    console.error("Supabase client not configured");
    return null;
  }

  try {
    // Fetch profile data
    const { data: profileData, error: profileError } = await supabase
      .from("contributor_profiles")
      .select("*")
      .eq("user_id", userId)
      .single();

    // Get user credits
    const credits = await getUserCredits(userId);

    // Get user devices
    const devices = await getUserDevices(userId);

    // Calculate stats from devices
    let totalTasksCompleted = 0;
    let totalComputeTime = 0;
    const deviceIds: string[] = [];

    for (const device of devices) {
      totalTasksCompleted += device.stats.tasksCompleted;
      totalComputeTime += device.stats.totalComputeTime;
      deviceIds.push(device.id);
    }

    const totalCreditsEarned = credits?.totalEarned || 0;
    const rank = calculateRank(totalTasksCompleted);

    // Check if profile exists in database
    let displayName = generateDisplayName(userId);
    let avatarUrl: string | undefined;
    let bio: string | undefined;
    let badges: Badge[] = [];
    let joinedAt = new Date().toISOString();

    if (profileData && !profileError) {
      const record = profileData as any;
      displayName = record.display_name;
      avatarUrl = record.avatar_url || undefined;
      bio = record.bio || undefined;
      badges = (record.badges as Badge[]) || [];
      joinedAt = record.joined_at;
    } else if (profileError && profileError.code !== "PGRST116") {
      console.error("Error fetching contributor profile:", profileError);
      return null;
    }

    // Check for new badges
    const hasPowerDevice = devices.some((d) => d.tier === "power");
    const eligibleBadgeIds = checkBadgeEligibility(
      totalTasksCompleted,
      totalCreditsEarned,
      hasPowerDevice
    );

    // Add any newly earned badges
    const existingBadgeIds = new Set(badges.map((b) => b.id));
    for (const badgeId of eligibleBadgeIds) {
      if (!existingBadgeIds.has(badgeId)) {
        // Find badge definition
        const badgeDef = Object.values(BADGE_DEFINITIONS).find(
          (def) => def.id === badgeId
        );
        if (badgeDef) {
          badges.push({
            ...badgeDef,
            earnedAt: new Date().toISOString(),
          });
        }
      }
    }

    // Update profile with new badges if needed
    if (badges.length > existingBadgeIds.size) {
      await updateContributorProfile(userId, { badges });
    }

    return {
      userId,
      displayName,
      avatarUrl,
      bio,
      rank,
      totalCreditsEarned,
      totalTasksCompleted,
      totalComputeTime,
      devices: deviceIds,
      badges,
      joinedAt,
    };
  } catch (error) {
    console.error("Error fetching contributor profile:", error);
    return null;
  }
}

/**
 * Create or update contributor profile
 */
export async function updateContributorProfile(
  userId: string,
  updates: {
    displayName?: string;
    avatarUrl?: string;
    bio?: string;
    badges?: Badge[];
  }
): Promise<boolean> {
  if (!supabase) {
    console.error("Supabase client not configured");
    return false;
  }

  const updateData: Record<string, any> = {
    updated_at: new Date().toISOString(),
  };

  if (updates.displayName !== undefined) {
    updateData.display_name = updates.displayName;
  }
  if (updates.avatarUrl !== undefined) {
    updateData.avatar_url = updates.avatarUrl;
  }
  if (updates.bio !== undefined) {
    updateData.bio = updates.bio;
  }
  if (updates.badges !== undefined) {
    updateData.badges = updates.badges;
  }

  // Try to update first
  const { error: updateError } = await supabase
    .from("contributor_profiles")
    // @ts-ignore - Type inference issue with Supabase update
    .update(updateData)
    .eq("user_id", userId);

  // If no rows updated, create new profile
  if (updateError && updateError.code === "PGRST116") {
    const insertData = {
      user_id: userId,
      display_name: updates.displayName || generateDisplayName(userId),
      avatar_url: updates.avatarUrl || null,
      bio: updates.bio || null,
      badges: updates.badges || [],
      joined_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };

    const { error: insertError } = await supabase
      .from("contributor_profiles")
      // @ts-ignore - Type inference issue with Supabase insert
      .insert(insertData);

    if (insertError) {
      console.error("Error creating contributor profile:", insertError);
      return false;
    }

    return true;
  }

  if (updateError) {
    console.error("Error updating contributor profile:", updateError);
    return false;
  }

  return true;
}

/**
 * Get leaderboard of top contributors
 */
export async function getLeaderboard(
  limit = 10
): Promise<ContributorProfile[]> {
  if (!supabase) {
    console.error("Supabase client not configured");
    return [];
  }

  try {
    // Get all user credits sorted by total earned
    const { data: creditsData, error: creditsError } = await supabase
      .from("user_credits")
      .select("user_id, total_earned")
      .order("total_earned", { ascending: false })
      .limit(limit);

    if (creditsError || !creditsData) {
      console.error("Error fetching leaderboard data:", creditsError);
      return [];
    }

    // Fetch full profiles for top users
    const profiles: ContributorProfile[] = [];
    for (const row of creditsData) {
      const record = row as any;
      const profile = await getContributorProfile(record.user_id);
      if (profile) {
        profiles.push(profile);
      }
    }

    return profiles;
  } catch (error) {
    console.error("Error fetching leaderboard:", error);
    return [];
  }
}

/**
 * Get global contribution statistics
 */
export async function getContributionStats(): Promise<{
  totalContributors: number;
  totalCreditsEarned: number;
  totalTasksCompleted: number;
}> {
  if (!supabase) {
    console.error("Supabase client not configured");
    return {
      totalContributors: 0,
      totalCreditsEarned: 0,
      totalTasksCompleted: 0,
    };
  }

  try {
    // Count unique contributors (users with credits)
    const { count: contributorCount } = await supabase
      .from("user_credits")
      .select("*", { count: "exact", head: true })
      .gt("total_earned", 0);

    // Sum total credits earned
    const { data: creditsSum } = await supabase
      .from("user_credits")
      .select("total_earned");

    const totalCreditsEarned = creditsSum?.reduce(
      (sum: number, row: any) => sum + (row.total_earned || 0),
      0
    ) || 0;

    // Sum total tasks completed
    const { data: devicesData } = await supabase
      .from("devices")
      .select("stats");

    const totalTasksCompleted = devicesData?.reduce(
      (sum: number, row: any) => sum + (row.stats?.tasksCompleted || 0),
      0
    ) || 0;

    return {
      totalContributors: contributorCount || 0,
      totalCreditsEarned,
      totalTasksCompleted,
    };
  } catch (error) {
    console.error("Error fetching contribution stats:", error);
    return {
      totalContributors: 0,
      totalCreditsEarned: 0,
      totalTasksCompleted: 0,
    };
  }
}

/**
 * Initialize contributor profile (call when user first registers a device)
 */
export async function initializeContributorProfile(
  userId: string,
  displayName?: string
): Promise<boolean> {
  if (!supabase) {
    console.error("Supabase client not configured");
    return false;
  }

  const name = displayName || generateDisplayName(userId);

  const { error } = await supabase
    .from("contributor_profiles")
    // @ts-ignore - Type inference issue with Supabase insert
    .insert({
      user_id: userId,
      display_name: name,
      avatar_url: null,
      bio: null,
      badges: [],
      joined_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    });

  if (error) {
    // Ignore if profile already exists
    if (error.code !== "23505") {
      console.error("Error initializing contributor profile:", error);
      return false;
    }
  }

  return true;
}
