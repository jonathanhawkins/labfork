/**
 * Contributor Profile Component
 *
 * Display contributor profile with stats, badges, and rank.
 */

"use client";

import { useContributor } from "@/lib/compute/useContributor";
import type { ContributorRank } from "@/lib/compute/user-types";

interface ContributorProfileProps {
  userId: string;
  showBadges?: boolean;
  showDevices?: boolean;
}

const RANK_COLORS: Record<ContributorRank, string> = {
  novice: "text-gray-600 bg-gray-100",
  contributor: "text-blue-600 bg-blue-100",
  expert: "text-purple-600 bg-purple-100",
  legend: "text-yellow-600 bg-yellow-100",
};

const RANK_LABELS: Record<ContributorRank, string> = {
  novice: "Novice",
  contributor: "Contributor",
  expert: "Expert",
  legend: "Legend",
};

export default function ContributorProfile({
  userId,
  showBadges = true,
  showDevices = false,
}: ContributorProfileProps) {
  const { profile, isLoading, error } = useContributor(userId);

  if (isLoading) {
    return (
      <div className="animate-pulse">
        <div className="h-24 bg-gray-200 rounded-lg"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-red-600 p-4 border border-red-200 rounded-lg">
        Failed to load profile: {error.message}
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="text-gray-500 p-4 border border-gray-200 rounded-lg">
        Profile not found
      </div>
    );
  }

  const rankColor = RANK_COLORS[profile.rank];
  const rankLabel = RANK_LABELS[profile.rank];

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-start gap-4">
        {/* Avatar */}
        <div className="w-16 h-16 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white text-2xl font-bold">
          {profile.displayName.charAt(0).toUpperCase()}
        </div>

        {/* Name and Rank */}
        <div className="flex-1">
          <h2 className="text-xl font-bold">{profile.displayName}</h2>
          <span className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${rankColor}`}>
            {rankLabel}
          </span>
          {profile.bio && (
            <p className="mt-2 text-gray-600">{profile.bio}</p>
          )}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <div className="text-2xl font-bold text-blue-600">
            {profile.totalCreditsEarned.toLocaleString()}
          </div>
          <div className="text-sm text-gray-600">Credits Earned</div>
        </div>
        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <div className="text-2xl font-bold text-green-600">
            {profile.totalTasksCompleted.toLocaleString()}
          </div>
          <div className="text-sm text-gray-600">Tasks Completed</div>
        </div>
        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <div className="text-2xl font-bold text-purple-600">
            {Math.round(profile.totalComputeTime / 3600)}h
          </div>
          <div className="text-sm text-gray-600">Compute Time</div>
        </div>
      </div>

      {/* Badges */}
      {showBadges && profile.badges.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Badges</h3>
          <div className="flex flex-wrap gap-2">
            {profile.badges.map((badge) => (
              <div
                key={badge.id}
                className="flex items-center gap-2 px-3 py-2 bg-white border border-gray-200 rounded-lg hover:border-gray-300 transition-colors"
                title={badge.description}
              >
                <span className="text-2xl">{badge.icon}</span>
                <div className="text-sm">
                  <div className="font-medium">{badge.name}</div>
                  <div className="text-gray-500 text-xs">
                    {new Date(badge.earnedAt).toLocaleDateString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Devices */}
      {showDevices && profile.devices.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">
            Devices ({profile.devices.length})
          </h3>
          <div className="flex flex-wrap gap-2">
            {profile.devices.map((deviceId) => (
              <div
                key={deviceId}
                className="px-3 py-1 bg-gray-100 text-gray-700 text-sm rounded-full font-mono"
              >
                {deviceId.slice(0, 12)}...
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Member Since */}
      <div className="text-sm text-gray-500">
        Member since {new Date(profile.joinedAt).toLocaleDateString()}
      </div>
    </div>
  );
}
