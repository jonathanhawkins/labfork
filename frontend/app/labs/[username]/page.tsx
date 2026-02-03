"use client";

/**
 * User Labs Page
 *
 * Lists all public labs by a specific user.
 */

import { useState, useEffect, useCallback } from "react";
import { useUser } from "@clerk/nextjs";
import { cn } from "@/lib/utils";
import {
  User,
  Loader2,
  Star,
  GitFork,
  Activity,
  Layers,
} from "lucide-react";
import { LabCard } from "@/components/labs/LabCard";
import type { Lab } from "@/lib/labs/types";
import { getClientUser } from "@/lib/auth/client";

interface UserLabsPageProps {
  params: {
    username: string;
  };
}

export default function UserLabsPage({ params }: UserLabsPageProps) {
  const { username } = params;

  // Clerk authentication
  const { user: clerkUser, isLoaded: isUserLoaded } = useUser();
  const currentUser = getClientUser(clerkUser);

  const [labs, setLabs] = useState<Lab[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isCurrentUser, setIsCurrentUser] = useState(false);

  // User stats
  const [totalStars, setTotalStars] = useState(0);
  const [totalForks, setTotalForks] = useState(0);

  const fetchUserLabs = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await fetch(`/api/labs/user/${username}`);
      const data = await response.json();

      if (data.success) {
        setLabs(data.labs);

        // Calculate total stars and forks
        const stars = data.labs.reduce((sum: number, lab: Lab) => sum + lab.stats.stars, 0);
        const forks = data.labs.reduce((sum: number, lab: Lab) => sum + lab.stats.forks, 0);
        setTotalStars(stars);
        setTotalForks(forks);

        // Check if this is the current user using proper auth
        setIsCurrentUser(currentUser?.username === username);
      } else {
        setError(data.error || "Failed to load labs");
      }
    } catch (err) {
      console.error("Failed to fetch user labs:", err);
      setError("Failed to load labs");
    } finally {
      setIsLoading(false);
    }
  }, [username, currentUser]);

  useEffect(() => {
    // Only fetch labs once user auth is loaded
    if (isUserLoaded) {
      fetchUserLabs();
    }
  }, [isUserLoaded, fetchUserLabs]);

  // Show loading state while user auth or labs are loading
  if (!isUserLoaded || isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-foreground-muted" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-foreground-bright mb-2">User not found</h1>
          <p className="text-foreground-muted mb-4">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Profile Header */}
      <div className="border-b border-border bg-gradient-to-b from-foreground-muted/5 to-transparent">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="flex items-center gap-4">
            {/* Avatar */}
            <div className="w-16 h-16 rounded-full bg-foreground-muted/10 border border-border flex items-center justify-center">
              <User className="w-8 h-8 text-foreground-subtle" />
            </div>

            {/* Info */}
            <div>
              <h1 className="text-2xl font-bold text-foreground-bright">
                {username}
                {isCurrentUser && (
                  <span className="ml-2 text-sm font-normal text-foreground-muted">(you)</span>
                )}
              </h1>
              <div className="flex items-center gap-4 mt-1 text-sm text-foreground-muted">
                <span className="flex items-center gap-1">
                  <Layers className="w-4 h-4" />
                  {labs.length} labs
                </span>
                <span className="flex items-center gap-1">
                  <Star className="w-4 h-4" />
                  {totalStars} stars
                </span>
                <span className="flex items-center gap-1">
                  <GitFork className="w-4 h-4" />
                  {totalForks} forks
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Labs Grid */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        <h2 className="text-lg font-medium text-foreground-bright mb-4">
          {isCurrentUser ? "Your Labs" : `${username}'s Labs`}
        </h2>

        {labs.length === 0 ? (
          <div className="text-center py-12 border border-border rounded-lg">
            <Layers className="w-12 h-12 mx-auto text-foreground-subtle mb-3" />
            <p className="text-sm text-foreground-muted">
              {isCurrentUser ? "You haven't created any labs yet" : "No public labs"}
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {labs.map((lab) => (
              <LabCard key={lab.id} lab={lab} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
