"use client";

/**
 * FeedSeeder Component
 *
 * Automatically seeds the activity feed with demo data on first visit.
 * This component renders nothing - it just handles the seeding logic.
 */

import { useEffect, useRef } from "react";

export interface FeedSeederProps {
  /** Called after seeding completes */
  onSeeded?: () => void;
}

export function FeedSeeder({ onSeeded }: FeedSeederProps) {
  const hasChecked = useRef(false);

  useEffect(() => {
    // Only run once
    if (hasChecked.current) return;
    hasChecked.current = true;

    async function checkAndSeed() {
      try {
        // Check if we have data
        const checkResponse = await fetch("/api/activity/seed");
        const checkData = await checkResponse.json();

        if (!checkData.hasData) {
          // No data - seed it
          const seedResponse = await fetch("/api/activity/seed", {
            method: "POST",
          });
          const seedData = await seedResponse.json();

          if (seedData.seeded) {
            console.log(`[FeedSeeder] Seeded ${seedData.count} demo activities`);
            onSeeded?.();
          }
        }
      } catch (error) {
        // Silently fail - seeding is optional
        console.error("[FeedSeeder] Failed to seed:", error);
      }
    }

    checkAndSeed();
  }, [onSeeded]);

  // Render nothing
  return null;
}

export default FeedSeeder;
