"use client";

/**
 * FeedSeeder Component
 *
 * Previously auto-seeded fake demo data. Now a no-op.
 * Kept for backwards compatibility with existing imports.
 */

export interface FeedSeederProps {
  onSeeded?: () => void;
}

export function FeedSeeder(_props: FeedSeederProps) {
  return null;
}

export default FeedSeeder;
