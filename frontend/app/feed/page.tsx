/**
 * Activity Feed Page
 *
 * Shows a personalized feed of activity from labs and users the viewer follows.
 * Auto-seeds demo data on first visit.
 */

import { FeedPageClient } from "@/components/social";

export const metadata = {
  title: "Activity Feed | LabFork",
  description: "Stay updated with the latest research activity from labs you follow",
};

export default function FeedPage() {
  return <FeedPageClient />;
}
