"use client";

import PublicLabView from "@/components/lab/PublicLabView";

/**
 * Public AI Lab Viewer
 *
 * This page shows the live lab view with:
 * - 3D visualization of AI agents working
 * - Real-time task progress from the compute network
 * - Activity feed
 * - Mobile-optimized with bottom navigation tabs
 *
 * The global Navigation component handles the header.
 */
export default function WatchPage() {
  return <PublicLabView />;
}
