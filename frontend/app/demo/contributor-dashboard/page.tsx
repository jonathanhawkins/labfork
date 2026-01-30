"use client";

import { ContributorDashboard } from "@/components/compute";

export default function ContributorDashboardDemo() {
  const handleSettingsClick = () => {
    console.log("Settings clicked - would open settings dialog");
  };

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="max-w-7xl mx-auto px-4">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground-bright mb-2">
            Contributor Dashboard Demo
          </h1>
          <p className="text-sm text-muted-foreground">
            Real-time compute contribution tracking with animated counters and live progress
          </p>
        </div>

        <ContributorDashboard onSettingsClick={handleSettingsClick} />
      </div>
    </div>
  );
}
