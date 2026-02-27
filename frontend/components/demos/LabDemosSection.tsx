"use client";

/**
 * LabDemosSection - Shows research demos belonging to a lab.
 *
 * Displayed on the lab Overview tab, above the 3D simulator.
 * Shows a grid of compact demo cards with a link to the full demos page.
 */

import Link from "next/link";
import { Beaker, ArrowRight } from "lucide-react";
import { getDemosForLab } from "@/lib/demos";
import { DemoCard } from "./DemoCard";

interface LabDemosSectionProps {
  labSlug: string;
}

export function LabDemosSection({ labSlug }: LabDemosSectionProps) {
  const demos = getDemosForLab(labSlug);

  if (demos.length === 0) return null;

  const readyDemos = demos.filter((d) => d.status === "ready");
  const totalReady = readyDemos.length;

  // Show up to 6 ready demos on the overview, prioritize variety
  const displayDemos = readyDemos.slice(0, 6);

  return (
    <div className="p-4 sm:p-6 rounded-lg border border-border">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Beaker className="w-4 h-4 text-foreground-muted" />
          <h3 className="text-base sm:text-lg font-medium text-foreground-bright">
            Research Demos
          </h3>
          <span className="text-xs text-foreground-subtle bg-foreground-muted/10 px-1.5 py-0.5 rounded-full">
            {totalReady}
          </span>
        </div>
        <Link
          href="/demos"
          className="inline-flex items-center gap-1 text-xs text-foreground-muted hover:text-foreground transition-colors min-h-[44px] px-2"
        >
          View all
          <ArrowRight className="w-3 h-3" />
        </Link>
      </div>

      {/* Demo grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {displayDemos.map((demo) => (
          <DemoCard key={demo.id} demo={demo} compact />
        ))}
      </div>

      {/* Overflow indicator */}
      {totalReady > 6 && (
        <div className="mt-3 text-center">
          <Link
            href="/demos"
            className="inline-flex items-center gap-1.5 text-xs text-foreground-muted hover:text-foreground-bright transition-colors min-h-[44px] px-4"
          >
            +{totalReady - 6} more demos
            <ArrowRight className="w-3 h-3" />
          </Link>
        </div>
      )}
    </div>
  );
}
