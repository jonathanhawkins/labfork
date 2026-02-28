"use client";

/**
 * LabShowcase - Domain-aware interactive showcase router.
 *
 * Maps lab slugs to their rich interactive showcase components.
 * Renders nothing if no showcase is available for the given lab.
 *
 * This is displayed prominently on the Overview tab of the generic lab page,
 * providing visitors with interactive content specific to each lab's domain.
 */

import dynamic from "next/dynamic";
import { Loader2 } from "lucide-react";
import type { Lab } from "@/lib/labs/types";

// Lazy-load showcases to avoid bundling all heavy 3D/simulation code
const WaterHarvesterShowcase = dynamic(
  () =>
    import("./showcases/WaterHarvesterShowcase").then(
      (mod) => mod.WaterHarvesterShowcase
    ),
  {
    ssr: false,
    loading: () => (
      <div className="h-[300px] rounded-lg border border-border bg-background-elevated flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin text-foreground-muted" />
      </div>
    ),
  }
);

/** Registry of lab slugs to their showcase components */
const SHOWCASE_REGISTRY: Record<string, React.ComponentType> = {
  "water-harvester": WaterHarvesterShowcase,
};

interface LabShowcaseProps {
  lab: Lab;
}

export function LabShowcase({ lab }: LabShowcaseProps) {
  const ShowcaseComponent = SHOWCASE_REGISTRY[lab.slug];

  if (!ShowcaseComponent) return null;

  return <ShowcaseComponent />;
}
