"use client";

/**
 * DemoBreadcrumb - Navigation breadcrumb for individual demo pages.
 *
 * Shows: Lab Name > Demos > Current Demo
 * Provides clear navigation context connecting demos to their parent lab.
 */

import Link from "next/link";
import { ChevronRight } from "lucide-react";
import { getDemoById, getLabRefForDemo } from "@/lib/demos";

interface DemoBreadcrumbProps {
  /** The demo ID (route segment) */
  demoId: string;
}

export function DemoBreadcrumb({ demoId }: DemoBreadcrumbProps) {
  const demo = getDemoById(demoId);
  const labRef = demo ? getLabRefForDemo(demo) : undefined;

  return (
    <nav className="flex items-center gap-1.5 text-sm mb-6 sm:mb-8 flex-wrap">
      {/* Lab link */}
      {labRef && (
        <>
          <Link
            href={labRef.path}
            className="text-foreground-muted hover:text-foreground transition-colors min-h-[44px] flex items-center"
          >
            {labRef.name}
          </Link>
          <ChevronRight className="w-3.5 h-3.5 text-foreground-subtle flex-shrink-0" />
        </>
      )}

      {/* Demos link */}
      <Link
        href="/demos"
        className="text-foreground-muted hover:text-foreground transition-colors min-h-[44px] flex items-center"
      >
        Demos
      </Link>

      {/* Current demo */}
      {demo && (
        <>
          <ChevronRight className="w-3.5 h-3.5 text-foreground-subtle flex-shrink-0" />
          <span className="text-foreground-bright">{demo.title}</span>
        </>
      )}
    </nav>
  );
}
