"use client";

import { Suspense } from "react";
import { useRouter } from "next/navigation";
import { LabWizard } from "@/components/lab-wizard";

/**
 * Loading fallback for the wizard
 */
function WizardLoading() {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="text-center space-y-4">
        <div className="w-12 h-12 border-4 border-foreground-bright/20 border-t-foreground-bright rounded-full animate-spin mx-auto" />
        <p className="text-sm text-foreground-muted">Loading wizard...</p>
      </div>
    </div>
  );
}

/**
 * Wizard page content (uses useSearchParams)
 */
function LabWizardContent() {
  const router = useRouter();

  return (
    <LabWizard
      onClose={() => {
        router.push("/lab");
      }}
    />
  );
}

/**
 * New Lab Page
 *
 * Multi-step wizard for creating a new research lab.
 *
 * URL parameters:
 * - step: Current wizard step (welcome, domain, hardware, research, review)
 * - domain: Pre-selected domain slug
 */
export default function NewLabPage() {
  return (
    <Suspense fallback={<WizardLoading />}>
      <LabWizardContent />
    </Suspense>
  );
}
