"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { DomainWizard, WizardDomainConfig } from "@/components/domain";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";

/**
 * New Domain Page - Create a new research domain using the wizard
 */
export default function NewDomainPage() {
  const router = useRouter();
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleComplete = useCallback(
    async (config: WizardDomainConfig) => {
      setIsSaving(true);
      setError(null);

      try {
        // In a real app, this would POST to an API to create the domain
        // For now, we'll just simulate success and redirect
        console.log("Creating domain with config:", config);

        // Simulate API call
        await new Promise((resolve) => setTimeout(resolve, 1000));

        // Redirect to the new domain's lab page
        router.push(`/lab?domain=${config.branding.slug}`);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to create domain"
        );
        setIsSaving(false);
      }
    },
    [router]
  );

  const handleCancel = useCallback(() => {
    router.push("/domains");
  }, [router]);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-background-card/50 backdrop-blur-sm">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center gap-4">
          <Link
            href="/domains"
            className="flex items-center gap-2 text-sm text-foreground-muted hover:text-foreground transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Domains
          </Link>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-4xl mx-auto px-4 py-8">
        {/* Error message */}
        {error && (
          <div className="mb-6 p-4 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
            {error}
          </div>
        )}

        {/* Wizard */}
        <DomainWizard
          onComplete={handleComplete}
          onCancel={handleCancel}
          isSaving={isSaving}
          className="max-w-2xl mx-auto"
        />

        {/* Help text */}
        <div className="mt-8 text-center">
          <p className="text-xs text-foreground-subtle">
            Need help?{" "}
            <Link
              href="/docs/domains"
              className="text-foreground-muted hover:text-foreground underline"
            >
              Read the domain creation guide
            </Link>
          </p>
        </div>
      </main>
    </div>
  );
}
