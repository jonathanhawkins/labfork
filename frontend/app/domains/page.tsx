import { Metadata } from "next";
import { DomainBrowser } from "@/components/domain";
import { Layers, Plus } from "lucide-react";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Research Domains | AI Research Lab",
  description:
    "Browse and select from available AI research domains including voice cloning, quantitative trading, robotics, and biotech NLP.",
};

/**
 * Domains Page - Browse all available research domains
 *
 * Features:
 * - Grid view of all domains
 * - Search and filter
 * - Click to open in lab
 * - Link to create new domain
 */
export default function DomainsPage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border">
        <div className="max-w-6xl mx-auto px-4 py-8">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <div
                  className="w-10 h-10 rounded-lg flex items-center justify-center"
                  style={{ backgroundColor: "rgba(59, 130, 246, 0.2)" }}
                >
                  <Layers className="w-5 h-5 text-blue-400" />
                </div>
                <h1 className="text-xl font-normal text-foreground-bright">
                  Research Domains
                </h1>
              </div>
              <p className="text-sm text-muted-foreground max-w-xl">
                Choose a research domain to start exploring. Each domain comes with
                its own 3D lab environment, research focus, and evaluation metrics.
              </p>
            </div>

            <Link
              href="/domains/new"
              className="flex items-center gap-2 px-4 py-2 text-sm bg-foreground text-background rounded-lg hover:bg-foreground-bright transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span>Create Domain</span>
            </Link>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-6xl mx-auto px-4 py-8">
        <DomainBrowser />
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-auto">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-foreground-muted">
            <p>
              Can&apos;t find what you&apos;re looking for?{" "}
              <Link
                href="/domains/new"
                className="text-foreground hover:text-foreground-bright underline"
              >
                Create a custom domain
              </Link>{" "}
              for your research area.
            </p>
            <p>
              Or browse the{" "}
              <Link
                href="https://github.com"
                className="text-foreground hover:text-foreground-bright underline"
              >
                community domains
              </Link>
              .
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
