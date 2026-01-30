"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  ChevronDown,
  Check,
  Layers,
  ExternalLink,
  Loader2,
} from "lucide-react";
import Link from "next/link";

/**
 * Domain summary for switcher
 */
interface DomainOption {
  name: string;
  slug: string;
  primaryColor: string;
}

/**
 * Props for DomainSwitcher
 */
export interface DomainSwitcherProps {
  /** Current domain slug */
  currentDomain?: string;
  /** Called when domain changes */
  onDomainChange?: (slug: string) => void;
  /** Whether to navigate on change */
  navigateOnChange?: boolean;
  /** Custom class name */
  className?: string;
  /** Compact mode for mobile */
  compact?: boolean;
}

/**
 * DomainSwitcher - Dropdown for switching between domains
 *
 * Used in the Navigation component to allow quick domain switching.
 */
export function DomainSwitcher({
  currentDomain,
  onDomainChange,
  navigateOnChange = true,
  className,
  compact = false,
}: DomainSwitcherProps) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [isOpen, setIsOpen] = useState(false);
  const [domains, setDomains] = useState<DomainOption[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Get current domain from URL if not provided
  const activeDomain = currentDomain || searchParams.get("domain") || "voice-clone";

  // Find current domain info
  const currentDomainInfo = domains.find((d) => d.slug === activeDomain);

  // Fetch domains on mount
  useEffect(() => {
    async function fetchDomains() {
      try {
        const response = await fetch("/api/domains");
        const data = await response.json();
        setDomains(
          data.domains?.map((d: { name: string; slug: string; primaryColor: string }) => ({
            name: d.name,
            slug: d.slug,
            primaryColor: d.primaryColor,
          })) || []
        );
      } catch (err) {
        console.error("Failed to fetch domains:", err);
      } finally {
        setIsLoading(false);
      }
    }

    fetchDomains();
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Handle domain selection
  const handleSelectDomain = (slug: string) => {
    setIsOpen(false);

    if (onDomainChange) {
      onDomainChange(slug);
    }

    if (navigateOnChange) {
      // Get current path and update domain param
      const params = new URLSearchParams(searchParams.toString());
      params.set("domain", slug);
      router.push(`/lab?${params.toString()}`);
    }
  };

  return (
    <div ref={dropdownRef} className={cn("relative", className)}>
      {/* Trigger button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors",
          "text-sm text-foreground hover:text-foreground-bright",
          "bg-background-card border border-border hover:border-foreground-muted",
          isOpen && "border-foreground-muted"
        )}
      >
        {isLoading ? (
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
        ) : (
          <>
            {/* Color indicator */}
            <div
              className="w-2 h-2 rounded-full"
              style={{
                backgroundColor: currentDomainInfo?.primaryColor || "#3b82f6",
              }}
            />

            {/* Domain name */}
            {!compact && (
              <span className="max-w-[120px] truncate">
                {currentDomainInfo?.name || "Select Domain"}
              </span>
            )}

            <ChevronDown
              className={cn(
                "w-3.5 h-3.5 text-foreground-muted transition-transform",
                isOpen && "rotate-180"
              )}
            />
          </>
        )}
      </button>

      {/* Dropdown menu */}
      {isOpen && (
        <div
          className={cn(
            "absolute top-full left-0 mt-1 z-50 min-w-[200px]",
            "bg-background-elevated border border-border rounded-lg shadow-lg",
            "py-1 overflow-hidden"
          )}
        >
          {/* Domain list */}
          <div className="max-h-[300px] overflow-y-auto">
            {domains.length === 0 ? (
              <div className="px-3 py-2 text-sm text-foreground-muted">
                No domains available
              </div>
            ) : (
              domains.map((domain) => (
                <button
                  key={domain.slug}
                  onClick={() => handleSelectDomain(domain.slug)}
                  className={cn(
                    "w-full flex items-center gap-2 px-3 py-2 text-left",
                    "text-sm transition-colors",
                    domain.slug === activeDomain
                      ? "bg-foreground-muted/20 text-foreground-bright"
                      : "text-foreground hover:bg-background-card hover:text-foreground-bright"
                  )}
                >
                  {/* Color indicator */}
                  <div
                    className="w-2 h-2 rounded-full flex-shrink-0"
                    style={{ backgroundColor: domain.primaryColor }}
                  />

                  {/* Name */}
                  <span className="flex-1 truncate">{domain.name}</span>

                  {/* Check mark for selected */}
                  {domain.slug === activeDomain && (
                    <Check className="w-3.5 h-3.5 text-foreground-bright flex-shrink-0" />
                  )}
                </button>
              ))
            )}
          </div>

          {/* Divider */}
          <div className="border-t border-border my-1" />

          {/* Browse all link */}
          <Link
            href="/domains"
            onClick={() => setIsOpen(false)}
            className={cn(
              "flex items-center gap-2 px-3 py-2",
              "text-sm text-foreground-muted hover:text-foreground hover:bg-background-card"
            )}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Browse All Domains</span>
            <ExternalLink className="w-3 h-3 ml-auto" />
          </Link>
        </div>
      )}
    </div>
  );
}

export default DomainSwitcher;
