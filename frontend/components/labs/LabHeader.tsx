"use client";

/**
 * LabHeader Component
 *
 * Header section for a lab page showing:
 * - Lab name and owner
 * - Domain badge
 * - Star/fork/task/paper counts
 * - Action buttons (star, fork, share)
 * - Forked-from link if applicable
 */

import { useState } from "react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import {
  Star,
  GitFork,
  Share2,
  Activity,
  FileText,
  Eye,
  Settings,
  MoreHorizontal,
  ExternalLink,
  Lock,
  Globe,
  Link as LinkIcon,
  Mic,
  LineChart,
  Bot,
  Dna,
  Layers,
  Sun,
  Network,
  Zap,
  Lightbulb,
} from "lucide-react";
import type { Lab } from "@/lib/labs/types";
import { getLabPath } from "@/lib/labs/types";
import { StarButton } from "./StarButton";
import { ForkDialog, quickForkLab } from "./ForkDialog";
import { useRouter } from "next/navigation";

export interface LabHeaderProps {
  /** Lab data */
  lab: Lab;
  /** Is starred by current user */
  isStarred?: boolean;
  /** Is the current user the owner */
  isOwner?: boolean;
  /** Star toggle handler */
  onStarToggle?: (starred: boolean, count: number) => void;
  /** Fork success handler */
  onForkSuccess?: (forkedLab: Lab) => void;
  /** Share click handler */
  onShareClick?: () => void;
  /** Settings click handler */
  onSettingsClick?: () => void;
  /** Custom class name */
  className?: string;
}

/**
 * Get domain icon
 */
function getDomainIcon(domainSlug: string) {
  switch (domainSlug) {
    case "voice-clone":
      return Mic;
    case "quant-trading":
      return LineChart;
    case "robotics":
      return Bot;
    case "biotech":
      return Dna;
    case "firefly-network":
      return Sun;
    case "mesh-networking":
      return Network;
    case "solar-energy":
      return Zap;
    default:
      return Layers;
  }
}

/**
 * Get domain color
 */
function getDomainColor(domainSlug: string): string {
  switch (domainSlug) {
    case "voice-clone":
      return "text-blue-400 bg-blue-500/10 border-blue-500/20";
    case "quant-trading":
      return "text-green-400 bg-green-500/10 border-green-500/20";
    case "robotics":
      return "text-purple-400 bg-purple-500/10 border-purple-500/20";
    case "biotech":
      return "text-pink-400 bg-pink-500/10 border-pink-500/20";
    case "firefly-network":
      return "text-amber-400 bg-amber-500/10 border-amber-500/20";
    case "solar-energy":
      return "text-yellow-400 bg-yellow-500/10 border-yellow-500/20";
    case "mesh-networking":
      return "text-cyan-400 bg-cyan-500/10 border-cyan-500/20";
    default:
      return "text-foreground-muted bg-foreground-muted/10 border-foreground-muted/20";
  }
}

/**
 * Get visibility icon
 */
function getVisibilityIcon(visibility: Lab["visibility"]) {
  switch (visibility) {
    case "public":
      return Globe;
    case "private":
      return Lock;
    case "unlisted":
      return LinkIcon;
  }
}

export function LabHeader({
  lab,
  isStarred = false,
  isOwner = false,
  onStarToggle,
  onForkSuccess,
  onShareClick,
  onSettingsClick,
  className,
}: LabHeaderProps) {
  const [isForkDialogOpen, setIsForkDialogOpen] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [isQuickForking, setIsQuickForking] = useState(false);
  const router = useRouter();

  const handleQuickFork = async () => {
    if (isQuickForking) return;
    setIsQuickForking(true);
    try {
      const result = await quickForkLab(lab);
      if (result.success && result.lab) {
        onForkSuccess?.(result.lab);
        // Redirect to the forked lab
        router.push(getLabPath(result.lab.owner.username, result.lab.slug));
      } else {
        // Fall back to dialog on error
        setIsForkDialogOpen(true);
      }
    } catch {
      setIsForkDialogOpen(true);
    } finally {
      setIsQuickForking(false);
    }
  };

  const DomainIcon = getDomainIcon(lab.domainSlug);
  const domainColor = getDomainColor(lab.domainSlug);
  const VisibilityIcon = getVisibilityIcon(lab.visibility);
  const isIdea = lab.status === "idea";
  const allStatsZero = lab.stats.stars === 0 && lab.stats.forks === 0 && lab.stats.tasks === 0 && lab.stats.papers === 0;

  return (
    <div className={cn("border-b border-border", className)}>
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Forked from indicator */}
        {lab.forkedFrom && (
          <div className="mb-3">
            <Link
              href={getLabPath(lab.forkedFrom.sourceOwner, lab.forkedFrom.sourceSlug)}
              className="inline-flex items-center gap-1.5 text-sm text-foreground-muted hover:text-foreground transition-colors"
            >
              <GitFork className="w-4 h-4" />
              Forked from{" "}
              <span className="font-medium">
                {lab.forkedFrom.sourceOwner}/{lab.forkedFrom.sourceSlug}
              </span>
              <ExternalLink className="w-3 h-3" />
            </Link>
          </div>
        )}

        {/* Main header row */}
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
          {/* Left side: Icon, name, owner, domain */}
          <div className="flex items-start gap-3 sm:gap-4 min-w-0 flex-1">
            {/* Domain Icon */}
            <div
              className={cn(
                "flex-shrink-0 w-10 h-10 sm:w-14 sm:h-14 rounded-lg sm:rounded-xl flex items-center justify-center border",
                domainColor
              )}
            >
              <DomainIcon className="w-5 h-5 sm:w-7 sm:h-7" />
            </div>

            {/* Name and meta */}
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 sm:gap-3 flex-wrap">
                <h1 className="text-xl sm:text-2xl font-semibold text-foreground-bright truncate">
                  {lab.name}
                </h1>

                {/* Visibility badge */}
                <span
                  className={cn(
                    "inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs border",
                    lab.visibility === "public"
                      ? "text-green-400 bg-green-500/10 border-green-500/20"
                      : lab.visibility === "private"
                      ? "text-red-400 bg-red-500/10 border-red-500/20"
                      : "text-yellow-400 bg-yellow-500/10 border-yellow-500/20"
                  )}
                >
                  <VisibilityIcon className="w-3 h-3" />
                  {lab.visibility}
                </span>

                {/* Featured badge */}
                {lab.isFeatured && (
                  <span className="px-2 py-0.5 rounded text-xs bg-gradient-to-r from-yellow-500/20 to-orange-500/20 text-yellow-400 border border-yellow-500/20">
                    Featured
                  </span>
                )}

                {/* Idea badge */}
                {isIdea && (
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs text-amber-400 bg-amber-500/10 border border-amber-500/20">
                    <Lightbulb className="w-3 h-3" />
                    Idea
                  </span>
                )}
              </div>

              {/* Owner and domain */}
              <div className="flex items-center gap-3 mt-1 text-sm text-foreground-muted">
                <Link
                  href={`/labs/${lab.owner.username}`}
                  className="hover:text-foreground transition-colors"
                >
                  {lab.owner.displayName}
                </Link>
                <span className="text-foreground-subtle">·</span>
                <span className={cn("px-2 py-0.5 rounded text-xs", domainColor)}>
                  {lab.domainName}
                </span>
              </div>

              {/* Description */}
              {lab.description && (
                <p className="mt-2 text-sm text-foreground-muted max-w-2xl">
                  {lab.description}
                </p>
              )}

              {/* Tags */}
              {lab.tags.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-3">
                  {lab.tags.map((tag) => (
                    <Link
                      key={tag}
                      href={`/explore?tag=${encodeURIComponent(tag)}`}
                      className="px-2 py-0.5 rounded text-xs bg-foreground-muted/10 text-foreground-subtle hover:bg-foreground-muted/20 transition-colors"
                    >
                      {tag}
                    </Link>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Right side: Actions */}
          <div className="flex items-center gap-2 flex-shrink-0 flex-wrap">
            {/* Star button */}
            <StarButton
              labId={lab.id}
              initialCount={lab.stats.stars}
              initialStarred={isStarred}
              onToggle={onStarToggle}
            />

            {/* Fork buttons */}
            <div className="flex items-center">
              {/* Quick Fork & Launch */}
              <button
                onClick={handleQuickFork}
                disabled={isQuickForking}
                className={cn(
                  "flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-l-lg border-y border-l transition-colors min-h-[44px]",
                  "bg-foreground text-background hover:bg-foreground-bright border-foreground",
                  "disabled:opacity-70"
                )}
                title="Fork and launch immediately"
              >
                {isQuickForking ? (
                  <span className="w-4 h-4 border-2 border-background/30 border-t-background rounded-full animate-spin" />
                ) : (
                  <GitFork className="w-4 h-4" />
                )}
                <span className="font-medium hidden sm:inline">{isIdea ? "Start This Research" : "Fork & Launch"}</span>
                <span className="font-medium sm:hidden">Fork</span>
              </button>

              {/* Fork count / dialog trigger */}
              <button
                onClick={() => setIsForkDialogOpen(true)}
                className={cn(
                  "flex items-center gap-1 px-2 py-1.5 text-sm rounded-r-lg border transition-colors min-h-[44px]",
                  "border-border bg-background hover:bg-foreground-muted/10 text-foreground-muted hover:text-foreground"
                )}
                title="Fork with custom name"
              >
                <span className="font-medium">{lab.stats.forks}</span>
              </button>
            </div>

            {/* Share button */}
            <button
              onClick={onShareClick}
              className={cn(
                "p-2 rounded-lg border transition-colors",
                "border-border hover:bg-foreground-muted/10 text-foreground-muted hover:text-foreground"
              )}
              title="Share lab"
            >
              <Share2 className="w-4 h-4" />
            </button>

            {/* More dropdown */}
            <div className="relative">
              <button
                onClick={() => setShowDropdown(!showDropdown)}
                className={cn(
                  "p-2 rounded-lg border transition-colors",
                  "border-border hover:bg-foreground-muted/10 text-foreground-muted hover:text-foreground"
                )}
              >
                <MoreHorizontal className="w-4 h-4" />
              </button>

              {showDropdown && (
                <>
                  <div
                    className="fixed inset-0 z-10"
                    onClick={() => setShowDropdown(false)}
                  />
                  <div className="absolute right-0 top-full mt-1 w-48 rounded-lg border border-border bg-background-elevated shadow-lg z-20">
                    <div className="py-1">
                      {isOwner && (
                        <button
                          onClick={() => {
                            onSettingsClick?.();
                            setShowDropdown(false);
                          }}
                          className="w-full px-3 py-2 text-sm text-left text-foreground-muted hover:bg-foreground-muted/10 flex items-center gap-2"
                        >
                          <Settings className="w-4 h-4" />
                          Settings
                        </button>
                      )}
                      <button
                        onClick={() => {
                          navigator.clipboard.writeText(window.location.href);
                          setShowDropdown(false);
                        }}
                        className="w-full px-3 py-2 text-sm text-left text-foreground-muted hover:bg-foreground-muted/10 flex items-center gap-2"
                      >
                        <LinkIcon className="w-4 h-4" />
                        Copy link
                      </button>
                      <button
                        onClick={() => {
                          window.open(window.location.href, "_blank");
                          setShowDropdown(false);
                        }}
                        className="w-full px-3 py-2 text-sm text-left text-foreground-muted hover:bg-foreground-muted/10 flex items-center gap-2"
                      >
                        <ExternalLink className="w-4 h-4" />
                        Open in new tab
                      </button>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Stats row */}
        <div className="flex items-center gap-4 sm:gap-6 mt-4 sm:mt-6 pt-3 sm:pt-4 border-t border-border/50 flex-wrap">
          {allStatsZero ? (
            <span className="text-sm text-foreground-subtle">No activity yet</span>
          ) : (
            <>
              {lab.stats.stars > 0 && (
                <div className="flex items-center gap-2 text-sm">
                  <Star className="w-4 h-4 text-yellow-400" />
                  <span className="font-medium text-foreground">{lab.stats.stars}</span>
                  <span className="text-foreground-muted">stars</span>
                </div>
              )}

              {lab.stats.forks > 0 && (
                <div className="flex items-center gap-2 text-sm">
                  <GitFork className="w-4 h-4 text-foreground-subtle" />
                  <span className="font-medium text-foreground">{lab.stats.forks}</span>
                  <span className="text-foreground-muted">forks</span>
                </div>
              )}

              {lab.stats.tasks > 0 && (
                <div className="flex items-center gap-2 text-sm">
                  <Activity className="w-4 h-4 text-foreground-subtle" />
                  <span className="font-medium text-foreground">{lab.stats.tasks}</span>
                  <span className="text-foreground-muted">tasks</span>
                </div>
              )}

              {lab.stats.papers > 0 && (
                <div className="flex items-center gap-2 text-sm">
                  <FileText className="w-4 h-4 text-foreground-subtle" />
                  <span className="font-medium text-foreground">{lab.stats.papers}</span>
                  <span className="text-foreground-muted">papers</span>
                </div>
              )}

              {lab.stats.viewers > 0 && (
                <div className="flex items-center gap-2 text-sm">
                  <Eye className="w-4 h-4 text-green-400" />
                  <span className="font-medium text-green-400">{lab.stats.viewers}</span>
                  <span className="text-foreground-muted">watching</span>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Fork Dialog */}
      <ForkDialog
        lab={lab}
        isOpen={isForkDialogOpen}
        onClose={() => setIsForkDialogOpen(false)}
        onSuccess={(forkedLab) => {
          setIsForkDialogOpen(false);
          onForkSuccess?.(forkedLab);
        }}
      />
    </div>
  );
}

export default LabHeader;
