"use client";

import { cn } from "@/lib/utils";
import Link from "next/link";
import {
  Beaker,
  Bot,
  Brain,
  ChevronRight,
  Cpu,
  Layers,
  Mic,
  TrendingUp,
} from "lucide-react";

/**
 * Props for DomainCard component
 */
export interface DomainCardProps {
  /** Domain name */
  name: string;
  /** Domain slug for URL */
  slug: string;
  /** Short description */
  description: string;
  /** Difficulty level */
  difficulty?: "beginner" | "intermediate" | "advanced";
  /** Primary brand color (hex) */
  primaryColor: string;
  /** Accent color (hex) */
  accentColor: string;
  /** Background style */
  backgroundStyle?: string;
  /** Tags for categorization */
  tags?: string[];
  /** Number of 3D props in scene */
  propsCount?: number;
  /** Number of evaluation metrics */
  metricsCount?: number;
  /** Whether this card is currently selected */
  isSelected?: boolean;
  /** Click handler (if provided, card is clickable without navigation) */
  onClick?: () => void;
  /** Whether to show as compact card */
  compact?: boolean;
}

/**
 * Get icon based on domain tags/name
 */
function getDomainIcon(name: string, tags?: string[]) {
  const tagStr = (tags || []).join(" ").toLowerCase();
  const nameLower = name.toLowerCase();

  if (tagStr.includes("voice") || tagStr.includes("tts") || tagStr.includes("speech")) {
    return Mic;
  }
  if (tagStr.includes("trading") || tagStr.includes("finance") || tagStr.includes("quant")) {
    return TrendingUp;
  }
  if (tagStr.includes("robot") || tagStr.includes("embodied")) {
    return Bot;
  }
  if (tagStr.includes("bio") || tagStr.includes("drug") || tagStr.includes("nlp")) {
    return Beaker;
  }
  if (tagStr.includes("ml") || tagStr.includes("machine-learning") || nameLower.includes("ml")) {
    return Brain;
  }

  return Cpu;
}

/**
 * Get difficulty badge color
 */
function getDifficultyColor(difficulty?: string) {
  switch (difficulty) {
    case "beginner":
      return "bg-green-500/20 text-green-400 border-green-500/30";
    case "intermediate":
      return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
    case "advanced":
      return "bg-red-500/20 text-red-400 border-red-500/30";
    default:
      return "bg-foreground-muted/20 text-foreground-muted border-foreground-muted/30";
  }
}

/**
 * DomainCard - Displays a domain in a card format
 *
 * Used in DomainBrowser grid and domain selection interfaces.
 */
export function DomainCard({
  name,
  slug,
  description,
  difficulty,
  primaryColor,
  accentColor,
  tags,
  propsCount,
  metricsCount,
  isSelected,
  onClick,
  compact = false,
}: DomainCardProps) {
  const Icon = getDomainIcon(name, tags);

  const cardContent = (
    <div
      className={cn(
        "relative group rounded-lg border transition-all duration-200",
        "bg-background-card hover:bg-background-elevated",
        isSelected
          ? "border-foreground-bright ring-1 ring-foreground-bright/50"
          : "border-border hover:border-foreground-muted",
        compact ? "p-3" : "p-4"
      )}
      style={{
        // Subtle gradient overlay using domain colors
        backgroundImage: isSelected
          ? `linear-gradient(135deg, ${primaryColor}10 0%, transparent 50%)`
          : undefined,
      }}
    >
      {/* Color accent bar */}
      <div
        className="absolute top-0 left-0 right-0 h-1 rounded-t-lg opacity-80"
        style={{ backgroundColor: primaryColor }}
      />

      {/* Header */}
      <div className={cn("flex items-start gap-3", compact ? "mt-1" : "mt-2")}>
        {/* Icon */}
        <div
          className={cn(
            "flex-shrink-0 rounded-lg flex items-center justify-center",
            compact ? "w-8 h-8" : "w-10 h-10"
          )}
          style={{ backgroundColor: `${primaryColor}20` }}
        >
          <Icon
            className={cn(compact ? "w-4 h-4" : "w-5 h-5")}
            style={{ color: primaryColor }}
          />
        </div>

        {/* Title and difficulty */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h3
              className={cn(
                "font-normal truncate",
                compact ? "text-sm" : "text-base",
                "text-foreground-bright group-hover:text-white transition-colors"
              )}
            >
              {name}
            </h3>
            {!compact && difficulty && (
              <span
                className={cn(
                  "text-[10px] px-1.5 py-0.5 rounded border",
                  getDifficultyColor(difficulty)
                )}
              >
                {difficulty}
              </span>
            )}
          </div>

          {!compact && (
            <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
              {description}
            </p>
          )}
        </div>

        {/* Arrow indicator */}
        <ChevronRight
          className={cn(
            "flex-shrink-0 text-foreground-muted transition-transform",
            "group-hover:translate-x-0.5 group-hover:text-foreground",
            compact ? "w-4 h-4" : "w-5 h-5"
          )}
        />
      </div>

      {/* Tags */}
      {!compact && tags && tags.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-3">
          {tags.slice(0, 4).map((tag) => (
            <span
              key={tag}
              className="text-[10px] px-1.5 py-0.5 rounded bg-background border border-border text-foreground-muted"
            >
              {tag}
            </span>
          ))}
          {tags.length > 4 && (
            <span className="text-[10px] text-foreground-subtle">
              +{tags.length - 4} more
            </span>
          )}
        </div>
      )}

      {/* Stats footer */}
      {!compact && (propsCount !== undefined || metricsCount !== undefined) && (
        <div className="flex items-center gap-4 mt-3 pt-3 border-t border-border">
          {propsCount !== undefined && (
            <div className="flex items-center gap-1.5 text-xs text-foreground-muted">
              <Layers className="w-3 h-3" />
              <span>{propsCount} props</span>
            </div>
          )}
          {metricsCount !== undefined && metricsCount > 0 && (
            <div className="flex items-center gap-1.5 text-xs text-foreground-muted">
              <TrendingUp className="w-3 h-3" />
              <span>{metricsCount} metrics</span>
            </div>
          )}
        </div>
      )}
    </div>
  );

  // If onClick is provided, render as button
  if (onClick) {
    return (
      <button onClick={onClick} className="w-full text-left">
        {cardContent}
      </button>
    );
  }

  // Otherwise, render as link
  return <Link href={`/lab?domain=${slug}`}>{cardContent}</Link>;
}

export default DomainCard;
