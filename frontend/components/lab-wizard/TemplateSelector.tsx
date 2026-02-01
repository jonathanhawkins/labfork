"use client";

/**
 * TemplateSelector Component
 *
 * Allows users to browse and select from pre-configured lab templates
 * for quick lab creation.
 */

import { useState, useMemo } from "react";
import { cn } from "@/lib/utils";
import {
  Search,
  Sparkles,
  Star,
  Mic,
  LineChart,
  Bot,
  Pill,
  Cloud,
  Eye,
  MessageSquare,
  Gamepad2,
  Dna,
  Music,
  Brain,
  FlaskConical,
  Cpu,
  Palette,
  Users,
  Check,
  ChevronRight,
  Zap,
  HardDrive,
} from "lucide-react";
import {
  LAB_TEMPLATES,
  FEATURED_TEMPLATES,
  TEMPLATE_CATEGORIES,
  getTemplatesByCategory,
  searchTemplates,
  type LabTemplate,
  type TemplateCategory,
  type TemplateDifficulty,
} from "@/lib/lab-wizard/templates";

export interface TemplateSelectorProps {
  /** Called when a template is selected */
  onSelect: (template: LabTemplate) => void;
  /** Called when user wants to start from scratch */
  onStartFromScratch?: () => void;
  /** Currently selected template ID */
  selectedTemplateId?: string;
  /** Available VRAM (to filter/warn) */
  availableVram?: number;
  /** Custom class name */
  className?: string;
}

/**
 * Get icon component from string name
 */
function getIconComponent(iconName: string) {
  const icons: Record<string, typeof Mic> = {
    Mic,
    LineChart,
    Bot,
    Pill,
    Cloud,
    Eye,
    MessageSquare,
    Gamepad2,
    Dna,
    Music,
    Sparkles,
    Brain,
    FlaskConical,
    Cpu,
    Palette,
    Users,
  };
  return icons[iconName] || Sparkles;
}

/**
 * Get category icon
 */
function getCategoryIcon(category: TemplateCategory) {
  const icons: Record<TemplateCategory, typeof Brain> = {
    "ai-ml": Brain,
    science: FlaskConical,
    engineering: Cpu,
    creative: Palette,
    community: Users,
  };
  return icons[category];
}

/**
 * Difficulty badge colors
 */
function getDifficultyStyle(difficulty: TemplateDifficulty) {
  switch (difficulty) {
    case "beginner":
      return "text-green-400 bg-green-500/10";
    case "intermediate":
      return "text-yellow-400 bg-yellow-500/10";
    case "advanced":
      return "text-red-400 bg-red-500/10";
  }
}

/**
 * Template Card Component
 */
function TemplateCard({
  template,
  isSelected,
  onSelect,
  availableVram,
}: {
  template: LabTemplate;
  isSelected: boolean;
  onSelect: () => void;
  availableVram?: number;
}) {
  const Icon = getIconComponent(template.icon);
  const needsMoreVram = availableVram !== undefined && template.minVram !== undefined && template.minVram > availableVram;

  return (
    <button
      onClick={onSelect}
      className={cn(
        "w-full text-left p-4 rounded-lg border transition-all",
        "hover:border-foreground-muted/50 hover:bg-foreground-muted/5",
        "focus:outline-none focus:ring-2 focus:ring-blue-500/50",
        "min-h-[120px] touch-manipulation",
        isSelected
          ? "border-blue-500 bg-blue-500/10"
          : "border-border bg-background-card"
      )}
    >
      <div className="flex items-start gap-3">
        {/* Icon */}
        <div
          className={cn(
            "w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0",
            isSelected ? "bg-blue-500/20" : "bg-foreground-muted/10"
          )}
        >
          <Icon
            className={cn(
              "w-5 h-5",
              isSelected ? "text-blue-400" : "text-foreground-muted"
            )}
          />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-medium text-foreground truncate">
              {template.name}
            </h3>
            {template.featured && (
              <Star className="w-3 h-3 text-yellow-400 flex-shrink-0" />
            )}
          </div>

          <p className="text-xs text-foreground-muted mt-1 line-clamp-2">
            {template.description}
          </p>

          {/* Meta info */}
          <div className="flex items-center gap-2 mt-2 flex-wrap">
            <span
              className={cn(
                "text-xs px-2 py-0.5 rounded-full",
                getDifficultyStyle(template.difficulty)
              )}
            >
              {template.difficulty}
            </span>

            {template.minVram && (
              <span
                className={cn(
                  "flex items-center gap-1 text-xs px-2 py-0.5 rounded-full",
                  needsMoreVram
                    ? "text-red-400 bg-red-500/10"
                    : "text-foreground-subtle bg-foreground-subtle/10"
                )}
              >
                <HardDrive className="w-3 h-3" />
                {template.minVram}GB
              </span>
            )}
          </div>
        </div>

        {/* Selection indicator */}
        {isSelected && (
          <div className="w-6 h-6 rounded-full bg-blue-500 flex items-center justify-center flex-shrink-0">
            <Check className="w-4 h-4 text-white" />
          </div>
        )}
      </div>
    </button>
  );
}

/**
 * TemplateSelector - Main component for browsing templates
 */
export function TemplateSelector({
  onSelect,
  onStartFromScratch,
  selectedTemplateId,
  availableVram,
  className,
}: TemplateSelectorProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [activeCategory, setActiveCategory] = useState<TemplateCategory | "all">("all");

  // Filter templates based on search and category
  const filteredTemplates = useMemo(() => {
    let templates = LAB_TEMPLATES;

    // Filter by category
    if (activeCategory !== "all") {
      templates = getTemplatesByCategory(activeCategory);
    }

    // Filter by search
    if (searchQuery.trim()) {
      const searchResults = searchTemplates(searchQuery);
      templates = templates.filter((t) => searchResults.includes(t));
    }

    return templates;
  }, [searchQuery, activeCategory]);

  // Get featured templates for hero section
  const featured = FEATURED_TEMPLATES.slice(0, 3);

  const categories: (TemplateCategory | "all")[] = [
    "all",
    "ai-ml",
    "science",
    "engineering",
    "creative",
    "community",
  ];

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="text-center space-y-2">
        <h2 className="text-xl font-medium text-foreground-bright">
          Choose a Template
        </h2>
        <p className="text-sm text-foreground-muted">
          Start with a pre-configured template or create from scratch
        </p>
      </div>

      {/* Featured templates */}
      {!searchQuery && activeCategory === "all" && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-sm text-foreground-muted">
            <Star className="w-4 h-4 text-yellow-400" />
            <span>Featured Templates</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {featured.map((template) => (
              <button
                key={template.id}
                onClick={() => onSelect(template)}
                className={cn(
                  "p-4 rounded-lg border text-left transition-all",
                  "hover:border-blue-500/50 hover:bg-blue-500/5",
                  "focus:outline-none focus:ring-2 focus:ring-blue-500/50",
                  "min-h-[100px] touch-manipulation",
                  selectedTemplateId === template.id
                    ? "border-blue-500 bg-blue-500/10"
                    : "border-border bg-background-card"
                )}
              >
                <div className="flex items-center gap-2 mb-2">
                  {(() => {
                    const Icon = getIconComponent(template.icon);
                    return <Icon className="w-5 h-5 text-blue-400" />;
                  })()}
                  <span className="text-sm font-medium text-foreground">
                    {template.name}
                  </span>
                </div>
                <p className="text-xs text-foreground-muted line-clamp-2">
                  {template.description}
                </p>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Search bar */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-foreground-muted" />
        <input
          type="text"
          placeholder="Search templates..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className={cn(
            "w-full pl-10 pr-4 py-2.5 rounded-lg",
            "bg-background border border-border",
            "text-sm text-foreground placeholder:text-foreground-subtle",
            "focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50"
          )}
        />
      </div>

      {/* Category tabs */}
      <div className="flex gap-2 overflow-x-auto pb-2 -mx-4 px-4 scrollbar-hide">
        {categories.map((cat) => {
          const isActive = activeCategory === cat;
          const Icon = cat === "all" ? Sparkles : getCategoryIcon(cat);
          const label = cat === "all" ? "All" : TEMPLATE_CATEGORIES[cat].name;

          return (
            <button
              key={cat}
              onClick={() => setActiveCategory(cat)}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs whitespace-nowrap",
                "transition-colors touch-manipulation min-h-[36px]",
                isActive
                  ? "bg-foreground-bright text-background"
                  : "bg-foreground-muted/10 text-foreground-muted hover:bg-foreground-muted/20"
              )}
            >
              <Icon className="w-3.5 h-3.5" />
              {label}
            </button>
          );
        })}
      </div>

      {/* Template grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {filteredTemplates.map((template) => (
          <TemplateCard
            key={template.id}
            template={template}
            isSelected={selectedTemplateId === template.id}
            onSelect={() => onSelect(template)}
            availableVram={availableVram}
          />
        ))}
      </div>

      {/* Empty state */}
      {filteredTemplates.length === 0 && (
        <div className="text-center py-8">
          <p className="text-foreground-muted">No templates found</p>
          <button
            onClick={() => {
              setSearchQuery("");
              setActiveCategory("all");
            }}
            className="mt-2 text-sm text-blue-400 hover:underline"
          >
            Clear filters
          </button>
        </div>
      )}

      {/* Start from scratch option */}
      {onStartFromScratch && (
        <div className="pt-4 border-t border-border">
          <button
            onClick={onStartFromScratch}
            className={cn(
              "w-full flex items-center justify-center gap-2 p-4 rounded-lg",
              "border border-dashed border-border",
              "text-sm text-foreground-muted hover:text-foreground",
              "hover:border-foreground-muted/50 hover:bg-foreground-muted/5",
              "transition-colors touch-manipulation"
            )}
          >
            <Zap className="w-4 h-4" />
            Start from scratch
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}

export default TemplateSelector;
