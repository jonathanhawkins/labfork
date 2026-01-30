"use client";

import { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { Palette, Type, FileText, Eye } from "lucide-react";

/**
 * Branding configuration for a domain
 */
export interface DomainBranding {
  name: string;
  slug: string;
  description: string;
  primaryColor: string;
  accentColor: string;
  difficulty: "beginner" | "intermediate" | "advanced";
}

/**
 * Preset color palettes
 */
const COLOR_PRESETS = [
  { name: "Teal", primary: "#4ecdc4", accent: "#66ffaa" },
  { name: "Emerald", primary: "#10b981", accent: "#f59e0b" },
  { name: "Orange", primary: "#f97316", accent: "#fbbf24" },
  { name: "Purple", primary: "#8b5cf6", accent: "#22c55e" },
  { name: "Blue", primary: "#3b82f6", accent: "#06b6d4" },
  { name: "Rose", primary: "#f43f5e", accent: "#fb7185" },
  { name: "Amber", primary: "#f59e0b", accent: "#fcd34d" },
  { name: "Cyan", primary: "#06b6d4", accent: "#22d3ee" },
];

export interface WizardStepBrandingProps {
  /** Current branding values */
  branding: DomainBranding;
  /** Called when branding changes */
  onBrandingChange: (branding: DomainBranding) => void;
  /** Validation errors */
  errors?: Partial<Record<keyof DomainBranding, string>>;
  /** Custom class name */
  className?: string;
}

/**
 * Generate slug from name
 */
function generateSlug(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 50);
}

/**
 * WizardStepBranding - Step 2: Configure name, description, and branding
 */
export function WizardStepBranding({
  branding,
  onBrandingChange,
  errors = {},
  className,
}: WizardStepBrandingProps) {
  const [autoSlug, setAutoSlug] = useState(true);

  const handleNameChange = useCallback(
    (name: string) => {
      const newBranding = { ...branding, name };
      if (autoSlug) {
        newBranding.slug = generateSlug(name);
      }
      onBrandingChange(newBranding);
    },
    [branding, onBrandingChange, autoSlug]
  );

  const handleSlugChange = useCallback(
    (slug: string) => {
      setAutoSlug(false);
      onBrandingChange({ ...branding, slug: generateSlug(slug) });
    },
    [branding, onBrandingChange]
  );

  const handleColorPreset = useCallback(
    (preset: { primary: string; accent: string }) => {
      onBrandingChange({
        ...branding,
        primaryColor: preset.primary,
        accentColor: preset.accent,
      });
    },
    [branding, onBrandingChange]
  );

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="text-center">
        <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-foreground-muted/10 mb-4">
          <Type className="w-6 h-6 text-foreground-muted" />
        </div>
        <h2 className="text-lg font-normal text-foreground-bright">
          Name & Branding
        </h2>
        <p className="text-sm text-foreground-muted mt-1">
          Give your domain a name and visual identity
        </p>
      </div>

      {/* Name field */}
      <div className="space-y-2">
        <label className="text-sm text-foreground-muted flex items-center gap-2">
          <Type className="w-4 h-4" />
          Domain Name
        </label>
        <input
          type="text"
          value={branding.name}
          onChange={(e) => handleNameChange(e.target.value)}
          placeholder="My Research Lab"
          className={cn(
            "w-full px-4 py-2.5 text-sm rounded-lg",
            "bg-background-card border",
            "text-foreground placeholder:text-foreground-subtle",
            "focus:outline-none focus:border-foreground-muted",
            errors.name ? "border-red-500/50" : "border-border"
          )}
        />
        {errors.name && (
          <p className="text-xs text-red-400">{errors.name}</p>
        )}
      </div>

      {/* Slug field */}
      <div className="space-y-2">
        <label className="text-sm text-foreground-muted flex items-center gap-2">
          <FileText className="w-4 h-4" />
          URL Slug
        </label>
        <div className="flex items-center gap-2">
          <span className="text-sm text-foreground-subtle">/lab?domain=</span>
          <input
            type="text"
            value={branding.slug}
            onChange={(e) => handleSlugChange(e.target.value)}
            placeholder="my-research-lab"
            className={cn(
              "flex-1 px-3 py-2 text-sm rounded-lg font-mono",
              "bg-background-card border",
              "text-foreground placeholder:text-foreground-subtle",
              "focus:outline-none focus:border-foreground-muted",
              errors.slug ? "border-red-500/50" : "border-border"
            )}
          />
        </div>
        {errors.slug && (
          <p className="text-xs text-red-400">{errors.slug}</p>
        )}
      </div>

      {/* Description field */}
      <div className="space-y-2">
        <label className="text-sm text-foreground-muted flex items-center gap-2">
          <FileText className="w-4 h-4" />
          Description
        </label>
        <textarea
          value={branding.description}
          onChange={(e) =>
            onBrandingChange({ ...branding, description: e.target.value })
          }
          placeholder="A brief description of your research focus..."
          rows={3}
          className={cn(
            "w-full px-4 py-2.5 text-sm rounded-lg resize-none",
            "bg-background-card border",
            "text-foreground placeholder:text-foreground-subtle",
            "focus:outline-none focus:border-foreground-muted",
            errors.description ? "border-red-500/50" : "border-border"
          )}
        />
        {errors.description && (
          <p className="text-xs text-red-400">{errors.description}</p>
        )}
      </div>

      {/* Difficulty level */}
      <div className="space-y-2">
        <label className="text-sm text-foreground-muted">Difficulty Level</label>
        <div className="flex gap-2">
          {(["beginner", "intermediate", "advanced"] as const).map((level) => (
            <button
              key={level}
              onClick={() => onBrandingChange({ ...branding, difficulty: level })}
              className={cn(
                "flex-1 px-4 py-2 text-sm rounded-lg border transition-colors",
                branding.difficulty === level
                  ? level === "beginner"
                    ? "bg-green-500/20 border-green-500/50 text-green-400"
                    : level === "intermediate"
                    ? "bg-yellow-500/20 border-yellow-500/50 text-yellow-400"
                    : "bg-red-500/20 border-red-500/50 text-red-400"
                  : "bg-background-card border-border text-foreground-muted hover:border-foreground-muted"
              )}
            >
              {level.charAt(0).toUpperCase() + level.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Color palette */}
      <div className="space-y-3">
        <label className="text-sm text-foreground-muted flex items-center gap-2">
          <Palette className="w-4 h-4" />
          Color Palette
        </label>

        {/* Color presets */}
        <div className="grid grid-cols-4 gap-2">
          {COLOR_PRESETS.map((preset) => (
            <button
              key={preset.name}
              onClick={() => handleColorPreset(preset)}
              className={cn(
                "group relative h-12 rounded-lg border transition-all overflow-hidden",
                branding.primaryColor === preset.primary
                  ? "border-foreground-bright ring-1 ring-foreground-bright/50"
                  : "border-border hover:border-foreground-muted"
              )}
              title={preset.name}
            >
              <div
                className="absolute inset-0"
                style={{
                  background: `linear-gradient(135deg, ${preset.primary} 0%, ${preset.accent} 100%)`,
                }}
              />
              <span className="absolute inset-0 flex items-center justify-center text-[10px] font-medium text-white opacity-0 group-hover:opacity-100 transition-opacity bg-black/40">
                {preset.name}
              </span>
            </button>
          ))}
        </div>

        {/* Custom color inputs */}
        <div className="flex gap-4">
          <div className="flex-1 space-y-1">
            <span className="text-xs text-foreground-subtle">Primary</span>
            <div className="flex items-center gap-2">
              <input
                type="color"
                value={branding.primaryColor}
                onChange={(e) =>
                  onBrandingChange({ ...branding, primaryColor: e.target.value })
                }
                className="w-8 h-8 rounded cursor-pointer border-0"
              />
              <input
                type="text"
                value={branding.primaryColor}
                onChange={(e) =>
                  onBrandingChange({ ...branding, primaryColor: e.target.value })
                }
                className="flex-1 px-2 py-1 text-xs font-mono rounded bg-background-card border border-border text-foreground"
              />
            </div>
          </div>
          <div className="flex-1 space-y-1">
            <span className="text-xs text-foreground-subtle">Accent</span>
            <div className="flex items-center gap-2">
              <input
                type="color"
                value={branding.accentColor}
                onChange={(e) =>
                  onBrandingChange({ ...branding, accentColor: e.target.value })
                }
                className="w-8 h-8 rounded cursor-pointer border-0"
              />
              <input
                type="text"
                value={branding.accentColor}
                onChange={(e) =>
                  onBrandingChange({ ...branding, accentColor: e.target.value })
                }
                className="flex-1 px-2 py-1 text-xs font-mono rounded bg-background-card border border-border text-foreground"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Preview */}
      <div className="space-y-2">
        <label className="text-sm text-foreground-muted flex items-center gap-2">
          <Eye className="w-4 h-4" />
          Preview
        </label>
        <div
          className="h-24 rounded-lg border border-border overflow-hidden"
          style={{
            background: `linear-gradient(135deg, ${branding.primaryColor}20 0%, ${branding.accentColor}10 100%)`,
          }}
        >
          <div
            className="h-1"
            style={{ backgroundColor: branding.primaryColor }}
          />
          <div className="p-4">
            <h3 className="text-sm font-normal text-foreground-bright">
              {branding.name || "Untitled Domain"}
            </h3>
            <p className="text-xs text-foreground-muted mt-1 line-clamp-2">
              {branding.description || "No description"}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default WizardStepBranding;
