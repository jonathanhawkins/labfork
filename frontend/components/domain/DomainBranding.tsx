"use client";

import React, { useEffect, ReactNode } from "react";
import { useDomainSafe } from "./DomainProvider";
import { BackgroundStyle } from "@/lib/domain/types";

/**
 * Props for DomainBranding
 */
interface DomainBrandingProps {
  /** Children to render */
  children: ReactNode;
  /** Additional className for the wrapper */
  className?: string;
  /** Override background style */
  backgroundStyle?: BackgroundStyle;
}

/**
 * Get CSS class for background style
 */
function getBackgroundClass(style: BackgroundStyle): string {
  switch (style) {
    case "sky":
      return "bg-gradient-to-b from-pink-50 via-rose-50 to-white";
    case "space":
      return "bg-gradient-to-b from-slate-900 via-purple-950 to-black";
    case "grid":
      return "bg-slate-900 domain-grid-bg";
    case "gradient":
      return "domain-custom-gradient";
    case "minimal":
      return "bg-white dark:bg-slate-900";
    default:
      return "bg-gradient-to-b from-pink-50 to-white";
  }
}

/**
 * Apply domain branding CSS variables
 */
function applyDomainCssVariables(
  primaryColor: string,
  accentColor: string,
  backgroundColor?: string,
  gradientColors?: [string, string]
) {
  const root = document.documentElement;

  root.style.setProperty("--domain-primary", primaryColor);
  root.style.setProperty("--domain-accent", accentColor);

  if (backgroundColor) {
    root.style.setProperty("--domain-bg", backgroundColor);
  }

  if (gradientColors) {
    root.style.setProperty("--domain-gradient-from", gradientColors[0]);
    root.style.setProperty("--domain-gradient-to", gradientColors[1]);
  }

  // Also set HSL values for alpha variations
  const hexToHsl = (hex: string): string => {
    const r = parseInt(hex.slice(1, 3), 16) / 255;
    const g = parseInt(hex.slice(3, 5), 16) / 255;
    const b = parseInt(hex.slice(5, 7), 16) / 255;

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h = 0;
    let s = 0;
    const l = (max + min) / 2;

    if (max !== min) {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);

      switch (max) {
        case r:
          h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
          break;
        case g:
          h = ((b - r) / d + 2) / 6;
          break;
        case b:
          h = ((r - g) / d + 4) / 6;
          break;
      }
    }

    return `${Math.round(h * 360)} ${Math.round(s * 100)}% ${Math.round(l * 100)}%`;
  };

  root.style.setProperty("--domain-primary-hsl", hexToHsl(primaryColor));
  root.style.setProperty("--domain-accent-hsl", hexToHsl(accentColor));
}

/**
 * Reset domain CSS variables to defaults
 */
function resetDomainCssVariables() {
  const root = document.documentElement;

  root.style.removeProperty("--domain-primary");
  root.style.removeProperty("--domain-accent");
  root.style.removeProperty("--domain-bg");
  root.style.removeProperty("--domain-gradient-from");
  root.style.removeProperty("--domain-gradient-to");
  root.style.removeProperty("--domain-primary-hsl");
  root.style.removeProperty("--domain-accent-hsl");
}

/**
 * Component that applies domain-specific branding styles
 *
 * Reads branding configuration from the domain context and applies:
 * - CSS variables for colors
 * - Background styles
 * - Optional custom gradients
 *
 * @example
 * ```tsx
 * <DomainProvider slug="voice-clone">
 *   <DomainBranding>
 *     <Lab3D />
 *   </DomainBranding>
 * </DomainProvider>
 * ```
 */
export function DomainBranding({
  children,
  className = "",
  backgroundStyle: overrideBackgroundStyle,
}: DomainBrandingProps) {
  const domain = useDomainSafe();
  const config = domain?.config;

  // Apply CSS variables when config changes
  useEffect(() => {
    if (config) {
      applyDomainCssVariables(
        config.branding.primaryColor,
        config.branding.accentColor,
        config.branding.backgroundColor,
        config.branding.gradientColors
      );
    }

    return () => {
      resetDomainCssVariables();
    };
  }, [config]);

  // Determine background style
  const bgStyle =
    overrideBackgroundStyle ||
    config?.branding.backgroundStyle ||
    "sky";

  const bgClass = getBackgroundClass(bgStyle);

  return (
    <div
      className={`domain-branding ${bgClass} ${className}`}
      style={
        config?.branding.backgroundColor && bgStyle === "minimal"
          ? { backgroundColor: config.branding.backgroundColor }
          : undefined
      }
    >
      {children}
    </div>
  );
}

/**
 * Hook to get current domain colors
 */
export function useDomainColors() {
  const domain = useDomainSafe();

  return {
    primary: domain?.config?.branding.primaryColor || "#3b82f6",
    accent: domain?.config?.branding.accentColor || "#22c55e",
    background: domain?.config?.branding.backgroundColor,
    backgroundStyle: domain?.config?.branding.backgroundStyle || "sky",
  };
}

/**
 * Component that renders domain-colored text
 */
export function DomainColoredText({
  children,
  color = "primary",
  className = "",
}: {
  children: ReactNode;
  color?: "primary" | "accent";
  className?: string;
}) {
  const colors = useDomainColors();
  const colorValue = color === "primary" ? colors.primary : colors.accent;

  return (
    <span className={className} style={{ color: colorValue }}>
      {children}
    </span>
  );
}

/**
 * Component that renders domain-colored background
 */
export function DomainColoredBg({
  children,
  color = "primary",
  className = "",
  opacity = 1,
}: {
  children: ReactNode;
  color?: "primary" | "accent";
  className?: string;
  opacity?: number;
}) {
  const colors = useDomainColors();
  const colorValue = color === "primary" ? colors.primary : colors.accent;

  // Convert hex to rgba for opacity
  const hexToRgba = (hex: string, alpha: number): string => {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  };

  return (
    <div
      className={className}
      style={{ backgroundColor: hexToRgba(colorValue, opacity) }}
    >
      {children}
    </div>
  );
}

export default DomainBranding;
