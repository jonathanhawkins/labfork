"use client";

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
  ReactNode,
} from "react";
import {
  DomainConfig,
  mergeWithDefaults,
  hexToNumber,
} from "@/lib/domain/types";

/**
 * Domain context value
 */
interface DomainContextValue {
  /** Current domain configuration */
  config: DomainConfig | null;
  /** Current domain slug */
  slug: string;
  /** Loading state */
  isLoading: boolean;
  /** Error if config failed to load */
  error: Error | null;
  /** Switch to a different domain */
  switchDomain: (slug: string) => Promise<void>;
  /** Get a color from the branding as a number for Three.js */
  getColorNumber: (colorKey: "primary" | "accent") => number;
  /** Get CSS variable string for a branding color */
  getCssColor: (colorKey: "primary" | "accent") => string;
}

/**
 * Domain context
 */
const DomainContext = createContext<DomainContextValue | null>(null);

/**
 * Props for DomainProvider
 */
interface DomainProviderProps {
  /** Domain slug to load */
  slug: string;
  /** Optional pre-loaded config for SSR */
  initialConfig?: DomainConfig;
  /** Children to render */
  children: ReactNode;
}

/**
 * Fetch domain config from API
 */
async function fetchDomainConfig(slug: string): Promise<DomainConfig> {
  // Try to fetch from API endpoint
  const response = await fetch(`/api/domain/${slug}`);

  if (!response.ok) {
    // If API fails, try loading a fallback default
    if (slug === "voice-clone") {
      // Return hardcoded voice-clone config as fallback
      return mergeWithDefaults({
        name: "Voice Clone Research",
        slug: "voice-clone",
        description: "Research prosody and emotion conditioning in TTS",
        branding: {
          primaryColor: "#4ecdc4",
          accentColor: "#66ffaa",
          backgroundStyle: "sky",
        },
        scene: {
          props: [
            { id: "supercomputer", type: "supercomputer", position: [-6, 0, -5], scale: 1.3 },
            { id: "microphone", type: "microphone", position: [-6, 0, 5], scale: 2.0 },
            { id: "speaker", type: "speaker", position: [6, 0, 5], scale: 2.0 },
            { id: "server", type: "server", position: [6, 0, -5], scale: 2.0 },
          ],
        },
        research: {
          arxivCategories: ["cs.SD", "cs.CL", "eess.AS"],
          keywords: ["prosody", "emotion TTS", "voice cloning"],
        },
      });
    }
    throw new Error(`Failed to load domain config for ${slug}`);
  }

  return response.json();
}

/**
 * Provider component for domain configuration
 *
 * Wraps components that need access to domain configuration via useDomain() hook.
 *
 * @example
 * ```tsx
 * <DomainProvider slug="voice-clone">
 *   <Lab3D />
 * </DomainProvider>
 * ```
 */
export function DomainProvider({
  slug,
  initialConfig,
  children,
}: DomainProviderProps) {
  const [config, setConfig] = useState<DomainConfig | null>(initialConfig || null);
  const [currentSlug, setCurrentSlug] = useState(slug);
  const [isLoading, setIsLoading] = useState(!initialConfig);
  const [error, setError] = useState<Error | null>(null);

  // Load config on mount or when slug changes
  useEffect(() => {
    // Skip if we have initial config and slug matches
    if (initialConfig && slug === currentSlug && config) {
      return;
    }

    // Skip if slug hasn't changed and we already have config
    if (slug === currentSlug && config && !error) {
      return;
    }

    let cancelled = false;

    async function loadConfig() {
      setIsLoading(true);
      setError(null);

      try {
        const loadedConfig = await fetchDomainConfig(slug);
        if (!cancelled) {
          setConfig(loadedConfig);
          setCurrentSlug(slug);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err : new Error(String(err)));
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    loadConfig();

    return () => {
      cancelled = true;
    };
  }, [slug, initialConfig, currentSlug, config, error]);

  // Switch domain function
  const switchDomain = useCallback(async (newSlug: string) => {
    if (newSlug === currentSlug) return;

    setIsLoading(true);
    setError(null);

    try {
      const loadedConfig = await fetchDomainConfig(newSlug);
      setConfig(loadedConfig);
      setCurrentSlug(newSlug);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [currentSlug]);

  // Get color as number for Three.js
  const getColorNumber = useCallback(
    (colorKey: "primary" | "accent"): number => {
      if (!config) return colorKey === "primary" ? 0x3b82f6 : 0x22c55e;
      const hex =
        colorKey === "primary"
          ? config.branding.primaryColor
          : config.branding.accentColor;
      return hexToNumber(hex);
    },
    [config]
  );

  // Get CSS color string
  const getCssColor = useCallback(
    (colorKey: "primary" | "accent"): string => {
      if (!config)
        return colorKey === "primary" ? "#3b82f6" : "#22c55e";
      return colorKey === "primary"
        ? config.branding.primaryColor
        : config.branding.accentColor;
    },
    [config]
  );

  // Memoize context value
  const value = useMemo<DomainContextValue>(
    () => ({
      config,
      slug: currentSlug,
      isLoading,
      error,
      switchDomain,
      getColorNumber,
      getCssColor,
    }),
    [config, currentSlug, isLoading, error, switchDomain, getColorNumber, getCssColor]
  );

  return (
    <DomainContext.Provider value={value}>
      {children}
    </DomainContext.Provider>
  );
}

/**
 * Hook to access domain configuration
 *
 * Must be used within a DomainProvider.
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { config, isLoading, error } = useDomain();
 *
 *   if (isLoading) return <Spinner />;
 *   if (error) return <Error message={error.message} />;
 *
 *   return <div>Domain: {config?.name}</div>;
 * }
 * ```
 */
export function useDomain(): DomainContextValue {
  const context = useContext(DomainContext);

  if (!context) {
    throw new Error("useDomain must be used within a DomainProvider");
  }

  return context;
}

/**
 * Hook to access domain configuration with a fallback
 *
 * Returns null instead of throwing if used outside provider.
 * Useful for components that can work with or without domain context.
 */
export function useDomainSafe(): DomainContextValue | null {
  return useContext(DomainContext);
}

/**
 * Hook to get a specific domain config value
 */
export function useDomainConfig<K extends keyof DomainConfig>(
  key: K
): DomainConfig[K] | undefined {
  const { config } = useDomain();
  return config?.[key];
}

/**
 * Default export
 */
export default DomainProvider;
