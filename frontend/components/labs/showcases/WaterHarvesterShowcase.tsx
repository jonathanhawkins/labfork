"use client";

/**
 * WaterHarvesterShowcase - Interactive mini-simulator for the water-harvester lab.
 *
 * Displays on the generic lab page Overview tab to showcase:
 * - 3D MOF Water Harvester scene with physics
 * - Humidity slider for real-time parameter changes
 * - Quick simulation results (daily yield, efficiency)
 * - Link to the full standalone lab page for deeper exploration
 */

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import {
  Droplets,
  Sun,
  Wind,
  Play,
  ArrowRight,
  Loader2,
  Settings2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  useSimulation,
  formatSimulationResults,
} from "@/lib/simulations/use-simulation";
import type { SimulationParams } from "@/lib/simulations/types";

// Dynamic import for 3D scene (client-side only, no SSR)
const MOFHarvesterScene = dynamic(
  () => import("@/components/MOFHarvesterScene"),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-full flex items-center justify-center bg-background-elevated">
        <div className="text-center">
          <div className="w-10 h-10 border-3 border-blue-400 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-xs text-foreground-muted">
            Loading 3D Harvester...
          </p>
        </div>
      </div>
    ),
  }
);

export function WaterHarvesterShowcase() {
  const [humidity, setHumidity] = useState(45);
  const [mirrorCount, setMirrorCount] = useState(4);
  const [surfacePattern, setSurfacePattern] = useState<"beetle" | "flat">(
    "beetle"
  );

  const {
    simulation,
    isLoading: isSimulating,
    isPolling,
    error: simError,
    runSimulation,
  } = useSimulation();

  // Run quick simulation when parameters change
  useEffect(() => {
    const params: SimulationParams = {
      type: "water_harvester",
      parameters: {
        sorbent_width_cm: 30,
        sorbent_depth_cm: 25,
        mirror_count: mirrorCount,
        mirror_angle: 45,
        humidity_percent: humidity,
        surface_pattern: surfacePattern,
        temperature_ambient_c: 25,
      },
      mode: "quick",
    };
    runSimulation(params);
  }, [humidity, mirrorCount, surfacePattern]); // eslint-disable-line react-hooks/exhaustive-deps

  const simResults = simulation?.results
    ? formatSimulationResults(simulation.results)
    : null;

  return (
    <div className="rounded-lg border border-cyan-500/30 bg-gradient-to-br from-cyan-500/5 via-background to-blue-500/5 overflow-hidden">
      {/* Header */}
      <div className="px-4 sm:px-6 pt-4 sm:pt-5 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 rounded-lg bg-cyan-500/15">
              <Droplets className="w-4 h-4 text-cyan-400" />
            </div>
            <div>
              <h3 className="text-sm sm:text-base font-medium text-foreground-bright">
                Interactive Simulator
              </h3>
              <p className="text-[11px] text-foreground-muted">
                Adjust parameters, see real-time physics
              </p>
            </div>
          </div>
          <Link
            href="/labs/labfork/water-harvester"
            className="inline-flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300 transition-colors min-h-[44px] px-2"
          >
            Full lab
            <ArrowRight className="w-3 h-3" />
          </Link>
        </div>
      </div>

      {/* Main content: 3D scene + controls side by side on desktop */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-0">
        {/* 3D Scene */}
        <div className="lg:col-span-3 relative aspect-[4/3] sm:aspect-[16/10] lg:aspect-auto lg:min-h-[320px]">
          <MOFHarvesterScene
            humidity={humidity}
            mirrorCount={mirrorCount}
            dailyYield={
              simulation?.results
                ? (
                    simulation.results as {
                      daily_yield_liters?: number;
                    }
                  ).daily_yield_liters
                : undefined
            }
            sorbentTemp={
              simulation?.results
                ? (
                    simulation.results as {
                      peak_temperature_c?: number;
                    }
                  ).peak_temperature_c
                : undefined
            }
            className="absolute inset-0"
          />
          {/* Overlay hint */}
          <div className="absolute bottom-2 left-2 text-[10px] text-foreground-subtle bg-background/60 px-2 py-1 rounded backdrop-blur-sm">
            Drag to rotate | Pinch to zoom
          </div>
        </div>

        {/* Controls + Results */}
        <div className="lg:col-span-2 p-4 sm:p-5 space-y-4 border-t lg:border-t-0 lg:border-l border-border/50">
          {/* Humidity slider */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs text-foreground-muted flex items-center gap-1.5">
                <Wind className="w-3 h-3" />
                Humidity
              </label>
              <span className="text-sm font-medium text-foreground-bright">
                {humidity}% RH
              </span>
            </div>
            <input
              type="range"
              min="10"
              max="80"
              value={humidity}
              onChange={(e) => setHumidity(parseInt(e.target.value))}
              className="w-full accent-cyan-400 h-2"
            />
            <div className="flex justify-between text-[10px] text-foreground-subtle mt-1">
              <span>Desert 10%</span>
              <span>Coastal 80%</span>
            </div>
          </div>

          {/* Mirror count */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs text-foreground-muted flex items-center gap-1.5">
                <Sun className="w-3 h-3" />
                Mirrors
              </label>
              <span className="text-sm font-medium text-foreground-bright">
                {mirrorCount}
              </span>
            </div>
            <input
              type="range"
              min="2"
              max="8"
              value={mirrorCount}
              onChange={(e) => setMirrorCount(parseInt(e.target.value))}
              className="w-full accent-yellow-400 h-2"
            />
          </div>

          {/* Surface pattern toggle */}
          <div>
            <label className="text-xs text-foreground-muted block mb-2">
              Condenser Surface
            </label>
            <div className="grid grid-cols-2 gap-2">
              {(["beetle", "flat"] as const).map((pattern) => (
                <button
                  key={pattern}
                  onClick={() => setSurfacePattern(pattern)}
                  className={cn(
                    "px-3 py-2 rounded-lg text-xs transition-colors capitalize min-h-[44px]",
                    surfacePattern === pattern
                      ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/40"
                      : "bg-background border border-border hover:border-foreground-muted/50 text-foreground-muted"
                  )}
                >
                  {pattern === "beetle" ? "Beetle (16x)" : "Flat"}
                </button>
              ))}
            </div>
          </div>

          {/* Divider */}
          <div className="border-t border-border/50" />

          {/* Simulation results */}
          <div>
            <div className="flex items-center gap-1.5 mb-3">
              <Settings2 className="w-3 h-3 text-foreground-muted" />
              <span className="text-xs font-medium text-foreground-muted">
                Simulation Results
              </span>
              {isSimulating && (
                <Loader2 className="w-3 h-3 animate-spin text-cyan-400 ml-auto" />
              )}
            </div>

            {simResults ? (
              <div className="grid grid-cols-2 gap-3">
                <div className="p-2.5 rounded-lg bg-background/50 border border-border/50">
                  <p className="text-lg font-bold text-cyan-400">
                    {simResults.dailyYield}
                  </p>
                  <p className="text-[10px] text-foreground-muted">
                    Daily yield
                  </p>
                </div>
                <div className="p-2.5 rounded-lg bg-background/50 border border-border/50">
                  <p className="text-lg font-bold text-foreground-bright">
                    {simResults.efficiency}
                  </p>
                  <p className="text-[10px] text-foreground-muted">
                    Efficiency
                  </p>
                </div>
                <div className="p-2.5 rounded-lg bg-background/50 border border-border/50">
                  <p className="text-sm font-semibold text-foreground">
                    {simResults.collectionRate}
                  </p>
                  <p className="text-[10px] text-foreground-muted">
                    Collection
                  </p>
                </div>
                <div className="p-2.5 rounded-lg bg-background/50 border border-border/50">
                  <p className="text-sm font-semibold text-foreground">
                    {simResults.peakTemp}
                  </p>
                  <p className="text-[10px] text-foreground-muted">Peak temp</p>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center py-6">
                <Loader2 className="w-5 h-5 animate-spin text-foreground-muted" />
              </div>
            )}
          </div>

          {/* CTA to full lab */}
          <Link
            href="/labs/labfork/water-harvester"
            className="flex items-center justify-center gap-2 w-full px-4 py-2.5 rounded-lg bg-cyan-500/15 text-cyan-400 hover:bg-cyan-500/25 transition-colors text-sm font-medium min-h-[44px]"
          >
            <Play className="w-3.5 h-3.5" />
            Open Full Simulator
          </Link>
        </div>
      </div>
    </div>
  );
}
