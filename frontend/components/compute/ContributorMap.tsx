"use client";

/**
 * Contributor Map Component
 *
 * Simple geographic visualization of contributor distribution.
 * Uses SVG world map with animated contributor density indicators.
 */

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { Globe, MapPin, TrendingUp } from "lucide-react";

/**
 * Geographic region
 */
export interface Region {
  id: string;
  name: string;
  contributors: number;
  x: number; // SVG coordinate
  y: number;
  color?: string;
}

/**
 * Props for ContributorMap
 */
export interface ContributorMapProps {
  /** Active contributors count */
  activeContributors: number;
  /** Show animation for new contributors */
  showNewJoin?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Mock regions (would come from real geolocation data)
 */
const MOCK_REGIONS: Region[] = [
  { id: "na-west", name: "North America West", contributors: 0, x: 120, y: 80 },
  { id: "na-east", name: "North America East", contributors: 0, x: 180, y: 90 },
  { id: "eu-west", name: "Europe West", contributors: 0, x: 280, y: 70 },
  { id: "eu-east", name: "Europe East", contributors: 0, x: 320, y: 65 },
  { id: "asia-east", name: "Asia East", contributors: 0, x: 400, y: 85 },
  { id: "asia-south", name: "Asia South", contributors: 0, x: 370, y: 110 },
  { id: "oceania", name: "Oceania", contributors: 0, x: 440, y: 150 },
];

/**
 * Contributor Map Component
 */
export function ContributorMap({
  activeContributors,
  showNewJoin = false,
  className,
}: ContributorMapProps) {
  const [regions, setRegions] = useState<Region[]>(MOCK_REGIONS);
  const [newJoinPulse, setNewJoinPulse] = useState(0);

  // Distribute contributors across regions (mock distribution)
  useEffect(() => {
    if (activeContributors === 0) {
      setRegions(MOCK_REGIONS);
      return;
    }

    // Simple distribution algorithm
    const updated = MOCK_REGIONS.map((region, i) => {
      const weight = [0.25, 0.2, 0.15, 0.1, 0.15, 0.1, 0.05][i] || 0.1;
      const contributors = Math.floor(activeContributors * weight);
      return { ...region, contributors };
    });

    setRegions(updated);
  }, [activeContributors]);

  // Trigger pulse animation when new contributors join
  useEffect(() => {
    if (showNewJoin) {
      setNewJoinPulse((p) => p + 1);
    }
  }, [showNewJoin]);

  const totalRegionsActive = regions.filter((r) => r.contributors > 0).length;

  return (
    <div className={cn("bg-card border border-border rounded-xl p-6", className)}>
      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <h3 className="text-sm font-semibold text-foreground-bright mb-1">
            Global Distribution
          </h3>
          <p className="text-xs text-muted-foreground">
            {totalRegionsActive} {totalRegionsActive === 1 ? "region" : "regions"} active
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Globe className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground-bright">
            {activeContributors}
          </span>
        </div>
      </div>

      {/* Simple World Map Visualization */}
      <div className="relative bg-muted/20 rounded-lg p-4 aspect-[2/1] overflow-hidden">
        {/* SVG World Map (Simplified) */}
        <svg
          viewBox="0 0 500 200"
          className="w-full h-full"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Simplified continents as shapes */}
          <g className="opacity-20 fill-muted-foreground">
            {/* North America */}
            <path d="M 100,50 L 200,50 L 210,120 L 180,140 L 120,130 L 90,100 Z" />
            {/* South America */}
            <path d="M 160,140 L 190,140 L 195,190 L 165,200 L 155,170 Z" />
            {/* Europe */}
            <path d="M 250,40 L 330,45 L 335,90 L 310,100 L 260,95 L 245,70 Z" />
            {/* Africa */}
            <path d="M 270,100 L 320,105 L 330,180 L 290,190 L 260,170 L 265,130 Z" />
            {/* Asia */}
            <path d="M 335,30 L 460,35 L 470,130 L 450,140 L 350,135 L 340,90 Z" />
            {/* Oceania */}
            <path d="M 430,140 L 480,145 L 485,175 L 465,180 L 425,170 Z" />
          </g>

          {/* Contributor Markers */}
          {regions.map((region, index) => (
            <ContributorMarker
              key={region.id}
              region={region}
              pulseKey={newJoinPulse}
              delay={index * 0.1}
            />
          ))}
        </svg>

        {/* New contributor animation overlay */}
        <AnimatePresence>
          {showNewJoin && (
            <motion.div
              key={newJoinPulse}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 1.2 }}
              transition={{ duration: 0.5 }}
              className="absolute top-4 right-4 flex items-center gap-2 bg-green-500/20 border border-green-500/30 rounded-lg px-3 py-1.5"
            >
              <motion.div
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 0.5, repeat: Infinity }}
                className="w-2 h-2 rounded-full bg-green-400"
              />
              <span className="text-xs font-medium text-green-400">
                New contributor joined
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Region List */}
      <div className="mt-4 space-y-2 max-h-48 overflow-y-auto">
        {regions
          .filter((r) => r.contributors > 0)
          .sort((a, b) => b.contributors - a.contributors)
          .map((region, index) => (
            <motion.div
              key={region.id}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className="flex items-center justify-between py-2 px-3 bg-muted/30 rounded-md"
            >
              <div className="flex items-center gap-2">
                <MapPin className="w-3 h-3 text-muted-foreground" />
                <span className="text-xs text-foreground">{region.name}</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="text-xs font-medium text-foreground-bright">
                  {region.contributors}
                </span>
                {index === 0 && (
                  <TrendingUp className="w-3 h-3 text-green-400 ml-1" />
                )}
              </div>
            </motion.div>
          ))}
      </div>

      {/* Empty State */}
      {activeContributors === 0 && (
        <div className="mt-4 text-center py-8">
          <Globe className="w-8 h-8 text-muted-foreground mx-auto mb-2 opacity-40" />
          <p className="text-sm text-muted-foreground">
            No active contributors yet
          </p>
        </div>
      )}
    </div>
  );
}

/**
 * Contributor Marker Component
 */
function ContributorMarker({
  region,
  pulseKey,
  delay,
}: {
  region: Region;
  pulseKey: number;
  delay: number;
}) {
  if (region.contributors === 0) return null;

  // Calculate marker size based on contributor count
  const baseSize = 8;
  const size = baseSize + Math.min(region.contributors * 2, 20);

  return (
    <g>
      {/* Outer pulse ring */}
      <motion.circle
        key={`pulse-${pulseKey}`}
        cx={region.x}
        cy={region.y}
        r={size}
        className="fill-purple-500/20 stroke-purple-500/40"
        strokeWidth="1"
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: [1, 1.5, 2], opacity: [0.5, 0.2, 0] }}
        transition={{
          duration: 2,
          repeat: Infinity,
          delay,
          ease: "easeOut",
        }}
      />

      {/* Main marker */}
      <motion.circle
        cx={region.x}
        cy={region.y}
        r={size / 2}
        className="fill-purple-500 stroke-purple-400"
        strokeWidth="1.5"
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        whileHover={{ scale: 1.2 }}
        transition={{ duration: 0.3, delay }}
      />

      {/* Inner glow */}
      <motion.circle
        cx={region.x}
        cy={region.y}
        r={size / 4}
        className="fill-purple-300"
        animate={{ opacity: [0.6, 1, 0.6] }}
        transition={{ duration: 2, repeat: Infinity }}
      />

      {/* Tooltip on hover (simplified) */}
      <title>
        {region.name}: {region.contributors} {region.contributors === 1 ? "contributor" : "contributors"}
      </title>
    </g>
  );
}
