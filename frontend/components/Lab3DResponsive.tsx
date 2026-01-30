"use client";

/**
 * Lab3DResponsive
 *
 * Wrapper that shows full 3D lab on desktop, simpler 2D fallback on mobile.
 * Improves performance and usability on phones/tablets.
 */

import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import type { ActivityWithConfig } from "./lab/activities";

// Lazy load the heavy 3D component
const Lab3D = dynamic(() => import("./Lab3D"), {
  ssr: false,
  loading: () => <Lab2DFallback isLoading />,
});

interface Agent {
  id: string;
  name: string;
  color: number;
  position: [number, number, number];
  task?: string;
  status: "idle" | "working" | "thinking";
}

interface Lab3DResponsiveProps {
  agents?: Agent[];
  activities?: ActivityWithConfig[];
  onAgentClick?: (agent: Agent) => void;
  onComputerClick?: () => void;
  showDemoProps?: boolean;
}

// Simple hook for mobile detection
function useIsMobile(breakpoint = 768) {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < breakpoint);
    };

    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, [breakpoint]);

  return isMobile;
}

// Check for reduced motion preference
function usePrefersReducedMotion() {
  const [prefersReduced, setPrefersReduced] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setPrefersReduced(mq.matches);

    const handler = (e: MediaQueryListEvent) => setPrefersReduced(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  return prefersReduced;
}

// 2D Fallback for mobile
function Lab2DFallback({ isLoading = false }: { isLoading?: boolean }) {
  const agents = [
    { id: "codex", name: "Codex", color: "#ffb3ba", status: "working" },
    { id: "opus", name: "Opus", color: "#bae1ff", status: "working" },
    { id: "explorer", name: "Explorer", color: "#ffffba", status: "thinking" },
    { id: "planner", name: "Planner", color: "#baffc9", status: "idle" },
    { id: "lab-manager", name: "Lab-Manager", color: "#4ecdc4", status: "working" },
  ];

  return (
    <div className="w-full h-full min-h-[300px] bg-gradient-to-b from-slate-900 to-slate-800 rounded-xl p-6 flex flex-col">
      {/* Header */}
      <div className="text-center mb-6">
        <h3 className="text-lg font-semibold text-white mb-1">
          {isLoading ? "Loading Lab..." : "Research Lab"}
        </h3>
        <p className="text-sm text-gray-400">
          {isLoading ? "Preparing 3D visualization" : "5 AI agents working"}
        </p>
      </div>

      {/* Agent Grid */}
      <div className="flex-1 grid grid-cols-2 sm:grid-cols-3 gap-3">
        {agents.map((agent) => (
          <button
            key={agent.id}
            className="flex flex-col items-center justify-center p-4 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 active:scale-[0.98] transition-all min-h-[100px]"
          >
            {/* Agent avatar */}
            <div
              className="w-12 h-12 rounded-full mb-2 flex items-center justify-center relative"
              style={{ backgroundColor: agent.color }}
            >
              <span className="text-xl">
                {agent.id === "codex" && "💻"}
                {agent.id === "opus" && "📊"}
                {agent.id === "explorer" && "🔍"}
                {agent.id === "planner" && "📋"}
                {agent.id === "lab-manager" && "🤖"}
              </span>

              {/* Status indicator */}
              <span
                className={`absolute -bottom-1 -right-1 w-4 h-4 rounded-full border-2 border-slate-800 ${
                  agent.status === "working"
                    ? "bg-green-500 animate-pulse"
                    : agent.status === "thinking"
                    ? "bg-yellow-500 animate-pulse"
                    : "bg-gray-500"
                }`}
              />
            </div>

            <span className="text-sm font-medium text-white">{agent.name}</span>
            <span className="text-xs text-gray-400 capitalize">{agent.status}</span>
          </button>
        ))}

        {/* Supercomputer card */}
        <button className="flex flex-col items-center justify-center p-4 rounded-xl bg-gradient-to-br from-purple-500/20 to-blue-500/20 border border-purple-500/30 hover:from-purple-500/30 hover:to-blue-500/30 active:scale-[0.98] transition-all min-h-[100px]">
          <div className="w-12 h-12 rounded-lg bg-slate-700 mb-2 flex items-center justify-center">
            <span className="text-xl">🖥️</span>
          </div>
          <span className="text-sm font-medium text-white">Supercomputer</span>
          <span className="text-xs text-green-400">Online</span>
        </button>
      </div>

      {/* Footer hint */}
      <p className="text-center text-xs text-gray-500 mt-4">
        Tap an agent to see details • Rotate to landscape for 3D view
      </p>
    </div>
  );
}

export function Lab3DResponsive(props: Lab3DResponsiveProps) {
  const isMobile = useIsMobile();
  const prefersReducedMotion = usePrefersReducedMotion();

  // Show 2D fallback on mobile or when user prefers reduced motion
  if (isMobile || prefersReducedMotion) {
    return <Lab2DFallback />;
  }

  return <Lab3D {...props} />;
}

export default Lab3DResponsive;
