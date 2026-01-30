/**
 * MetaAgentFeature
 *
 * Feature section showcasing the 5 meta-agents and their
 * collective intelligence capabilities.
 */

"use client";

import React, { useState } from "react";

interface MetaAgent {
  name: string;
  displayName: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  capabilities: string[];
  exampleDiscovery: {
    title: string;
    details: string;
  };
}

const metaAgents: MetaAgent[] = [
  {
    name: "synergy-detector",
    displayName: "Synergy Detector",
    description: "Finds powerful combinations of techniques from different papers",
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    color: "#f59e0b",
    capabilities: [
      "Analyzes technique compatibility",
      "Scores potential synergies",
      "Suggests experiment combinations",
    ],
    exampleDiscovery: {
      title: "Voice + Sentiment Synergy",
      details: "Emotion vectors from voice cloning can predict trading sentiment",
    },
  },
  {
    name: "pattern-recognizer",
    displayName: "Pattern Recognizer",
    description: "Identifies emerging research trends and paradigm shifts",
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
    color: "#8b5cf6",
    capabilities: [
      "Tracks citation velocity",
      "Identifies hot topics",
      "Predicts breakout papers",
    ],
    exampleDiscovery: {
      title: "Diffusion Trend",
      details: "Detected diffusion models spreading from images to audio/3D/robotics",
    },
  },
  {
    name: "gap-analyzer",
    displayName: "Gap Analyzer",
    description: "Discovers unexplored research opportunities",
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
    ),
    color: "#ec4899",
    capabilities: [
      "Maps research landscape",
      "Finds underexplored areas",
      "Suggests novel experiments",
    ],
    exampleDiscovery: {
      title: "Prosody Gap",
      details: "No papers on combining prosody contours with reinforcement learning",
    },
  },
  {
    name: "evolution-engine",
    displayName: "Evolution Engine",
    description: "Evolves techniques through genetic algorithms",
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
      </svg>
    ),
    color: "#10b981",
    capabilities: [
      "Crossover of successful techniques",
      "Mutation for exploration",
      "Fitness-based selection",
    ],
    exampleDiscovery: {
      title: "Evolved Architecture",
      details: "Combined attention from ViT + convolutions from ResNet = hybrid outperforms both",
    },
  },
  {
    name: "transfer-agent",
    displayName: "Transfer Agent",
    description: "Bridges knowledge across different research domains",
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
      </svg>
    ),
    color: "#3b82f6",
    capabilities: [
      "Domain similarity analysis",
      "Technique adaptation rules",
      "Cross-domain validation",
    ],
    exampleDiscovery: {
      title: "Game to Finance Transfer",
      details: "Multi-agent RL from StarCraft adapted for portfolio management",
    },
  },
];

export function MetaAgentFeature() {
  const [selectedAgent, setSelectedAgent] = useState<MetaAgent>(metaAgents[0]);

  return (
    <section className="py-24 bg-slate-900">
      <div className="max-w-6xl mx-auto px-6">
        {/* Section Header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-amber-500/10 text-amber-400 text-sm mb-4">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            Meta-Research Intelligence
          </div>
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            5 AI Agents Working Together
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            A team of specialized agents that analyze, discover, and evolve research
            techniques automatically.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-12 items-start">
          {/* Agent Selector */}
          <div className="space-y-3">
            {metaAgents.map((agent) => (
              <button
                key={agent.name}
                onClick={() => setSelectedAgent(agent)}
                className={`w-full flex items-center gap-4 p-4 rounded-xl transition-all text-left ${
                  selectedAgent.name === agent.name
                    ? "bg-white/10 border border-white/20"
                    : "bg-white/5 border border-transparent hover:bg-white/10"
                }`}
              >
                <div
                  className="p-3 rounded-lg"
                  style={{
                    backgroundColor: `${agent.color}20`,
                    color: agent.color,
                  }}
                >
                  {agent.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <h4 className="font-semibold text-white">{agent.displayName}</h4>
                  <p className="text-sm text-gray-400 truncate">{agent.description}</p>
                </div>
                <svg
                  className={`w-5 h-5 transition-transform ${
                    selectedAgent.name === agent.name ? "rotate-90" : ""
                  }`}
                  style={{ color: selectedAgent.name === agent.name ? agent.color : "#6b7280" }}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
            ))}
          </div>

          {/* Agent Details */}
          <div className="lg:sticky lg:top-8">
            <div
              className="rounded-2xl border border-white/10 overflow-hidden"
              style={{
                background: `linear-gradient(135deg, ${selectedAgent.color}10, transparent)`,
              }}
            >
              {/* Header */}
              <div className="p-6 border-b border-white/10">
                <div className="flex items-center gap-4 mb-4">
                  <div
                    className="p-4 rounded-xl"
                    style={{
                      backgroundColor: `${selectedAgent.color}20`,
                      color: selectedAgent.color,
                    }}
                  >
                    {selectedAgent.icon}
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white">
                      {selectedAgent.displayName}
                    </h3>
                    <p className="text-gray-400">{selectedAgent.description}</p>
                  </div>
                </div>
              </div>

              {/* Capabilities */}
              <div className="p-6 border-b border-white/10">
                <h4 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
                  Capabilities
                </h4>
                <ul className="space-y-3">
                  {selectedAgent.capabilities.map((cap, idx) => (
                    <li key={idx} className="flex items-center gap-3">
                      <div
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: selectedAgent.color }}
                      />
                      <span className="text-gray-300">{cap}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Example Discovery */}
              <div className="p-6">
                <h4 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
                  Example Discovery
                </h4>
                <div
                  className="p-4 rounded-xl"
                  style={{ backgroundColor: `${selectedAgent.color}10` }}
                >
                  <div
                    className="font-semibold mb-1"
                    style={{ color: selectedAgent.color }}
                  >
                    {selectedAgent.exampleDiscovery.title}
                  </div>
                  <p className="text-sm text-gray-400">
                    {selectedAgent.exampleDiscovery.details}
                  </p>
                </div>
              </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-4 mt-6">
              <div className="p-4 rounded-xl bg-white/5 text-center">
                <div className="text-2xl font-bold text-white">1,247</div>
                <div className="text-xs text-gray-500">Synergies Found</div>
              </div>
              <div className="p-4 rounded-xl bg-white/5 text-center">
                <div className="text-2xl font-bold text-white">89</div>
                <div className="text-xs text-gray-500">Techniques Evolved</div>
              </div>
              <div className="p-4 rounded-xl bg-white/5 text-center">
                <div className="text-2xl font-bold text-white">23</div>
                <div className="text-xs text-gray-500">Cross-Domain Transfers</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default MetaAgentFeature;
