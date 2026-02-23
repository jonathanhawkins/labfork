/**
 * DomainShowcase
 *
 * Grid display of all available research domains with
 * animated cards and live agent counts.
 * Fully internationalized for global accessibility.
 */

"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useTranslations } from "next-intl";

export interface DomainInfo {
  slug: string;
  nameKey: string;
  descKey: string;
  primaryColor: string;
  accentColor: string;
  icon: React.ReactNode;
  recentBreakthrough?: string;
  tags: string[];
}

interface DomainShowcaseProps {
  domains?: DomainInfo[];
}

const domainIcons: Record<string, React.ReactNode> = {
  "voice-clone": (
    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
    </svg>
  ),
  "quant-trading": (
    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
    </svg>
  ),
  "game-ai": (
    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  "robotics-ml": (
    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
    </svg>
  ),
  "drug-discovery": (
    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
    </svg>
  ),
  "climate-modeling": (
    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  "nlp-research": (
    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>
  ),
  "computer-vision": (
    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
    </svg>
  ),
  "biotech-nlp": (
    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
    </svg>
  ),
};

export const defaultDomains: DomainInfo[] = [
  {
    slug: "voice-clone",
    nameKey: "voiceClone",
    descKey: "voiceCloneDesc",
    primaryColor: "#4ecdc4",
    accentColor: "#66ffaa",
    icon: domainIcons["voice-clone"],
    recentBreakthrough: "Emotion transfer without paired data",
    tags: ["TTS", "Prosody", "Emotion"],
  },
  {
    slug: "quant-trading",
    nameKey: "quantTrading",
    descKey: "quantTradingDesc",
    primaryColor: "#10b981",
    accentColor: "#f59e0b",
    icon: domainIcons["quant-trading"],
    recentBreakthrough: "Transformer-based momentum signals",
    tags: ["Finance", "RL", "Time Series"],
  },
  {
    slug: "game-ai",
    nameKey: "gameAI",
    descKey: "gameAIDesc",
    primaryColor: "#ec4899",
    accentColor: "#8b5cf6",
    icon: domainIcons["game-ai"],
    recentBreakthrough: "Zero-shot transfer to new game genres",
    tags: ["RL", "Games", "Multi-Agent"],
  },
  {
    slug: "robotics-ml",
    nameKey: "roboticsML",
    descKey: "roboticsMLDesc",
    primaryColor: "#f97316",
    accentColor: "#06b6d4",
    icon: domainIcons["robotics-ml"],
    recentBreakthrough: "Sim-to-real with 10x less data",
    tags: ["Robotics", "Manipulation", "Sim2Real"],
  },
  {
    slug: "drug-discovery",
    nameKey: "drugDiscovery",
    descKey: "drugDiscoveryDesc",
    primaryColor: "#14b8a6",
    accentColor: "#f43f5e",
    icon: domainIcons["drug-discovery"],
    recentBreakthrough: "Novel EGFR inhibitor candidates",
    tags: ["Molecules", "Proteins", "Graph NN"],
  },
  {
    slug: "climate-modeling",
    nameKey: "climateModeling",
    descKey: "climateModelingDesc",
    primaryColor: "#0ea5e9",
    accentColor: "#22c55e",
    icon: domainIcons["climate-modeling"],
    recentBreakthrough: "10-day forecast matching ECMWF",
    tags: ["Climate", "Weather", "Geospatial"],
  },
  {
    slug: "nlp-research",
    nameKey: "nlpResearch",
    descKey: "nlpResearchDesc",
    primaryColor: "#6366f1",
    accentColor: "#f59e0b",
    icon: domainIcons["nlp-research"],
    recentBreakthrough: "Reasoning chains with 3x fewer tokens",
    tags: ["LLMs", "RAG", "Transformers"],
  },
  {
    slug: "computer-vision",
    nameKey: "computerVision",
    descKey: "computerVisionDesc",
    primaryColor: "#ef4444",
    accentColor: "#3b82f6",
    icon: domainIcons["computer-vision"],
    recentBreakthrough: "Real-time segmentation at 60fps",
    tags: ["Detection", "Segmentation", "ViT"],
  },
  {
    slug: "biotech-nlp",
    nameKey: "biotechNLP",
    descKey: "biotechNLPDesc",
    primaryColor: "#8b5cf6",
    accentColor: "#22c55e",
    icon: domainIcons["biotech-nlp"],
    recentBreakthrough: "98% NER accuracy on clinical notes",
    tags: ["BioNLP", "Drug Discovery", "Clinical"],
  },
];

function DomainCard({ domain }: { domain: DomainInfo }) {
  const t = useTranslations("domains");
  const [isHovered, setIsHovered] = useState(false);

  return (
    <Link
      href={`/lab?domain=${domain.slug}`}
      className="group relative block"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div
        className="relative overflow-hidden rounded-2xl border border-white/10 bg-white/5 backdrop-blur-sm transition-all duration-300 hover:border-white/20 hover:bg-white/10 hover:shadow-xl"
        style={{
          boxShadow: isHovered
            ? `0 20px 40px -20px ${domain.primaryColor}40`
            : undefined,
        }}
      >
        {/* Gradient accent */}
        <div
          className="absolute top-0 left-0 right-0 h-1 opacity-80"
          style={{
            background: `linear-gradient(90deg, ${domain.primaryColor}, ${domain.accentColor})`,
          }}
        />

        <div className="p-6">
          {/* Header */}
          <div className="flex items-start justify-between mb-4">
            <div
              className="p-3 rounded-xl"
              style={{ backgroundColor: `${domain.primaryColor}20` }}
            >
              <div style={{ color: domain.primaryColor }}>{domain.icon}</div>
            </div>
          </div>

          {/* Title & Description */}
          <h3 className="text-lg font-semibold text-white mb-2 group-hover:text-blue-400 transition-colors">
            {t(domain.nameKey)}
          </h3>
          <p className="text-sm text-gray-400 mb-4 line-clamp-2">
            {t(domain.descKey)}
          </p>

          {/* Breakthrough */}
          {domain.recentBreakthrough && (
            <div className="mb-4 p-3 rounded-lg bg-white/5 border border-white/5">
              <div className="text-xs text-gray-500 mb-1">{t("latestBreakthrough")}</div>
              <div className="text-sm text-gray-300 line-clamp-1">
                {domain.recentBreakthrough}
              </div>
            </div>
          )}

          {/* Tags */}
          <div className="flex flex-wrap gap-2 mb-4">
            {domain.tags.slice(0, 3).map((tag) => (
              <span
                key={tag}
                className="px-2 py-0.5 text-xs rounded-full bg-white/5 text-gray-400"
              >
                {tag}
              </span>
            ))}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-end text-sm">
            <span
              className="font-medium transition-colors"
              style={{ color: isHovered ? domain.primaryColor : "#9ca3af" }}
            >
              {t("viewLab")} &rarr;
            </span>
          </div>
        </div>
      </div>
    </Link>
  );
}

export function DomainShowcase({ domains = defaultDomains }: DomainShowcaseProps) {
  const t = useTranslations("domains");

  return (
    <section className="py-24 bg-slate-950">
      <div className="max-w-7xl mx-auto px-6">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            {t("title")}
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            {t("subtitle")}
          </p>
        </div>

        {/* Domain Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {domains.map((domain) => (
            <DomainCard key={domain.slug} domain={domain} />
          ))}
        </div>

        {/* Custom Domain CTA */}
        <div className="mt-12 text-center">
          <p className="text-gray-500 mb-4">
            {t("customDomainCTA")}
          </p>
          <Link
            href="/domains/new"
            className="inline-flex items-center gap-2 px-6 py-3 border border-white/20 text-white rounded-xl hover:bg-white/5 transition-colors"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 6v6m0 0v6m0-6h6m-6 0H6"
              />
            </svg>
            {t("createCustomDomain")}
          </Link>
        </div>
      </div>
    </section>
  );
}

export default DomainShowcase;
