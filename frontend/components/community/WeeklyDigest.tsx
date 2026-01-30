/**
 * WeeklyDigest
 *
 * Newsletter-style display of the weekly community digest
 * with breakthroughs, trends, synergies, and highlights.
 */

"use client";

import React, { useState } from "react";
import {
  WeeklyDigest as DigestType,
  Breakthrough,
  TrendingTechnique,
  SynergyHighlight,
  EvolutionHighlight,
  ContributorHighlight,
  SignificanceLevel,
} from "@/lib/meta/community/types";

interface WeeklyDigestProps {
  digest: DigestType;
  onShare?: () => void;
  onSubscribe?: () => void;
}

const significanceColors: Record<SignificanceLevel, string> = {
  critical: "bg-red-100 text-red-700 border-red-200",
  high: "bg-orange-100 text-orange-700 border-orange-200",
  medium: "bg-blue-100 text-blue-700 border-blue-200",
  low: "bg-gray-100 text-gray-600 border-gray-200",
};

export function WeeklyDigest({ digest, onShare, onSubscribe }: WeeklyDigestProps) {
  const [expandedSection, setExpandedSection] = useState<string | null>("breakthroughs");

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden max-w-4xl mx-auto">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="text-sm opacity-80">
            Week {digest.weekNumber}, {digest.year}
          </div>
          <div className="flex items-center gap-2">
            {onShare && (
              <button
                onClick={onShare}
                className="px-3 py-1 bg-white/20 hover:bg-white/30 rounded text-sm transition-colors"
              >
                Share
              </button>
            )}
            {onSubscribe && (
              <button
                onClick={onSubscribe}
                className="px-3 py-1 bg-white hover:bg-gray-100 text-indigo-600 rounded text-sm font-medium transition-colors"
              >
                Subscribe
              </button>
            )}
          </div>
        </div>

        <h1 className="text-2xl font-bold mb-2">{digest.summary.headline}</h1>

        <div className="flex flex-wrap gap-4 text-sm opacity-90">
          <span>{formatDate(digest.startDate)} - {formatDate(digest.endDate)}</span>
          <span>{digest.summary.readTimeMinutes} min read</span>
        </div>
      </div>

      {/* Key Highlights */}
      <div className="p-6 border-b">
        <h2 className="text-lg font-semibold mb-3">Key Highlights</h2>
        <ul className="space-y-2">
          {digest.summary.keyHighlights.map((highlight, i) => (
            <li key={i} className="flex items-start gap-2">
              <span className="text-indigo-500 mt-1">*</span>
              <span>{highlight}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Stats Bar */}
      <div className="grid grid-cols-4 gap-4 p-4 bg-gray-50 border-b">
        <StatBox label="Active Labs" value={digest.stats.activeLabs} />
        <StatBox label="New Techniques" value={digest.stats.newTechniques} />
        <StatBox label="Synergies" value={digest.stats.synergiesDiscovered} />
        <StatBox label="Collaborations" value={digest.stats.collaborationsStarted} />
      </div>

      {/* Sections */}
      <div className="divide-y">
        {/* Breakthroughs */}
        {digest.breakthroughs.length > 0 && (
          <DigestSection
            title="Breakthroughs"
            icon="star"
            count={digest.breakthroughs.length}
            isExpanded={expandedSection === "breakthroughs"}
            onToggle={() =>
              setExpandedSection(expandedSection === "breakthroughs" ? null : "breakthroughs")
            }
          >
            <div className="space-y-4">
              {digest.breakthroughs.map((b) => (
                <BreakthroughCard key={b.id} breakthrough={b} />
              ))}
            </div>
          </DigestSection>
        )}

        {/* Trending Techniques */}
        {digest.trendingTechniques.length > 0 && (
          <DigestSection
            title="Trending Techniques"
            icon="trending"
            count={digest.trendingTechniques.length}
            isExpanded={expandedSection === "trending"}
            onToggle={() =>
              setExpandedSection(expandedSection === "trending" ? null : "trending")
            }
          >
            <div className="space-y-3">
              {digest.trendingTechniques.map((t, i) => (
                <TrendingCard key={t.id} technique={t} rank={i + 1} />
              ))}
            </div>
          </DigestSection>
        )}

        {/* Synergies */}
        {digest.newSynergies.length > 0 && (
          <DigestSection
            title="New Synergies"
            icon="link"
            count={digest.newSynergies.length}
            isExpanded={expandedSection === "synergies"}
            onToggle={() =>
              setExpandedSection(expandedSection === "synergies" ? null : "synergies")
            }
          >
            <div className="space-y-3">
              {digest.newSynergies.map((s) => (
                <SynergyCard key={s.id} synergy={s} />
              ))}
            </div>
          </DigestSection>
        )}

        {/* Evolution Highlights */}
        {digest.evolutionHighlights.length > 0 && (
          <DigestSection
            title="Evolution Highlights"
            icon="dna"
            count={digest.evolutionHighlights.length}
            isExpanded={expandedSection === "evolution"}
            onToggle={() =>
              setExpandedSection(expandedSection === "evolution" ? null : "evolution")
            }
          >
            <div className="space-y-3">
              {digest.evolutionHighlights.map((e) => (
                <EvolutionCard key={e.id} evolution={e} />
              ))}
            </div>
          </DigestSection>
        )}

        {/* Top Contributors */}
        {digest.topContributors.length > 0 && (
          <DigestSection
            title="Top Contributors"
            icon="users"
            count={digest.topContributors.length}
            isExpanded={expandedSection === "contributors"}
            onToggle={() =>
              setExpandedSection(expandedSection === "contributors" ? null : "contributors")
            }
          >
            <div className="space-y-2">
              {digest.topContributors.slice(0, 5).map((c) => (
                <ContributorRow key={c.labId} contributor={c} />
              ))}
            </div>
          </DigestSection>
        )}
      </div>

      {/* Footer */}
      <div className="p-6 bg-gray-50 text-center text-sm text-gray-500">
        <p>
          Generated on {formatDate(digest.generatedAt)}
          {digest.publishedAt && ` | Published ${formatDate(digest.publishedAt)}`}
        </p>
      </div>
    </div>
  );
}

function StatBox({ label, value }: { label: string; value: number }) {
  return (
    <div className="text-center">
      <div className="text-2xl font-bold text-indigo-600">{value}</div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  );
}

interface DigestSectionProps {
  title: string;
  icon: string;
  count: number;
  isExpanded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}

function DigestSection({
  title,
  icon,
  count,
  isExpanded,
  onToggle,
  children,
}: DigestSectionProps) {
  return (
    <div>
      <button
        onClick={onToggle}
        className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <SectionIcon name={icon} />
          <span className="font-semibold">{title}</span>
          <span className="px-2 py-0.5 bg-gray-100 rounded-full text-sm text-gray-600">
            {count}
          </span>
        </div>
        <svg
          className={`w-5 h-5 text-gray-400 transition-transform ${
            isExpanded ? "rotate-180" : ""
          }`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>

      {isExpanded && <div className="px-6 pb-6">{children}</div>}
    </div>
  );
}

function SectionIcon({ name }: { name: string }) {
  const icons: Record<string, React.ReactNode> = {
    star: (
      <svg className="w-5 h-5 text-yellow-500" fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
      </svg>
    ),
    trending: (
      <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    ),
    link: (
      <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
      </svg>
    ),
    dna: (
      <svg className="w-5 h-5 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
    users: (
      <svg className="w-5 h-5 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m12 5.197v-1a6 6 0 00-6-6" />
      </svg>
    ),
  };

  return <>{icons[name] || null}</>;
}

function BreakthroughCard({ breakthrough }: { breakthrough: Breakthrough }) {
  return (
    <div className="border rounded-lg p-4">
      <div className="flex items-start justify-between mb-2">
        <h3 className="font-semibold">{breakthrough.title}</h3>
        <span
          className={`px-2 py-0.5 text-xs rounded border ${
            significanceColors[breakthrough.significance]
          }`}
        >
          {breakthrough.significance}
        </span>
      </div>
      <p className="text-gray-600 text-sm mb-3">{breakthrough.description}</p>
      <div className="flex items-center justify-between text-sm">
        <span className="text-gray-500">by {breakthrough.labName}</span>
        {breakthrough.metrics.improvementPercent > 0 && (
          <span className="text-green-600 font-medium">
            +{breakthrough.metrics.improvementPercent}% improvement
          </span>
        )}
      </div>
    </div>
  );
}

function TrendingCard({ technique, rank }: { technique: TrendingTechnique; rank: number }) {
  return (
    <div className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
      <span className="w-8 h-8 flex items-center justify-center bg-indigo-100 text-indigo-600 font-bold rounded-full">
        {rank}
      </span>
      <div className="flex-1">
        <div className="font-medium">{technique.name}</div>
        <div className="text-sm text-gray-500">
          {technique.category} | {technique.usageCount} uses
        </div>
      </div>
      <div className="text-right">
        <div
          className={`font-semibold ${
            technique.weeklyGrowth > 0 ? "text-green-600" : "text-red-600"
          }`}
        >
          {technique.weeklyGrowth > 0 ? "+" : ""}
          {technique.weeklyGrowth.toFixed(0)}%
        </div>
        <div className="text-xs text-gray-400">this week</div>
      </div>
    </div>
  );
}

function SynergyCard({ synergy }: { synergy: SynergyHighlight }) {
  return (
    <div className="border-l-4 border-blue-400 pl-4 py-2">
      <div className="flex items-center gap-2 mb-1">
        {synergy.techniques.map((t, i) => (
          <React.Fragment key={t}>
            <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-sm">
              {t}
            </span>
            {i < synergy.techniques.length - 1 && (
              <span className="text-gray-400">+</span>
            )}
          </React.Fragment>
        ))}
      </div>
      <p className="text-gray-600 text-sm">{synergy.description}</p>
      <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
        <span>Score: {(synergy.synergyScore * 100).toFixed(0)}%</span>
        <span>by {synergy.discoveredBy}</span>
      </div>
    </div>
  );
}

function EvolutionCard({ evolution }: { evolution: EvolutionHighlight }) {
  return (
    <div className="flex items-center gap-4 p-3 border rounded-lg">
      <div className="w-12 h-12 flex items-center justify-center bg-purple-100 text-purple-600 font-bold rounded-lg">
        G{evolution.generation}
      </div>
      <div className="flex-1">
        <div className="font-medium">{evolution.techniqueName}</div>
        <div className="text-sm text-gray-500">
          {evolution.newCapabilities.slice(0, 2).join(", ")}
          {evolution.newCapabilities.length > 2 && "..."}
        </div>
      </div>
      <div className="text-green-600 font-semibold">
        +{evolution.fitnessImprovement.toFixed(1)}%
      </div>
    </div>
  );
}

function ContributorRow({ contributor }: { contributor: ContributorHighlight }) {
  return (
    <div className="flex items-center gap-4 py-2">
      <span
        className={`w-8 h-8 flex items-center justify-center rounded-full font-bold ${
          contributor.rank === 1
            ? "bg-yellow-100 text-yellow-700"
            : contributor.rank === 2
            ? "bg-gray-200 text-gray-700"
            : contributor.rank === 3
            ? "bg-orange-100 text-orange-700"
            : "bg-gray-100 text-gray-600"
        }`}
      >
        {contributor.rank}
      </span>
      <div className="flex-1">
        <div className="font-medium">{contributor.labName}</div>
        <div className="text-sm text-gray-500">
          {contributor.discoveries} discoveries | {contributor.contributions} contributions
        </div>
      </div>
      <div className="text-indigo-600 font-semibold">{contributor.weeklyScore} pts</div>
    </div>
  );
}

export default WeeklyDigest;
