/**
 * OpportunityBoard
 *
 * Searchable board displaying research opportunities with
 * filtering, claiming, and progress tracking.
 */

"use client";

import React, { useState, useMemo } from "react";
import {
  ResearchOpportunity,
  DifficultyLevel,
  SignificanceLevel,
  OpportunityLeaderboard,
  LeaderboardEntry,
  GapType,
} from "@/lib/meta/community/types";

interface OpportunityBoardProps {
  opportunities: ResearchOpportunity[];
  leaderboard?: OpportunityLeaderboard;
  currentLabId?: string;
  onClaim?: (opportunityId: string) => void;
  onUnclaim?: (opportunityId: string) => void;
  onViewDetails?: (opportunity: ResearchOpportunity) => void;
}

const difficultyColors: Record<DifficultyLevel, string> = {
  beginner: "bg-green-100 text-green-700",
  intermediate: "bg-blue-100 text-blue-700",
  advanced: "bg-orange-100 text-orange-700",
  expert: "bg-red-100 text-red-700",
};

const impactColors: Record<SignificanceLevel, string> = {
  critical: "bg-red-100 text-red-700",
  high: "bg-orange-100 text-orange-700",
  medium: "bg-blue-100 text-blue-700",
  low: "bg-gray-100 text-gray-600",
};

const gapTypeLabels: Record<GapType, string> = {
  "missing-technique": "Missing Technique",
  "unexplored-combination": "Unexplored Combination",
  "domain-adaptation": "Domain Adaptation",
  "performance-optimization": "Performance Optimization",
  scalability: "Scalability",
  "theoretical-gap": "Theoretical Gap",
  tooling: "Tooling",
  documentation: "Documentation",
};

export function OpportunityBoard({
  opportunities,
  leaderboard,
  currentLabId,
  onClaim,
  onUnclaim,
  onViewDetails,
}: OpportunityBoardProps) {
  const [search, setSearch] = useState("");
  const [filterDomain, setFilterDomain] = useState<string>("");
  const [filterDifficulty, setFilterDifficulty] = useState<DifficultyLevel | "">("");
  const [filterImpact, setFilterImpact] = useState<SignificanceLevel | "">("");
  const [showBountyOnly, setShowBountyOnly] = useState(false);
  const [showLeaderboard, setShowLeaderboard] = useState(false);

  // Get unique domains
  const domains = useMemo(() => {
    const domainSet = new Set(opportunities.map((o) => o.domain));
    return Array.from(domainSet).sort();
  }, [opportunities]);

  // Filter opportunities
  const filteredOpportunities = useMemo(() => {
    return opportunities.filter((opp) => {
      if (search && !opp.title.toLowerCase().includes(search.toLowerCase()) &&
          !opp.description.toLowerCase().includes(search.toLowerCase())) {
        return false;
      }
      if (filterDomain && opp.domain !== filterDomain) return false;
      if (filterDifficulty && opp.difficulty !== filterDifficulty) return false;
      if (filterImpact && opp.impact !== filterImpact) return false;
      if (showBountyOnly && !opp.hasBounty) return false;
      return true;
    });
  }, [opportunities, search, filterDomain, filterDifficulty, filterImpact, showBountyOnly]);

  // Count by status
  const statusCounts = useMemo(() => {
    return opportunities.reduce(
      (acc, opp) => {
        acc[opp.status] = (acc[opp.status] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    );
  }, [opportunities]);

  return (
    <div className="bg-white rounded-lg shadow-lg">
      {/* Header */}
      <div className="p-6 border-b">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold">Research Opportunities</h2>
            <p className="text-gray-500 text-sm">
              {filteredOpportunities.length} opportunities available
            </p>
          </div>
          {leaderboard && (
            <button
              onClick={() => setShowLeaderboard(!showLeaderboard)}
              className={`px-4 py-2 rounded-lg transition-colors ${
                showLeaderboard
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200"
              }`}
            >
              Leaderboard
            </button>
          )}
        </div>

        {/* Status Summary */}
        <div className="flex flex-wrap gap-4 mb-4">
          <StatusBadge label="Open" count={statusCounts.open || 0} color="green" />
          <StatusBadge label="Claimed" count={statusCounts.claimed || 0} color="blue" />
          <StatusBadge label="In Progress" count={statusCounts["in-progress"] || 0} color="yellow" />
          <StatusBadge label="Completed" count={statusCounts.completed || 0} color="gray" />
        </div>

        {/* Search and Filters */}
        <div className="flex flex-wrap gap-3">
          <input
            type="text"
            placeholder="Search opportunities..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="flex-1 min-w-[200px] px-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          />

          <select
            value={filterDomain}
            onChange={(e) => setFilterDomain(e.target.value)}
            className="px-3 py-2 border rounded-lg bg-white"
          >
            <option value="">All Domains</option>
            {domains.map((domain) => (
              <option key={domain} value={domain}>
                {domain}
              </option>
            ))}
          </select>

          <select
            value={filterDifficulty}
            onChange={(e) => setFilterDifficulty(e.target.value as DifficultyLevel | "")}
            className="px-3 py-2 border rounded-lg bg-white"
          >
            <option value="">All Difficulties</option>
            <option value="beginner">Beginner</option>
            <option value="intermediate">Intermediate</option>
            <option value="advanced">Advanced</option>
            <option value="expert">Expert</option>
          </select>

          <select
            value={filterImpact}
            onChange={(e) => setFilterImpact(e.target.value as SignificanceLevel | "")}
            className="px-3 py-2 border rounded-lg bg-white"
          >
            <option value="">All Impact</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>

          <label className="flex items-center gap-2 px-3 py-2 border rounded-lg cursor-pointer hover:bg-gray-50">
            <input
              type="checkbox"
              checked={showBountyOnly}
              onChange={(e) => setShowBountyOnly(e.target.checked)}
              className="rounded text-indigo-600"
            />
            <span className="text-sm">Bounty Only</span>
          </label>
        </div>
      </div>

      {/* Leaderboard Panel */}
      {showLeaderboard && leaderboard && (
        <div className="p-6 bg-gradient-to-r from-indigo-50 to-purple-50 border-b">
          <h3 className="font-semibold mb-4">Top Contributors</h3>
          <div className="space-y-2">
            {leaderboard.entries.slice(0, 5).map((entry) => (
              <LeaderboardRow
                key={entry.labId}
                entry={entry}
                isCurrentLab={entry.labId === currentLabId}
              />
            ))}
          </div>
        </div>
      )}

      {/* Opportunities List */}
      <div className="divide-y">
        {filteredOpportunities.map((opportunity) => (
          <OpportunityCard
            key={opportunity.id}
            opportunity={opportunity}
            isClaimedByUser={opportunity.claimedBy?.labId === currentLabId}
            onClaim={onClaim}
            onUnclaim={onUnclaim}
            onViewDetails={onViewDetails}
          />
        ))}

        {filteredOpportunities.length === 0 && (
          <div className="p-12 text-center text-gray-500">
            <svg
              className="w-16 h-16 mx-auto mb-4 text-gray-300"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
              />
            </svg>
            <p className="text-lg font-medium mb-1">No opportunities found</p>
            <p className="text-sm">Try adjusting your filters</p>
          </div>
        )}
      </div>
    </div>
  );
}

function StatusBadge({
  label,
  count,
  color,
}: {
  label: string;
  count: number;
  color: string;
}) {
  const colorClasses: Record<string, string> = {
    green: "bg-green-100 text-green-700",
    blue: "bg-blue-100 text-blue-700",
    yellow: "bg-yellow-100 text-yellow-700",
    gray: "bg-gray-100 text-gray-600",
  };

  return (
    <div className={`px-3 py-1 rounded-full text-sm ${colorClasses[color]}`}>
      <span className="font-semibold">{count}</span> {label}
    </div>
  );
}

function LeaderboardRow({
  entry,
  isCurrentLab,
}: {
  entry: LeaderboardEntry;
  isCurrentLab: boolean;
}) {
  const rankStyles: Record<number, string> = {
    1: "bg-yellow-100 text-yellow-700",
    2: "bg-gray-200 text-gray-700",
    3: "bg-orange-100 text-orange-700",
  };

  return (
    <div
      className={`flex items-center gap-4 p-2 rounded-lg ${
        isCurrentLab ? "bg-blue-50 border border-blue-200" : ""
      }`}
    >
      <span
        className={`w-8 h-8 flex items-center justify-center rounded-full font-bold ${
          rankStyles[entry.rank] || "bg-gray-100 text-gray-600"
        }`}
      >
        {entry.rank}
      </span>
      <div className="flex-1">
        <div className="font-medium">
          {entry.labName}
          {isCurrentLab && (
            <span className="ml-2 text-xs text-blue-500">(You)</span>
          )}
        </div>
        <div className="text-xs text-gray-500">
          {entry.opportunitiesCompleted} completed | {entry.streak} streak
        </div>
      </div>
      <div className="text-right">
        <div className="font-semibold text-indigo-600">{entry.impactScore}</div>
        <div className="text-xs text-gray-400">impact</div>
      </div>
      {entry.badges.length > 0 && (
        <div className="flex gap-1">
          {entry.badges.slice(0, 3).map((badge) => (
            <span
              key={badge}
              className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs rounded"
              title={badge}
            >
              {badge.charAt(0).toUpperCase()}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

interface OpportunityCardProps {
  opportunity: ResearchOpportunity;
  isClaimedByUser: boolean;
  onClaim?: (opportunityId: string) => void;
  onUnclaim?: (opportunityId: string) => void;
  onViewDetails?: (opportunity: ResearchOpportunity) => void;
}

function OpportunityCard({
  opportunity,
  isClaimedByUser,
  onClaim,
  onUnclaim,
  onViewDetails,
}: OpportunityCardProps) {
  const isOpen = opportunity.status === "open";
  const isClaimed = opportunity.status === "claimed" || opportunity.status === "in-progress";

  return (
    <div className="p-6 hover:bg-gray-50 transition-colors">
      <div className="flex items-start gap-4">
        {/* Status Indicator */}
        <div
          className={`w-3 h-3 rounded-full mt-1.5 flex-shrink-0 ${
            isOpen
              ? "bg-green-500"
              : isClaimed
              ? "bg-blue-500"
              : "bg-gray-300"
          }`}
        />

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-4 mb-2">
            <div>
              <h3 className="font-semibold text-lg">{opportunity.title}</h3>
              <div className="flex flex-wrap items-center gap-2 mt-1">
                <span className="text-sm text-gray-500">{opportunity.domain}</span>
                <span className="text-gray-300">|</span>
                <span className="text-sm text-gray-500">
                  {gapTypeLabels[opportunity.gapType]}
                </span>
              </div>
            </div>

            {opportunity.hasBounty && opportunity.bounty && (
              <div className="px-3 py-1 bg-yellow-100 text-yellow-700 rounded-lg text-sm font-semibold flex-shrink-0">
                ${opportunity.bounty.amount} bounty
              </div>
            )}
          </div>

          <p className="text-gray-600 mb-3 line-clamp-2">
            {opportunity.description}
          </p>

          {/* Tags */}
          <div className="flex flex-wrap gap-2 mb-3">
            <span className={`px-2 py-0.5 text-xs rounded ${difficultyColors[opportunity.difficulty]}`}>
              {opportunity.difficulty}
            </span>
            <span className={`px-2 py-0.5 text-xs rounded ${impactColors[opportunity.impact]}`}>
              {opportunity.impact} impact
            </span>
            <span className="px-2 py-0.5 text-xs rounded bg-gray-100 text-gray-600">
              {opportunity.estimatedEffort.minHours}-{opportunity.estimatedEffort.maxHours}h
            </span>
            {opportunity.tags.slice(0, 3).map((tag) => (
              <span
                key={tag}
                className="px-2 py-0.5 text-xs rounded bg-indigo-50 text-indigo-600"
              >
                {tag}
              </span>
            ))}
          </div>

          {/* Progress bar for claimed opportunities */}
          {isClaimed && opportunity.progress > 0 && (
            <div className="mb-3">
              <div className="flex items-center justify-between text-sm mb-1">
                <span className="text-gray-500">Progress</span>
                <span className="font-medium">{opportunity.progress}%</span>
              </div>
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-indigo-600 rounded-full transition-all"
                  style={{ width: `${opportunity.progress}%` }}
                />
              </div>
            </div>
          )}

          {/* Claimed by info */}
          {opportunity.claimedBy && (
            <div className="text-sm text-gray-500 mb-3">
              Claimed by{" "}
              <span className="font-medium">{opportunity.claimedBy.labName}</span>
              {isClaimedByUser && " (You)"}
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center gap-2">
            {isOpen && onClaim && (
              <button
                onClick={() => onClaim(opportunity.id)}
                className="px-4 py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 transition-colors"
              >
                Claim Opportunity
              </button>
            )}
            {isClaimedByUser && onUnclaim && (
              <button
                onClick={() => onUnclaim(opportunity.id)}
                className="px-4 py-2 border border-red-300 text-red-600 text-sm rounded-lg hover:bg-red-50 transition-colors"
              >
                Release Claim
              </button>
            )}
            {onViewDetails && (
              <button
                onClick={() => onViewDetails(opportunity)}
                className="px-4 py-2 text-gray-600 text-sm hover:bg-gray-100 rounded-lg transition-colors"
              >
                View Details
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default OpportunityBoard;
