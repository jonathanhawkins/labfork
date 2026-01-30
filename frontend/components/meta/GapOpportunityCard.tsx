"use client";

/**
 * GapOpportunityCard Component
 *
 * Displays a research gap opportunity with impact score, effort estimate,
 * and action buttons for pursuing the opportunity.
 */

import React, { useState } from "react";
import {
  GapOpportunity,
  ResearchGap,
  getGapTypeLabel,
  getGapSeverityColor,
  getOpportunityTypeIcon,
  getEffortLevelLabel,
} from "@/lib/meta/gaps";

interface GapOpportunityCardProps {
  opportunity: GapOpportunity;
  gap?: ResearchGap;
  onPursue?: (opportunityId: string) => void;
  onDismiss?: (opportunityId: string) => void;
  onViewDetails?: (opportunityId: string) => void;
  showGapInfo?: boolean;
  compact?: boolean;
  className?: string;
}

export function GapOpportunityCard({
  opportunity,
  gap,
  onPursue,
  onDismiss,
  onViewDetails,
  showGapInfo = true,
  compact = false,
  className = "",
}: GapOpportunityCardProps) {
  const [expanded, setExpanded] = useState(false);

  const impactPercentage = Math.round(opportunity.impactScore * 100);
  const priorityPercentage = Math.round(opportunity.priorityScore * 100);

  const getImpactColor = (score: number): string => {
    if (score >= 0.8) return "text-green-600";
    if (score >= 0.6) return "text-blue-600";
    if (score >= 0.4) return "text-yellow-600";
    return "text-gray-600";
  };

  const getEffortColor = (level: string): string => {
    switch (level) {
      case "trivial":
        return "bg-green-100 text-green-800";
      case "small":
        return "bg-blue-100 text-blue-800";
      case "medium":
        return "bg-yellow-100 text-yellow-800";
      case "large":
        return "bg-orange-100 text-orange-800";
      case "research":
        return "bg-red-100 text-red-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  if (compact) {
    return (
      <div
        className={`p-3 border rounded-lg hover:shadow-md transition-shadow cursor-pointer ${className}`}
        onClick={() => onViewDetails?.(opportunity.id)}
      >
        <div className="flex items-center justify-between">
          <div className="flex-1 min-w-0">
            <h4 className="font-medium text-sm truncate">{opportunity.title}</h4>
            <div className="flex items-center gap-2 mt-1">
              <span
                className={`text-xs px-2 py-0.5 rounded ${getEffortColor(opportunity.effort.level)}`}
              >
                {opportunity.effort.level}
              </span>
              <span className={`text-sm font-semibold ${getImpactColor(opportunity.impactScore)}`}>
                {impactPercentage}%
              </span>
            </div>
          </div>
          <div className="text-2xl font-bold text-blue-600">{priorityPercentage}</div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white border rounded-lg shadow-sm ${className}`}>
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs px-2 py-0.5 bg-blue-100 text-blue-800 rounded capitalize">
                {opportunity.type.replace("_", " ")}
              </span>
              <span
                className={`text-xs px-2 py-0.5 rounded ${getEffortColor(opportunity.effort.level)}`}
              >
                {getEffortLevelLabel(opportunity.effort.level)}
              </span>
            </div>
            <h3 className="font-semibold text-lg">{opportunity.title}</h3>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-blue-600">{priorityPercentage}</div>
            <div className="text-xs text-gray-500">Priority</div>
          </div>
        </div>
      </div>

      {/* Scores */}
      <div className="px-4 py-3 bg-gray-50 border-b">
        <div className="flex items-center gap-6">
          <div>
            <div className="text-xs text-gray-500 uppercase">Impact</div>
            <div className={`text-lg font-semibold ${getImpactColor(opportunity.impactScore)}`}>
              {impactPercentage}%
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500 uppercase">Effort</div>
            <div className="text-lg font-semibold text-gray-700">
              {opportunity.effort.personWeeks}w
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500 uppercase">Confidence</div>
            <div className="text-lg font-semibold text-gray-700">
              {Math.round(opportunity.confidence * 100)}%
            </div>
          </div>
        </div>
      </div>

      {/* Description */}
      <div className="p-4">
        <p className="text-sm text-gray-600">{opportunity.description}</p>
      </div>

      {/* Expandable Details */}
      <div className="border-t">
        <button
          className="w-full px-4 py-2 text-sm text-gray-600 hover:bg-gray-50 flex items-center justify-center gap-2"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? "Hide Details" : "Show Details"}
          <svg
            className={`w-4 h-4 transition-transform ${expanded ? "rotate-180" : ""}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {expanded && (
          <div className="px-4 pb-4 space-y-4">
            {/* Prerequisites */}
            {opportunity.prerequisites.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-2">Prerequisites</h4>
                <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                  {opportunity.prerequisites.map((prereq, idx) => (
                    <li key={idx}>{prereq}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Potential Outcomes */}
            {opportunity.potentialOutcomes.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-2">Potential Outcomes</h4>
                <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                  {opportunity.potentialOutcomes.map((outcome, idx) => (
                    <li key={idx}>{outcome}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Skills Required */}
            {opportunity.effort.skills.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-2">Skills Required</h4>
                <div className="flex flex-wrap gap-2">
                  {opportunity.effort.skills.map((skill, idx) => (
                    <span
                      key={idx}
                      className="text-xs px-2 py-1 bg-gray-100 text-gray-700 rounded"
                    >
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Suggested Techniques */}
            {opportunity.suggestedTechniques.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-2">Suggested Techniques</h4>
                <div className="flex flex-wrap gap-2">
                  {opportunity.suggestedTechniques.slice(0, 5).map((techId, idx) => (
                    <span
                      key={idx}
                      className="text-xs px-2 py-1 bg-blue-50 text-blue-700 rounded"
                    >
                      {techId.slice(0, 20)}...
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Gap Info */}
            {showGapInfo && gap && (
              <div className="p-3 bg-gray-50 rounded-lg">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Related Gap</h4>
                <div className="flex items-center gap-2">
                  <span
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: getGapSeverityColor(gap.severity) }}
                  />
                  <span className="text-sm text-gray-600">{gap.title}</span>
                </div>
                <p className="text-xs text-gray-500 mt-1">{getGapTypeLabel(gap.type)}</p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="px-4 py-3 border-t bg-gray-50 flex items-center gap-2">
        {onPursue && (
          <button
            onClick={() => onPursue(opportunity.id)}
            className="flex-1 px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded hover:bg-blue-700 transition-colors"
          >
            Pursue Opportunity
          </button>
        )}
        {onDismiss && (
          <button
            onClick={() => onDismiss(opportunity.id)}
            className="px-4 py-2 text-gray-600 text-sm font-medium hover:bg-gray-200 rounded transition-colors"
          >
            Dismiss
          </button>
        )}
        {onViewDetails && (
          <button
            onClick={() => onViewDetails(opportunity.id)}
            className="px-4 py-2 text-blue-600 text-sm font-medium hover:bg-blue-50 rounded transition-colors"
          >
            View Details
          </button>
        )}
      </div>
    </div>
  );
}

/**
 * Gap Card Component
 */
interface GapCardProps {
  gap: ResearchGap;
  opportunities?: GapOpportunity[];
  onViewOpportunities?: (gapId: string) => void;
  className?: string;
}

export function GapCard({
  gap,
  opportunities = [],
  onViewOpportunities,
  className = "",
}: GapCardProps) {
  return (
    <div className={`bg-white border rounded-lg shadow-sm p-4 ${className}`}>
      <div className="flex items-start gap-3">
        <div
          className="w-4 h-4 rounded-full mt-1 flex-shrink-0"
          style={{ backgroundColor: getGapSeverityColor(gap.severity) }}
        />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs px-2 py-0.5 bg-gray-100 text-gray-700 rounded capitalize">
              {getGapTypeLabel(gap.type)}
            </span>
            <span
              className="text-xs px-2 py-0.5 rounded capitalize"
              style={{
                backgroundColor: `${getGapSeverityColor(gap.severity)}20`,
                color: getGapSeverityColor(gap.severity),
              }}
            >
              {gap.severity}
            </span>
          </div>
          <h3 className="font-medium text-gray-900">{gap.title}</h3>
          <p className="text-sm text-gray-600 mt-1 line-clamp-2">{gap.description}</p>

          {gap.domains.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {gap.domains.map((domain, idx) => (
                <span
                  key={idx}
                  className="text-xs px-2 py-0.5 bg-blue-50 text-blue-700 rounded"
                >
                  {domain}
                </span>
              ))}
            </div>
          )}

          {opportunities.length > 0 && (
            <div className="mt-3 pt-3 border-t">
              <button
                onClick={() => onViewOpportunities?.(gap.id)}
                className="text-sm text-blue-600 hover:text-blue-800"
              >
                {opportunities.length} opportunit{opportunities.length === 1 ? "y" : "ies"} available
              </button>
            </div>
          )}
        </div>
        <div className="text-right">
          <div className="text-xl font-bold text-gray-700">
            {Math.round(gap.confidence * 100)}%
          </div>
          <div className="text-xs text-gray-500">Confidence</div>
        </div>
      </div>
    </div>
  );
}

export default GapOpportunityCard;
