"use client";

/**
 * TransferProposal Component
 *
 * Displays cross-domain transfer proposals with feasibility scores,
 * implementation steps, and success predictions.
 */

import React, { useState } from "react";

interface TransferProposalData {
  id: string;
  sourceTechnique: {
    id: string;
    name: string;
  };
  sourceDomain: string;
  targetDomain: string;
  principle: {
    name: string;
    level: number;
    coreInsight: string;
    confidence: number;
  };
  feasibility: {
    overallScore: number;
    level: "trivial" | "straightforward" | "moderate" | "challenging" | "infeasible";
    components: {
      technical: number;
      data: number;
      computational: number;
      knowledge: number;
      resources: number;
    };
    risks: Array<{
      name: string;
      score: number;
      mitigation: string;
    }>;
    effort: {
      personDays: number;
      range: { min: number; max: number };
    };
    recommendations: string[];
  };
  implementationGuide?: {
    steps: Array<{
      step: number;
      title: string;
      description: string;
      estimatedHours: number;
    }>;
    successCriteria: string[];
  };
  successPrediction: {
    probability: number;
    confidenceInterval: { low: number; high: number };
    successFactors: string[];
    failureRisks: string[];
  };
  status: string;
  createdAt: Date;
}

interface TransferProposalProps {
  proposal: TransferProposalData;
  onApprove?: (proposalId: string) => void;
  onReject?: (proposalId: string) => void;
  onStartImplementation?: (proposalId: string) => void;
  className?: string;
}

export function TransferProposal({
  proposal,
  onApprove,
  onReject,
  onStartImplementation,
  className = "",
}: TransferProposalProps) {
  const [activeTab, setActiveTab] = useState<"overview" | "feasibility" | "implementation" | "prediction">("overview");

  return (
    <div className={`bg-white border rounded-lg shadow-sm ${className}`}>
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-start justify-between">
          <div>
            <h3 className="font-semibold text-lg">
              Transfer: {proposal.sourceTechnique.name}
            </h3>
            <p className="text-sm text-gray-500">
              {proposal.sourceDomain} → {proposal.targetDomain}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <FeasibilityBadge level={proposal.feasibility.level} />
            <StatusBadge status={proposal.status} />
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b flex">
        <TabButton
          label="Overview"
          active={activeTab === "overview"}
          onClick={() => setActiveTab("overview")}
        />
        <TabButton
          label="Feasibility"
          active={activeTab === "feasibility"}
          onClick={() => setActiveTab("feasibility")}
        />
        <TabButton
          label="Implementation"
          active={activeTab === "implementation"}
          onClick={() => setActiveTab("implementation")}
        />
        <TabButton
          label="Prediction"
          active={activeTab === "prediction"}
          onClick={() => setActiveTab("prediction")}
        />
      </div>

      {/* Content */}
      <div className="p-4">
        {activeTab === "overview" && (
          <OverviewTab proposal={proposal} />
        )}
        {activeTab === "feasibility" && (
          <FeasibilityTab feasibility={proposal.feasibility} />
        )}
        {activeTab === "implementation" && (
          <ImplementationTab guide={proposal.implementationGuide} />
        )}
        {activeTab === "prediction" && (
          <PredictionTab prediction={proposal.successPrediction} />
        )}
      </div>

      {/* Actions */}
      {proposal.status === "proposed" && (
        <div className="p-4 border-t bg-gray-50 flex justify-end gap-2">
          {onReject && (
            <button
              onClick={() => onReject(proposal.id)}
              className="px-4 py-2 text-sm border rounded hover:bg-gray-100"
            >
              Reject
            </button>
          )}
          {onApprove && (
            <button
              onClick={() => onApprove(proposal.id)}
              className="px-4 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Approve
            </button>
          )}
        </div>
      )}
      {proposal.status === "validated" && onStartImplementation && (
        <div className="p-4 border-t bg-gray-50 flex justify-end">
          <button
            onClick={() => onStartImplementation(proposal.id)}
            className="px-4 py-2 text-sm bg-green-600 text-white rounded hover:bg-green-700"
          >
            Start Implementation
          </button>
        </div>
      )}
    </div>
  );
}

// Sub-components

function TabButton({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 text-sm font-medium border-b-2 ${
        active
          ? "border-blue-600 text-blue-600"
          : "border-transparent text-gray-600 hover:text-gray-900"
      }`}
    >
      {label}
    </button>
  );
}

function FeasibilityBadge({
  level,
}: {
  level: "trivial" | "straightforward" | "moderate" | "challenging" | "infeasible";
}) {
  const colors: Record<string, string> = {
    trivial: "bg-green-100 text-green-700",
    straightforward: "bg-lime-100 text-lime-700",
    moderate: "bg-yellow-100 text-yellow-700",
    challenging: "bg-orange-100 text-orange-700",
    infeasible: "bg-red-100 text-red-700",
  };

  return (
    <span className={`px-2 py-1 text-xs rounded-full ${colors[level]}`}>
      {level.charAt(0).toUpperCase() + level.slice(1)}
    </span>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    proposed: "bg-gray-100 text-gray-700",
    analyzing: "bg-blue-100 text-blue-700",
    validated: "bg-purple-100 text-purple-700",
    in_progress: "bg-yellow-100 text-yellow-700",
    completed: "bg-green-100 text-green-700",
    failed: "bg-red-100 text-red-700",
    abandoned: "bg-gray-100 text-gray-500",
  };

  return (
    <span className={`px-2 py-1 text-xs rounded-full ${colors[status] || colors.proposed}`}>
      {status.replace("_", " ")}
    </span>
  );
}

function OverviewTab({ proposal }: { proposal: TransferProposalData }) {
  return (
    <div className="space-y-4">
      {/* Principle */}
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-2">Core Principle</h4>
        <div className="p-3 bg-blue-50 rounded-lg">
          <div className="flex items-center justify-between mb-1">
            <span className="font-medium">{proposal.principle.name}</span>
            <span className="text-xs text-gray-500">
              Level {proposal.principle.level}
            </span>
          </div>
          <p className="text-sm text-gray-700">{proposal.principle.coreInsight}</p>
          <div className="mt-2 text-xs text-gray-500">
            Confidence: {(proposal.principle.confidence * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-3 gap-4">
        <QuickStat
          label="Feasibility"
          value={`${(proposal.feasibility.overallScore * 100).toFixed(0)}%`}
          color={proposal.feasibility.overallScore >= 0.6 ? "green" : proposal.feasibility.overallScore >= 0.4 ? "yellow" : "red"}
        />
        <QuickStat
          label="Success Probability"
          value={`${(proposal.successPrediction.probability * 100).toFixed(0)}%`}
          color={proposal.successPrediction.probability >= 0.6 ? "green" : proposal.successPrediction.probability >= 0.4 ? "yellow" : "red"}
        />
        <QuickStat
          label="Effort"
          value={`${proposal.feasibility.effort.personDays} days`}
          color="blue"
        />
      </div>

      {/* Recommendations */}
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-2">Recommendations</h4>
        <ul className="space-y-1">
          {proposal.feasibility.recommendations.slice(0, 3).map((rec, idx) => (
            <li key={idx} className="text-sm text-gray-600 flex items-start gap-2">
              <span className="text-blue-500 mt-1">*</span>
              {rec}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function QuickStat({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: "green" | "yellow" | "red" | "blue";
}) {
  const colorClasses: Record<string, string> = {
    green: "text-green-600",
    yellow: "text-yellow-600",
    red: "text-red-600",
    blue: "text-blue-600",
  };

  return (
    <div className="text-center p-3 bg-gray-50 rounded-lg">
      <div className="text-xs text-gray-500">{label}</div>
      <div className={`text-lg font-semibold ${colorClasses[color]}`}>{value}</div>
    </div>
  );
}

function FeasibilityTab({
  feasibility,
}: {
  feasibility: TransferProposalData["feasibility"];
}) {
  return (
    <div className="space-y-4">
      {/* Component Scores */}
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-2">Component Scores</h4>
        <div className="space-y-2">
          {Object.entries(feasibility.components).map(([key, value]) => (
            <div key={key} className="flex items-center gap-2">
              <span className="text-sm text-gray-600 w-28 capitalize">{key}</span>
              <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${
                    value >= 0.7
                      ? "bg-green-500"
                      : value >= 0.4
                      ? "bg-yellow-500"
                      : "bg-red-500"
                  }`}
                  style={{ width: `${value * 100}%` }}
                />
              </div>
              <span className="text-sm text-gray-600 w-12 text-right">
                {(value * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Risks */}
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-2">
          Risks ({feasibility.risks.length})
        </h4>
        <div className="space-y-2">
          {feasibility.risks.slice(0, 5).map((risk, idx) => (
            <div key={idx} className="p-2 bg-red-50 rounded text-sm">
              <div className="flex items-center justify-between">
                <span className="font-medium text-red-700">{risk.name}</span>
                <span className="text-xs text-red-500">
                  Score: {risk.score.toFixed(1)}
                </span>
              </div>
              <p className="text-red-600 text-xs mt-1">{risk.mitigation}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Effort Estimate */}
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-2">Effort Estimate</h4>
        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="text-2xl font-semibold text-center">
            {feasibility.effort.personDays} person-days
          </div>
          <div className="text-xs text-gray-500 text-center mt-1">
            Range: {feasibility.effort.range.min} - {feasibility.effort.range.max} days
          </div>
        </div>
      </div>
    </div>
  );
}

function ImplementationTab({
  guide,
}: {
  guide?: TransferProposalData["implementationGuide"];
}) {
  if (!guide) {
    return (
      <div className="text-center text-gray-500 py-8">
        Implementation guide not available
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Steps */}
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-2">Implementation Steps</h4>
        <div className="space-y-3">
          {guide.steps.map((step) => (
            <div key={step.step} className="flex gap-3">
              <div className="w-6 h-6 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center text-sm font-medium flex-shrink-0">
                {step.step}
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className="font-medium text-sm">{step.title}</span>
                  <span className="text-xs text-gray-500">
                    ~{step.estimatedHours}h
                  </span>
                </div>
                <p className="text-sm text-gray-600 mt-0.5">{step.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Success Criteria */}
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-2">Success Criteria</h4>
        <ul className="space-y-1">
          {guide.successCriteria.map((criterion, idx) => (
            <li key={idx} className="text-sm text-gray-600 flex items-start gap-2">
              <span className="text-green-500 mt-0.5">&#10003;</span>
              {criterion}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function PredictionTab({
  prediction,
}: {
  prediction: TransferProposalData["successPrediction"];
}) {
  const probabilityColor =
    prediction.probability >= 0.7
      ? "text-green-600"
      : prediction.probability >= 0.5
      ? "text-yellow-600"
      : "text-red-600";

  return (
    <div className="space-y-4">
      {/* Probability */}
      <div className="text-center p-4 bg-gray-50 rounded-lg">
        <div className="text-xs text-gray-500 mb-1">Success Probability</div>
        <div className={`text-4xl font-bold ${probabilityColor}`}>
          {(prediction.probability * 100).toFixed(0)}%
        </div>
        <div className="text-xs text-gray-500 mt-1">
          Confidence: {(prediction.confidenceInterval.low * 100).toFixed(0)}% -{" "}
          {(prediction.confidenceInterval.high * 100).toFixed(0)}%
        </div>
      </div>

      {/* Success Factors */}
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-2">Success Factors</h4>
        <div className="space-y-1">
          {prediction.successFactors.map((factor, idx) => (
            <div
              key={idx}
              className="px-3 py-1.5 bg-green-50 text-green-700 text-sm rounded"
            >
              {factor}
            </div>
          ))}
        </div>
      </div>

      {/* Failure Risks */}
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-2">Failure Risks</h4>
        <div className="space-y-1">
          {prediction.failureRisks.map((risk, idx) => (
            <div
              key={idx}
              className="px-3 py-1.5 bg-red-50 text-red-700 text-sm rounded"
            >
              {risk}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/**
 * TransferProposalList Component
 *
 * List view of multiple transfer proposals.
 */

interface TransferProposalListProps {
  proposals: TransferProposalData[];
  onSelect?: (proposal: TransferProposalData) => void;
  selectedId?: string;
  className?: string;
}

export function TransferProposalList({
  proposals,
  onSelect,
  selectedId,
  className = "",
}: TransferProposalListProps) {
  return (
    <div className={`space-y-2 ${className}`}>
      {proposals.length === 0 ? (
        <div className="text-center text-gray-500 py-8">No proposals</div>
      ) : (
        proposals.map((proposal) => (
          <div
            key={proposal.id}
            onClick={() => onSelect?.(proposal)}
            className={`p-4 border rounded-lg cursor-pointer transition-colors ${
              selectedId === proposal.id
                ? "border-blue-500 bg-blue-50"
                : "hover:bg-gray-50"
            }`}
          >
            <div className="flex items-center justify-between">
              <div>
                <span className="font-medium">{proposal.sourceTechnique.name}</span>
                <span className="text-gray-400 mx-2">→</span>
                <span className="text-gray-600">{proposal.targetDomain}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-mono">
                  {(proposal.feasibility.overallScore * 100).toFixed(0)}%
                </span>
                <FeasibilityBadge level={proposal.feasibility.level} />
              </div>
            </div>
            <p className="text-sm text-gray-500 mt-1 truncate">
              {proposal.principle.coreInsight}
            </p>
          </div>
        ))
      )}
    </div>
  );
}

export default TransferProposal;
