/**
 * PlatformMetrics
 *
 * Platform-wide statistics display with charts and trends
 * showing labs, papers, techniques, and community metrics.
 */

"use client";

import React, { useState } from "react";
import {
  PlatformMetrics as MetricsType,
  MetricsSummary,
  MetricsPeriod,
} from "@/lib/meta/community";

interface PlatformMetricsProps {
  metrics: MetricsType;
  summary?: MetricsSummary;
  trends?: { timestamp: string; value: number }[];
  onPeriodChange?: (period: MetricsPeriod) => void;
}

export function PlatformMetrics({
  metrics,
  summary,
  trends,
  onPeriodChange,
}: PlatformMetricsProps) {
  const [selectedPeriod, setSelectedPeriod] = useState<MetricsPeriod>(metrics.period);

  const handlePeriodChange = (period: MetricsPeriod) => {
    setSelectedPeriod(period);
    onPeriodChange?.(period);
  };

  return (
    <div className="bg-white rounded-lg shadow-lg">
      {/* Header */}
      <div className="p-6 border-b">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold">Platform Metrics</h2>
            <p className="text-gray-500 text-sm">
              Last updated: {new Date(metrics.timestamp).toLocaleString()}
            </p>
          </div>
          <div className="flex gap-2">
            {(["daily", "weekly", "monthly", "all-time"] as MetricsPeriod[]).map(
              (period) => (
                <button
                  key={period}
                  onClick={() => handlePeriodChange(period)}
                  className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                    selectedPeriod === period
                      ? "bg-indigo-600 text-white"
                      : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                  }`}
                >
                  {period.charAt(0).toUpperCase() + period.slice(1).replace("-", " ")}
                </button>
              )
            )}
          </div>
        </div>

        {/* Summary Alerts */}
        {summary && (
          <div className="grid grid-cols-3 gap-4">
            {summary.highlights.length > 0 && (
              <div className="p-3 bg-green-50 rounded-lg">
                <div className="text-sm font-medium text-green-700 mb-1">Highlights</div>
                <ul className="text-sm text-green-600 space-y-1">
                  {summary.highlights.slice(0, 2).map((h, i) => (
                    <li key={i}>* {h}</li>
                  ))}
                </ul>
              </div>
            )}
            {summary.concerns.length > 0 && (
              <div className="p-3 bg-yellow-50 rounded-lg">
                <div className="text-sm font-medium text-yellow-700 mb-1">Concerns</div>
                <ul className="text-sm text-yellow-600 space-y-1">
                  {summary.concerns.slice(0, 2).map((c, i) => (
                    <li key={i}>* {c}</li>
                  ))}
                </ul>
              </div>
            )}
            {summary.recommendations.length > 0 && (
              <div className="p-3 bg-blue-50 rounded-lg">
                <div className="text-sm font-medium text-blue-700 mb-1">Recommendations</div>
                <ul className="text-sm text-blue-600 space-y-1">
                  {summary.recommendations.slice(0, 2).map((r, i) => (
                    <li key={i}>* {r}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Main Stats Grid */}
      <div className="p-6 grid grid-cols-4 gap-6">
        {/* Labs */}
        <MetricCard
          title="Labs"
          icon="users"
          mainValue={metrics.labs.total}
          subValues={[
            { label: "Active", value: metrics.labs.active },
            { label: "New", value: metrics.labs.new, highlight: true },
            { label: "Avg Techniques", value: metrics.labs.avgTechniquesPerLab.toFixed(1) },
          ]}
          color="indigo"
        />

        {/* Papers */}
        <MetricCard
          title="Papers"
          icon="document"
          mainValue={metrics.papers.total}
          subValues={[
            { label: "Processed", value: metrics.papers.processed },
            { label: "Pending", value: metrics.papers.pending },
            { label: "Failed", value: metrics.papers.failed, danger: metrics.papers.failed > 10 },
          ]}
          color="blue"
        />

        {/* Techniques */}
        <MetricCard
          title="Techniques"
          icon="code"
          mainValue={metrics.techniques.total}
          subValues={[
            { label: "Active", value: metrics.techniques.active },
            { label: "Experimental", value: metrics.techniques.experimental },
            { label: "Gen " + metrics.techniques.evolutionGenerations, value: "max" },
          ]}
          color="purple"
        />

        {/* Collaborations */}
        <MetricCard
          title="Collaborations"
          icon="link"
          mainValue={metrics.collaborations.total}
          subValues={[
            { label: "Active", value: metrics.collaborations.active },
            { label: "Completed", value: metrics.collaborations.completed },
            { label: "Success", value: `${metrics.collaborations.successRate.toFixed(0)}%` },
          ]}
          color="green"
        />
      </div>

      {/* Detailed Sections */}
      <div className="border-t">
        <div className="grid grid-cols-2 divide-x">
          {/* Growth Metrics */}
          <div className="p-6">
            <h3 className="font-semibold mb-4">Growth</h3>
            <div className="space-y-3">
              <GrowthRow
                label="Labs"
                value={metrics.growth.labsGrowth}
              />
              <GrowthRow
                label="Papers"
                value={metrics.growth.papersGrowth}
              />
              <GrowthRow
                label="Techniques"
                value={metrics.growth.techniquesGrowth}
              />
              <GrowthRow
                label="Collaborations"
                value={metrics.growth.collaborationsGrowth}
              />
            </div>
          </div>

          {/* Success Rates */}
          <div className="p-6">
            <h3 className="font-semibold mb-4">Success Rates</h3>
            <div className="space-y-3">
              <ProgressRow
                label="Task Completion"
                value={metrics.successRates.taskCompletion}
              />
              <ProgressRow
                label="Collaboration Success"
                value={metrics.successRates.collaborationSuccess}
              />
              <ProgressRow
                label="Gap Resolution"
                value={metrics.successRates.gapResolution}
              />
              <ProgressRow
                label="Evolution Improvement"
                value={metrics.successRates.evolutionImprovement}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Adoption Rates */}
      <div className="border-t p-6">
        <h3 className="font-semibold mb-4">Adoption This Period</h3>
        <div className="grid grid-cols-4 gap-4">
          <AdoptionCard
            label="New Techniques"
            value={metrics.adoptionRates.newTechniques}
            icon="plus"
          />
          <AdoptionCard
            label="Synergy Discoveries"
            value={metrics.adoptionRates.synergyDiscoveries}
            icon="link"
          />
          <AdoptionCard
            label="Evolution Experiments"
            value={metrics.adoptionRates.evolutionExperiments}
            icon="dna"
          />
          <AdoptionCard
            label="Collaboration Joins"
            value={metrics.adoptionRates.collaborationJoins}
            icon="users"
          />
        </div>
      </div>

      {/* Top Categories */}
      <div className="border-t">
        <div className="grid grid-cols-2 divide-x">
          {/* Top Domains */}
          <div className="p-6">
            <h3 className="font-semibold mb-4">Top Domains</h3>
            <div className="space-y-2">
              {metrics.labs.topDomains.slice(0, 5).map((domain) => (
                <div key={domain.domain} className="flex items-center justify-between">
                  <span className="text-gray-700">{domain.domain}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-gray-400 text-sm">{domain.count}</span>
                    <div className="w-24 h-2 bg-gray-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-indigo-500 rounded-full"
                        style={{ width: `${domain.percentage}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Top Categories */}
          <div className="p-6">
            <h3 className="font-semibold mb-4">Top Categories</h3>
            <div className="space-y-2">
              {metrics.techniques.topCategories.slice(0, 5).map((cat) => (
                <div key={cat.category} className="flex items-center justify-between">
                  <span className="text-gray-700">{cat.category}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-gray-400 text-sm">{cat.count}</span>
                    <div className="w-24 h-2 bg-gray-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-purple-500 rounded-full"
                        style={{ width: `${cat.percentage}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Cost Metrics */}
      <div className="border-t p-6 bg-gray-50">
        <h3 className="font-semibold mb-4">Cost Metrics</h3>
        <div className="grid grid-cols-5 gap-4">
          <CostCard
            label="API Calls"
            value={metrics.costs.totalApiCalls.toLocaleString()}
          />
          <CostCard
            label="API Cost"
            value={`$${metrics.costs.apiCostEstimate.toFixed(2)}`}
          />
          <CostCard
            label="Compute Hours"
            value={metrics.costs.computeHours.toFixed(1)}
          />
          <CostCard
            label="Cost/Paper"
            value={`$${metrics.costs.avgCostPerPaper.toFixed(3)}`}
          />
          <CostCard
            label="Cost/Discovery"
            value={`$${metrics.costs.avgCostPerDiscovery.toFixed(3)}`}
          />
        </div>
      </div>
    </div>
  );
}

interface MetricCardProps {
  title: string;
  icon: string;
  mainValue: number;
  subValues: { label: string; value: string | number; highlight?: boolean; danger?: boolean }[];
  color: string;
}

function MetricCard({ title, icon, mainValue, subValues, color }: MetricCardProps) {
  const colorClasses: Record<string, string> = {
    indigo: "bg-indigo-50 text-indigo-600",
    blue: "bg-blue-50 text-blue-600",
    purple: "bg-purple-50 text-purple-600",
    green: "bg-green-50 text-green-600",
  };

  return (
    <div className="border rounded-lg p-4">
      <div className="flex items-center gap-2 mb-3">
        <div className={`p-2 rounded-lg ${colorClasses[color]}`}>
          <MetricIcon name={icon} />
        </div>
        <span className="font-medium">{title}</span>
      </div>
      <div className="text-3xl font-bold mb-3">{mainValue.toLocaleString()}</div>
      <div className="space-y-1">
        {subValues.map((sv) => (
          <div key={sv.label} className="flex justify-between text-sm">
            <span className="text-gray-500">{sv.label}</span>
            <span
              className={
                sv.highlight
                  ? "text-green-600 font-medium"
                  : sv.danger
                  ? "text-red-600"
                  : ""
              }
            >
              {sv.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function MetricIcon({ name }: { name: string }) {
  const icons: Record<string, React.ReactNode> = {
    users: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
      </svg>
    ),
    document: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
    code: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
      </svg>
    ),
    link: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
      </svg>
    ),
  };

  return <>{icons[name] || null}</>;
}

function GrowthRow({ label, value }: { label: string; value: number }) {
  const isPositive = value >= 0;

  return (
    <div className="flex items-center justify-between">
      <span className="text-gray-600">{label}</span>
      <span
        className={`font-semibold ${
          isPositive ? "text-green-600" : "text-red-600"
        }`}
      >
        {isPositive ? "+" : ""}
        {value.toFixed(1)}%
      </span>
    </div>
  );
}

function ProgressRow({ label, value }: { label: string; value: number }) {
  const color =
    value >= 80 ? "bg-green-500" : value >= 50 ? "bg-yellow-500" : "bg-red-500";

  return (
    <div>
      <div className="flex items-center justify-between text-sm mb-1">
        <span className="text-gray-600">{label}</span>
        <span className="font-medium">{value.toFixed(0)}%</span>
      </div>
      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={`h-full ${color} rounded-full transition-all`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}

function AdoptionCard({
  label,
  value,
  icon,
}: {
  label: string;
  value: number;
  icon: string;
}) {
  return (
    <div className="text-center p-4 bg-gray-50 rounded-lg">
      <div className="text-3xl font-bold text-indigo-600">{value}</div>
      <div className="text-sm text-gray-500 mt-1">{label}</div>
    </div>
  );
}

function CostCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center">
      <div className="text-lg font-semibold">{value}</div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  );
}

export default PlatformMetrics;
