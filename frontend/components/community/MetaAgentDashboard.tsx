/**
 * MetaAgentDashboard
 *
 * Live monitoring dashboard for meta-agents with status,
 * activity logs, metrics, and admin controls.
 */

"use client";

import React, { useState } from "react";
import {
  MetaAgentDashboard as DashboardType,
  MetaAgentStatus,
  MetaAgentName,
  AgentStatus,
  HealthStatus,
  Discovery,
  ActivityLogEntry,
  SignificanceLevel,
} from "@/lib/meta/community/types";

interface MetaAgentDashboardProps {
  dashboard: DashboardType;
  onPauseAgent?: (agentName: MetaAgentName) => void;
  onResumeAgent?: (agentName: MetaAgentName) => void;
  onRefresh?: () => void;
}

const statusColors: Record<AgentStatus, string> = {
  running: "bg-green-500",
  idle: "bg-blue-500",
  paused: "bg-yellow-500",
  error: "bg-red-500",
  initializing: "bg-purple-500",
};

const healthColors: Record<HealthStatus, string> = {
  healthy: "text-green-600",
  degraded: "text-yellow-600",
  unhealthy: "text-red-600",
  unknown: "text-gray-400",
};

const significanceColors: Record<SignificanceLevel, string> = {
  critical: "bg-red-100 text-red-700",
  high: "bg-orange-100 text-orange-700",
  medium: "bg-blue-100 text-blue-700",
  low: "bg-gray-100 text-gray-600",
};

export function MetaAgentDashboard({
  dashboard,
  onPauseAgent,
  onResumeAgent,
  onRefresh,
}: MetaAgentDashboardProps) {
  const [selectedAgent, setSelectedAgent] = useState<MetaAgentName | null>(null);
  const [activeTab, setActiveTab] = useState<"overview" | "activity" | "discoveries">("overview");

  const selectedAgentData = selectedAgent
    ? dashboard.agents.find((a) => a.name === selectedAgent)
    : null;

  return (
    <div className="bg-white rounded-lg shadow-lg">
      {/* Header */}
      <div className="p-6 border-b">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold">Meta-Agent Dashboard</h2>
            <p className="text-gray-500 text-sm">
              Last updated: {new Date(dashboard.updatedAt).toLocaleTimeString()}
            </p>
          </div>
          <div className="flex items-center gap-4">
            <SystemHealthBadge health={dashboard.systemHealth} />
            {onRefresh && (
              <button
                onClick={onRefresh}
                className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
              >
                Refresh
              </button>
            )}
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-4 gap-4">
          <SummaryCard
            label="Total Discoveries"
            value={dashboard.summary.totalDiscoveries}
            subValue={`+${dashboard.summary.discoveriesToday} today`}
            color="indigo"
          />
          <SummaryCard
            label="Active Agents"
            value={`${dashboard.summary.activeAgents}/${dashboard.agents.length}`}
            subValue={
              dashboard.summary.pausedAgents > 0
                ? `${dashboard.summary.pausedAgents} paused`
                : "All running"
            }
            color="green"
          />
          <SummaryCard
            label="Pending Tasks"
            value={dashboard.summary.pendingTasks}
            subValue="in queue"
            color="blue"
          />
          <SummaryCard
            label="Alerts Today"
            value={dashboard.summary.alertsToday}
            subValue="generated"
            color="orange"
          />
        </div>
      </div>

      {/* Main Content */}
      <div className="flex">
        {/* Agent List */}
        <div className="w-80 border-r">
          <div className="p-4 border-b">
            <h3 className="font-semibold">Agents</h3>
          </div>
          <div className="divide-y">
            {dashboard.agents.map((agent) => (
              <AgentListItem
                key={agent.name}
                agent={agent}
                isSelected={selectedAgent === agent.name}
                onClick={() => setSelectedAgent(agent.name)}
                onPause={onPauseAgent}
                onResume={onResumeAgent}
              />
            ))}
          </div>
        </div>

        {/* Agent Details */}
        <div className="flex-1">
          {selectedAgentData ? (
            <div>
              {/* Tabs */}
              <div className="border-b px-6">
                <div className="flex gap-4">
                  {(["overview", "activity", "discoveries"] as const).map((tab) => (
                    <button
                      key={tab}
                      onClick={() => setActiveTab(tab)}
                      className={`py-4 px-2 border-b-2 transition-colors capitalize ${
                        activeTab === tab
                          ? "border-indigo-600 text-indigo-600"
                          : "border-transparent text-gray-500 hover:text-gray-700"
                      }`}
                    >
                      {tab}
                    </button>
                  ))}
                </div>
              </div>

              {/* Tab Content */}
              <div className="p-6">
                {activeTab === "overview" && (
                  <AgentOverview agent={selectedAgentData} />
                )}
                {activeTab === "activity" && (
                  <AgentActivityLog log={selectedAgentData.activityLog} />
                )}
                {activeTab === "discoveries" && (
                  <AgentDiscoveries discoveries={selectedAgentData.recentDiscoveries} />
                )}
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-96 text-gray-500">
              Select an agent to view details
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function SystemHealthBadge({ health }: { health: DashboardType["systemHealth"] }) {
  return (
    <div className="flex items-center gap-4 px-4 py-2 bg-gray-50 rounded-lg">
      <div className="flex items-center gap-2">
        <span
          className={`w-2 h-2 rounded-full ${
            health.overall === "healthy"
              ? "bg-green-500"
              : health.overall === "degraded"
              ? "bg-yellow-500"
              : "bg-red-500"
          }`}
        />
        <span className="text-sm font-medium capitalize">{health.overall}</span>
      </div>
      <div className="text-xs text-gray-500">
        CPU: {health.cpu.toFixed(0)}% | Mem: {health.memory.toFixed(0)}%
      </div>
    </div>
  );
}

function SummaryCard({
  label,
  value,
  subValue,
  color,
}: {
  label: string;
  value: string | number;
  subValue: string;
  color: string;
}) {
  const colorClasses: Record<string, string> = {
    indigo: "bg-indigo-50 text-indigo-600",
    green: "bg-green-50 text-green-600",
    blue: "bg-blue-50 text-blue-600",
    orange: "bg-orange-50 text-orange-600",
  };

  return (
    <div className={`p-4 rounded-lg ${colorClasses[color]}`}>
      <div className="text-2xl font-bold">{value}</div>
      <div className="text-sm font-medium">{label}</div>
      <div className="text-xs opacity-75">{subValue}</div>
    </div>
  );
}

interface AgentListItemProps {
  agent: MetaAgentStatus;
  isSelected: boolean;
  onClick: () => void;
  onPause?: (agentName: MetaAgentName) => void;
  onResume?: (agentName: MetaAgentName) => void;
}

function AgentListItem({
  agent,
  isSelected,
  onClick,
  onPause,
  onResume,
}: AgentListItemProps) {
  const isRunning = agent.status === "running";
  const canControl = agent.status !== "error" && agent.status !== "initializing";

  return (
    <div
      className={`p-4 cursor-pointer transition-colors ${
        isSelected ? "bg-indigo-50" : "hover:bg-gray-50"
      }`}
      onClick={onClick}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${statusColors[agent.status]}`} />
          <span className="font-medium">{agent.displayName}</span>
        </div>
        <span className={`text-xs ${healthColors[agent.health]}`}>
          {agent.health}
        </span>
      </div>

      <div className="text-sm text-gray-500 mb-2">
        {agent.currentTask || (agent.status === "idle" ? "Waiting for tasks" : agent.status)}
      </div>

      <div className="flex items-center justify-between text-xs">
        <span className="text-gray-400">
          {agent.metrics.discoveriesCount} discoveries
        </span>
        {canControl && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              if (agent.isEnabled && onPause) {
                onPause(agent.name);
              } else if (!agent.isEnabled && onResume) {
                onResume(agent.name);
              }
            }}
            className={`px-2 py-0.5 rounded ${
              agent.isEnabled
                ? "bg-yellow-100 text-yellow-700 hover:bg-yellow-200"
                : "bg-green-100 text-green-700 hover:bg-green-200"
            }`}
          >
            {agent.isEnabled ? "Pause" : "Resume"}
          </button>
        )}
      </div>
    </div>
  );
}

function AgentOverview({ agent }: { agent: MetaAgentStatus }) {
  return (
    <div className="space-y-6">
      {/* Status */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="text-sm text-gray-500 mb-1">Status</div>
          <div className="flex items-center gap-2">
            <span className={`w-3 h-3 rounded-full ${statusColors[agent.status]}`} />
            <span className="font-semibold capitalize">{agent.status}</span>
          </div>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="text-sm text-gray-500 mb-1">Health</div>
          <div className={`font-semibold capitalize ${healthColors[agent.health]}`}>
            {agent.health}
          </div>
        </div>
      </div>

      {/* Current Task */}
      {agent.currentTask && (
        <div className="p-4 border rounded-lg">
          <div className="text-sm text-gray-500 mb-1">Current Task</div>
          <div className="font-medium">{agent.currentTask}</div>
        </div>
      )}

      {/* Metrics */}
      <div>
        <h4 className="font-semibold mb-3">Performance Metrics</h4>
        <div className="grid grid-cols-3 gap-4">
          <MetricBox
            label="Total Runs"
            value={agent.metrics.totalRuns}
          />
          <MetricBox
            label="Success Rate"
            value={`${((agent.metrics.successfulRuns / Math.max(agent.metrics.totalRuns, 1)) * 100).toFixed(1)}%`}
          />
          <MetricBox
            label="Avg Time"
            value={`${(agent.metrics.avgExecutionTimeMs / 1000).toFixed(1)}s`}
          />
          <MetricBox
            label="Discoveries"
            value={agent.metrics.discoveriesCount}
          />
          <MetricBox
            label="Last Hour"
            value={agent.metrics.lastHourActivity}
          />
          <MetricBox
            label="Error Rate"
            value={`${agent.metrics.errorRate.toFixed(1)}%`}
            danger={agent.metrics.errorRate > 10}
          />
        </div>
      </div>

      {/* Queue */}
      {agent.queuedTasks > 0 && (
        <div className="p-4 bg-blue-50 rounded-lg">
          <div className="flex items-center justify-between">
            <span className="font-medium">Queued Tasks</span>
            <span className="text-2xl font-bold text-blue-600">
              {agent.queuedTasks}
            </span>
          </div>
        </div>
      )}

      {/* Pause Reason */}
      {agent.pauseReason && (
        <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="text-sm text-yellow-700 font-medium mb-1">Pause Reason</div>
          <div className="text-yellow-800">{agent.pauseReason}</div>
          {agent.pausedAt && (
            <div className="text-xs text-yellow-600 mt-1">
              Paused at {new Date(agent.pausedAt).toLocaleString()}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function MetricBox({
  label,
  value,
  danger = false,
}: {
  label: string;
  value: string | number;
  danger?: boolean;
}) {
  return (
    <div className="p-3 bg-gray-50 rounded-lg">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-xl font-bold ${danger ? "text-red-600" : ""}`}>
        {value}
      </div>
    </div>
  );
}

function AgentActivityLog({ log }: { log: ActivityLogEntry[] }) {
  if (log.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        No activity recorded yet
      </div>
    );
  }

  return (
    <div className="space-y-2 max-h-96 overflow-y-auto">
      {log.map((entry) => (
        <div
          key={entry.id}
          className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg"
        >
          <span
            className={`w-2 h-2 rounded-full mt-1.5 ${
              entry.result === "success"
                ? "bg-green-500"
                : entry.result === "failure"
                ? "bg-red-500"
                : "bg-yellow-500"
            }`}
          />
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between">
              <span className="font-medium">{entry.action}</span>
              <span className="text-xs text-gray-400">
                {new Date(entry.timestamp).toLocaleTimeString()}
              </span>
            </div>
            <p className="text-sm text-gray-600 truncate">{entry.details}</p>
            {entry.duration && (
              <span className="text-xs text-gray-400">
                Duration: {entry.duration}ms
              </span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

function AgentDiscoveries({ discoveries }: { discoveries: Discovery[] }) {
  if (discoveries.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        No discoveries yet
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {discoveries.map((discovery) => (
        <div key={discovery.id} className="p-4 border rounded-lg">
          <div className="flex items-start justify-between mb-2">
            <span className="font-medium capitalize">{discovery.type}</span>
            <span
              className={`px-2 py-0.5 text-xs rounded ${
                significanceColors[discovery.significance]
              }`}
            >
              {discovery.significance}
            </span>
          </div>
          <p className="text-gray-600">{discovery.description}</p>
          <div className="text-xs text-gray-400 mt-2">
            {new Date(discovery.timestamp).toLocaleString()}
          </div>
        </div>
      ))}
    </div>
  );
}

export default MetaAgentDashboard;
