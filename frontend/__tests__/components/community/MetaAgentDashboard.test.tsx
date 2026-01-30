/**
 * Tests for MetaAgentDashboard Component
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MetaAgentDashboard } from "@/components/community/MetaAgentDashboard";
import type { MetaAgentDashboard as DashboardType } from "@/lib/meta/community/types";

describe("MetaAgentDashboard", () => {
  const mockDashboard: DashboardType = {
    agents: [
      {
        id: "agent-synergy-detector",
        name: "synergy-detector",
        displayName: "Synergy Detector",
        status: "running",
        health: "healthy",
        lastActivity: new Date().toISOString(),
        currentTask: "Scanning for synergies...",
        queuedTasks: 3,
        metrics: {
          totalRuns: 150,
          successfulRuns: 145,
          failedRuns: 5,
          avgExecutionTimeMs: 2500,
          discoveriesCount: 45,
          lastHourActivity: 12,
          last24HourActivity: 85,
          uptime: 99.5,
          errorRate: 3.3,
        },
        recentDiscoveries: [
          {
            id: "disc-1",
            type: "synergy",
            description: "Found synergy between attention and codec",
            significance: "high",
            timestamp: new Date().toISOString(),
            metadata: {},
          },
        ],
        activityLog: [
          {
            id: "log-1",
            timestamp: new Date().toISOString(),
            action: "scan",
            details: "Completed routine scan",
            result: "success",
            duration: 1500,
          },
        ],
        isEnabled: true,
      },
      {
        id: "agent-pattern-recognizer",
        name: "pattern-recognizer",
        displayName: "Pattern Recognizer",
        status: "idle",
        health: "healthy",
        lastActivity: new Date().toISOString(),
        queuedTasks: 0,
        metrics: {
          totalRuns: 100,
          successfulRuns: 98,
          failedRuns: 2,
          avgExecutionTimeMs: 3000,
          discoveriesCount: 30,
          lastHourActivity: 5,
          last24HourActivity: 50,
          uptime: 99.8,
          errorRate: 2,
        },
        recentDiscoveries: [],
        activityLog: [],
        isEnabled: true,
      },
      {
        id: "agent-gap-analyzer",
        name: "gap-analyzer",
        displayName: "Gap Analyzer",
        status: "paused",
        health: "degraded",
        lastActivity: new Date(Date.now() - 3600000).toISOString(),
        queuedTasks: 5,
        metrics: {
          totalRuns: 80,
          successfulRuns: 70,
          failedRuns: 10,
          avgExecutionTimeMs: 5000,
          discoveriesCount: 20,
          lastHourActivity: 0,
          last24HourActivity: 30,
          uptime: 95,
          errorRate: 12.5,
        },
        recentDiscoveries: [],
        activityLog: [],
        isEnabled: false,
        pausedAt: new Date(Date.now() - 3600000).toISOString(),
        pauseReason: "High error rate",
      },
      {
        id: "agent-evolution-engine",
        name: "evolution-engine",
        displayName: "Evolution Engine",
        status: "running",
        health: "healthy",
        lastActivity: new Date().toISOString(),
        currentTask: "Evolving techniques...",
        queuedTasks: 2,
        metrics: {
          totalRuns: 60,
          successfulRuns: 58,
          failedRuns: 2,
          avgExecutionTimeMs: 10000,
          discoveriesCount: 15,
          lastHourActivity: 3,
          last24HourActivity: 25,
          uptime: 99,
          errorRate: 3.3,
        },
        recentDiscoveries: [],
        activityLog: [],
        isEnabled: true,
      },
      {
        id: "agent-transfer-agent",
        name: "transfer-agent",
        displayName: "Transfer Agent",
        status: "idle",
        health: "healthy",
        lastActivity: new Date().toISOString(),
        queuedTasks: 0,
        metrics: {
          totalRuns: 40,
          successfulRuns: 38,
          failedRuns: 2,
          avgExecutionTimeMs: 4000,
          discoveriesCount: 10,
          lastHourActivity: 2,
          last24HourActivity: 20,
          uptime: 99.5,
          errorRate: 5,
        },
        recentDiscoveries: [],
        activityLog: [],
        isEnabled: true,
      },
    ],
    systemHealth: {
      overall: "healthy",
      cpu: 25,
      memory: 45,
      activeConnections: 2,
      queueDepth: 10,
      errorRate: 5,
      latency: 3500,
    },
    recentAlerts: [
      {
        id: "alert-1",
        type: "breakthrough",
        significance: "high",
        title: "New Breakthrough",
        description: "A major discovery",
        relatedIds: [],
        metadata: {},
        createdAt: new Date().toISOString(),
        read: false,
        dismissed: false,
      },
    ],
    summary: {
      totalDiscoveries: 120,
      discoveriesToday: 8,
      activeAgents: 2,
      pausedAgents: 1,
      pendingTasks: 10,
      alertsToday: 5,
    },
    updatedAt: new Date().toISOString(),
  };

  describe("Header", () => {
    it("should render dashboard title", () => {
      render(<MetaAgentDashboard dashboard={mockDashboard} />);

      expect(screen.getByText("Meta-Agent Dashboard")).toBeInTheDocument();
    });

    it("should render system health badge", () => {
      render(<MetaAgentDashboard dashboard={mockDashboard} />);

      // System health badge contains "healthy" with font-medium class
      const healthBadge = screen.getAllByText("healthy").find(
        (el) => el.className.includes("font-medium")
      );
      expect(healthBadge).toBeInTheDocument();
      expect(screen.getByText(/CPU:/)).toBeInTheDocument();
    });

    it("should render refresh button when handler provided", () => {
      const onRefresh = vi.fn();
      render(<MetaAgentDashboard dashboard={mockDashboard} onRefresh={onRefresh} />);

      const refreshBtn = screen.getByText("Refresh");
      expect(refreshBtn).toBeInTheDocument();

      fireEvent.click(refreshBtn);
      expect(onRefresh).toHaveBeenCalled();
    });
  });

  describe("Summary Stats", () => {
    it("should render total discoveries", () => {
      render(<MetaAgentDashboard dashboard={mockDashboard} />);

      expect(screen.getByText("120")).toBeInTheDocument();
      expect(screen.getByText("Total Discoveries")).toBeInTheDocument();
    });

    it("should render active agents count", () => {
      render(<MetaAgentDashboard dashboard={mockDashboard} />);

      expect(screen.getByText("2/5")).toBeInTheDocument();
      expect(screen.getByText("Active Agents")).toBeInTheDocument();
    });

    it("should render pending tasks", () => {
      render(<MetaAgentDashboard dashboard={mockDashboard} />);

      expect(screen.getByText("10")).toBeInTheDocument();
      expect(screen.getByText("Pending Tasks")).toBeInTheDocument();
    });

    it("should render alerts today", () => {
      render(<MetaAgentDashboard dashboard={mockDashboard} />);

      expect(screen.getByText("5")).toBeInTheDocument();
      expect(screen.getByText("Alerts Today")).toBeInTheDocument();
    });
  });

  describe("Agent List", () => {
    it("should render all agents", () => {
      render(<MetaAgentDashboard dashboard={mockDashboard} />);

      expect(screen.getByText("Synergy Detector")).toBeInTheDocument();
      expect(screen.getByText("Pattern Recognizer")).toBeInTheDocument();
      expect(screen.getByText("Gap Analyzer")).toBeInTheDocument();
      expect(screen.getByText("Evolution Engine")).toBeInTheDocument();
      expect(screen.getByText("Transfer Agent")).toBeInTheDocument();
    });

    it("should show agent status indicators", () => {
      render(<MetaAgentDashboard dashboard={mockDashboard} />);

      // Running agents should have green dots
      // We can verify agents are listed with their tasks
      expect(screen.getByText("Scanning for synergies...")).toBeInTheDocument();
      expect(screen.getByText("Evolving techniques...")).toBeInTheDocument();
    });

    it("should show pause/resume buttons", () => {
      render(
        <MetaAgentDashboard
          dashboard={mockDashboard}
          onPauseAgent={vi.fn()}
          onResumeAgent={vi.fn()}
        />
      );

      expect(screen.getAllByText("Pause").length).toBeGreaterThan(0);
      expect(screen.getAllByText("Resume").length).toBeGreaterThan(0);
    });
  });

  describe("Agent Selection", () => {
    it("should show agent details when selected", () => {
      render(<MetaAgentDashboard dashboard={mockDashboard} />);

      // Click on Synergy Detector
      fireEvent.click(screen.getByText("Synergy Detector"));

      // Should show details
      expect(screen.getByText("Status")).toBeInTheDocument();
      expect(screen.getByText("Health")).toBeInTheDocument();
      expect(screen.getByText("Performance Metrics")).toBeInTheDocument();
    });

    it("should show tabs when agent selected", () => {
      render(<MetaAgentDashboard dashboard={mockDashboard} />);

      fireEvent.click(screen.getByText("Synergy Detector"));

      expect(screen.getByRole("button", { name: "overview" })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "activity" })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "discoveries" })).toBeInTheDocument();
    });

    it("should switch tabs", () => {
      render(<MetaAgentDashboard dashboard={mockDashboard} />);

      fireEvent.click(screen.getByText("Synergy Detector"));

      // Click activity tab
      fireEvent.click(screen.getByRole("button", { name: "activity" }));

      // Should show activity log content
      expect(screen.getByText("Completed routine scan")).toBeInTheDocument();
    });

    it("should show discoveries tab content", () => {
      render(<MetaAgentDashboard dashboard={mockDashboard} />);

      fireEvent.click(screen.getByText("Synergy Detector"));
      fireEvent.click(screen.getByRole("button", { name: "discoveries" }));

      expect(screen.getByText("Found synergy between attention and codec")).toBeInTheDocument();
    });
  });

  describe("Agent Overview", () => {
    it("should show metrics", () => {
      render(<MetaAgentDashboard dashboard={mockDashboard} />);

      fireEvent.click(screen.getByText("Synergy Detector"));

      expect(screen.getByText("Total Runs")).toBeInTheDocument();
      expect(screen.getByText("150")).toBeInTheDocument();
      expect(screen.getByText("Success Rate")).toBeInTheDocument();
      expect(screen.getByText("45")).toBeInTheDocument(); // discoveries
    });

    it("should show current task", () => {
      render(<MetaAgentDashboard dashboard={mockDashboard} />);

      fireEvent.click(screen.getByText("Synergy Detector"));

      // Current Task is shown in the overview panel
      const currentTaskLabels = screen.getAllByText("Current Task");
      expect(currentTaskLabels.length).toBeGreaterThan(0);
      // The task description is shown in multiple places (list + details)
      expect(screen.getAllByText("Scanning for synergies...").length).toBeGreaterThan(0);
    });

    it("should show pause reason for paused agents", () => {
      render(<MetaAgentDashboard dashboard={mockDashboard} />);

      fireEvent.click(screen.getByText("Gap Analyzer"));

      expect(screen.getByText("Pause Reason")).toBeInTheDocument();
      expect(screen.getByText("High error rate")).toBeInTheDocument();
    });
  });

  describe("Agent Controls", () => {
    it("should call onPauseAgent when pause clicked", () => {
      const onPauseAgent = vi.fn();
      render(
        <MetaAgentDashboard
          dashboard={mockDashboard}
          onPauseAgent={onPauseAgent}
        />
      );

      const pauseButtons = screen.getAllByText("Pause");
      fireEvent.click(pauseButtons[0]);

      expect(onPauseAgent).toHaveBeenCalled();
    });

    it("should call onResumeAgent when resume clicked", () => {
      const onResumeAgent = vi.fn();
      render(
        <MetaAgentDashboard
          dashboard={mockDashboard}
          onResumeAgent={onResumeAgent}
        />
      );

      const resumeBtn = screen.getByText("Resume");
      fireEvent.click(resumeBtn);

      expect(onResumeAgent).toHaveBeenCalledWith("gap-analyzer");
    });
  });

  describe("Empty States", () => {
    it("should show placeholder when no agent selected", () => {
      render(<MetaAgentDashboard dashboard={mockDashboard} />);

      expect(screen.getByText("Select an agent to view details")).toBeInTheDocument();
    });

    it("should show empty activity log message", () => {
      render(<MetaAgentDashboard dashboard={mockDashboard} />);

      fireEvent.click(screen.getByText("Pattern Recognizer"));
      fireEvent.click(screen.getByRole("button", { name: "activity" }));

      expect(screen.getByText("No activity recorded yet")).toBeInTheDocument();
    });

    it("should show empty discoveries message", () => {
      render(<MetaAgentDashboard dashboard={mockDashboard} />);

      fireEvent.click(screen.getByText("Pattern Recognizer"));
      fireEvent.click(screen.getByRole("button", { name: "discoveries" }));

      expect(screen.getByText("No discoveries yet")).toBeInTheDocument();
    });
  });

  describe("Health Indicators", () => {
    it("should show degraded health styling", () => {
      render(<MetaAgentDashboard dashboard={mockDashboard} />);

      // Gap Analyzer has degraded health
      expect(screen.getByText("degraded")).toBeInTheDocument();
    });

    it("should show error rate warning when high", () => {
      render(<MetaAgentDashboard dashboard={mockDashboard} />);

      fireEvent.click(screen.getByText("Gap Analyzer"));

      expect(screen.getByText("Error Rate")).toBeInTheDocument();
      expect(screen.getByText("12.5%")).toBeInTheDocument();
    });
  });
});
