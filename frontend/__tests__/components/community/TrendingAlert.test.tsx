/**
 * Tests for TrendingAlert Component
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TrendingAlert, TrendingAlertList } from "@/components/community/TrendingAlert";
import type { TrendingAlert as AlertType } from "@/lib/meta/community/types";

describe("TrendingAlert", () => {
  const mockAlert: AlertType = {
    id: "alert-1",
    type: "breakthrough",
    significance: "high",
    title: "Major Discovery",
    description: "A significant breakthrough in prosody transfer techniques",
    relatedIds: ["tech-1", "tech-2"],
    metadata: {
      sourceLabId: "lab-1",
      sourceLabName: "Voice Research Lab",
      domain: "TTS",
      impactScore: 85,
      actionUrl: "/discoveries/1",
    },
    createdAt: new Date(Date.now() - 30 * 60 * 1000).toISOString(), // 30 min ago
    read: false,
    dismissed: false,
  };

  describe("Full Alert Rendering", () => {
    it("should render alert title and description", () => {
      render(<TrendingAlert alert={mockAlert} />);

      expect(screen.getByText("Major Discovery")).toBeInTheDocument();
      expect(
        screen.getByText("A significant breakthrough in prosody transfer techniques")
      ).toBeInTheDocument();
    });

    it("should render alert type", () => {
      render(<TrendingAlert alert={mockAlert} />);

      expect(screen.getByText("breakthrough")).toBeInTheDocument();
    });

    it("should render significance badge", () => {
      render(<TrendingAlert alert={mockAlert} />);

      expect(screen.getByText("high")).toBeInTheDocument();
    });

    it("should render metadata", () => {
      render(<TrendingAlert alert={mockAlert} />);

      expect(screen.getByText("Lab: Voice Research Lab")).toBeInTheDocument();
      expect(screen.getByText("Domain: TTS")).toBeInTheDocument();
      expect(screen.getByText("Impact: 85")).toBeInTheDocument();
    });

    it("should render relative time", () => {
      render(<TrendingAlert alert={mockAlert} />);

      expect(screen.getByText("30m ago")).toBeInTheDocument();
    });

    it("should show unread indicator for unread alerts", () => {
      render(<TrendingAlert alert={mockAlert} />);

      // Unread indicator is a blue dot
      const unreadDot = document.querySelector(".bg-blue-500.rounded-full");
      expect(unreadDot).toBeInTheDocument();
    });

    it("should not show unread indicator for read alerts", () => {
      const readAlert = { ...mockAlert, read: true };
      render(<TrendingAlert alert={readAlert} />);

      const unreadDots = document.querySelectorAll(".bg-blue-500.rounded-full.w-2.h-2");
      expect(unreadDots.length).toBe(0);
    });
  });

  describe("Compact Alert Rendering", () => {
    it("should render compact version", () => {
      render(<TrendingAlert alert={mockAlert} compact />);

      expect(screen.getByText("Major Discovery")).toBeInTheDocument();
      expect(screen.getByText("30m ago")).toBeInTheDocument();
    });

    it("should truncate description in compact mode", () => {
      render(<TrendingAlert alert={mockAlert} compact />);

      // Description should be truncated (has truncate class)
      const titleElement = screen.getByText("Major Discovery");
      expect(titleElement.className).toContain("truncate");
    });
  });

  describe("Actions", () => {
    it("should call onMarkRead when mark read clicked", () => {
      const onMarkRead = vi.fn();
      render(<TrendingAlert alert={mockAlert} onMarkRead={onMarkRead} />);

      const markReadBtn = screen.getByText("Mark Read");
      fireEvent.click(markReadBtn);

      expect(onMarkRead).toHaveBeenCalledWith(mockAlert.id);
    });

    it("should call onDismiss when dismiss clicked", () => {
      const onDismiss = vi.fn();
      render(<TrendingAlert alert={mockAlert} onDismiss={onDismiss} />);

      const dismissBtn = screen.getByText("Dismiss");
      fireEvent.click(dismissBtn);

      expect(onDismiss).toHaveBeenCalledWith(mockAlert.id);
    });

    it("should call onAction when view details clicked", () => {
      const onAction = vi.fn();
      render(<TrendingAlert alert={mockAlert} onAction={onAction} />);

      const viewBtn = screen.getByText("View Details");
      fireEvent.click(viewBtn);

      expect(onAction).toHaveBeenCalledWith(mockAlert.id);
    });

    it("should not show mark read for already read alerts", () => {
      const readAlert = { ...mockAlert, read: true };
      const onMarkRead = vi.fn();
      render(<TrendingAlert alert={readAlert} onMarkRead={onMarkRead} />);

      expect(screen.queryByText("Mark Read")).not.toBeInTheDocument();
    });

    it("should not show dismiss for already dismissed alerts", () => {
      const dismissedAlert = { ...mockAlert, dismissed: true };
      const onDismiss = vi.fn();
      render(<TrendingAlert alert={dismissedAlert} onDismiss={onDismiss} />);

      expect(screen.queryByText("Dismiss")).not.toBeInTheDocument();
    });
  });

  describe("Alert Types", () => {
    const alertTypes = [
      "synergy",
      "breakthrough",
      "trend",
      "gap-filled",
      "collaboration",
      "evolution",
      "opportunity",
    ] as const;

    alertTypes.forEach((type) => {
      it(`should render ${type} alert correctly`, () => {
        const alert = { ...mockAlert, type };
        render(<TrendingAlert alert={alert} />);

        expect(screen.getByText(type)).toBeInTheDocument();
      });
    });
  });

  describe("Significance Levels", () => {
    const levels = ["critical", "high", "medium", "low"] as const;

    levels.forEach((significance) => {
      it(`should render ${significance} significance correctly`, () => {
        const alert = { ...mockAlert, significance };
        render(<TrendingAlert alert={alert} />);

        expect(screen.getByText(significance)).toBeInTheDocument();
      });
    });
  });

  describe("Dismissed State", () => {
    it("should apply opacity to dismissed alerts", () => {
      const dismissedAlert = { ...mockAlert, dismissed: true };
      const { container } = render(<TrendingAlert alert={dismissedAlert} />);

      expect(container.firstChild).toHaveClass("opacity-50");
    });
  });
});

describe("TrendingAlertList", () => {
  const mockAlerts: AlertType[] = [
    {
      id: "alert-1",
      type: "breakthrough",
      significance: "high",
      title: "Breakthrough 1",
      description: "Description 1",
      relatedIds: [],
      metadata: {},
      createdAt: new Date().toISOString(),
      read: false,
      dismissed: false,
    },
    {
      id: "alert-2",
      type: "synergy",
      significance: "medium",
      title: "Synergy Found",
      description: "Description 2",
      relatedIds: [],
      metadata: {},
      createdAt: new Date().toISOString(),
      read: true,
      dismissed: false,
    },
    {
      id: "alert-3",
      type: "trend",
      significance: "low",
      title: "New Trend",
      description: "Description 3",
      relatedIds: [],
      metadata: {},
      createdAt: new Date().toISOString(),
      read: false,
      dismissed: false,
    },
  ];

  describe("Rendering", () => {
    it("should render all alerts", () => {
      render(<TrendingAlertList alerts={mockAlerts} />);

      expect(screen.getByText("Breakthrough 1")).toBeInTheDocument();
      expect(screen.getByText("Synergy Found")).toBeInTheDocument();
      expect(screen.getByText("New Trend")).toBeInTheDocument();
    });

    it("should respect maxItems limit", () => {
      render(<TrendingAlertList alerts={mockAlerts} maxItems={2} />);

      expect(screen.getByText("Breakthrough 1")).toBeInTheDocument();
      expect(screen.getByText("Synergy Found")).toBeInTheDocument();
      expect(screen.queryByText("New Trend")).not.toBeInTheDocument();
    });

    it("should show remaining count when limited", () => {
      render(<TrendingAlertList alerts={mockAlerts} maxItems={2} />);

      expect(screen.getByText("+1 more alerts")).toBeInTheDocument();
    });
  });

  describe("Empty State", () => {
    it("should show empty message when no alerts", () => {
      render(<TrendingAlertList alerts={[]} />);

      expect(screen.getByText("No alerts to show")).toBeInTheDocument();
    });

    it("should not show empty message when showEmpty is false", () => {
      render(<TrendingAlertList alerts={[]} showEmpty={false} />);

      expect(screen.queryByText("No alerts to show")).not.toBeInTheDocument();
    });
  });

  describe("Compact Mode", () => {
    it("should render compact alerts", () => {
      render(<TrendingAlertList alerts={mockAlerts} compact />);

      // All alerts should render in compact mode
      expect(screen.getByText("Breakthrough 1")).toBeInTheDocument();
    });
  });

  describe("Event Handlers", () => {
    it("should pass onMarkRead to individual alerts", () => {
      const onMarkRead = vi.fn();
      render(<TrendingAlertList alerts={mockAlerts} onMarkRead={onMarkRead} />);

      const markReadButtons = screen.getAllByText("Mark Read");
      fireEvent.click(markReadButtons[0]);

      expect(onMarkRead).toHaveBeenCalled();
    });

    it("should pass onDismiss to individual alerts", () => {
      const onDismiss = vi.fn();
      render(<TrendingAlertList alerts={mockAlerts} onDismiss={onDismiss} />);

      const dismissButtons = screen.getAllByText("Dismiss");
      fireEvent.click(dismissButtons[0]);

      expect(onDismiss).toHaveBeenCalled();
    });
  });
});
