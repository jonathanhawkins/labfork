/**
 * ParticipantList Component Tests
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ParticipantList } from "@/components/collaboration/ParticipantList";
import {
  Participant,
  ParticipantRole,
  ParticipantStatus,
  createParticipantId,
} from "@/lib/meta/collaboration/types";

describe("ParticipantList", () => {
  const createMockParticipant = (
    overrides: Partial<Participant> = {}
  ): Participant => ({
    id: createParticipantId(),
    labId: "lab-1",
    labName: "Test Lab",
    role: "contributor",
    status: "active",
    expertise: ["machine-learning", "nlp"],
    commitment: {
      hoursPerWeek: 10,
      durationWeeks: 4,
      resources: ["GPU"],
      responsibilities: ["Data processing"],
    },
    joinedAt: new Date().toISOString(),
    lastActiveAt: new Date().toISOString(),
    contributions: [],
    assignedObjectives: ["obj-1"],
    ...overrides,
  });

  describe("Rendering", () => {
    it("should render header with participant count", () => {
      const participants = [
        createMockParticipant({ labName: "Lab 1" }),
        createMockParticipant({ labId: "lab-2", labName: "Lab 2" }),
      ];

      render(<ParticipantList participants={participants} />);

      expect(screen.getByText("Participants")).toBeInTheDocument();
      expect(screen.getByText(/2 active of 2 total/)).toBeInTheDocument();
    });

    it("should render participant names", () => {
      const participants = [
        createMockParticipant({ labName: "Alpha Lab" }),
        createMockParticipant({ labId: "lab-2", labName: "Beta Lab" }),
      ];

      render(<ParticipantList participants={participants} />);

      expect(screen.getByText("Alpha Lab")).toBeInTheDocument();
      expect(screen.getByText("Beta Lab")).toBeInTheDocument();
    });

    it("should render role badges", () => {
      const participants = [
        createMockParticipant({ role: "lead", labName: "Lead Lab" }),
        createMockParticipant({ role: "contributor", labId: "lab-2", labName: "Contrib Lab" }),
        createMockParticipant({ role: "advisor", labId: "lab-3", labName: "Advisor Lab" }),
      ];

      render(<ParticipantList participants={participants} />);

      expect(screen.getByText("Lead")).toBeInTheDocument();
      expect(screen.getByText("Contributor")).toBeInTheDocument();
      expect(screen.getByText("Advisor")).toBeInTheDocument();
    });

    it("should show expertise tags", () => {
      const participants = [
        createMockParticipant({ expertise: ["nlp", "transformers", "pytorch"] }),
      ];

      render(<ParticipantList participants={participants} />);

      // Should show first 2 and a "+1" indicator
      expect(screen.getByText("nlp")).toBeInTheDocument();
      expect(screen.getByText("transformers")).toBeInTheDocument();
      expect(screen.getByText("+1")).toBeInTheDocument();
    });

    it("should show contribution count", () => {
      const participants = [
        createMockParticipant({
          contributions: [
            {
              id: "contrib-1",
              type: "research",
              description: "Did research",
              timestamp: new Date().toISOString(),
              artifacts: [],
              impactScore: 5,
            },
            {
              id: "contrib-2",
              type: "implementation",
              description: "Built feature",
              timestamp: new Date().toISOString(),
              artifacts: [],
              impactScore: 10,
            },
          ],
        }),
      ];

      render(<ParticipantList participants={participants} />);

      expect(screen.getByText("2 contributions")).toBeInTheDocument();
    });

    it("should show empty state when no participants", () => {
      render(<ParticipantList participants={[]} />);

      expect(screen.getByText("No participants yet")).toBeInTheDocument();
    });
  });

  describe("Sorting", () => {
    it("should show lead first", () => {
      const participants = [
        createMockParticipant({ labId: "lab-2", labName: "Other Lab", role: "contributor" }),
        createMockParticipant({ labId: "lab-1", labName: "Lead Lab", role: "lead" }),
      ];

      render(<ParticipantList participants={participants} />);

      const labs = screen.getAllByText(/Lab$/);
      expect(labs[0].textContent).toBe("Lead Lab");
    });

    it("should show co-lead after lead", () => {
      const participants = [
        createMockParticipant({ labId: "lab-3", labName: "Regular Lab", role: "contributor" }),
        createMockParticipant({ labId: "lab-2", labName: "Co-Lead Lab", role: "co-lead" }),
        createMockParticipant({ labId: "lab-1", labName: "Lead Lab", role: "lead" }),
      ];

      render(<ParticipantList participants={participants} />);

      const labs = screen.getAllByText(/Lab$/);
      expect(labs[0].textContent).toBe("Lead Lab");
      expect(labs[1].textContent).toBe("Co-Lead Lab");
    });
  });

  describe("Current User Indicator", () => {
    it("should show (You) for current user", () => {
      const participants = [
        createMockParticipant({ labId: "my-lab", labName: "My Lab" }),
        createMockParticipant({ labId: "other-lab", labName: "Other Lab" }),
      ];

      render(<ParticipantList participants={participants} currentLabId="my-lab" />);

      expect(screen.getByText("(You)")).toBeInTheDocument();
    });

    it("should highlight current user row", () => {
      const participants = [
        createMockParticipant({ labId: "my-lab", labName: "My Lab" }),
      ];

      render(<ParticipantList participants={participants} currentLabId="my-lab" />);

      // Row wrapper should have blue background (bg-blue-50 class)
      // The structure is: wrapper div (bg-blue-50) > main row div > info div > name span
      const nameElement = screen.getByText("My Lab");
      // Traverse up to find the wrapper with bg-blue-50
      let wrapper = nameElement.closest("div");
      while (wrapper && !wrapper.className.includes("bg-blue")) {
        wrapper = wrapper.parentElement?.closest("div") || null;
      }
      expect(wrapper?.className).toContain("bg-blue");
    });
  });

  describe("Expandable Details", () => {
    it("should expand on click", () => {
      const participants = [
        createMockParticipant({
          commitment: {
            hoursPerWeek: 15,
            durationWeeks: 8,
            resources: ["GPU", "Dataset"],
            responsibilities: ["Lead research"],
          },
        }),
      ];

      render(<ParticipantList participants={participants} />);

      // Click expand button
      const expandButton = screen.getByRole("button", { name: /expand/i });
      fireEvent.click(expandButton);

      // Should show detailed commitment info
      expect(screen.getByText(/15h\/week/)).toBeInTheDocument();
      expect(screen.getByText(/8 weeks/)).toBeInTheDocument();
    });

    it("should show resources when expanded", () => {
      const participants = [
        createMockParticipant({
          commitment: {
            hoursPerWeek: 10,
            durationWeeks: 4,
            resources: ["GPU Cluster", "Training Data"],
            responsibilities: [],
          },
        }),
      ];

      render(<ParticipantList participants={participants} />);

      const expandButton = screen.getByRole("button", { name: /expand/i });
      fireEvent.click(expandButton);

      expect(screen.getByText("Resources")).toBeInTheDocument();
      expect(screen.getByText("GPU Cluster")).toBeInTheDocument();
      expect(screen.getByText("Training Data")).toBeInTheDocument();
    });

    it("should show recent contributions when expanded", () => {
      const participants = [
        createMockParticipant({
          contributions: [
            {
              id: "c1",
              type: "research",
              description: "Completed literature review",
              timestamp: new Date().toISOString(),
              artifacts: [],
              impactScore: 8,
            },
          ],
        }),
      ];

      render(<ParticipantList participants={participants} />);

      const expandButton = screen.getByRole("button", { name: /expand/i });
      fireEvent.click(expandButton);

      expect(screen.getByText("Recent Contributions")).toBeInTheDocument();
      expect(screen.getByText("Completed literature review")).toBeInTheDocument();
    });

    it("should collapse when clicked again", () => {
      const participants = [createMockParticipant()];

      render(<ParticipantList participants={participants} />);

      const expandButton = screen.getByRole("button", { name: /expand/i });
      fireEvent.click(expandButton);
      fireEvent.click(expandButton);

      // Commitment details should no longer be visible
      expect(screen.queryByText("Commitment:")).not.toBeInTheDocument();
    });
  });

  describe("Management Actions", () => {
    it("should show remove button when canManage is true", () => {
      const participants = [
        createMockParticipant({ role: "contributor", labName: "Removable Lab" }),
      ];

      render(
        <ParticipantList
          participants={participants}
          canManage
          onRemoveParticipant={vi.fn()}
        />
      );

      expect(screen.getByRole("button", { name: /remove/i })).toBeInTheDocument();
    });

    it("should not show remove button for lead", () => {
      const participants = [
        createMockParticipant({ role: "lead", labName: "Lead Lab" }),
      ];

      render(
        <ParticipantList
          participants={participants}
          canManage
          onRemoveParticipant={vi.fn()}
        />
      );

      expect(screen.queryByRole("button", { name: /remove/i })).not.toBeInTheDocument();
    });

    it("should call onRemoveParticipant when remove clicked", () => {
      const onRemove = vi.fn();
      const participantId = createParticipantId();
      const participants = [
        createMockParticipant({ id: participantId, role: "contributor" }),
      ];

      render(
        <ParticipantList
          participants={participants}
          canManage
          onRemoveParticipant={onRemove}
        />
      );

      fireEvent.click(screen.getByRole("button", { name: /remove/i }));
      expect(onRemove).toHaveBeenCalledWith(participantId);
    });

    it("should show role menu when canManage is true", () => {
      const participants = [
        createMockParticipant({ role: "contributor" }),
      ];

      render(
        <ParticipantList
          participants={participants}
          canManage
          onUpdateRole={vi.fn()}
        />
      );

      // Should have settings/role button
      expect(screen.getByRole("button", { name: /change role/i })).toBeInTheDocument();
    });

    it("should call onUpdateRole when role selected", () => {
      const onUpdateRole = vi.fn();
      const participantId = createParticipantId();
      const participants = [
        createMockParticipant({ id: participantId, role: "contributor" }),
      ];

      render(
        <ParticipantList
          participants={participants}
          canManage
          onUpdateRole={onUpdateRole}
        />
      );

      // Open role menu
      fireEvent.click(screen.getByRole("button", { name: /change role/i }));

      // Select new role
      fireEvent.click(screen.getByText("Advisor"));

      expect(onUpdateRole).toHaveBeenCalledWith(participantId, "advisor");
    });
  });

  describe("Status Indicators", () => {
    it("should show status indicator for active participants", () => {
      const participants = [createMockParticipant({ status: "active" })];

      render(<ParticipantList participants={participants} />);

      // Should have green status indicator
      const indicator = document.querySelector(".text-green-500");
      expect(indicator).toBeInTheDocument();
    });

    it("should count only active participants", () => {
      const participants = [
        createMockParticipant({ status: "active", labId: "lab-1" }),
        createMockParticipant({ status: "active", labId: "lab-2" }),
        createMockParticipant({ status: "withdrawn", labId: "lab-3" }),
      ];

      render(<ParticipantList participants={participants} />);

      expect(screen.getByText(/2 active of 3 total/)).toBeInTheDocument();
    });
  });
});
