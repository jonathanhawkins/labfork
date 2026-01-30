/**
 * ProjectIdeasSection Component Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ProjectIdeasSection, projectIdeas } from "@/components/landing/ProjectIdeasSection";

describe("ProjectIdeasSection", () => {
  describe("Rendering", () => {
    it("should render section header", () => {
      render(<ProjectIdeasSection />);

      expect(
        screen.getByText("Build Projects That Change The World")
      ).toBeInTheDocument();
    });

    it("should render section badge", () => {
      render(<ProjectIdeasSection />);

      expect(
        screen.getByText("Inspire Your Next Breakthrough")
      ).toBeInTheDocument();
    });

    it("should render section description", () => {
      render(<ProjectIdeasSection />);

      expect(
        screen.getByText(/Use AI Research Labs to accelerate breakthrough projects/i)
      ).toBeInTheDocument();
    });

    it("should render bottom CTA", () => {
      render(<ProjectIdeasSection />);

      expect(screen.getByText("Start Your Own Project")).toBeInTheDocument();
    });

    it("should render custom idea prompt", () => {
      render(<ProjectIdeasSection />);

      expect(
        screen.getByText("Have your own world-changing idea?")
      ).toBeInTheDocument();
    });
  });

  describe("Project Cards", () => {
    it("should render all 10 project names", () => {
      render(<ProjectIdeasSection />);

      // Each project appears twice (desktop grid + mobile carousel)
      expect(screen.getAllByText("Atmospheric Water Harvester").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("AI Micro-Housing Factory").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("Precision Micro-Farming").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("Community Mesh Power Grid").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("Open Medical Diagnostic AI").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("Plastic-to-Fuel Converter").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("Disaster Response Drones").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("Regenerative Agriculture AI").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("Local Manufacturing Platform").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("AI Tutor for Every Child").length).toBeGreaterThanOrEqual(1);
    });

    it("should render project descriptions", () => {
      render(<ProjectIdeasSection />);

      // Descriptions appear in both desktop and mobile views
      expect(
        screen.getAllByText("Extract clean water from air in any climate").length
      ).toBeGreaterThanOrEqual(1);
      expect(
        screen.getAllByText("Automated robots build $5K homes in 48 hours").length
      ).toBeGreaterThanOrEqual(1);
    });

    it("should render project emojis", () => {
      render(<ProjectIdeasSection />);

      // Check for emoji role="img" elements (appear twice: desktop + mobile)
      const waterEmojis = screen.getAllByRole("img", { name: "Atmospheric Water Harvester" });
      expect(waterEmojis.length).toBeGreaterThanOrEqual(1);
      expect(waterEmojis[0]).toHaveTextContent("🌊");

      const housingEmojis = screen.getAllByRole("img", { name: "AI Micro-Housing Factory" });
      expect(housingEmojis.length).toBeGreaterThanOrEqual(1);
      expect(housingEmojis[0]).toHaveTextContent("🏠");
    });

    it("should render impact stats", () => {
      render(<ProjectIdeasSection />);

      // Stats appear in both desktop and mobile views
      expect(screen.getAllByText("2B people").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("Food security").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("Healthcare for billions").length).toBeGreaterThanOrEqual(1);
    });

    it("should render cost stats", () => {
      render(<ProjectIdeasSection />);

      // Multiple projects have "Free" cost, shown in both views
      expect(screen.getAllByText("Free").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("<$100").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("<$500").length).toBeGreaterThanOrEqual(1);
    });

    it("should render domain badges", () => {
      render(<ProjectIdeasSection />);

      // Multiple projects have these domains, shown in both views
      expect(screen.getAllByText("Robotics ML").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("Computer Vision").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("NLP Research").length).toBeGreaterThanOrEqual(1);
    });
  });

  describe("Project Data", () => {
    it("should have exactly 10 projects", () => {
      expect(projectIdeas).toHaveLength(10);
    });

    it("should have unique slugs for all projects", () => {
      const slugs = projectIdeas.map((p) => p.slug);
      const uniqueSlugs = new Set(slugs);
      expect(uniqueSlugs.size).toBe(10);
    });

    it("should have all required fields for each project", () => {
      projectIdeas.forEach((project) => {
        expect(project.slug).toBeDefined();
        expect(project.emoji).toBeDefined();
        expect(project.name).toBeDefined();
        expect(project.description).toBeDefined();
        expect(project.impact).toBeDefined();
        expect(project.cost).toBeDefined();
        expect(project.domain).toBeDefined();
        expect(project.domainSlug).toBeDefined();
        expect(project.color).toBeDefined();
        expect(project.accentColor).toBeDefined();
      });
    });

    it("should have valid hex colors", () => {
      const hexColorRegex = /^#[0-9a-fA-F]{6}$/;
      projectIdeas.forEach((project) => {
        expect(project.color).toMatch(hexColorRegex);
        expect(project.accentColor).toMatch(hexColorRegex);
      });
    });
  });

  describe("Links", () => {
    it("should link Start Your Own Project to /lab/new", () => {
      render(<ProjectIdeasSection />);

      const link = screen.getByText("Start Your Own Project").closest("a");
      expect(link).toHaveAttribute("href", "/lab/new");
    });

    it("should have Start Building links with correct project and domain params", () => {
      render(<ProjectIdeasSection />);

      // Find all Start Building links (20 total: 10 desktop + 10 mobile)
      const startBuildingLinks = screen.getAllByText("Start Building");
      expect(startBuildingLinks.length).toBe(20);

      // Check first project link
      const firstLink = startBuildingLinks[0].closest("a");
      expect(firstLink).toHaveAttribute(
        "href",
        "/lab/new?project=atmospheric-water-harvester&domain=climate-modeling"
      );
    });
  });

  describe("Custom Projects", () => {
    it("should render custom projects when provided", () => {
      const customProjects = [
        {
          slug: "custom-project",
          emoji: "🔬",
          name: "Custom Research Project",
          description: "A custom project for testing",
          impact: "High impact",
          cost: "$1K",
          domain: "Custom Domain",
          domainSlug: "custom",
          color: "#ff0000",
          accentColor: "#00ff00",
        },
      ];

      render(<ProjectIdeasSection projects={customProjects} />);

      // Custom projects appear twice (desktop grid + mobile carousel)
      expect(screen.getAllByText("Custom Research Project").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("A custom project for testing").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("High impact").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("$1K").length).toBeGreaterThanOrEqual(1);
    });

    it("should handle empty projects array", () => {
      render(<ProjectIdeasSection projects={[]} />);

      expect(
        screen.getByText("Build Projects That Change The World")
      ).toBeInTheDocument();
      expect(screen.queryByText("Start Building")).not.toBeInTheDocument();
    });
  });

  describe("Mobile Carousel", () => {
    it("should render scroll buttons", () => {
      render(<ProjectIdeasSection />);

      expect(screen.getByLabelText("Scroll left")).toBeInTheDocument();
      expect(screen.getByLabelText("Scroll right")).toBeInTheDocument();
    });
  });

  describe("Accessibility", () => {
    it("should have proper aria labels for emojis", () => {
      render(<ProjectIdeasSection />);

      // Each emoji appears twice (desktop + mobile)
      projectIdeas.forEach((project) => {
        const emojis = screen.getAllByRole("img", { name: project.name });
        expect(emojis.length).toBeGreaterThanOrEqual(1);
      });
    });

    it("should have accessible scroll buttons", () => {
      render(<ProjectIdeasSection />);

      const leftButton = screen.getByLabelText("Scroll left");
      const rightButton = screen.getByLabelText("Scroll right");

      expect(leftButton.tagName).toBe("BUTTON");
      expect(rightButton.tagName).toBe("BUTTON");
    });
  });
});
