/**
 * SocialPreview Component Tests
 */

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { SocialPreview } from "@/components/landing/SocialPreview";

describe("SocialPreview", () => {
  describe("Rendering", () => {
    it("should render section header", () => {
      render(<SocialPreview />);

      expect(screen.getByText("Research is Social")).toBeInTheDocument();
    });

    it("should render section badge", () => {
      render(<SocialPreview />);

      expect(screen.getByText("Collaborative Research")).toBeInTheDocument();
    });

    it("should render section description", () => {
      render(<SocialPreview />);

      expect(
        screen.getByText(/Star labs you admire, fork techniques/i)
      ).toBeInTheDocument();
    });
  });

  describe("Feature List", () => {
    it("should render Star & Follow feature", () => {
      render(<SocialPreview />);

      expect(screen.getByText("Star & Follow")).toBeInTheDocument();
      expect(
        screen.getByText(/Bookmark interesting labs/i)
      ).toBeInTheDocument();
    });

    it("should render Fork & Extend feature", () => {
      render(<SocialPreview />);

      expect(screen.getByText("Fork & Extend")).toBeInTheDocument();
      expect(
        screen.getByText(/Clone any public lab/i)
      ).toBeInTheDocument();
    });

    it("should render Collaborate feature", () => {
      render(<SocialPreview />);

      expect(screen.getByText("Collaborate")).toBeInTheDocument();
      expect(
        screen.getByText(/Invite others to your lab/i)
      ).toBeInTheDocument();
    });

    it("should render Weekly Digest feature", () => {
      render(<SocialPreview />);

      expect(screen.getByText("Weekly Digest")).toBeInTheDocument();
      expect(
        screen.getByText(/Get personalized updates/i)
      ).toBeInTheDocument();
    });
  });

  describe("Live Activity Feed", () => {
    it("should render Live Activity header", () => {
      render(<SocialPreview />);

      expect(screen.getByText("Live Activity")).toBeInTheDocument();
      expect(screen.getByText("Global Feed")).toBeInTheDocument();
    });

    it("should render activity items", () => {
      render(<SocialPreview />);

      expect(screen.getByText("MIT Lab")).toBeInTheDocument();
      expect(screen.getByText("Stanford NLP")).toBeInTheDocument();
      expect(screen.getByText("DeepMind")).toBeInTheDocument();
      expect(screen.getByText("OpenAI")).toBeInTheDocument();
      expect(screen.getByText("Berkeley AI")).toBeInTheDocument();
    });

    it("should show activity types", () => {
      render(<SocialPreview />);

      expect(screen.getByText("discovery")).toBeInTheDocument();
      expect(screen.getByText("star")).toBeInTheDocument();
      expect(screen.getByText("fork")).toBeInTheDocument();
      expect(screen.getByText("collaboration")).toBeInTheDocument();
      expect(screen.getByText("comment")).toBeInTheDocument();
    });

    it("should show activity targets", () => {
      render(<SocialPreview />);

      expect(
        screen.getByText("Voice emotion transfer without paired data")
      ).toBeInTheDocument();
      expect(screen.getByText("Quant Trading Lab")).toBeInTheDocument();
      expect(screen.getByText("Game AI Lab")).toBeInTheDocument();
    });

    it("should show activity timestamps", () => {
      render(<SocialPreview />);

      expect(screen.getByText("2 min ago")).toBeInTheDocument();
      expect(screen.getByText("5 min ago")).toBeInTheDocument();
      expect(screen.getByText("12 min ago")).toBeInTheDocument();
    });

    it("should show activity details when available", () => {
      render(<SocialPreview />);

      expect(
        screen.getByText("Using contrastive learning to align prosody embeddings")
      ).toBeInTheDocument();
      expect(
        screen.getByText("Adapting for multi-agent environments")
      ).toBeInTheDocument();
    });
  });
});
