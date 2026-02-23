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
    });

    it("should show empty state when no activities", () => {
      render(<SocialPreview />);

      expect(
        screen.getByText("Activity will appear here as research progresses")
      ).toBeInTheDocument();
    });
  });
});
