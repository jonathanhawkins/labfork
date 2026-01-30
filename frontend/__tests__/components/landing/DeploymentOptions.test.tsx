/**
 * DeploymentOptions Component Tests
 */

import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { DeploymentOptions } from "@/components/landing/DeploymentOptions";

describe("DeploymentOptions", () => {
  describe("Rendering", () => {
    it("should render section header", () => {
      render(<DeploymentOptions />);

      expect(screen.getByText("Deploy Your Way")).toBeInTheDocument();
    });

    it("should render section badge", () => {
      render(<DeploymentOptions />);

      expect(screen.getByText("Easy Deployment")).toBeInTheDocument();
    });

    it("should render section description", () => {
      render(<DeploymentOptions />);

      expect(
        screen.getByText(/Cloud, container, or local/i)
      ).toBeInTheDocument();
    });
  });

  describe("Deployment Options", () => {
    it("should render One-Click Cloud option", () => {
      render(<DeploymentOptions />);

      expect(screen.getByText("One-Click Cloud")).toBeInTheDocument();
      expect(screen.getByText("Deploy to AWS, GCP, or Azure in minutes")).toBeInTheDocument();
    });

    it("should render Docker Compose option", () => {
      render(<DeploymentOptions />);

      expect(screen.getByText("Docker Compose")).toBeInTheDocument();
      expect(screen.getByText("Run anywhere with containers")).toBeInTheDocument();
    });

    it("should render Local Development option", () => {
      render(<DeploymentOptions />);

      expect(screen.getByText("Local Development")).toBeInTheDocument();
      expect(screen.getByText("Run on your own hardware")).toBeInTheDocument();
    });

    it("should show Recommended badge for cloud option", () => {
      render(<DeploymentOptions />);

      expect(screen.getByText("Recommended")).toBeInTheDocument();
    });

    it("should show Free badge for local option", () => {
      render(<DeploymentOptions />);

      expect(screen.getByText("Free")).toBeInTheDocument();
    });
  });

  describe("Cloud Features", () => {
    it("should list cloud deployment features", () => {
      render(<DeploymentOptions />);

      expect(screen.getByText("Auto-scaling GPU instances")).toBeInTheDocument();
      expect(screen.getByText("Managed databases included")).toBeInTheDocument();
      expect(screen.getByText("SSL certificates automated")).toBeInTheDocument();
      expect(screen.getByText("Cost monitoring dashboard")).toBeInTheDocument();
    });
  });

  describe("Docker Features", () => {
    it("should list Docker deployment features", () => {
      render(<DeploymentOptions />);

      expect(screen.getByText("Single docker-compose.yml")).toBeInTheDocument();
      expect(screen.getByText("NVIDIA GPU support")).toBeInTheDocument();
      expect(screen.getByText("Persistent volumes")).toBeInTheDocument();
      expect(screen.getByText("Easy environment config")).toBeInTheDocument();
    });
  });

  describe("Local Features", () => {
    it("should list local deployment features", () => {
      render(<DeploymentOptions />);

      expect(screen.getByText("Mac M-series optimized")).toBeInTheDocument();
      expect(screen.getByText("CUDA support for NVIDIA")).toBeInTheDocument();
      expect(screen.getByText("Minimal dependencies")).toBeInTheDocument();
      expect(screen.getByText("Full source access")).toBeInTheDocument();
    });
  });

  describe("CTAs", () => {
    it("should have Deploy Now CTA for cloud", () => {
      render(<DeploymentOptions />);

      expect(screen.getByText("Deploy Now")).toBeInTheDocument();
    });

    it("should have Get Config CTA for Docker", () => {
      render(<DeploymentOptions />);

      expect(screen.getByText("Get Config")).toBeInTheDocument();
    });

    it("should have Setup Guide CTA for local", () => {
      render(<DeploymentOptions />);

      expect(screen.getByText("Setup Guide")).toBeInTheDocument();
    });
  });

  describe("Pricing Section", () => {
    it("should render pricing header", () => {
      render(<DeploymentOptions />);

      expect(screen.getByText("Simple Pricing")).toBeInTheDocument();
      expect(screen.getByText("Start free, scale when ready")).toBeInTheDocument();
    });

    it("should render Hobby tier", () => {
      render(<DeploymentOptions />);

      expect(screen.getByText("$0")).toBeInTheDocument();
      expect(screen.getByText("Hobby")).toBeInTheDocument();
      expect(screen.getByText("Local Mac/PC")).toBeInTheDocument();
    });

    it("should render Pro tier", () => {
      render(<DeploymentOptions />);

      expect(screen.getByText("$49/mo")).toBeInTheDocument();
      expect(screen.getByText("Pro")).toBeInTheDocument();
      expect(screen.getByText("Cloud GPU (T4)")).toBeInTheDocument();
    });

    it("should render Team tier", () => {
      render(<DeploymentOptions />);

      expect(screen.getByText("$199/mo")).toBeInTheDocument();
      expect(screen.getByText("Team")).toBeInTheDocument();
      expect(screen.getByText("Cloud GPU (A100)")).toBeInTheDocument();
    });

    it("should show tier features", () => {
      render(<DeploymentOptions />);

      expect(screen.getByText("Unlimited research")).toBeInTheDocument();
      expect(screen.getByText("GPU training")).toBeInTheDocument();
      expect(screen.getByText("Priority support")).toBeInTheDocument();
    });
  });

  describe("Option Selection", () => {
    it("should highlight cloud option by default", () => {
      render(<DeploymentOptions />);

      // Cloud option should have the highlighted style
      const cloudOption = screen.getByText("One-Click Cloud").closest("button");
      expect(cloudOption?.className).toContain("border-white/20");
    });

    it("should switch selection when clicking docker option", () => {
      render(<DeploymentOptions />);

      const dockerOption = screen.getByText("Docker Compose").closest("button");
      fireEvent.click(dockerOption!);

      // Docker option should now be highlighted
      expect(dockerOption?.className).toContain("border-white/20");
    });

    it("should switch selection when clicking local option", () => {
      render(<DeploymentOptions />);

      const localOption = screen.getByText("Local Development").closest("button");
      fireEvent.click(localOption!);

      // Local option should now be highlighted
      expect(localOption?.className).toContain("border-white/20");
    });
  });
});
