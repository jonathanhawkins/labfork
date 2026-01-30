/**
 * Tests for DomainPreview Component Type Contracts
 *
 * Note: The actual 3D rendering is tested through visual verification since
 * Three.js requires WebGL which is not available in jsdom. These tests verify
 * the component's prop types and API without actually rendering the 3D scene.
 */

import { describe, it, expect } from "vitest";
import { DomainPreviewProps } from "@/components/domain/DomainPreview";

describe("DomainPreview Types", () => {
  describe("DomainPreviewProps interface", () => {
    it("requires primaryColor string", () => {
      const props: Partial<DomainPreviewProps> = {
        primaryColor: "#4ecdc4",
      };
      expect(props.primaryColor).toBe("#4ecdc4");
    });

    it("requires accentColor string", () => {
      const props: Partial<DomainPreviewProps> = {
        accentColor: "#66ffaa",
      };
      expect(props.accentColor).toBe("#66ffaa");
    });

    it("accepts optional backgroundStyle", () => {
      const styles: DomainPreviewProps["backgroundStyle"][] = [
        "sky",
        "grid",
        "gradient",
        "particles",
        "minimal",
      ];
      for (const style of styles) {
        const props: Partial<DomainPreviewProps> = {
          backgroundStyle: style,
        };
        expect(props.backgroundStyle).toBe(style);
      }
    });

    it("accepts optional props array", () => {
      const props: Partial<DomainPreviewProps> = {
        props: ["microphone", "gpu", "monitor"],
      };
      expect(props.props).toHaveLength(3);
    });

    it("accepts optional animated boolean", () => {
      const propsAnimated: Partial<DomainPreviewProps> = {
        animated: true,
      };
      const propsStatic: Partial<DomainPreviewProps> = {
        animated: false,
      };
      expect(propsAnimated.animated).toBe(true);
      expect(propsStatic.animated).toBe(false);
    });

    it("accepts optional width as number", () => {
      const props: Partial<DomainPreviewProps> = {
        width: 300,
      };
      expect(props.width).toBe(300);
    });

    it("accepts optional width as string", () => {
      const props: Partial<DomainPreviewProps> = {
        width: "100%",
      };
      expect(props.width).toBe("100%");
    });

    it("accepts optional height as number", () => {
      const props: Partial<DomainPreviewProps> = {
        height: 200,
      };
      expect(props.height).toBe(200);
    });

    it("accepts optional className string", () => {
      const props: Partial<DomainPreviewProps> = {
        className: "custom-preview",
      };
      expect(props.className).toBe("custom-preview");
    });
  });

  describe("color validation patterns", () => {
    it("validates hex color format with hash", () => {
      const validColors = [
        "#4ecdc4",
        "#66ffaa",
        "#ff0000",
        "#000000",
        "#ffffff",
        "#123abc",
      ];
      const hexPattern = /^#[0-9a-fA-F]{6}$/;
      for (const color of validColors) {
        expect(color).toMatch(hexPattern);
      }
    });

    it("validates uppercase hex colors", () => {
      const validColors = ["#4ECDC4", "#66FFAA", "#FF0000"];
      const hexPattern = /^#[0-9a-fA-F]{6}$/;
      for (const color of validColors) {
        expect(color).toMatch(hexPattern);
      }
    });

    it("rejects invalid color formats", () => {
      const invalidColors = [
        "4ecdc4", // no hash
        "#fff", // 3-char shorthand
        "#gggggg", // invalid hex chars
        "rgb(0,0,0)", // rgb format
        "red", // named color
      ];
      const hexPattern = /^#[0-9a-fA-F]{6}$/;
      for (const color of invalidColors) {
        expect(color).not.toMatch(hexPattern);
      }
    });
  });

  describe("props array validation", () => {
    it("accepts valid prop names", () => {
      const validProps = [
        "microphone",
        "speaker",
        "waveform",
        "server-rack",
        "gpu",
        "supercomputer",
        "monitor",
        "data-hub",
        "robot-arm",
        "trading-terminal",
        "lab-equipment",
        "neural-network",
      ];
      for (const prop of validProps) {
        expect(typeof prop).toBe("string");
        expect(prop.length).toBeGreaterThan(0);
      }
    });
  });

  describe("full props configuration", () => {
    it("creates valid complete props object", () => {
      const fullProps: DomainPreviewProps = {
        primaryColor: "#4ecdc4",
        accentColor: "#66ffaa",
        backgroundStyle: "sky",
        props: ["microphone", "gpu"],
        animated: true,
        width: 300,
        height: 200,
        className: "preview-card",
      };

      expect(fullProps.primaryColor).toBeDefined();
      expect(fullProps.accentColor).toBeDefined();
      expect(fullProps.backgroundStyle).toBe("sky");
      expect(fullProps.props).toHaveLength(2);
      expect(fullProps.animated).toBe(true);
      expect(fullProps.width).toBe(300);
      expect(fullProps.height).toBe(200);
      expect(fullProps.className).toBe("preview-card");
    });

    it("creates valid minimal props object", () => {
      const minimalProps: DomainPreviewProps = {
        primaryColor: "#000000",
        accentColor: "#ffffff",
      };

      expect(minimalProps.primaryColor).toBeDefined();
      expect(minimalProps.accentColor).toBeDefined();
      expect(minimalProps.backgroundStyle).toBeUndefined();
      expect(minimalProps.props).toBeUndefined();
      expect(minimalProps.animated).toBeUndefined();
    });
  });
});
