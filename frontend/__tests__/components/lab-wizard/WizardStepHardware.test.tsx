/**
 * WizardStepHardware Component Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { WizardStepHardware } from "@/components/lab-wizard/WizardStepHardware";
import type { HardwareConfig } from "@/lib/lab-wizard/types";

// Mock the CLOUD_PROVIDERS
vi.mock("@/lib/lab-wizard/types", async () => {
  const actual = await vi.importActual("@/lib/lab-wizard/types");
  return {
    ...actual,
    CLOUD_PROVIDERS: [
      { id: "runpod", name: "RunPod", icon: "🚀" },
      { id: "aws", name: "AWS", icon: "☁️" },
      { id: "gcp", name: "GCP", icon: "🌐" },
      { id: "lambda-labs", name: "Lambda", icon: "λ" },
    ],
  };
});

describe("WizardStepHardware", () => {
  const defaultConfig: HardwareConfig = {
    type: "local",
  };

  const defaultProps = {
    config: defaultConfig,
    onConfigChange: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    // Mock fetch to return empty result by default to prevent warnings
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ success: false }),
    }));
    // Mock environment variable for quick connect
    vi.stubGlobal("process", {
      ...process,
      env: {
        ...process.env,
        NEXT_PUBLIC_REMOTE_GPU_HOST: "192.0.2.100",
        NEXT_PUBLIC_REMOTE_GPU_USER: "doc",
      },
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe("rendering", () => {
    it("renders info banner", () => {
      render(<WizardStepHardware {...defaultProps} />);
      expect(screen.getByText(/Configure where your research will run/)).toBeDefined();
    });

    it("renders hardware type selector", () => {
      render(<WizardStepHardware {...defaultProps} />);
      expect(screen.getByText("Select Hardware Type")).toBeDefined();
    });

    it("renders all hardware options", () => {
      render(<WizardStepHardware {...defaultProps} />);
      expect(screen.getByText("Local Machine")).toBeDefined();
      expect(screen.getByText("Remote SSH")).toBeDefined();
      expect(screen.getByText("Cloud Provider")).toBeDefined();
    });

    it("renders hardware option descriptions", () => {
      render(<WizardStepHardware {...defaultProps} />);
      expect(screen.getByText(/Use your current computer's GPU/)).toBeDefined();
      expect(screen.getByText(/Connect to a remote GPU server/)).toBeDefined();
      expect(screen.getByText(/Use RunPod, AWS, or other cloud GPUs/)).toBeDefined();
    });

    it("shows GPU requirement for selected domain", () => {
      render(<WizardStepHardware {...defaultProps} selectedDomain="voice-clone" />);
      expect(screen.getByText(/24GB VRAM recommended/)).toBeDefined();
    });
  });

  describe("hardware type selection", () => {
    it("calls onConfigChange when hardware type is changed", () => {
      const onConfigChange = vi.fn();
      render(
        <WizardStepHardware {...defaultProps} onConfigChange={onConfigChange} />
      );

      const remoteOption = screen.getByText("Remote SSH").closest("button");
      fireEvent.click(remoteOption!);

      expect(onConfigChange).toHaveBeenCalledWith(
        expect.objectContaining({
          type: "remote-ssh",
        })
      );
    });

    it("highlights selected hardware type", () => {
      const { container } = render(<WizardStepHardware {...defaultProps} />);

      // Local should be selected by default
      const buttons = container.querySelectorAll("button");
      const localButton = Array.from(buttons).find((btn) =>
        btn.textContent?.includes("Local Machine")
      );
      expect(localButton?.className).toContain("border-foreground-bright");
    });
  });

  describe("local hardware configuration", () => {
    it("renders local hardware section when type is local", () => {
      render(<WizardStepHardware {...defaultProps} />);
      expect(screen.getByText("Local Hardware")).toBeDefined();
    });

    it("renders Detect button", () => {
      render(<WizardStepHardware {...defaultProps} />);
      expect(screen.getByText("Detect")).toBeDefined();
    });

    it("shows No GPU Detected initially", () => {
      render(<WizardStepHardware {...defaultProps} />);
      expect(screen.getByText("No GPU Detected")).toBeDefined();
    });

    it("shows Ollama status", () => {
      render(<WizardStepHardware {...defaultProps} />);
      expect(screen.getByText("Ollama")).toBeDefined();
    });

    it("calls fetch when Detect button is clicked", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            success: true,
            gpu: { name: "RTX 4090", vram: 24, available: true },
          }),
      });

      render(<WizardStepHardware {...defaultProps} />);

      const detectButton = screen.getByText("Detect");
      fireEvent.click(detectButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith("/api/lab/hardware");
      });
    });

    it("updates config with detected GPU", async () => {
      const onConfigChange = vi.fn();
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            success: true,
            gpu: { name: "RTX 4090", vram: 24, available: true },
          }),
      });

      render(
        <WizardStepHardware {...defaultProps} onConfigChange={onConfigChange} />
      );

      const detectButton = screen.getByText("Detect");
      fireEvent.click(detectButton);

      await waitFor(() => {
        expect(onConfigChange).toHaveBeenCalledWith(
          expect.objectContaining({
            type: "local",
            local: expect.objectContaining({
              gpu: expect.objectContaining({
                name: "RTX 4090",
                vram: 24,
              }),
            }),
          })
        );
      });
    });
  });

  describe("SSH configuration", () => {
    const sshConfig: HardwareConfig = {
      type: "remote-ssh",
      ssh: {
        host: "",
        port: 22,
        user: "",
      },
    };

    it("renders SSH connection section when type is remote-ssh", () => {
      render(<WizardStepHardware {...defaultProps} config={sshConfig} />);
      expect(screen.getByText("SSH Connection")).toBeDefined();
    });

    it("renders host input", () => {
      render(<WizardStepHardware {...defaultProps} config={sshConfig} />);
      expect(screen.getByPlaceholderText(/192.168.1.100 or hostname/)).toBeDefined();
    });

    it("renders port input", () => {
      render(<WizardStepHardware {...defaultProps} config={sshConfig} />);
      expect(screen.getByPlaceholderText("22")).toBeDefined();
    });

    it("renders username input", () => {
      render(<WizardStepHardware {...defaultProps} config={sshConfig} />);
      expect(screen.getByPlaceholderText("user")).toBeDefined();
    });

    it("renders SSH key path input", () => {
      render(<WizardStepHardware {...defaultProps} config={sshConfig} />);
      expect(screen.getByPlaceholderText("~/.ssh/id_rsa")).toBeDefined();
    });

    it("renders Quick Connect section", () => {
      render(<WizardStepHardware {...defaultProps} config={sshConfig} />);
      expect(screen.getByText("Quick Connect")).toBeDefined();
    });

    it("renders RTX 4090 quick connect button", () => {
      render(<WizardStepHardware {...defaultProps} config={sshConfig} />);
      expect(screen.getByText("RTX 4090 (Tailscale)")).toBeDefined();
    });

    it("fills in RTX 4090 credentials when quick connect is clicked", () => {
      const onConfigChange = vi.fn();
      render(
        <WizardStepHardware
          {...defaultProps}
          config={sshConfig}
          onConfigChange={onConfigChange}
        />
      );

      const quickConnect = screen.getByText("RTX 4090 (Tailscale)");
      fireEvent.click(quickConnect);

      expect(onConfigChange).toHaveBeenCalledWith(
        expect.objectContaining({
          ssh: expect.objectContaining({
            host: "192.0.2.100",
            user: "doc",
          }),
        })
      );
    });

    it("renders Test Connection button", () => {
      render(<WizardStepHardware {...defaultProps} config={sshConfig} />);
      expect(screen.getByText("Test Connection")).toBeDefined();
    });

    it("disables Test Connection when host is empty", () => {
      render(<WizardStepHardware {...defaultProps} config={sshConfig} />);
      const testButton = screen.getByText("Test Connection").closest("button");
      expect(testButton?.disabled).toBe(true);
    });

    it("enables Test Connection when host and user are provided", () => {
      const filledConfig: HardwareConfig = {
        type: "remote-ssh",
        ssh: {
          host: "192.168.1.100",
          port: 22,
          user: "admin",
        },
      };
      render(<WizardStepHardware {...defaultProps} config={filledConfig} />);
      const testButton = screen.getByText("Test Connection").closest("button");
      expect(testButton?.disabled).toBe(false);
    });

    it("shows validation warning for unverified SSH", () => {
      render(<WizardStepHardware {...defaultProps} config={sshConfig} />);
      expect(screen.getByText(/Please test your SSH connection/)).toBeDefined();
    });
  });

  describe("cloud configuration", () => {
    const cloudConfig: HardwareConfig = {
      type: "cloud",
      cloud: {
        provider: "",
      },
    };

    it("renders Cloud Provider section when type is cloud", () => {
      render(<WizardStepHardware {...defaultProps} config={cloudConfig} />);
      // Cloud Provider appears in both selector and section heading
      const elements = screen.getAllByText("Cloud Provider");
      expect(elements.length).toBeGreaterThan(0);
    });

    it("renders provider options", () => {
      render(<WizardStepHardware {...defaultProps} config={cloudConfig} />);
      expect(screen.getByText("RunPod")).toBeDefined();
      expect(screen.getByText("AWS")).toBeDefined();
      expect(screen.getByText("GCP")).toBeDefined();
      expect(screen.getByText("Lambda")).toBeDefined();
    });

    it("calls onConfigChange when provider is selected", () => {
      const onConfigChange = vi.fn();
      render(
        <WizardStepHardware
          {...defaultProps}
          config={cloudConfig}
          onConfigChange={onConfigChange}
        />
      );

      const runpodOption = screen.getByText("RunPod").closest("button");
      fireEvent.click(runpodOption!);

      expect(onConfigChange).toHaveBeenCalledWith(
        expect.objectContaining({
          cloud: expect.objectContaining({
            provider: "runpod",
          }),
        })
      );
    });

    it("shows API key input when provider is selected", () => {
      const configWithProvider: HardwareConfig = {
        type: "cloud",
        cloud: {
          provider: "runpod",
        },
      };
      render(<WizardStepHardware {...defaultProps} config={configWithProvider} />);
      expect(screen.getByPlaceholderText(/runpod API key/)).toBeDefined();
    });

    it("shows API key required warning", () => {
      const configWithProvider: HardwareConfig = {
        type: "cloud",
        cloud: {
          provider: "runpod",
        },
      };
      render(<WizardStepHardware {...defaultProps} config={configWithProvider} />);
      expect(screen.getByText(/API key required/)).toBeDefined();
    });

    it("shows provider-specific info for RunPod", () => {
      const configWithProvider: HardwareConfig = {
        type: "cloud",
        cloud: {
          provider: "runpod",
        },
      };
      render(<WizardStepHardware {...defaultProps} config={configWithProvider} />);
      expect(screen.getByText(/RunPod - Serverless GPUs/)).toBeDefined();
    });
  });

  describe("GPU requirements by domain", () => {
    it("shows correct requirement for voice-clone", () => {
      render(<WizardStepHardware {...defaultProps} selectedDomain="voice-clone" />);
      expect(screen.getByText(/24GB VRAM recommended for training/)).toBeDefined();
    });

    it("shows correct requirement for quant-trading", () => {
      render(<WizardStepHardware {...defaultProps} selectedDomain="quant-trading" />);
      expect(screen.getByText(/8GB VRAM minimum/)).toBeDefined();
    });

    it("shows correct requirement for robotics", () => {
      render(<WizardStepHardware {...defaultProps} selectedDomain="robotics" />);
      expect(screen.getByText(/24GB VRAM recommended for simulation/)).toBeDefined();
    });

    it("shows correct requirement for biotech", () => {
      render(<WizardStepHardware {...defaultProps} selectedDomain="biotech" />);
      expect(screen.getByText(/48GB\+ VRAM for molecular modeling/)).toBeDefined();
    });

    it("shows default requirement for unknown domain", () => {
      render(<WizardStepHardware {...defaultProps} selectedDomain="unknown" />);
      expect(screen.getByText(/8GB VRAM minimum recommended/)).toBeDefined();
    });
  });

  describe("styling", () => {
    it("applies custom className", () => {
      const { container } = render(
        <WizardStepHardware {...defaultProps} className="custom-class" />
      );
      expect(container.firstChild?.className).toContain("custom-class");
    });
  });
});
