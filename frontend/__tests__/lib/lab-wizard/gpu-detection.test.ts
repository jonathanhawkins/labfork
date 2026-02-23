/**
 * GPU Detection Utility Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  detectLocalGpu,
  detectLocalSystem,
  systemInfoToLocalConfig,
  DOMAIN_GPU_REQUIREMENTS,
  checkGpuMeetsDomain,
  formatGpuInfo,
  getGpuRecommendation,
  MOCK_GPU_INFO,
  MOCK_SYSTEM_INFO,
} from "@/lib/lab-wizard/gpu-detection";

describe("gpu-detection", () => {
  describe("detectLocalGpu", () => {
    beforeEach(() => {
      vi.stubGlobal("fetch", vi.fn());
    });

    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it("should return mock GPU info when API returns success", async () => {
      const mockResponse = {
        success: true,
        gpu: MOCK_GPU_INFO,
      };

      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await detectLocalGpu();

      expect(result).toBeDefined();
      expect(result?.name).toBe(MOCK_GPU_INFO.name);
      expect(result?.vram).toBe(MOCK_GPU_INFO.vram);
    });

    it("should return null when API returns error", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: false }),
      });

      const result = await detectLocalGpu();

      expect(result).toBeNull();
    });

    it("should return null when fetch throws", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error("Network error")
      );

      const result = await detectLocalGpu();

      expect(result).toBeNull();
    });
  });

  describe("detectLocalSystem", () => {
    beforeEach(() => {
      vi.stubGlobal("fetch", vi.fn());
    });

    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it("should return system info when API returns success", async () => {
      const mockResponse = {
        success: true,
        gpu: MOCK_GPU_INFO,
        ram: 64,
        platform: "darwin",
        ollamaInstalled: true,
      };

      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await detectLocalSystem();

      expect(result).toBeDefined();
      expect(result?.platform).toBe("darwin");
    });

    it("should return null on error", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error("Network error")
      );

      const result = await detectLocalSystem();

      expect(result).toBeNull();
    });
  });

  describe("systemInfoToLocalConfig", () => {
    it("should convert system info to local config", () => {
      const result = systemInfoToLocalConfig(MOCK_SYSTEM_INFO, MOCK_GPU_INFO);

      expect(result.gpu).toEqual(MOCK_GPU_INFO);
      expect(result.systemInfo).toBeDefined();
    });

    it("should handle missing GPU info", () => {
      const result = systemInfoToLocalConfig(MOCK_SYSTEM_INFO, null);

      // Should fall back to GPU from system info
      expect(result.gpu).toEqual(MOCK_SYSTEM_INFO.gpu);
    });
  });

  describe("DOMAIN_GPU_REQUIREMENTS", () => {
    it("should have requirements for voice-clone domain", () => {
      const req = DOMAIN_GPU_REQUIREMENTS["voice-clone"];

      expect(req).toBeDefined();
      expect(req.minVram).toBeGreaterThan(0);
      expect(req.recommendedVram).toBeGreaterThanOrEqual(req.minVram);
    });

    it("should have requirements for quant-trading domain", () => {
      const req = DOMAIN_GPU_REQUIREMENTS["quant-trading"];

      expect(req).toBeDefined();
      expect(req.minVram).toBeGreaterThan(0);
    });

    it("should have requirements for robotics domain", () => {
      const req = DOMAIN_GPU_REQUIREMENTS["robotics"];

      expect(req).toBeDefined();
      expect(req.minVram).toBeGreaterThan(0);
    });

    it("should have requirements for biotech domain", () => {
      const req = DOMAIN_GPU_REQUIREMENTS["biotech"];

      expect(req).toBeDefined();
      expect(req.minVram).toBeGreaterThan(0);
    });
  });

  describe("checkGpuMeetsDomain", () => {
    it("should return meets=true when GPU has enough VRAM", () => {
      const gpu = { ...MOCK_GPU_INFO, vram: 24 };
      const result = checkGpuMeetsDomain(gpu, "voice-clone");

      expect(result.meets).toBe(true);
      expect(result.warning).toBeUndefined();
    });

    it("should return meets=false when GPU has insufficient VRAM", () => {
      const gpu = { ...MOCK_GPU_INFO, vram: 1 };  // 1GB is below all minimums
      const result = checkGpuMeetsDomain(gpu, "biotech");  // biotech needs 16GB min

      expect(result.meets).toBe(false);
      expect(result.warning).toBeDefined();
    });

    it("should handle unknown domain gracefully", () => {
      const result = checkGpuMeetsDomain(MOCK_GPU_INFO, "unknown-domain");

      expect(result.meets).toBe(true);
    });

    it("should return meets=false when GPU is not available", () => {
      const gpu = { ...MOCK_GPU_INFO, available: false };
      const result = checkGpuMeetsDomain(gpu, "voice-clone");

      expect(result.meets).toBe(false);
    });
  });

  describe("formatGpuInfo", () => {
    it("should format GPU info as string", () => {
      const result = formatGpuInfo(MOCK_GPU_INFO);

      expect(result).toContain(MOCK_GPU_INFO.name);
      expect(result).toContain(String(MOCK_GPU_INFO.vram));
      expect(result).toContain("GB");
    });

    it("should include CUDA version if present", () => {
      const gpu = { ...MOCK_GPU_INFO, cudaVersion: "12.1" };
      const result = formatGpuInfo(gpu);

      expect(result).toContain("CUDA 12.1");
    });

    it("should handle GPU without CUDA version", () => {
      const gpu = { ...MOCK_GPU_INFO, cudaVersion: undefined };
      const result = formatGpuInfo(gpu);

      expect(result).not.toContain("CUDA");
    });
  });

  describe("getGpuRecommendation", () => {
    it("should recommend GPU for voice-clone domain", () => {
      const result = getGpuRecommendation("voice-clone");

      expect(result).toContain("24GB");
    });

    it("should recommend GPU for biotech domain", () => {
      const result = getGpuRecommendation("biotech");

      expect(result).toContain("48GB");
    });

    it("should return default recommendation for unknown domain", () => {
      const result = getGpuRecommendation("unknown");

      expect(result).toBeDefined();
      expect(typeof result).toBe("string");
    });
  });
});
