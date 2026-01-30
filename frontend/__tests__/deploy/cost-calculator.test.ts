/**
 * Cost Calculator Tests
 *
 * Tests for the deployment cost estimation logic
 */

import { describe, it, expect } from "vitest";

// Pricing constants (matching CostCalculator.tsx)
const PRICING = {
  gpu: {
    none: 0,
    rtx4090: 0.44,
    a100: 1.89,
    "apple-silicon": 0,
  },
  storage: {
    local: 0,
    cloud: 0.023,
  },
  bandwidth: {
    free: 0,
    paid: 0.09,
  },
  platform: {
    local: 0,
    "vercel-free": 0,
    "vercel-pro": 20,
    railway: 5,
    runpod: 0,
    aws: 0,
  },
};

interface UsageConfig {
  hoursPerWeek: number;
  gpuType: keyof typeof PRICING.gpu;
  deploymentMethod: keyof typeof PRICING.platform;
  storageGB: number;
  teamSize: number;
}

interface CostBreakdown {
  compute: number;
  storage: number;
  bandwidth: number;
  platform: number;
  total: number;
  perUser: number;
}

// Cost calculation function (extracted from component logic)
function calculateCosts(config: UsageConfig): CostBreakdown {
  const weeksPerMonth = 4.33;
  const hoursPerMonth = config.hoursPerWeek * weeksPerMonth;

  let compute = 0;
  if (config.deploymentMethod === "runpod" || config.deploymentMethod === "aws") {
    compute = hoursPerMonth * PRICING.gpu[config.gpuType];
  }

  let storage = 0;
  if (config.deploymentMethod !== "local") {
    storage = config.storageGB * PRICING.storage.cloud;
  }

  let bandwidth = 0;
  if (config.deploymentMethod === "vercel-pro" || config.deploymentMethod === "aws") {
    bandwidth = 10 * config.teamSize * PRICING.bandwidth.paid;
  }

  const platform = PRICING.platform[config.deploymentMethod] || 0;
  const total = compute + storage + bandwidth + platform;
  const perUser = config.teamSize > 0 ? total / config.teamSize : 0;

  return { compute, storage, bandwidth, platform, total, perUser };
}

describe("Cost Calculator", () => {
  describe("Local Deployment", () => {
    it("should be free for local deployment", () => {
      const config: UsageConfig = {
        hoursPerWeek: 40,
        gpuType: "none",
        deploymentMethod: "local",
        storageGB: 50,
        teamSize: 1,
      };

      const costs = calculateCosts(config);

      expect(costs.compute).toBe(0);
      expect(costs.storage).toBe(0);
      expect(costs.bandwidth).toBe(0);
      expect(costs.platform).toBe(0);
      expect(costs.total).toBe(0);
    });

    it("should be free with Apple Silicon locally", () => {
      const config: UsageConfig = {
        hoursPerWeek: 168, // 24/7
        gpuType: "apple-silicon",
        deploymentMethod: "local",
        storageGB: 100,
        teamSize: 3,
      };

      const costs = calculateCosts(config);
      expect(costs.total).toBe(0);
    });
  });

  describe("Vercel Deployment", () => {
    it("should be free for Vercel free tier", () => {
      const config: UsageConfig = {
        hoursPerWeek: 20,
        gpuType: "none",
        deploymentMethod: "vercel-free",
        storageGB: 20,
        teamSize: 1,
      };

      const costs = calculateCosts(config);

      expect(costs.platform).toBe(0);
      expect(costs.storage).toBeCloseTo(20 * 0.023);
      expect(costs.bandwidth).toBe(0);
    });

    it("should include platform fee for Vercel Pro", () => {
      const config: UsageConfig = {
        hoursPerWeek: 20,
        gpuType: "none",
        deploymentMethod: "vercel-pro",
        storageGB: 50,
        teamSize: 5,
      };

      const costs = calculateCosts(config);

      expect(costs.platform).toBe(20);
      expect(costs.storage).toBeCloseTo(50 * 0.023);
      expect(costs.bandwidth).toBeCloseTo(10 * 5 * 0.09);
    });
  });

  describe("GPU Cloud (RunPod)", () => {
    it("should calculate RTX 4090 costs correctly", () => {
      const config: UsageConfig = {
        hoursPerWeek: 10,
        gpuType: "rtx4090",
        deploymentMethod: "runpod",
        storageGB: 30,
        teamSize: 1,
      };

      const costs = calculateCosts(config);
      const expectedCompute = 10 * 4.33 * 0.44;

      expect(costs.compute).toBeCloseTo(expectedCompute);
      expect(costs.platform).toBe(0);
      expect(costs.total).toBeCloseTo(expectedCompute + 30 * 0.023);
    });

    it("should calculate A100 costs correctly", () => {
      const config: UsageConfig = {
        hoursPerWeek: 40,
        gpuType: "a100",
        deploymentMethod: "runpod",
        storageGB: 100,
        teamSize: 2,
      };

      const costs = calculateCosts(config);
      const expectedCompute = 40 * 4.33 * 1.89;

      expect(costs.compute).toBeCloseTo(expectedCompute);
      expect(costs.perUser).toBeCloseTo(costs.total / 2);
    });

    it("should have no GPU cost for CPU-only", () => {
      const config: UsageConfig = {
        hoursPerWeek: 100,
        gpuType: "none",
        deploymentMethod: "runpod",
        storageGB: 50,
        teamSize: 1,
      };

      const costs = calculateCosts(config);
      expect(costs.compute).toBe(0);
    });
  });

  describe("Railway Deployment", () => {
    it("should include $5 base platform fee", () => {
      const config: UsageConfig = {
        hoursPerWeek: 0,
        gpuType: "none",
        deploymentMethod: "railway",
        storageGB: 10,
        teamSize: 1,
      };

      const costs = calculateCosts(config);

      expect(costs.platform).toBe(5);
      expect(costs.compute).toBe(0);
    });
  });

  describe("AWS Deployment", () => {
    it("should include bandwidth costs", () => {
      const config: UsageConfig = {
        hoursPerWeek: 20,
        gpuType: "rtx4090",
        deploymentMethod: "aws",
        storageGB: 100,
        teamSize: 10,
      };

      const costs = calculateCosts(config);

      expect(costs.bandwidth).toBeCloseTo(10 * 10 * 0.09);
      expect(costs.compute).toBeGreaterThan(0);
    });
  });

  describe("Team Size Calculations", () => {
    it("should calculate per-user cost correctly", () => {
      const config: UsageConfig = {
        hoursPerWeek: 40,
        gpuType: "rtx4090",
        deploymentMethod: "runpod",
        storageGB: 100,
        teamSize: 5,
      };

      const costs = calculateCosts(config);

      expect(costs.perUser).toBeCloseTo(costs.total / 5);
    });

    it("should handle single user", () => {
      const config: UsageConfig = {
        hoursPerWeek: 10,
        gpuType: "none",
        deploymentMethod: "vercel-pro",
        storageGB: 20,
        teamSize: 1,
      };

      const costs = calculateCosts(config);

      expect(costs.perUser).toBe(costs.total);
    });
  });

  describe("Edge Cases", () => {
    it("should handle zero hours", () => {
      const config: UsageConfig = {
        hoursPerWeek: 0,
        gpuType: "a100",
        deploymentMethod: "runpod",
        storageGB: 50,
        teamSize: 1,
      };

      const costs = calculateCosts(config);
      expect(costs.compute).toBe(0);
    });

    it("should handle 24/7 usage (168 hours)", () => {
      const config: UsageConfig = {
        hoursPerWeek: 168,
        gpuType: "rtx4090",
        deploymentMethod: "runpod",
        storageGB: 200,
        teamSize: 1,
      };

      const costs = calculateCosts(config);
      const expectedCompute = 168 * 4.33 * 0.44;

      expect(costs.compute).toBeCloseTo(expectedCompute);
      expect(costs.total).toBeGreaterThan(300); // ~$320/month for 24/7 4090
    });

    it("should handle large team", () => {
      const config: UsageConfig = {
        hoursPerWeek: 40,
        gpuType: "a100",
        deploymentMethod: "aws",
        storageGB: 500,
        teamSize: 20,
      };

      const costs = calculateCosts(config);

      expect(costs.bandwidth).toBeCloseTo(10 * 20 * 0.09);
      expect(costs.perUser).toBeLessThan(costs.total);
    });
  });

  describe("Realistic Scenarios", () => {
    it("should calculate development tier costs", () => {
      // Solo developer, local, no GPU
      const config: UsageConfig = {
        hoursPerWeek: 20,
        gpuType: "apple-silicon",
        deploymentMethod: "local",
        storageGB: 20,
        teamSize: 1,
      };

      const costs = calculateCosts(config);
      expect(costs.total).toBe(0);
    });

    it("should calculate light research tier costs", () => {
      // Small team, Vercel + occasional GPU
      const config: UsageConfig = {
        hoursPerWeek: 10,
        gpuType: "rtx4090",
        deploymentMethod: "runpod",
        storageGB: 50,
        teamSize: 2,
      };

      const costs = calculateCosts(config);

      // ~$20-30/month expected
      expect(costs.total).toBeGreaterThan(15);
      expect(costs.total).toBeLessThan(50);
    });

    it("should calculate heavy research tier costs", () => {
      // Team, Vercel Pro + heavy GPU
      const config: UsageConfig = {
        hoursPerWeek: 40,
        gpuType: "a100",
        deploymentMethod: "aws",
        storageGB: 200,
        teamSize: 5,
      };

      const costs = calculateCosts(config);

      // ~$350-400/month expected
      expect(costs.total).toBeGreaterThan(300);
      expect(costs.total).toBeLessThan(500);
    });
  });
});

describe("Pricing Constants", () => {
  it("should have correct GPU pricing", () => {
    expect(PRICING.gpu.rtx4090).toBe(0.44);
    expect(PRICING.gpu.a100).toBe(1.89);
    expect(PRICING.gpu.none).toBe(0);
    expect(PRICING.gpu["apple-silicon"]).toBe(0);
  });

  it("should have correct platform pricing", () => {
    expect(PRICING.platform.local).toBe(0);
    expect(PRICING.platform["vercel-free"]).toBe(0);
    expect(PRICING.platform["vercel-pro"]).toBe(20);
    expect(PRICING.platform.railway).toBe(5);
  });

  it("should have correct storage pricing", () => {
    expect(PRICING.storage.cloud).toBe(0.023);
    expect(PRICING.storage.local).toBe(0);
  });
});
