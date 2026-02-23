/**
 * Results Tests
 *
 * Tests for result types and repository functions.
 */

import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  generateResultId,
  generateMediaId,
  formatMetricValue,
  calculateImprovement,
  getResultPath,
  canEditResult,
  canViewResult,
  isResult,
  isModelMetadata,
  isDemoMetadata,
  isFindingMetadata,
  isComparisonMetadata,
  DEFAULT_RESULT_STATS,
  RESULT_TYPE_LABELS,
  Result,
  ResultMetric,
  ResultAuthor,
} from "@/lib/social/results/types";

// Mock the json-store module
vi.mock("@/lib/db/json-store", () => ({
  findOne: vi.fn(),
  findMany: vi.fn(),
  insertOne: vi.fn(),
  updateOne: vi.fn(),
  deleteOne: vi.fn(),
  deleteMany: vi.fn(),
  count: vi.fn(),
  exists: vi.fn(),
  findPaginated: vi.fn(),
}));

describe("Result Types", () => {
  describe("generateResultId", () => {
    it("should generate unique IDs", () => {
      const id1 = generateResultId();
      const id2 = generateResultId();
      expect(id1).not.toBe(id2);
      expect(id1).toMatch(/^res_[a-z0-9]+$/);
    });
  });

  describe("generateMediaId", () => {
    it("should generate unique media IDs", () => {
      const id1 = generateMediaId();
      const id2 = generateMediaId();
      expect(id1).not.toBe(id2);
      expect(id1).toMatch(/^med_[a-z0-9]+$/);
    });
  });

  describe("formatMetricValue", () => {
    it("should format percentage values", () => {
      const metric: ResultMetric = { name: "accuracy", value: 0.95, unit: "%" };
      expect(formatMetricValue(metric)).toBe("95.0%");
    });

    it("should format large numbers with K suffix", () => {
      const metric: ResultMetric = { name: "params", value: 1500, unit: "params" };
      expect(formatMetricValue(metric)).toBe("1.5K params");
    });

    it("should format millions with M suffix", () => {
      const metric: ResultMetric = { name: "params", value: 1500000 };
      expect(formatMetricValue(metric)).toBe("1.5M");
    });

    it("should format small integers without suffix", () => {
      const metric: ResultMetric = { name: "epochs", value: 100, unit: "epochs" };
      expect(formatMetricValue(metric)).toBe("100 epochs");
    });

    it("should format decimals to 2 places", () => {
      const metric: ResultMetric = { name: "loss", value: 0.123456 };
      expect(formatMetricValue(metric)).toBe("0.12");
    });
  });

  describe("calculateImprovement", () => {
    it("should calculate positive improvement for higher-is-better metric", () => {
      const metric: ResultMetric = {
        name: "MOS",
        value: 4.5,
        baseline: 4.0,
        higherIsBetter: true,
      };
      expect(calculateImprovement(metric)).toBeCloseTo(12.5);
    });

    it("should calculate negative improvement for lower-is-better metric", () => {
      const metric: ResultMetric = {
        name: "WER",
        value: 5,
        baseline: 10,
        higherIsBetter: false,
      };
      expect(calculateImprovement(metric)).toBeCloseTo(50);
    });

    it("should return null if no baseline", () => {
      const metric: ResultMetric = { name: "MOS", value: 4.5 };
      expect(calculateImprovement(metric)).toBeNull();
    });

    it("should return null if baseline is zero", () => {
      const metric: ResultMetric = { name: "MOS", value: 4.5, baseline: 0 };
      expect(calculateImprovement(metric)).toBeNull();
    });
  });

  describe("getResultPath", () => {
    it("should return correct path", () => {
      expect(getResultPath("res_123")).toBe("/results/res_123");
    });
  });

  describe("canEditResult", () => {
    const mockResult: Result = {
      id: "res_123",
      type: "finding",
      title: "Test",
      description: "Test",
      visibility: "public",
      status: "published",
      author: { id: "user_1", username: "test", displayName: "Test" },
      labId: "lab_1",
      media: [],
      tags: [],
      stats: DEFAULT_RESULT_STATS,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    it("should return true if user is author", () => {
      expect(canEditResult(mockResult, "user_1")).toBe(true);
    });

    it("should return false if user is not author", () => {
      expect(canEditResult(mockResult, "user_2")).toBe(false);
    });

    it("should return false if no user", () => {
      expect(canEditResult(mockResult, undefined)).toBe(false);
    });
  });

  describe("canViewResult", () => {
    const createResult = (
      visibility: "public" | "private" | "unlisted",
      status: "draft" | "published" | "archived"
    ): Result => ({
      id: "res_123",
      type: "finding",
      title: "Test",
      description: "Test",
      visibility,
      status,
      author: { id: "user_1", username: "test", displayName: "Test" },
      labId: "lab_1",
      media: [],
      tags: [],
      stats: DEFAULT_RESULT_STATS,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    });

    it("should allow viewing public published results", () => {
      expect(canViewResult(createResult("public", "published"))).toBe(true);
    });

    it("should allow viewing unlisted published results", () => {
      expect(canViewResult(createResult("unlisted", "published"))).toBe(true);
    });

    it("should not allow anonymous viewing of private results", () => {
      expect(canViewResult(createResult("private", "published"))).toBe(false);
    });

    it("should allow author to view private results", () => {
      expect(canViewResult(createResult("private", "published"), "user_1")).toBe(true);
    });
  });

  describe("isResult type guard", () => {
    it("should return true for valid result", () => {
      const result: Result = {
        id: "res_123",
        type: "finding",
        title: "Test",
        description: "Test",
        visibility: "public",
        status: "published",
        author: { id: "user_1", username: "test", displayName: "Test" },
        labId: "lab_1",
        media: [],
        tags: [],
        stats: DEFAULT_RESULT_STATS,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };
      expect(isResult(result)).toBe(true);
    });

    it("should return false for invalid object", () => {
      expect(isResult({})).toBe(false);
      expect(isResult(null)).toBe(false);
      expect(isResult(undefined)).toBe(false);
      expect(isResult({ id: "test" })).toBe(false);
    });
  });

  describe("Metadata type guards", () => {
    it("should identify model metadata", () => {
      expect(isModelMetadata({ modelName: "csm-1b" })).toBe(true);
      expect(isModelMetadata({ demoUrl: "http://example.com" })).toBe(false);
    });

    it("should identify demo metadata", () => {
      expect(isDemoMetadata({ demoUrl: "http://example.com", isInteractive: true })).toBe(true);
      expect(isDemoMetadata({ modelName: "csm-1b" })).toBe(false);
    });

    it("should identify finding metadata", () => {
      expect(isFindingMetadata({ keyInsight: "Test insight" })).toBe(true);
      expect(isFindingMetadata({ modelName: "csm-1b" })).toBe(false);
    });

    it("should identify comparison metadata", () => {
      expect(
        isComparisonMetadata({
          modelsCompared: ["a", "b"],
          criteria: ["speed"],
          summary: "A is faster",
        })
      ).toBe(true);
      expect(isComparisonMetadata({ keyInsight: "Test" })).toBe(false);
    });
  });

  describe("Constants", () => {
    it("should have correct result type labels", () => {
      expect(RESULT_TYPE_LABELS.model).toBe("Model");
      expect(RESULT_TYPE_LABELS.demo).toBe("Demo");
      expect(RESULT_TYPE_LABELS.finding).toBe("Finding");
      expect(RESULT_TYPE_LABELS.comparison).toBe("Comparison");
    });

    it("should have correct default stats", () => {
      expect(DEFAULT_RESULT_STATS).toEqual({
        likes: 0,
        comments: 0,
        shares: 0,
        views: 0,
        saves: 0,
      });
    });
  });
});
