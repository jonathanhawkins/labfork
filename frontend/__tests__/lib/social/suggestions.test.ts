/**
 * Suggestions Tests
 *
 * Tests for suggestion types and utilities.
 */

import { describe, it, expect, vi } from "vitest";
import {
  generateSuggestionId,
  getNetVotes,
  isClosedStatus,
  canEditSuggestion,
  isSuggestion,
  DEFAULT_SUGGESTION_STATS,
  CATEGORY_LABELS,
  STATUS_LABELS,
  STATUS_COLORS,
  PRIORITY_LABELS,
  Suggestion,
  SuggestionStatus,
} from "@/lib/social/suggestions/types";

// Mock the json-store module
vi.mock("@/lib/db/json-store", () => ({
  findOne: vi.fn(),
  findMany: vi.fn(),
  insertOne: vi.fn(),
  updateOne: vi.fn(),
  deleteOne: vi.fn(),
  count: vi.fn(),
  exists: vi.fn(),
  findPaginated: vi.fn(),
}));

describe("Suggestion Types", () => {
  describe("generateSuggestionId", () => {
    it("should generate unique IDs", () => {
      const id1 = generateSuggestionId();
      const id2 = generateSuggestionId();
      expect(id1).not.toBe(id2);
      expect(id1).toMatch(/^sug_[a-z0-9]+$/);
    });
  });

  describe("getNetVotes", () => {
    it("should calculate net votes correctly", () => {
      expect(getNetVotes({ upvotes: 10, downvotes: 3, comments: 0 })).toBe(7);
      expect(getNetVotes({ upvotes: 5, downvotes: 8, comments: 0 })).toBe(-3);
      expect(getNetVotes({ upvotes: 5, downvotes: 5, comments: 0 })).toBe(0);
    });
  });

  describe("isClosedStatus", () => {
    it("should return true for closed statuses", () => {
      expect(isClosedStatus("completed")).toBe(true);
      expect(isClosedStatus("declined")).toBe(true);
      expect(isClosedStatus("duplicate")).toBe(true);
    });

    it("should return false for open statuses", () => {
      expect(isClosedStatus("open")).toBe(false);
      expect(isClosedStatus("under_review")).toBe(false);
      expect(isClosedStatus("planned")).toBe(false);
      expect(isClosedStatus("in_progress")).toBe(false);
    });
  });

  describe("canEditSuggestion", () => {
    const createSuggestion = (authorId: string, status: SuggestionStatus): Suggestion => ({
      id: "sug_123",
      labId: "lab_1",
      author: { id: authorId, username: "test", displayName: "Test" },
      title: "Test Suggestion",
      description: "Test description",
      category: "feature_request",
      status,
      priority: "medium",
      tags: [],
      stats: DEFAULT_SUGGESTION_STATS,
      statusHistory: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    });

    it("should return true if user is author and suggestion is open", () => {
      expect(canEditSuggestion(createSuggestion("user_1", "open"), "user_1")).toBe(true);
    });

    it("should return false if user is not author", () => {
      expect(canEditSuggestion(createSuggestion("user_1", "open"), "user_2")).toBe(false);
    });

    it("should return false if suggestion is closed", () => {
      expect(canEditSuggestion(createSuggestion("user_1", "completed"), "user_1")).toBe(false);
      expect(canEditSuggestion(createSuggestion("user_1", "declined"), "user_1")).toBe(false);
    });

    it("should return false if no user", () => {
      expect(canEditSuggestion(createSuggestion("user_1", "open"), undefined)).toBe(false);
    });
  });

  describe("isSuggestion type guard", () => {
    it("should return true for valid suggestion", () => {
      const suggestion: Suggestion = {
        id: "sug_123",
        labId: "lab_1",
        author: { id: "user_1", username: "test", displayName: "Test" },
        title: "Test Suggestion",
        description: "Test description",
        category: "feature_request",
        status: "open",
        priority: "medium",
        tags: [],
        stats: DEFAULT_SUGGESTION_STATS,
        statusHistory: [],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };
      expect(isSuggestion(suggestion)).toBe(true);
    });

    it("should return false for invalid object", () => {
      expect(isSuggestion({})).toBe(false);
      expect(isSuggestion(null)).toBe(false);
      expect(isSuggestion({ id: "test" })).toBe(false);
    });
  });

  describe("Constants", () => {
    it("should have correct category labels", () => {
      expect(CATEGORY_LABELS.research_direction).toBe("Research Direction");
      expect(CATEGORY_LABELS.improvement).toBe("Improvement");
      expect(CATEGORY_LABELS.bug_report).toBe("Bug Report");
      expect(CATEGORY_LABELS.feature_request).toBe("Feature Request");
    });

    it("should have correct status labels", () => {
      expect(STATUS_LABELS.open).toBe("Open");
      expect(STATUS_LABELS.under_review).toBe("Under Review");
      expect(STATUS_LABELS.completed).toBe("Completed");
    });

    it("should have status colors", () => {
      expect(STATUS_COLORS.open).toBeDefined();
      expect(STATUS_COLORS.completed).toBeDefined();
    });

    it("should have correct priority labels", () => {
      expect(PRIORITY_LABELS.low).toBe("Low");
      expect(PRIORITY_LABELS.medium).toBe("Medium");
      expect(PRIORITY_LABELS.high).toBe("High");
      expect(PRIORITY_LABELS.critical).toBe("Critical");
    });

    it("should have correct default stats", () => {
      expect(DEFAULT_SUGGESTION_STATS).toEqual({
        upvotes: 0,
        downvotes: 0,
        comments: 0,
      });
    });
  });
});
