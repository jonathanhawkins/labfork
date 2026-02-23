/**
 * Comments Tests
 *
 * Tests for comment types and utilities.
 */

import { describe, it, expect, vi } from "vitest";
import {
  generateCommentId,
  extractMentions,
  stripMarkdown,
  getTotalReactions,
  canEditComment,
  canDeleteComment,
  isComment,
  DEFAULT_REACTION_COUNTS,
  REACTION_LABELS,
  REACTION_EMOJIS,
  MAX_COMMENT_LENGTH,
  MAX_REPLY_DEPTH,
  Comment,
  ReactionCounts,
} from "@/lib/social/comments/types";

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

describe("Comment Types", () => {
  describe("generateCommentId", () => {
    it("should generate unique IDs", () => {
      const id1 = generateCommentId();
      const id2 = generateCommentId();
      expect(id1).not.toBe(id2);
      expect(id1).toMatch(/^cmt_[a-z0-9]+$/);
    });
  });

  describe("extractMentions", () => {
    it("should extract single mention", () => {
      const mentions = extractMentions("Hello @john");
      expect(mentions).toHaveLength(1);
      expect(mentions[0].username).toBe("john");
      expect(mentions[0].startIndex).toBe(6);
    });

    it("should extract multiple mentions", () => {
      const mentions = extractMentions("Thanks @alice and @bob");
      expect(mentions).toHaveLength(2);
      expect(mentions[0].username).toBe("alice");
      expect(mentions[1].username).toBe("bob");
    });

    it("should handle no mentions", () => {
      const mentions = extractMentions("Hello world");
      expect(mentions).toHaveLength(0);
    });

    it("should handle mentions with hyphens and underscores", () => {
      const mentions = extractMentions("Hi @john-doe and @jane_doe");
      expect(mentions).toHaveLength(2);
      expect(mentions[0].username).toBe("john-doe");
      expect(mentions[1].username).toBe("jane_doe");
    });
  });

  describe("stripMarkdown", () => {
    it("should remove markdown formatting", () => {
      expect(stripMarkdown("**bold** text")).toBe("bold text");
      expect(stripMarkdown("*italic* text")).toBe("italic text");
      expect(stripMarkdown("`code` text")).toBe("code text");
    });

    it("should convert links to text", () => {
      expect(stripMarkdown("[link](http://example.com)")).toBe("link");
    });

    it("should remove images", () => {
      expect(stripMarkdown("![alt](http://example.com/img.png)")).toBe("");
    });

    it("should collapse newlines", () => {
      expect(stripMarkdown("line1\n\nline2")).toBe("line1 line2");
    });
  });

  describe("getTotalReactions", () => {
    it("should sum all reaction counts", () => {
      const counts: ReactionCounts = {
        like: 5,
        love: 3,
        insightful: 2,
        celebrate: 1,
        curious: 1,
        disagree: 0,
      };
      expect(getTotalReactions(counts)).toBe(12);
    });

    it("should return 0 for empty counts", () => {
      expect(getTotalReactions(DEFAULT_REACTION_COUNTS)).toBe(0);
    });
  });

  describe("canEditComment", () => {
    const createComment = (authorId: string, status: "active" | "deleted" | "hidden"): Comment => ({
      id: "cmt_123",
      entityType: "result",
      entityId: "res_123",
      author: { id: authorId, username: "test", displayName: "Test" },
      content: "Test comment",
      contentPlain: "Test comment",
      mentions: [],
      status,
      reactionCounts: DEFAULT_REACTION_COUNTS,
      replyCount: 0,
      depth: 0,
      createdAt: new Date().toISOString(),
      isEdited: false,
    });

    it("should return true if user is author and comment is active", () => {
      expect(canEditComment(createComment("user_1", "active"), "user_1")).toBe(true);
    });

    it("should return false if user is not author", () => {
      expect(canEditComment(createComment("user_1", "active"), "user_2")).toBe(false);
    });

    it("should return false if comment is deleted", () => {
      expect(canEditComment(createComment("user_1", "deleted"), "user_1")).toBe(false);
    });

    it("should return false if no user", () => {
      expect(canEditComment(createComment("user_1", "active"), undefined)).toBe(false);
    });
  });

  describe("canDeleteComment", () => {
    const createComment = (authorId: string, status: "active" | "deleted" | "hidden"): Comment => ({
      id: "cmt_123",
      entityType: "result",
      entityId: "res_123",
      author: { id: authorId, username: "test", displayName: "Test" },
      content: "Test comment",
      contentPlain: "Test comment",
      mentions: [],
      status,
      reactionCounts: DEFAULT_REACTION_COUNTS,
      replyCount: 0,
      depth: 0,
      createdAt: new Date().toISOString(),
      isEdited: false,
    });

    it("should return true if user is author and comment is not deleted", () => {
      expect(canDeleteComment(createComment("user_1", "active"), "user_1")).toBe(true);
    });

    it("should return false if comment is already deleted", () => {
      expect(canDeleteComment(createComment("user_1", "deleted"), "user_1")).toBe(false);
    });
  });

  describe("isComment type guard", () => {
    it("should return true for valid comment", () => {
      const comment: Comment = {
        id: "cmt_123",
        entityType: "result",
        entityId: "res_123",
        author: { id: "user_1", username: "test", displayName: "Test" },
        content: "Test comment",
        contentPlain: "Test comment",
        mentions: [],
        status: "active",
        reactionCounts: DEFAULT_REACTION_COUNTS,
        replyCount: 0,
        depth: 0,
        createdAt: new Date().toISOString(),
        isEdited: false,
      };
      expect(isComment(comment)).toBe(true);
    });

    it("should return false for invalid object", () => {
      expect(isComment({})).toBe(false);
      expect(isComment(null)).toBe(false);
      expect(isComment({ id: "test" })).toBe(false);
    });
  });

  describe("Constants", () => {
    it("should have correct reaction labels", () => {
      expect(REACTION_LABELS.like).toBe("Like");
      expect(REACTION_LABELS.love).toBe("Love");
      expect(REACTION_LABELS.insightful).toBe("Insightful");
    });

    it("should have reaction emojis", () => {
      expect(REACTION_EMOJIS.like).toBeDefined();
      expect(REACTION_EMOJIS.love).toBeDefined();
    });

    it("should have correct max values", () => {
      expect(MAX_COMMENT_LENGTH).toBe(10000);
      expect(MAX_REPLY_DEPTH).toBe(5);
    });

    it("should have correct default reaction counts", () => {
      expect(DEFAULT_REACTION_COUNTS).toEqual({
        like: 0,
        love: 0,
        insightful: 0,
        celebrate: 0,
        curious: 0,
        disagree: 0,
      });
    });
  });
});
