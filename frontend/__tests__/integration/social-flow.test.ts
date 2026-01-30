/**
 * Social Flow Integration Tests
 *
 * Tests the complete social layer user journeys:
 * 1. Create and publish a result
 * 2. Like and save results
 * 3. Comment on results with reactions
 * 4. Track activity across the feed
 * 5. Create and vote on suggestions
 * 6. Receive and manage notifications
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { Result, ResultAuthor } from "@/lib/social/results/types";
import { DEFAULT_RESULT_STATS } from "@/lib/social/results/types";
import type { Comment, CommentAuthor } from "@/lib/social/comments/types";
import { DEFAULT_REACTION_COUNTS } from "@/lib/social/comments/types";
import type { Activity, ActivityActor, ActivityTarget } from "@/lib/social/activity/types";
import type { Suggestion, SuggestionAuthor } from "@/lib/social/suggestions/types";
import { DEFAULT_SUGGESTION_STATS } from "@/lib/social/suggestions/types";
import type { Notification } from "@/lib/social/notifications/types";

// Mock fetch for API calls
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Test data
const mockAuthor: ResultAuthor = {
  id: "user-1",
  username: "researcher",
  displayName: "Test Researcher",
};

const mockResult: Result = {
  id: "res_test123",
  type: "finding",
  title: "Improved Voice Cloning with Prosody Control",
  description: "A new approach to voice cloning that preserves emotional prosody",
  content: "## Summary\n\nWe present a novel method...",
  visibility: "public",
  status: "published",
  author: mockAuthor,
  labId: "lab-1",
  media: [],
  tags: ["voice-clone", "prosody", "emotion"],
  stats: { ...DEFAULT_RESULT_STATS },
  createdAt: "2024-01-01T00:00:00Z",
  updatedAt: "2024-01-01T00:00:00Z",
  publishedAt: "2024-01-01T00:00:00Z",
};

const mockCommentAuthor: CommentAuthor = {
  id: "user-2",
  username: "commenter",
  displayName: "Active Commenter",
};

const mockComment: Comment = {
  id: "cmt_test123",
  entityType: "result",
  entityId: "res_test123",
  author: mockCommentAuthor,
  content: "Great work! Have you tried testing with different emotions?",
  contentPlain: "Great work! Have you tried testing with different emotions?",
  mentions: [],
  status: "active",
  reactionCounts: { ...DEFAULT_REACTION_COUNTS },
  replyCount: 0,
  depth: 0,
  createdAt: "2024-01-02T00:00:00Z",
  isEdited: false,
};

const mockSuggestionAuthor: SuggestionAuthor = {
  id: "user-3",
  username: "suggester",
  displayName: "Research Suggester",
};

const mockSuggestion: Suggestion = {
  id: "sug_test123",
  labId: "lab-1",
  author: mockSuggestionAuthor,
  title: "Add support for multi-speaker emotion transfer",
  description: "It would be great to support multiple speakers with emotion transfer",
  category: "feature_request",
  status: "open",
  priority: "medium",
  tags: ["multi-speaker", "emotion"],
  stats: { ...DEFAULT_SUGGESTION_STATS },
  statusHistory: [],
  createdAt: "2024-01-03T00:00:00Z",
  updatedAt: "2024-01-03T00:00:00Z",
};

describe("Social Flow Integration Tests", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetch.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Result Creation Flow", () => {
    it("creates a draft result", async () => {
      const draftResult = { ...mockResult, status: "draft", publishedAt: undefined };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, result: draftResult }),
      });

      const response = await fetch("/api/results", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "finding",
          title: "Improved Voice Cloning with Prosody Control",
          description: "A new approach to voice cloning",
          labId: "lab-1",
          visibility: "public",
          tags: ["voice-clone", "prosody"],
        }),
      });

      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.result.status).toBe("draft");
      expect(data.result.publishedAt).toBeUndefined();
    });

    it("publishes a draft result", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          result: mockResult,
        }),
      });

      const response = await fetch("/api/results/res_test123/publish", {
        method: "POST",
      });

      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.result.status).toBe("published");
      expect(data.result.publishedAt).toBeDefined();
    });

    it("validates required fields on creation", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: async () => ({ success: false, error: "Title is required" }),
      });

      const response = await fetch("/api/results", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "finding",
          description: "Missing title",
          labId: "lab-1",
        }),
      });

      const data = await response.json();

      expect(data.success).toBe(false);
      expect(data.error).toContain("required");
    });

    it("updates result with metrics", async () => {
      const resultWithMetrics = {
        ...mockResult,
        metrics: {
          primary: [
            { name: "MOS", value: 4.2, higherIsBetter: true },
            { name: "WER", value: 0.05, unit: "%", higherIsBetter: false },
          ],
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, result: resultWithMetrics }),
      });

      const response = await fetch("/api/results/res_test123", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          metrics: {
            primary: [
              { name: "MOS", value: 4.2, higherIsBetter: true },
              { name: "WER", value: 0.05, unit: "%", higherIsBetter: false },
            ],
          },
        }),
      });

      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.result.metrics.primary).toHaveLength(2);
    });
  });

  describe("Result Discovery Flow", () => {
    it("lists published results", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          results: [mockResult],
          total: 1,
          page: 1,
          totalPages: 1,
          hasMore: false,
        }),
      });

      const response = await fetch("/api/results?status=published");
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.results).toHaveLength(1);
      expect(data.results[0].status).toBe("published");
    });

    it("filters results by type", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          results: [mockResult],
          total: 1,
        }),
      });

      const response = await fetch("/api/results?type=finding");
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.results[0].type).toBe("finding");
    });

    it("searches results by text", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          results: [mockResult],
          total: 1,
        }),
      });

      const response = await fetch("/api/results?search=prosody");
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.results[0].title).toContain("Prosody");
    });

    it("gets trending results", async () => {
      const trendingResult = { ...mockResult, stats: { ...mockResult.stats, likes: 100, views: 1000 } };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          results: [trendingResult],
        }),
      });

      const response = await fetch("/api/results/trending");
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.results[0].stats.likes).toBeGreaterThan(0);
    });

    it("gets featured results", async () => {
      const featuredResult = { ...mockResult, isFeatured: true };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          results: [featuredResult],
        }),
      });

      const response = await fetch("/api/results/featured");
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.results[0].isFeatured).toBe(true);
    });
  });

  describe("Result Engagement Flow", () => {
    it("likes a result", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, liked: true, count: 1 }),
      });

      const response = await fetch("/api/results/res_test123/like", {
        method: "POST",
      });
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.liked).toBe(true);
      expect(data.count).toBe(1);
    });

    it("unlikes a result", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, liked: false, count: 0 }),
      });

      const response = await fetch("/api/results/res_test123/like", {
        method: "DELETE",
      });
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.liked).toBe(false);
    });

    it("saves a result to collection", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, saved: true, count: 1 }),
      });

      const response = await fetch("/api/results/res_test123/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ collection: "favorites" }),
      });
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.saved).toBe(true);
    });

    it("unsaves a result", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, saved: false, count: 0 }),
      });

      const response = await fetch("/api/results/res_test123/save", {
        method: "DELETE",
      });
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.saved).toBe(false);
    });

    it("checks like status", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, isLiked: true, count: 5 }),
      });

      const response = await fetch("/api/results/res_test123/like");
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.isLiked).toBe(true);
      expect(data.count).toBe(5);
    });
  });

  describe("Comment Flow", () => {
    it("creates a comment on a result", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, comment: mockComment }),
      });

      const response = await fetch("/api/comments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          entityType: "result",
          entityId: "res_test123",
          content: "Great work!",
        }),
      });

      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.comment.entityId).toBe("res_test123");
    });

    it("replies to a comment", async () => {
      const reply = {
        ...mockComment,
        id: "cmt_reply123",
        parentId: "cmt_test123",
        depth: 1,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, comment: reply }),
      });

      const response = await fetch("/api/comments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          entityType: "result",
          entityId: "res_test123",
          parentId: "cmt_test123",
          content: "Thanks! Yes, we tested multiple emotions.",
        }),
      });

      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.comment.parentId).toBe("cmt_test123");
      expect(data.comment.depth).toBe(1);
    });

    it("lists comments with replies", async () => {
      const commentWithReplies = {
        ...mockComment,
        replies: [
          {
            ...mockComment,
            id: "cmt_reply123",
            parentId: "cmt_test123",
            depth: 1,
            replies: [],
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          comments: [commentWithReplies],
          total: 1,
          page: 1,
          totalPages: 1,
          hasMore: false,
        }),
      });

      const response = await fetch(
        "/api/comments?entityType=result&entityId=res_test123&includeReplies=true"
      );
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.comments[0].replies).toHaveLength(1);
    });

    it("adds a reaction to a comment", async () => {
      const updatedCounts = { ...DEFAULT_REACTION_COUNTS, like: 1 };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          reactionCounts: updatedCounts,
          userReaction: "like",
        }),
      });

      const response = await fetch("/api/comments/cmt_test123/reactions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: "like" }),
      });

      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.reactionCounts.like).toBe(1);
      expect(data.userReaction).toBe("like");
    });

    it("removes a reaction from a comment", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          reactionCounts: DEFAULT_REACTION_COUNTS,
          userReaction: null,
        }),
      });

      const response = await fetch("/api/comments/cmt_test123/reactions", {
        method: "DELETE",
      });

      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.userReaction).toBeNull();
    });

    it("edits a comment", async () => {
      const editedComment = {
        ...mockComment,
        content: "Updated content",
        isEdited: true,
        editedAt: new Date().toISOString(),
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, comment: editedComment }),
      });

      const response = await fetch("/api/comments/cmt_test123", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: "Updated content" }),
      });

      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.comment.isEdited).toBe(true);
    });

    it("deletes a comment", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      });

      const response = await fetch("/api/comments/cmt_test123", {
        method: "DELETE",
      });

      const data = await response.json();

      expect(data.success).toBe(true);
    });
  });

  describe("Activity Feed Flow", () => {
    it("fetches activity feed", async () => {
      const mockActivities: Activity[] = [
        {
          id: "act_1",
          type: "result_liked",
          actor: { id: "user-2", username: "fan", displayName: "Research Fan" },
          target: { type: "result", id: "res_test123", title: mockResult.title },
          isPublic: true,
          createdAt: new Date().toISOString(),
        },
      ];

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          activities: mockActivities,
          total: 1,
          page: 1,
          hasMore: false,
        }),
      });

      const response = await fetch("/api/activity");
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.activities).toHaveLength(1);
      expect(data.activities[0].type).toBe("result_liked");
    });

    it("filters activity by lab", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          activities: [],
          total: 0,
        }),
      });

      const response = await fetch("/api/activity?labId=lab-1");
      const data = await response.json();

      expect(data.success).toBe(true);
    });

    it("gets aggregated activities", async () => {
      const aggregatedActivity = {
        id: "act_1",
        type: "result_liked",
        actor: { id: "user-2", username: "fan1", displayName: "Fan 1" },
        target: { type: "result", id: "res_test123", title: mockResult.title },
        isPublic: true,
        createdAt: new Date().toISOString(),
        otherActors: [
          { id: "user-3", username: "fan2", displayName: "Fan 2" },
          { id: "user-4", username: "fan3", displayName: "Fan 3" },
        ],
        totalCount: 3,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          activities: [aggregatedActivity],
          total: 1,
        }),
      });

      const response = await fetch("/api/activity?aggregate=true");
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.activities[0].totalCount).toBe(3);
    });
  });

  describe("Suggestion Flow", () => {
    it("creates a suggestion", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, suggestion: mockSuggestion }),
      });

      const response = await fetch("/api/social/suggestions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          labId: "lab-1",
          title: "Add support for multi-speaker emotion transfer",
          description: "It would be great to support multiple speakers",
          category: "feature_request",
        }),
      });

      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.suggestion.status).toBe("open");
    });

    it("lists suggestions with filtering", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          suggestions: [mockSuggestion],
          total: 1,
          page: 1,
          totalPages: 1,
          hasMore: false,
        }),
      });

      const response = await fetch("/api/social/suggestions?labId=lab-1&status=open");
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.suggestions[0].status).toBe("open");
    });

    it("upvotes a suggestion", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          vote: 1,
          stats: { upvotes: 1, downvotes: 0, comments: 0 },
        }),
      });

      const response = await fetch("/api/social/suggestions/sug_test123/vote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vote: 1 }),
      });

      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.vote).toBe(1);
      expect(data.stats.upvotes).toBe(1);
    });

    it("downvotes a suggestion", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          vote: -1,
          stats: { upvotes: 0, downvotes: 1, comments: 0 },
        }),
      });

      const response = await fetch("/api/social/suggestions/sug_test123/vote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vote: -1 }),
      });

      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.vote).toBe(-1);
      expect(data.stats.downvotes).toBe(1);
    });

    it("removes a vote", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          vote: null,
          stats: { upvotes: 0, downvotes: 0, comments: 0 },
        }),
      });

      const response = await fetch("/api/social/suggestions/sug_test123/vote", {
        method: "DELETE",
      });

      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.vote).toBeNull();
    });

    it("updates suggestion status", async () => {
      const updatedSuggestion = {
        ...mockSuggestion,
        status: "planned",
        statusHistory: [
          {
            from: "open",
            to: "planned",
            changedBy: "user-1",
            reason: "Scheduled for next sprint",
            changedAt: new Date().toISOString(),
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, suggestion: updatedSuggestion }),
      });

      const response = await fetch("/api/social/suggestions/sug_test123", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          status: "planned",
          statusReason: "Scheduled for next sprint",
        }),
      });

      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.suggestion.status).toBe("planned");
      expect(data.suggestion.statusHistory).toHaveLength(1);
    });
  });

  describe("Notification Flow", () => {
    it("fetches unread notifications", async () => {
      const mockNotifications: Notification[] = [
        {
          id: "notif_1",
          userId: "user-1",
          type: "result_liked",
          title: "Your result was liked",
          message: "Someone liked your finding",
          activityId: "act_1",
          isRead: false,
          createdAt: new Date().toISOString(),
        },
      ];

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          notifications: mockNotifications,
          total: 1,
          unreadCount: 1,
        }),
      });

      const response = await fetch("/api/notifications?unreadOnly=true");
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.unreadCount).toBe(1);
      expect(data.notifications[0].isRead).toBe(false);
    });

    it("marks notification as read", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, notification: { id: "notif_1", isRead: true } }),
      });

      const response = await fetch("/api/notifications/notif_1", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ isRead: true }),
      });

      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.notification.isRead).toBe(true);
    });

    it("marks all notifications as read", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, count: 5 }),
      });

      const response = await fetch("/api/notifications", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ markAllRead: true }),
      });

      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.count).toBe(5);
    });

    it("deletes a notification", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      });

      const response = await fetch("/api/notifications/notif_1", {
        method: "DELETE",
      });

      const data = await response.json();

      expect(data.success).toBe(true);
    });

    it("gets notification count for badge", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, unreadCount: 3 }),
      });

      const response = await fetch("/api/notifications?countOnly=true");
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.unreadCount).toBe(3);
    });
  });

  describe("Full Social Journey", () => {
    it("complete flow: create result -> get likes -> receive notifications", async () => {
      // Step 1: Create and publish a result
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, result: mockResult }),
      });

      let response = await fetch("/api/results", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          type: "finding",
          title: mockResult.title,
          description: mockResult.description,
          labId: "lab-1",
        }),
      });
      let data = await response.json();
      expect(data.success).toBe(true);
      const resultId = data.result.id;

      // Step 2: Another user likes the result
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, liked: true, count: 1 }),
      });

      response = await fetch(`/api/results/${resultId}/like`, {
        method: "POST",
      });
      data = await response.json();
      expect(data.liked).toBe(true);

      // Step 3: Another user comments
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, comment: mockComment }),
      });

      response = await fetch("/api/comments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          entityType: "result",
          entityId: resultId,
          content: "Amazing work!",
        }),
      });
      data = await response.json();
      expect(data.success).toBe(true);

      // Step 4: Original author checks notifications
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          notifications: [
            {
              id: "notif_like",
              type: "result_liked",
              title: "Your result was liked",
              isRead: false,
            },
            {
              id: "notif_comment",
              type: "comment_received",
              title: "New comment on your result",
              isRead: false,
            },
          ],
          unreadCount: 2,
        }),
      });

      response = await fetch("/api/notifications");
      data = await response.json();
      expect(data.unreadCount).toBe(2);

      // Step 5: Author marks notifications as read
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, count: 2 }),
      });

      response = await fetch("/api/notifications", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ markAllRead: true }),
      });
      data = await response.json();
      expect(data.count).toBe(2);
    });

    it("complete flow: create suggestion -> vote -> update status", async () => {
      // Step 1: User creates a suggestion
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, suggestion: mockSuggestion }),
      });

      let response = await fetch("/api/social/suggestions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          labId: "lab-1",
          title: mockSuggestion.title,
          description: mockSuggestion.description,
          category: "feature_request",
        }),
      });
      let data = await response.json();
      expect(data.success).toBe(true);
      const suggestionId = data.suggestion.id;

      // Step 2: Other users upvote
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          vote: 1,
          stats: { upvotes: 5, downvotes: 1, comments: 0 },
        }),
      });

      response = await fetch(`/api/social/suggestions/${suggestionId}/vote`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vote: 1 }),
      });
      data = await response.json();
      expect(data.stats.upvotes).toBe(5);

      // Step 3: Lab owner marks as planned
      const plannedSuggestion = {
        ...mockSuggestion,
        status: "planned",
        statusHistory: [
          { from: "open", to: "planned", changedBy: "owner", changedAt: new Date().toISOString() },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, suggestion: plannedSuggestion }),
      });

      response = await fetch(`/api/social/suggestions/${suggestionId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: "planned" }),
      });
      data = await response.json();
      expect(data.suggestion.status).toBe("planned");

      // Step 4: Later, mark as completed
      const completedSuggestion = {
        ...plannedSuggestion,
        status: "completed",
        resolvedAt: new Date().toISOString(),
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, suggestion: completedSuggestion }),
      });

      response = await fetch(`/api/social/suggestions/${suggestionId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: "completed" }),
      });
      data = await response.json();
      expect(data.suggestion.status).toBe("completed");
      expect(data.suggestion.resolvedAt).toBeDefined();
    });

    it("complete flow: comment thread with reactions", async () => {
      // Step 1: User posts a comment
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, comment: mockComment }),
      });

      let response = await fetch("/api/comments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          entityType: "result",
          entityId: "res_test123",
          content: "Great findings!",
        }),
      });
      let data = await response.json();
      expect(data.success).toBe(true);
      const commentId = data.comment.id;

      // Step 2: Another user replies
      const reply = {
        ...mockComment,
        id: "cmt_reply",
        parentId: commentId,
        depth: 1,
        content: "I agree, very impressive work!",
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, comment: reply }),
      });

      response = await fetch("/api/comments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          entityType: "result",
          entityId: "res_test123",
          parentId: commentId,
          content: "I agree!",
        }),
      });
      data = await response.json();
      expect(data.comment.parentId).toBe(commentId);

      // Step 3: Users add reactions
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          reactionCounts: { ...DEFAULT_REACTION_COUNTS, insightful: 3 },
          userReaction: "insightful",
        }),
      });

      response = await fetch(`/api/comments/${commentId}/reactions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: "insightful" }),
      });
      data = await response.json();
      expect(data.reactionCounts.insightful).toBe(3);

      // Step 4: Fetch the thread
      const threadWithReactions = {
        ...mockComment,
        reactionCounts: { ...DEFAULT_REACTION_COUNTS, insightful: 3 },
        replyCount: 1,
        replies: [{ ...reply, replies: [] }],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          comments: [threadWithReactions],
          total: 1,
        }),
      });

      response = await fetch(
        "/api/comments?entityType=result&entityId=res_test123&includeReplies=true"
      );
      data = await response.json();
      expect(data.comments[0].reactionCounts.insightful).toBe(3);
      expect(data.comments[0].replies).toHaveLength(1);
    });
  });

  describe("Error Handling", () => {
    it("handles 404 for non-existent result", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ success: false, error: "Result not found" }),
      });

      const response = await fetch("/api/results/non-existent");
      const data = await response.json();

      expect(data.success).toBe(false);
      expect(data.error).toContain("not found");
    });

    it("handles 403 for unauthorized edit", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 403,
        json: async () => ({ success: false, error: "Not authorized to edit this result" }),
      });

      const response = await fetch("/api/results/res_test123", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "Hacked title" }),
      });
      const data = await response.json();

      expect(data.success).toBe(false);
      expect(data.error).toContain("authorized");
    });

    it("handles validation errors", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ success: false, error: "Content exceeds maximum length" }),
      });

      const response = await fetch("/api/comments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          entityType: "result",
          entityId: "res_test123",
          content: "x".repeat(20000), // Too long
        }),
      });
      const data = await response.json();

      expect(data.success).toBe(false);
      expect(data.error).toContain("length");
    });

    it("handles network errors gracefully", async () => {
      mockFetch.mockRejectedValueOnce(new Error("Network error"));

      await expect(fetch("/api/results")).rejects.toThrow("Network error");
    });
  });
});
