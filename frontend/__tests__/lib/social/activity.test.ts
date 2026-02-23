/**
 * Activity Tests
 *
 * Tests for activity types and utilities.
 */

import { describe, it, expect, vi } from "vitest";
import {
  generateActivityId,
  getActivityDescription,
  getAggregatedDescription,
  formatActivityTime,
  groupActivitiesByDate,
  getDateGroupLabel,
  shouldNotify,
  isActivity,
  ACTIVITY_TYPE_LABELS,
  ACTIVITY_TYPE_ICONS,
  Activity,
  AggregatedActivity,
  ActivityActor,
  ActivityTarget,
} from "@/lib/social/activity/types";

// Mock the json-store module
vi.mock("@/lib/db/json-store", () => ({
  findMany: vi.fn(),
  insertOne: vi.fn(),
  findPaginated: vi.fn(),
  count: vi.fn(),
}));

describe("Activity Types", () => {
  const mockActor: ActivityActor = {
    id: "user_1",
    username: "testuser",
    displayName: "Test User",
  };

  const mockTarget: ActivityTarget = {
    type: "result",
    id: "res_123",
    title: "Test Result",
    url: "/results/res_123",
  };

  describe("generateActivityId", () => {
    it("should generate unique IDs", () => {
      const id1 = generateActivityId();
      const id2 = generateActivityId();
      expect(id1).not.toBe(id2);
      expect(id1).toMatch(/^act_[a-z0-9]+$/);
    });
  });

  describe("getActivityDescription", () => {
    it("should format activity description correctly", () => {
      const activity: Activity = {
        id: "act_123",
        type: "result_liked",
        actor: mockActor,
        target: mockTarget,
        isPublic: true,
        createdAt: new Date().toISOString(),
      };
      const description = getActivityDescription(activity);
      expect(description).toBe('Test User liked a result "Test Result"');
    });
  });

  describe("getAggregatedDescription", () => {
    it("should format single activity", () => {
      const activity: AggregatedActivity = {
        id: "act_123",
        type: "result_liked",
        actor: mockActor,
        target: mockTarget,
        isPublic: true,
        createdAt: new Date().toISOString(),
      };
      const description = getAggregatedDescription(activity);
      expect(description).toBe('Test User liked a result "Test Result"');
    });

    it("should format aggregated activity with multiple actors", () => {
      const activity: AggregatedActivity = {
        id: "act_123",
        type: "result_liked",
        actor: mockActor,
        target: mockTarget,
        isPublic: true,
        createdAt: new Date().toISOString(),
        otherActors: [
          { id: "user_2", username: "user2", displayName: "User 2" },
          { id: "user_3", username: "user3", displayName: "User 3" },
        ],
        totalCount: 3,
      };
      const description = getAggregatedDescription(activity);
      expect(description).toBe('Test User and 2 others liked a result "Test Result"');
    });

    it("should handle singular 'other'", () => {
      const activity: AggregatedActivity = {
        id: "act_123",
        type: "result_liked",
        actor: mockActor,
        target: mockTarget,
        isPublic: true,
        createdAt: new Date().toISOString(),
        otherActors: [{ id: "user_2", username: "user2", displayName: "User 2" }],
        totalCount: 2,
      };
      const description = getAggregatedDescription(activity);
      expect(description).toBe('Test User and 1 other liked a result "Test Result"');
    });
  });

  describe("formatActivityTime", () => {
    it("should format 'just now' for very recent activities", () => {
      const now = new Date().toISOString();
      expect(formatActivityTime(now)).toBe("just now");
    });

    it("should format minutes ago", () => {
      const tenMinutesAgo = new Date(Date.now() - 10 * 60 * 1000).toISOString();
      expect(formatActivityTime(tenMinutesAgo)).toBe("10m ago");
    });

    it("should format hours ago", () => {
      const fiveHoursAgo = new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString();
      expect(formatActivityTime(fiveHoursAgo)).toBe("5h ago");
    });

    it("should format days ago", () => {
      const threeDaysAgo = new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString();
      expect(formatActivityTime(threeDaysAgo)).toBe("3d ago");
    });

    it("should format weeks ago", () => {
      const twoWeeksAgo = new Date(Date.now() - 14 * 24 * 60 * 60 * 1000).toISOString();
      expect(formatActivityTime(twoWeeksAgo)).toBe("2w ago");
    });
  });

  describe("groupActivitiesByDate", () => {
    it("should group activities by date", () => {
      const today = new Date().toISOString();
      const yesterday = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();

      const activities: Activity[] = [
        {
          id: "act_1",
          type: "result_liked",
          actor: mockActor,
          target: mockTarget,
          isPublic: true,
          createdAt: today,
        },
        {
          id: "act_2",
          type: "lab_starred",
          actor: mockActor,
          target: mockTarget,
          isPublic: true,
          createdAt: yesterday,
        },
      ];

      const groups = groupActivitiesByDate(activities);
      expect(groups.size).toBe(2);
    });
  });

  describe("getDateGroupLabel", () => {
    it("should return 'Today' for today", () => {
      const today = new Date().toISOString().split("T")[0];
      expect(getDateGroupLabel(today)).toBe("Today");
    });

    it("should return 'Yesterday' for yesterday", () => {
      const yesterday = new Date(Date.now() - 24 * 60 * 60 * 1000)
        .toISOString()
        .split("T")[0];
      expect(getDateGroupLabel(yesterday)).toBe("Yesterday");
    });
  });

  describe("shouldNotify", () => {
    it("should not notify user of their own actions", () => {
      const activity: Activity = {
        id: "act_123",
        type: "result_liked",
        actor: mockActor,
        target: mockTarget,
        isPublic: true,
        createdAt: new Date().toISOString(),
      };
      expect(shouldNotify(activity, "user_1")).toBe(false);
    });

    it("should notify for result_liked by another user", () => {
      const activity: Activity = {
        id: "act_123",
        type: "result_liked",
        actor: mockActor,
        target: mockTarget,
        isPublic: true,
        createdAt: new Date().toISOString(),
      };
      expect(shouldNotify(activity, "user_2")).toBe(true);
    });

    it("should notify for user_mentioned", () => {
      const activity: Activity = {
        id: "act_123",
        type: "user_mentioned",
        actor: mockActor,
        target: { type: "user", id: "user_2", title: "User 2" },
        isPublic: true,
        createdAt: new Date().toISOString(),
      };
      expect(shouldNotify(activity, "user_2")).toBe(true);
    });

    it("should not notify for result_created", () => {
      const activity: Activity = {
        id: "act_123",
        type: "result_created",
        actor: mockActor,
        target: mockTarget,
        isPublic: true,
        createdAt: new Date().toISOString(),
      };
      expect(shouldNotify(activity, "user_2")).toBe(false);
    });
  });

  describe("isActivity type guard", () => {
    it("should return true for valid activity", () => {
      const activity: Activity = {
        id: "act_123",
        type: "result_liked",
        actor: mockActor,
        target: mockTarget,
        isPublic: true,
        createdAt: new Date().toISOString(),
      };
      expect(isActivity(activity)).toBe(true);
    });

    it("should return false for invalid object", () => {
      expect(isActivity({})).toBe(false);
      expect(isActivity(null)).toBe(false);
      expect(isActivity({ id: "test" })).toBe(false);
    });
  });

  describe("Constants", () => {
    it("should have correct activity type labels", () => {
      expect(ACTIVITY_TYPE_LABELS.result_created).toBe("created a result");
      expect(ACTIVITY_TYPE_LABELS.result_liked).toBe("liked a result");
      expect(ACTIVITY_TYPE_LABELS.lab_starred).toBe("starred a lab");
    });

    it("should have activity type icons", () => {
      expect(ACTIVITY_TYPE_ICONS.result_liked).toBe("Heart");
      expect(ACTIVITY_TYPE_ICONS.lab_starred).toBe("Star");
    });
  });
});
