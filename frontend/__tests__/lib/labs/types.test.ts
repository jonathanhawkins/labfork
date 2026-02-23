import { describe, it, expect } from "vitest";
import {
  generateLabId,
  isValidLabSlug,
  slugify,
  getLabPath,
  getLabApiPath,
  formatLabStats,
  canEditLab,
  canViewLab,
  isLab,
  DEFAULT_LAB_STATS,
  type Lab,
  type LabStats,
} from "@/lib/labs/types";

describe("Lab Types", () => {
  describe("generateLabId", () => {
    it("generates unique IDs", () => {
      const id1 = generateLabId();
      const id2 = generateLabId();

      expect(id1).not.toBe(id2);
    });

    it("generates IDs with correct prefix", () => {
      const id = generateLabId();
      expect(id.startsWith("lab_")).toBe(true);
    });

    it("generates IDs of reasonable length", () => {
      const id = generateLabId();
      // lab_ + timestamp (8-10 chars) + random (6 chars)
      expect(id.length).toBeGreaterThan(10);
      expect(id.length).toBeLessThan(30);
    });
  });

  describe("isValidLabSlug", () => {
    it("accepts valid slugs", () => {
      expect(isValidLabSlug("my-lab")).toBe(true);
      expect(isValidLabSlug("voice-clone-v2")).toBe(true);
      expect(isValidLabSlug("lab123")).toBe(true);
      expect(isValidLabSlug("a1b")).toBe(true);
    });

    it("rejects slugs that are too short", () => {
      expect(isValidLabSlug("ab")).toBe(false);
      expect(isValidLabSlug("a")).toBe(false);
    });

    it("rejects slugs starting with hyphen", () => {
      expect(isValidLabSlug("-my-lab")).toBe(false);
    });

    it("rejects slugs ending with hyphen", () => {
      expect(isValidLabSlug("my-lab-")).toBe(false);
    });

    it("rejects slugs with uppercase", () => {
      expect(isValidLabSlug("My-Lab")).toBe(false);
      expect(isValidLabSlug("MYLAB")).toBe(false);
    });

    it("rejects slugs with special characters", () => {
      expect(isValidLabSlug("my_lab")).toBe(false);
      expect(isValidLabSlug("my.lab")).toBe(false);
      expect(isValidLabSlug("my@lab")).toBe(false);
    });

    it("rejects slugs with spaces", () => {
      expect(isValidLabSlug("my lab")).toBe(false);
    });
  });

  describe("slugify", () => {
    it("converts to lowercase", () => {
      expect(slugify("My Lab")).toBe("my-lab");
      expect(slugify("UPPERCASE")).toBe("uppercase");
    });

    it("replaces spaces with hyphens", () => {
      expect(slugify("my cool lab")).toBe("my-cool-lab");
    });

    it("removes special characters", () => {
      expect(slugify("My Lab!@#$%")).toBe("my-lab");
    });

    it("removes leading and trailing hyphens", () => {
      expect(slugify("  My Lab  ")).toBe("my-lab");
      expect(slugify("---my-lab---")).toBe("my-lab");
    });

    it("truncates to 50 characters", () => {
      const longName = "a".repeat(100);
      expect(slugify(longName).length).toBeLessThanOrEqual(50);
    });

    it("handles empty strings", () => {
      expect(slugify("")).toBe("");
    });

    it("handles strings with only special characters", () => {
      expect(slugify("@#$%^&")).toBe("");
    });
  });

  describe("getLabPath", () => {
    it("returns correct path", () => {
      expect(getLabPath("testuser", "my-lab")).toBe("/labs/testuser/my-lab");
    });

    it("handles special characters in username/slug", () => {
      expect(getLabPath("user-123", "lab-456")).toBe("/labs/user-123/lab-456");
    });
  });

  describe("getLabApiPath", () => {
    it("returns correct API path", () => {
      expect(getLabApiPath("lab_123")).toBe("/api/labs/lab_123");
    });
  });

  describe("formatLabStats", () => {
    it("formats stats with all values", () => {
      const stats: LabStats = {
        stars: 10,
        forks: 5,
        tasks: 3,
        papers: 2,
        experiments: 1,
        viewers: 1,
      };

      const result = formatLabStats(stats);
      expect(result).toContain("10 stars");
      expect(result).toContain("5 forks");
      expect(result).toContain("3 tasks");
    });

    it("omits zero values", () => {
      const stats: LabStats = {
        stars: 10,
        forks: 0,
        tasks: 0,
        papers: 0,
        experiments: 0,
        viewers: 0,
      };

      const result = formatLabStats(stats);
      expect(result).toBe("10 stars");
      expect(result).not.toContain("forks");
    });

    it("returns No activity for empty stats", () => {
      expect(formatLabStats(DEFAULT_LAB_STATS)).toBe("No activity");
    });
  });

  describe("canEditLab", () => {
    const mockLab: Lab = {
      id: "lab-1",
      slug: "test",
      name: "Test",
      description: "",
      domainSlug: "voice-clone",
      domainName: "Voice Clone",
      owner: {
        id: "user-1",
        username: "testuser",
        displayName: "Test User",
      },
      visibility: "public",
      status: "active",
      stats: DEFAULT_LAB_STATS,
      tags: [],
      createdAt: "",
      updatedAt: "",
      lastActivityAt: "",
    };

    it("returns true for owner", () => {
      expect(canEditLab(mockLab, "user-1")).toBe(true);
    });

    it("returns false for non-owner", () => {
      expect(canEditLab(mockLab, "user-2")).toBe(false);
    });

    it("returns false for undefined user", () => {
      expect(canEditLab(mockLab, undefined)).toBe(false);
    });
  });

  describe("canViewLab", () => {
    const createLab = (visibility: "public" | "private" | "unlisted"): Lab => ({
      id: "lab-1",
      slug: "test",
      name: "Test",
      description: "",
      domainSlug: "voice-clone",
      domainName: "Voice Clone",
      owner: {
        id: "user-1",
        username: "testuser",
        displayName: "Test User",
      },
      visibility,
      status: "active",
      stats: DEFAULT_LAB_STATS,
      tags: [],
      createdAt: "",
      updatedAt: "",
      lastActivityAt: "",
    });

    it("public labs are viewable by anyone", () => {
      const lab = createLab("public");
      expect(canViewLab(lab, undefined)).toBe(true);
      expect(canViewLab(lab, "user-2")).toBe(true);
    });

    it("unlisted labs are viewable by anyone", () => {
      const lab = createLab("unlisted");
      expect(canViewLab(lab, undefined)).toBe(true);
      expect(canViewLab(lab, "user-2")).toBe(true);
    });

    it("private labs are only viewable by owner", () => {
      const lab = createLab("private");
      expect(canViewLab(lab, undefined)).toBe(false);
      expect(canViewLab(lab, "user-2")).toBe(false);
      expect(canViewLab(lab, "user-1")).toBe(true);
    });
  });

  describe("isLab", () => {
    it("returns true for valid lab objects", () => {
      const lab: Lab = {
        id: "lab-1",
        slug: "test",
        name: "Test",
        description: "",
        domainSlug: "voice-clone",
        domainName: "Voice Clone",
        owner: { id: "1", username: "test", displayName: "Test" },
        visibility: "public",
        status: "active",
        stats: DEFAULT_LAB_STATS,
        tags: [],
        createdAt: "",
        updatedAt: "",
        lastActivityAt: "",
      };

      expect(isLab(lab)).toBe(true);
    });

    it("returns false for null", () => {
      expect(isLab(null)).toBe(false);
    });

    it("returns false for undefined", () => {
      expect(isLab(undefined)).toBe(false);
    });

    it("returns false for non-objects", () => {
      expect(isLab("string")).toBe(false);
      expect(isLab(123)).toBe(false);
      expect(isLab([])).toBe(false);
    });

    it("returns false for objects missing required fields", () => {
      expect(isLab({ id: "1" })).toBe(false);
      expect(isLab({ id: "1", slug: "test" })).toBe(false);
      expect(isLab({ id: "1", slug: "test", name: "Test" })).toBe(false);
    });
  });

  describe("DEFAULT_LAB_STATS", () => {
    it("has all stats set to zero", () => {
      expect(DEFAULT_LAB_STATS.stars).toBe(0);
      expect(DEFAULT_LAB_STATS.forks).toBe(0);
      expect(DEFAULT_LAB_STATS.tasks).toBe(0);
      expect(DEFAULT_LAB_STATS.papers).toBe(0);
      expect(DEFAULT_LAB_STATS.experiments).toBe(0);
      expect(DEFAULT_LAB_STATS.viewers).toBe(0);
    });
  });
});
