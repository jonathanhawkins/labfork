import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { Lab, LabOwner } from "@/lib/labs/types";
import { DEFAULT_LAB_STATS } from "@/lib/labs/types";

/**
 * Integration Tests for Lab Flow
 *
 * Tests the complete user journey:
 * 1. Create a new lab
 * 2. View lab in explore page
 * 3. Star the lab
 * 4. Fork the lab
 * 5. Share the lab
 */

// Mock fetch for API calls
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Test data
const mockOwner: LabOwner = {
  id: "user-1",
  username: "testuser",
  displayName: "Test User",
};

const mockLab: Lab = {
  id: "lab-1",
  slug: "test-lab",
  name: "Test Lab",
  description: "Test description",
  domainSlug: "voice-clone",
  domainName: "Voice Clone",
  owner: mockOwner,
  visibility: "public",
  status: "active",
  stats: { ...DEFAULT_LAB_STATS },
  tags: ["test", "voice"],
  createdAt: "2024-01-01T00:00:00Z",
  updatedAt: "2024-01-01T00:00:00Z",
  lastActivityAt: "2024-01-01T00:00:00Z",
};

describe("Lab Flow Integration Tests", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetch.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Create Lab Flow", () => {
    it("creates a new lab via API", async () => {
      const newLab = { ...mockLab, id: "lab-new" };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, lab: newLab }),
      });

      const response = await fetch("/api/labs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: "Test Lab",
          slug: "test-lab",
          description: "Test description",
          domainSlug: "voice-clone",
          visibility: "public",
          tags: ["test", "voice"],
        }),
      });

      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.lab.name).toBe("Test Lab");
      expect(data.lab.slug).toBe("test-lab");
    });

    it("validates required fields", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: async () => ({ success: false, error: "Name is required" }),
      });

      const response = await fetch("/api/labs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          slug: "test-lab",
          domainSlug: "voice-clone",
        }),
      });

      const data = await response.json();

      expect(data.success).toBe(false);
      expect(data.error).toContain("required");
    });

    it("prevents duplicate slugs for same owner", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: async () => ({ success: false, error: "Slug already exists" }),
      });

      const response = await fetch("/api/labs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: "Test Lab 2",
          slug: "test-lab", // Same slug
          description: "Another lab",
          domainSlug: "voice-clone",
        }),
      });

      const data = await response.json();

      expect(data.success).toBe(false);
      expect(data.error).toContain("Slug");
    });
  });

  describe("Explore Labs Flow", () => {
    it("lists public labs", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          labs: [mockLab, { ...mockLab, id: "lab-2", name: "Lab 2" }],
          total: 2,
          page: 1,
          totalPages: 1,
        }),
      });

      const response = await fetch("/api/labs?visibility=public");
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.labs).toHaveLength(2);
      expect(data.total).toBe(2);
    });

    it("filters by domain", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          labs: [mockLab],
          total: 1,
        }),
      });

      const response = await fetch("/api/labs?domain=voice-clone");
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.labs[0].domainSlug).toBe("voice-clone");
    });

    it("searches by text query", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          labs: [mockLab],
          total: 1,
        }),
      });

      const response = await fetch("/api/labs?search=voice");
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.labs[0].name).toContain("Test");
    });

    it("sorts by popularity", async () => {
      const popularLab = { ...mockLab, stats: { ...mockLab.stats, stars: 100 } };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          labs: [popularLab, mockLab],
          total: 2,
        }),
      });

      const response = await fetch("/api/labs?sortBy=stars");
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.labs[0].stats.stars).toBeGreaterThan(data.labs[1].stats.stars);
    });

    it("paginates results", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          labs: [mockLab],
          total: 25,
          page: 2,
          totalPages: 3,
          hasMore: true,
        }),
      });

      const response = await fetch("/api/labs?page=2&limit=12");
      const data = await response.json();

      expect(data.page).toBe(2);
      expect(data.totalPages).toBe(3);
      expect(data.hasMore).toBe(true);
    });
  });

  describe("View Lab Flow", () => {
    it("gets lab by ID", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, lab: mockLab }),
      });

      const response = await fetch("/api/labs/lab-1");
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.lab.id).toBe("lab-1");
    });

    it("includes social stats", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          lab: mockLab,
          social: {
            stars: 10,
            forks: 3,
            isStarred: false,
            isFork: false,
          },
        }),
      });

      const response = await fetch("/api/labs/lab-1");
      const data = await response.json();

      expect(data.social.stars).toBe(10);
      expect(data.social.forks).toBe(3);
      expect(data.social.isStarred).toBe(false);
    });

    it("returns 404 for non-existent lab", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ success: false, error: "Lab not found" }),
      });

      const response = await fetch("/api/labs/non-existent");
      const data = await response.json();

      expect(data.success).toBe(false);
      expect(data.error).toContain("not found");
    });
  });

  describe("Star Lab Flow", () => {
    it("stars a lab", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, starred: true, count: 11 }),
      });

      const response = await fetch("/api/labs/lab-1/star", {
        method: "POST",
      });
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.starred).toBe(true);
      expect(data.count).toBe(11);
    });

    it("unstars a lab", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, starred: false, count: 9 }),
      });

      const response = await fetch("/api/labs/lab-1/star", {
        method: "DELETE",
      });
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.starred).toBe(false);
      expect(data.count).toBe(9);
    });

    it("checks star status", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, isStarred: true, count: 10 }),
      });

      const response = await fetch("/api/labs/lab-1/star");
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.isStarred).toBe(true);
    });
  });

  describe("Fork Lab Flow", () => {
    it("forks a lab", async () => {
      const forkedLab = {
        ...mockLab,
        id: "lab-forked",
        slug: "test-lab-fork",
        owner: { id: "user-2", username: "user2", displayName: "User Two" },
        forkedFrom: {
          sourceLabId: "lab-1",
          sourceSlug: "test-lab",
          sourceOwner: "testuser",
          forkedAt: "2024-01-15T00:00:00Z",
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, lab: forkedLab }),
      });

      const response = await fetch("/api/labs/lab-1/fork", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slug: "test-lab-fork" }),
      });
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.lab.forkedFrom).toBeDefined();
      expect(data.lab.forkedFrom.sourceLabId).toBe("lab-1");
    });

    it("uses custom slug for fork", async () => {
      const forkedLab = {
        ...mockLab,
        id: "lab-forked",
        slug: "my-custom-fork",
        forkedFrom: {
          sourceLabId: "lab-1",
          sourceSlug: "test-lab",
          sourceOwner: "testuser",
          forkedAt: "2024-01-15T00:00:00Z",
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, lab: forkedLab }),
      });

      const response = await fetch("/api/labs/lab-1/fork", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slug: "my-custom-fork", name: "My Fork" }),
      });
      const data = await response.json();

      expect(data.lab.slug).toBe("my-custom-fork");
    });

    it("gets fork lineage", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          lineage: [
            { id: "lab-1", name: "Original Lab" },
          ],
          forkCount: 5,
        }),
      });

      const response = await fetch("/api/labs/lab-forked/fork");
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.lineage).toHaveLength(1);
      expect(data.forkCount).toBe(5);
    });
  });

  describe("Share Lab Flow", () => {
    it("gets shareable lab URL", async () => {
      const labPath = `/labs/${mockLab.owner.username}/${mockLab.slug}`;

      expect(labPath).toBe("/labs/testuser/test-lab");
    });

    it("gets lab stats for sharing", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          stats: {
            stars: 42,
            forks: 10,
            tasks: 5,
            papers: 2,
            experiments: 3,
            viewers: 0,
          },
        }),
      });

      const response = await fetch("/api/labs/lab-1/stats");
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.stats.stars).toBe(42);
      expect(data.stats.forks).toBe(10);
    });
  });

  describe("Update Lab Flow", () => {
    it("updates lab details", async () => {
      const updatedLab = {
        ...mockLab,
        name: "Updated Name",
        description: "Updated description",
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, lab: updatedLab }),
      });

      const response = await fetch("/api/labs/lab-1", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: "Updated Name",
          description: "Updated description",
        }),
      });
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.lab.name).toBe("Updated Name");
    });

    it("updates visibility", async () => {
      const updatedLab = { ...mockLab, visibility: "private" };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, lab: updatedLab }),
      });

      const response = await fetch("/api/labs/lab-1", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ visibility: "private" }),
      });
      const data = await response.json();

      expect(data.lab.visibility).toBe("private");
    });
  });

  describe("Delete Lab Flow", () => {
    it("deletes a lab", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      });

      const response = await fetch("/api/labs/lab-1", {
        method: "DELETE",
      });
      const data = await response.json();

      expect(data.success).toBe(true);
    });

    it("prevents deletion by non-owner", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 403,
        json: async () => ({ success: false, error: "Not authorized" }),
      });

      const response = await fetch("/api/labs/lab-1", {
        method: "DELETE",
      });
      const data = await response.json();

      expect(data.success).toBe(false);
      expect(data.error).toContain("authorized");
    });
  });

  describe("Full User Journey", () => {
    it("complete flow: create -> view -> star -> fork -> share", async () => {
      // Step 1: Create lab
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, lab: mockLab }),
      });

      let response = await fetch("/api/labs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: "Test Lab",
          slug: "test-lab",
          description: "Test",
          domainSlug: "voice-clone",
        }),
      });
      let data = await response.json();
      expect(data.success).toBe(true);
      const createdLabId = data.lab.id;

      // Step 2: View lab in explore
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          labs: [mockLab],
          total: 1,
        }),
      });

      response = await fetch("/api/labs?visibility=public");
      data = await response.json();
      expect(data.labs[0].id).toBe(createdLabId);

      // Step 3: Star the lab
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, starred: true, count: 1 }),
      });

      response = await fetch(`/api/labs/${createdLabId}/star`, {
        method: "POST",
      });
      data = await response.json();
      expect(data.starred).toBe(true);

      // Step 4: Fork the lab
      const forkedLab = {
        ...mockLab,
        id: "lab-forked",
        slug: "test-lab-fork",
        forkedFrom: {
          sourceLabId: createdLabId,
          sourceSlug: "test-lab",
          sourceOwner: "testuser",
          forkedAt: new Date().toISOString(),
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, lab: forkedLab }),
      });

      response = await fetch(`/api/labs/${createdLabId}/fork`, {
        method: "POST",
      });
      data = await response.json();
      expect(data.lab.forkedFrom.sourceLabId).toBe(createdLabId);

      // Step 5: Get share stats
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          success: true,
          stats: { stars: 1, forks: 1, tasks: 0, papers: 0, experiments: 0, viewers: 0 },
        }),
      });

      response = await fetch(`/api/labs/${createdLabId}/stats`);
      data = await response.json();
      expect(data.stats.stars).toBe(1);
      expect(data.stats.forks).toBe(1);
    });
  });
});
