import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { Lab, Star, Fork, LabOwner } from "@/lib/labs/types";
import { DEFAULT_LAB_STATS } from "@/lib/labs/types";

// Mock the json-store module
vi.mock("@/lib/db/json-store", () => {
  let mockStars: Star[] = [];
  let mockForks: Fork[] = [];
  let mockLabs: Lab[] = [];

  return {
    COLLECTIONS: {
      LABS: "labs",
      STARS: "stars",
      FORKS: "forks",
    },
    findOne: vi.fn(async <T>(collection: string, predicate: (item: T) => boolean) => {
      if (collection === "stars") {
        return mockStars.find(predicate as (item: Star) => boolean) || null;
      }
      if (collection === "forks") {
        return mockForks.find(predicate as (item: Fork) => boolean) || null;
      }
      if (collection === "labs") {
        return mockLabs.find(predicate as (item: Lab) => boolean) || null;
      }
      return null;
    }),
    findMany: vi.fn(async <T>(collection: string, predicate: (item: T) => boolean) => {
      if (collection === "stars") {
        return mockStars.filter(predicate as (item: Star) => boolean);
      }
      if (collection === "forks") {
        return mockForks.filter(predicate as (item: Fork) => boolean);
      }
      if (collection === "labs") {
        return mockLabs.filter(predicate as (item: Lab) => boolean);
      }
      return [];
    }),
    insertOne: vi.fn(async <T>(collection: string, item: T) => {
      if (collection === "stars") {
        mockStars.push(item as Star);
      }
      if (collection === "forks") {
        mockForks.push(item as Fork);
      }
      if (collection === "labs") {
        mockLabs.push(item as Lab);
      }
      return item;
    }),
    updateOne: vi.fn(async <T>(collection: string, predicate: (item: T) => boolean, updates: Partial<T>) => {
      if (collection === "labs") {
        const index = mockLabs.findIndex(predicate as (item: Lab) => boolean);
        if (index >= 0) {
          mockLabs[index] = { ...mockLabs[index], ...updates } as Lab;
          return true;
        }
      }
      return false;
    }),
    deleteOne: vi.fn(async <T>(collection: string, predicate: (item: T) => boolean) => {
      if (collection === "stars") {
        const index = mockStars.findIndex(predicate as (item: Star) => boolean);
        if (index >= 0) {
          mockStars.splice(index, 1);
          return true;
        }
      }
      return false;
    }),
    count: vi.fn(async <T>(collection: string, predicate: (item: T) => boolean) => {
      if (collection === "stars") {
        return mockStars.filter(predicate as (item: Star) => boolean).length;
      }
      if (collection === "forks") {
        return mockForks.filter(predicate as (item: Fork) => boolean).length;
      }
      return 0;
    }),
    exists: vi.fn(async <T>(collection: string, predicate: (item: T) => boolean) => {
      if (collection === "stars") {
        return mockStars.some(predicate as (item: Star) => boolean);
      }
      return false;
    }),
    // Helper to reset mocks
    __reset: () => {
      mockStars = [];
      mockForks = [];
      mockLabs = [];
    },
    __setLabs: (labs: Lab[]) => {
      mockLabs = labs;
    },
    __setStars: (stars: Star[]) => {
      mockStars = stars;
    },
    __setForks: (forks: Fork[]) => {
      mockForks = forks;
    },
  };
});

// Mock the repository module
vi.mock("@/lib/labs/repository", () => ({
  getLabById: vi.fn(),
  getLabBySlug: vi.fn(),
  updateLabStats: vi.fn(),
  createLab: vi.fn(),
}));

// Import after mocks
import {
  starLab,
  unstarLab,
  isLabStarred,
  getStarCount,
  getStarredLabs,
  getLabStargazers,
  forkLab,
  getForkCount,
  getLabForks,
  getLabLineage,
  isLabFork,
  getOriginalLab,
  getLabSocialStats,
  toggleStar,
} from "@/lib/labs/social";
import { getLabById, updateLabStats, createLab } from "@/lib/labs/repository";
import * as jsonStore from "@/lib/db/json-store";

const mockJsonStore = jsonStore as typeof jsonStore & {
  __reset: () => void;
  __setLabs: (labs: Lab[]) => void;
  __setStars: (stars: Star[]) => void;
  __setForks: (forks: Fork[]) => void;
};

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
  stats: { ...DEFAULT_LAB_STATS, stars: 5, forks: 2 },
  tags: ["test"],
  createdAt: "2024-01-01T00:00:00Z",
  updatedAt: "2024-01-01T00:00:00Z",
  lastActivityAt: "2024-01-01T00:00:00Z",
};

describe("Lab Social Services", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockJsonStore.__reset();
  });

  describe("Star Operations", () => {
    describe("starLab", () => {
      it("stars a lab successfully", async () => {
        vi.mocked(getLabById).mockResolvedValue(mockLab);

        const result = await starLab("user-2", "lab-1");

        expect(result).toBe(true);
        expect(jsonStore.insertOne).toHaveBeenCalledWith(
          "stars",
          expect.objectContaining({
            userId: "user-2",
            labId: "lab-1",
          })
        );
        expect(updateLabStats).toHaveBeenCalledWith("lab-1", { stars: 6 });
      });

      it("returns false if already starred", async () => {
        mockJsonStore.__setStars([
          { userId: "user-2", labId: "lab-1", createdAt: "2024-01-01T00:00:00Z" },
        ]);

        const result = await starLab("user-2", "lab-1");

        expect(result).toBe(false);
        expect(jsonStore.insertOne).not.toHaveBeenCalled();
      });
    });

    describe("unstarLab", () => {
      it("unstars a lab successfully", async () => {
        mockJsonStore.__setStars([
          { userId: "user-2", labId: "lab-1", createdAt: "2024-01-01T00:00:00Z" },
        ]);
        vi.mocked(getLabById).mockResolvedValue(mockLab);

        const result = await unstarLab("user-2", "lab-1");

        expect(result).toBe(true);
        expect(updateLabStats).toHaveBeenCalledWith("lab-1", { stars: 4 });
      });

      it("returns false if not starred", async () => {
        const result = await unstarLab("user-2", "lab-1");

        expect(result).toBe(false);
        expect(updateLabStats).not.toHaveBeenCalled();
      });
    });

    describe("isLabStarred", () => {
      it("returns true if starred", async () => {
        mockJsonStore.__setStars([
          { userId: "user-2", labId: "lab-1", createdAt: "2024-01-01T00:00:00Z" },
        ]);

        const result = await isLabStarred("user-2", "lab-1");

        expect(result).toBe(true);
      });

      it("returns false if not starred", async () => {
        const result = await isLabStarred("user-2", "lab-1");

        expect(result).toBe(false);
      });
    });

    describe("getStarCount", () => {
      it("returns correct count", async () => {
        mockJsonStore.__setStars([
          { userId: "user-1", labId: "lab-1", createdAt: "2024-01-01T00:00:00Z" },
          { userId: "user-2", labId: "lab-1", createdAt: "2024-01-02T00:00:00Z" },
          { userId: "user-3", labId: "lab-2", createdAt: "2024-01-03T00:00:00Z" },
        ]);

        const result = await getStarCount("lab-1");

        expect(result).toBe(2);
      });

      it("returns 0 for no stars", async () => {
        const result = await getStarCount("lab-1");

        expect(result).toBe(0);
      });
    });

    describe("getStarredLabs", () => {
      it("returns labs starred by user", async () => {
        mockJsonStore.__setStars([
          { userId: "user-1", labId: "lab-1", createdAt: "2024-01-01T00:00:00Z" },
          { userId: "user-1", labId: "lab-2", createdAt: "2024-01-02T00:00:00Z" },
        ]);

        const lab2 = { ...mockLab, id: "lab-2", name: "Lab 2" };
        vi.mocked(getLabById)
          .mockResolvedValueOnce(mockLab)
          .mockResolvedValueOnce(lab2);

        const result = await getStarredLabs("user-1");

        expect(result).toHaveLength(2);
        expect(result[0].id).toBe("lab-1");
        expect(result[1].id).toBe("lab-2");
      });

      it("returns empty array if no starred labs", async () => {
        const result = await getStarredLabs("user-1");

        expect(result).toHaveLength(0);
      });
    });

    describe("getLabStargazers", () => {
      it("returns users who starred the lab", async () => {
        mockJsonStore.__setStars([
          { userId: "user-1", labId: "lab-1", createdAt: "2024-01-01T00:00:00Z" },
          { userId: "user-2", labId: "lab-1", createdAt: "2024-01-02T00:00:00Z" },
        ]);

        const result = await getLabStargazers("lab-1");

        expect(result).toHaveLength(2);
        expect(result[0].userId).toBe("user-1");
        expect(result[1].userId).toBe("user-2");
      });
    });

    describe("toggleStar", () => {
      it("stars if not starred", async () => {
        vi.mocked(getLabById).mockResolvedValue(mockLab);

        const result = await toggleStar("user-2", "lab-1");

        expect(result.starred).toBe(true);
      });

      it("unstars if already starred", async () => {
        mockJsonStore.__setStars([
          { userId: "user-2", labId: "lab-1", createdAt: "2024-01-01T00:00:00Z" },
        ]);
        vi.mocked(getLabById).mockResolvedValue(mockLab);

        const result = await toggleStar("user-2", "lab-1");

        expect(result.starred).toBe(false);
      });
    });
  });

  describe("Fork Operations", () => {
    describe("forkLab", () => {
      it("forks a lab successfully", async () => {
        vi.mocked(getLabById).mockResolvedValue(mockLab);
        vi.mocked(createLab).mockResolvedValue({
          ...mockLab,
          id: "lab-forked",
          slug: "test-lab-fork",
          owner: { ...mockOwner, id: "user-2", username: "user2" },
        });

        const newOwner: LabOwner = {
          id: "user-2",
          username: "user2",
          displayName: "User Two",
        };

        const result = await forkLab("lab-1", newOwner);

        expect(result.id).toBe("lab-forked");
        expect(result.forkedFrom).toBeDefined();
        expect(result.forkedFrom?.sourceLabId).toBe("lab-1");
        expect(updateLabStats).toHaveBeenCalledWith("lab-1", { forks: 3 });
      });

      it("throws error for non-existent lab", async () => {
        vi.mocked(getLabById).mockResolvedValue(null);

        const newOwner: LabOwner = {
          id: "user-2",
          username: "user2",
          displayName: "User Two",
        };

        await expect(forkLab("lab-1", newOwner)).rejects.toThrow("Source lab not found");
      });

      it("throws error for private lab by non-owner", async () => {
        const privateLab = { ...mockLab, visibility: "private" as const };
        vi.mocked(getLabById).mockResolvedValue(privateLab);

        const newOwner: LabOwner = {
          id: "user-2",
          username: "user2",
          displayName: "User Two",
        };

        await expect(forkLab("lab-1", newOwner)).rejects.toThrow("Cannot fork private lab");
      });

      it("uses custom slug if provided", async () => {
        vi.mocked(getLabById).mockResolvedValue(mockLab);
        vi.mocked(createLab).mockImplementation(async (input) => ({
          ...mockLab,
          id: "lab-forked",
          slug: input.slug,
          owner: { ...mockOwner, id: "user-2", username: "user2" },
        }));

        const newOwner: LabOwner = {
          id: "user-2",
          username: "user2",
          displayName: "User Two",
        };

        const result = await forkLab("lab-1", newOwner, "my-custom-fork");

        expect(createLab).toHaveBeenCalledWith(
          expect.objectContaining({ slug: "my-custom-fork" }),
          newOwner
        );
      });
    });

    describe("getForkCount", () => {
      it("returns correct fork count", async () => {
        mockJsonStore.__setForks([
          { id: "fork-1", sourceLabId: "lab-1", forkedLabId: "lab-2", userId: "user-2", createdAt: "" },
          { id: "fork-2", sourceLabId: "lab-1", forkedLabId: "lab-3", userId: "user-3", createdAt: "" },
        ]);

        const result = await getForkCount("lab-1");

        expect(result).toBe(2);
      });
    });

    describe("getLabForks", () => {
      it("returns all forks of a lab", async () => {
        mockJsonStore.__setForks([
          { id: "fork-1", sourceLabId: "lab-1", forkedLabId: "lab-2", userId: "user-2", createdAt: "" },
          { id: "fork-2", sourceLabId: "lab-1", forkedLabId: "lab-3", userId: "user-3", createdAt: "" },
        ]);

        const lab2 = { ...mockLab, id: "lab-2" };
        const lab3 = { ...mockLab, id: "lab-3" };
        vi.mocked(getLabById)
          .mockResolvedValueOnce(lab2)
          .mockResolvedValueOnce(lab3);

        const result = await getLabForks("lab-1");

        expect(result).toHaveLength(2);
      });
    });

    describe("getLabLineage", () => {
      it("returns chain of parent labs", async () => {
        const lab3: Lab = {
          ...mockLab,
          id: "lab-3",
          forkedFrom: {
            sourceLabId: "lab-2",
            sourceSlug: "lab-2",
            sourceOwner: "user-2",
            forkedAt: "",
          },
        };
        const lab2: Lab = {
          ...mockLab,
          id: "lab-2",
          forkedFrom: {
            sourceLabId: "lab-1",
            sourceSlug: "lab-1",
            sourceOwner: "user-1",
            forkedAt: "",
          },
        };
        const lab1: Lab = {
          ...mockLab,
          id: "lab-1",
        };

        vi.mocked(getLabById)
          .mockResolvedValueOnce(lab3)  // initial fetch
          .mockResolvedValueOnce(lab2)  // first parent
          .mockResolvedValueOnce(lab1); // second parent (original)

        const result = await getLabLineage("lab-3");

        expect(result).toHaveLength(2);
        expect(result[0].id).toBe("lab-2");
        expect(result[1].id).toBe("lab-1");
      });

      it("returns empty array for non-forked lab", async () => {
        vi.mocked(getLabById).mockResolvedValue(mockLab);

        const result = await getLabLineage("lab-1");

        expect(result).toHaveLength(0);
      });
    });

    describe("isLabFork", () => {
      it("returns true for forked lab", async () => {
        const forkedLab = {
          ...mockLab,
          forkedFrom: {
            sourceLabId: "lab-1",
            sourceSlug: "test-lab",
            sourceOwner: "user-1",
            forkedAt: "",
          },
        };
        vi.mocked(getLabById).mockResolvedValue(forkedLab);

        const result = await isLabFork("lab-2");

        expect(result).toBe(true);
      });

      it("returns false for original lab", async () => {
        vi.mocked(getLabById).mockResolvedValue(mockLab);

        const result = await isLabFork("lab-1");

        expect(result).toBe(false);
      });
    });

    describe("getOriginalLab", () => {
      it("returns original lab in fork chain", async () => {
        const lab3: Lab = {
          ...mockLab,
          id: "lab-3",
          forkedFrom: {
            sourceLabId: "lab-2",
            sourceSlug: "lab-2",
            sourceOwner: "user-2",
            forkedAt: "",
          },
        };
        const lab2: Lab = {
          ...mockLab,
          id: "lab-2",
          forkedFrom: {
            sourceLabId: "lab-1",
            sourceSlug: "lab-1",
            sourceOwner: "user-1",
            forkedAt: "",
          },
        };
        const lab1: Lab = {
          ...mockLab,
          id: "lab-1",
        };

        vi.mocked(getLabById)
          .mockResolvedValueOnce(lab3)
          .mockResolvedValueOnce(lab2)
          .mockResolvedValueOnce(lab1);

        const result = await getOriginalLab("lab-3");

        expect(result?.id).toBe("lab-1");
      });

      it("returns null for non-forked lab", async () => {
        vi.mocked(getLabById).mockResolvedValue(mockLab);

        const result = await getOriginalLab("lab-1");

        expect(result).toBe(null);
      });
    });
  });

  describe("Combined Social Stats", () => {
    describe("getLabSocialStats", () => {
      it("returns all social stats for a lab", async () => {
        mockJsonStore.__setStars([
          { userId: "user-2", labId: "lab-1", createdAt: "" },
          { userId: "user-3", labId: "lab-1", createdAt: "" },
        ]);
        mockJsonStore.__setForks([
          { id: "fork-1", sourceLabId: "lab-1", forkedLabId: "lab-2", userId: "user-2", createdAt: "" },
        ]);
        vi.mocked(getLabById).mockResolvedValue(mockLab);

        const result = await getLabSocialStats("lab-1", "user-2");

        expect(result.stars).toBe(2);
        expect(result.forks).toBe(1);
        expect(result.isStarred).toBe(true);
        expect(result.isFork).toBe(false);
      });

      it("handles anonymous user", async () => {
        vi.mocked(getLabById).mockResolvedValue(mockLab);

        const result = await getLabSocialStats("lab-1");

        expect(result.isStarred).toBe(false);
      });

      it("includes fork info for forked lab", async () => {
        const forkedLab = {
          ...mockLab,
          forkedFrom: {
            sourceLabId: "lab-0",
            sourceSlug: "original",
            sourceOwner: "creator",
            forkedAt: "2024-01-01T00:00:00Z",
          },
        };
        vi.mocked(getLabById).mockResolvedValue(forkedLab);

        const result = await getLabSocialStats("lab-1");

        expect(result.isFork).toBe(true);
        expect(result.forkedFrom?.sourceLabId).toBe("lab-0");
      });
    });
  });
});
