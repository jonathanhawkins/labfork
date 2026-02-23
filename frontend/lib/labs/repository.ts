/**
 * Lab Repository
 *
 * CRUD operations for labs using JSON file storage.
 */

import {
  readCollection,
  writeCollection,
  findOne,
  findMany,
  insertOne,
  updateOne,
  deleteOne,
  findPaginated,
  count,
  exists,
  COLLECTIONS,
} from "@/lib/db/json-store";
import type {
  Lab,
  LabStats,
  LabOwner,
  CreateLabInput,
  UpdateLabInput,
  LabListOptions,
  LabListResult,
  LabVisibility,
} from "./types";
import {
  generateLabId,
  isValidLabSlug,
  slugify,
  DEFAULT_LAB_STATS,
} from "./types";

/**
 * Create a new lab
 */
export async function createLab(
  input: CreateLabInput,
  owner: LabOwner
): Promise<Lab> {
  // Validate slug
  const slug = input.slug || slugify(input.name);
  if (!isValidLabSlug(slug)) {
    throw new Error(`Invalid lab slug: ${slug}`);
  }

  // Check if slug is already taken by this user
  const existingLab = await findOne<Lab>(
    COLLECTIONS.LABS,
    (lab) => lab.owner.id === owner.id && lab.slug === slug
  );
  if (existingLab) {
    throw new Error(`Lab with slug "${slug}" already exists for this user`);
  }

  const now = new Date().toISOString();
  const lab: Lab = {
    id: generateLabId(),
    slug,
    name: input.name,
    description: input.description,
    readme: input.readme,
    domainSlug: input.domainSlug,
    domainName: getDomainName(input.domainSlug),
    owner,
    visibility: input.visibility || "public",
    status: "active",
    stats: { ...DEFAULT_LAB_STATS },
    tags: input.tags || [],
    primaryColor: input.primaryColor,
    createdAt: now,
    updatedAt: now,
    lastActivityAt: now,
  };

  await insertOne(COLLECTIONS.LABS, lab);
  return lab;
}

/**
 * Get domain name from slug
 */
function getDomainName(slug: string): string {
  const names: Record<string, string> = {
    "voice-clone": "Voice Cloning",
    "quant-trading": "Quantitative Trading",
    robotics: "Robotics",
    biotech: "Biotech",
    "firefly-network": "Firefly Network",
    "solar-energy": "Solar Energy",
    "mesh-networking": "Mesh Networking",
    "swarm-intelligence": "Swarm Intelligence",
  };
  return names[slug] || slug.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Get lab by ID
 */
export async function getLabById(id: string): Promise<Lab | null> {
  return findOne<Lab>(COLLECTIONS.LABS, (lab) => lab.id === id);
}

/**
 * Get lab by owner and slug
 */
export async function getLabBySlug(
  ownerUsername: string,
  slug: string
): Promise<Lab | null> {
  return findOne<Lab>(
    COLLECTIONS.LABS,
    (lab) => lab.owner.username === ownerUsername && lab.slug === slug
  );
}

/**
 * Update a lab
 */
export async function updateLab(
  id: string,
  updates: UpdateLabInput
): Promise<Lab | null> {
  const updated = await updateOne<Lab>(
    COLLECTIONS.LABS,
    (lab) => lab.id === id,
    {
      ...updates,
      updatedAt: new Date().toISOString(),
    }
  );
  return updated;
}

/**
 * Update lab stats
 */
export async function updateLabStats(
  id: string,
  statsUpdates: Partial<LabStats>
): Promise<Lab | null> {
  const lab = await getLabById(id);
  if (!lab) return null;

  const updated = await updateOne<Lab>(COLLECTIONS.LABS, (l) => l.id === id, {
    stats: { ...lab.stats, ...statsUpdates },
    updatedAt: new Date().toISOString(),
  });
  return updated;
}

/**
 * Update lab activity timestamp
 */
export async function updateLabActivity(id: string): Promise<void> {
  await updateOne<Lab>(COLLECTIONS.LABS, (lab) => lab.id === id, {
    lastActivityAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  });
}

/**
 * Delete a lab
 */
export async function deleteLab(id: string): Promise<boolean> {
  return deleteOne<Lab>(COLLECTIONS.LABS, (lab) => lab.id === id);
}

/**
 * List labs with filtering, sorting, and pagination
 */
export async function listLabs(options: LabListOptions = {}): Promise<LabListResult> {
  // Build filter function
  const filter = (lab: Lab): boolean => {
    // Visibility filter (public by default for listings)
    if (options.visibility) {
      if (lab.visibility !== options.visibility) return false;
    } else if (!options.owner) {
      // If no owner specified, only show public labs
      if (lab.visibility !== "public") return false;
    }

    // Owner filter
    if (options.owner && lab.owner.username !== options.owner) return false;

    // Slug filter
    if (options.slug && lab.slug !== options.slug) return false;

    // Domain filter
    if (options.domain && lab.domainSlug !== options.domain) return false;

    // Status filter
    if (options.status && lab.status !== options.status) return false;

    // Tags filter (any match)
    if (options.tags && options.tags.length > 0) {
      const hasMatch = options.tags.some((tag) => lab.tags.includes(tag));
      if (!hasMatch) return false;
    }

    // Search filter
    if (options.search) {
      const search = options.search.toLowerCase();
      const matches =
        lab.name.toLowerCase().includes(search) ||
        lab.description.toLowerCase().includes(search) ||
        lab.tags.some((tag) => tag.toLowerCase().includes(search));
      if (!matches) return false;
    }

    return true;
  };

  // Build sort function
  const sort = (a: Lab, b: Lab): number => {
    const dir = options.sortDir === "asc" ? 1 : -1;

    switch (options.sortBy) {
      case "stars":
        return (b.stats.stars - a.stats.stars) * dir;
      case "forks":
        return (b.stats.forks - a.stats.forks) * dir;
      case "activity":
        return (
          (new Date(b.lastActivityAt).getTime() -
            new Date(a.lastActivityAt).getTime()) *
          dir
        );
      case "name":
        return a.name.localeCompare(b.name) * dir;
      case "created":
      default:
        return (
          (new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()) *
          dir
        );
    }
  };

  const result = await findPaginated<Lab>(COLLECTIONS.LABS, {
    filter,
    sort,
    page: options.page || 1,
    limit: options.limit || 20,
  });

  return {
    labs: result.items,
    total: result.total,
    page: result.page,
    totalPages: result.totalPages,
    hasMore: result.hasMore,
  };
}

/**
 * Get labs by user
 */
export async function getLabsByUser(
  username: string,
  includePrivate: boolean = false
): Promise<Lab[]> {
  return findMany<Lab>(COLLECTIONS.LABS, (lab) => {
    if (lab.owner.username !== username) return false;
    if (!includePrivate && lab.visibility === "private") return false;
    return true;
  });
}

/**
 * Get public labs
 */
export async function getPublicLabs(limit: number = 20): Promise<Lab[]> {
  const result = await listLabs({
    visibility: "public",
    status: "active",
    sortBy: "stars",
    limit,
  });
  return result.labs;
}

/**
 * Get featured labs (most starred)
 */
export async function getFeaturedLabs(limit: number = 6): Promise<Lab[]> {
  const result = await listLabs({
    visibility: "public",
    status: "active",
    sortBy: "stars",
    limit,
  });
  return result.labs;
}

/**
 * Get recent labs
 */
export async function getRecentLabs(limit: number = 10): Promise<Lab[]> {
  const result = await listLabs({
    visibility: "public",
    status: "active",
    sortBy: "created",
    limit,
  });
  return result.labs;
}

/**
 * Get active labs (recent activity)
 */
export async function getActiveLabs(limit: number = 10): Promise<Lab[]> {
  const result = await listLabs({
    visibility: "public",
    status: "active",
    sortBy: "activity",
    limit,
  });
  return result.labs;
}

/**
 * Search labs
 */
export async function searchLabs(
  query: string,
  options: Partial<LabListOptions> = {}
): Promise<LabListResult> {
  return listLabs({
    ...options,
    search: query,
    visibility: "public",
    status: "active",
  });
}

/**
 * Get labs by domain
 */
export async function getLabsByDomain(
  domainSlug: string,
  limit: number = 20
): Promise<Lab[]> {
  const result = await listLabs({
    domain: domainSlug,
    visibility: "public",
    status: "active",
    sortBy: "stars",
    limit,
  });
  return result.labs;
}

/**
 * Check if lab slug is available for user
 */
export async function isSlugAvailable(
  username: string,
  slug: string
): Promise<boolean> {
  const existing = await findOne<Lab>(
    COLLECTIONS.LABS,
    (lab) => lab.owner.username === username && lab.slug === slug
  );
  return !existing;
}

/**
 * Get lab count for user
 */
export async function getUserLabCount(username: string): Promise<number> {
  return count<Lab>(COLLECTIONS.LABS, (lab) => lab.owner.username === username);
}

/**
 * Get total lab count
 */
export async function getTotalLabCount(): Promise<number> {
  return count<Lab>(COLLECTIONS.LABS, (lab) => lab.visibility === "public");
}

/**
 * Increment lab view count
 */
export async function incrementLabViews(id: string): Promise<void> {
  const lab = await getLabById(id);
  if (lab) {
    await updateLabStats(id, { views: lab.stats.views + 1 });
  }
}

/**
 * Set lab viewer count (for live viewers)
 */
export async function setLabViewers(id: string, count: number): Promise<void> {
  await updateLabStats(id, { viewers: count });
}
