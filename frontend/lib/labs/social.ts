/**
 * Lab Social Services
 *
 * Star and fork functionality for labs.
 */

import {
  findOne,
  findMany,
  insertOne,
  deleteOne,
  count,
  exists,
  COLLECTIONS,
} from "@/lib/db/json-store";
import {
  getLabById,
  updateLabStats,
  createLab,
  getLabBySlug,
} from "./repository";
import type {
  Lab,
  Star,
  Fork,
  ForkInfo,
  CreateLabInput,
  LabOwner,
} from "./types";
import { generateLabId } from "./types";

// ==================== STAR OPERATIONS ====================

/**
 * Star a lab
 */
export async function starLab(userId: string, labId: string): Promise<boolean> {
  // Check if already starred
  const alreadyStarred = await exists<Star>(
    COLLECTIONS.STARS,
    (star) => star.userId === userId && star.labId === labId
  );

  if (alreadyStarred) {
    return false;
  }

  // Create star record
  const star: Star = {
    userId,
    labId,
    createdAt: new Date().toISOString(),
  };
  await insertOne(COLLECTIONS.STARS, star);

  // Update lab star count
  const lab = await getLabById(labId);
  if (lab) {
    await updateLabStats(labId, { stars: lab.stats.stars + 1 });
  }

  return true;
}

/**
 * Unstar a lab
 */
export async function unstarLab(userId: string, labId: string): Promise<boolean> {
  const deleted = await deleteOne<Star>(
    COLLECTIONS.STARS,
    (star) => star.userId === userId && star.labId === labId
  );

  if (deleted) {
    // Update lab star count
    const lab = await getLabById(labId);
    if (lab && lab.stats.stars > 0) {
      await updateLabStats(labId, { stars: lab.stats.stars - 1 });
    }
  }

  return deleted;
}

/**
 * Check if user has starred a lab
 */
export async function isLabStarred(
  userId: string,
  labId: string
): Promise<boolean> {
  return exists<Star>(
    COLLECTIONS.STARS,
    (star) => star.userId === userId && star.labId === labId
  );
}

/**
 * Get star count for a lab
 */
export async function getStarCount(labId: string): Promise<number> {
  return count<Star>(COLLECTIONS.STARS, (star) => star.labId === labId);
}

/**
 * Get labs starred by a user
 */
export async function getStarredLabs(userId: string): Promise<Lab[]> {
  const stars = await findMany<Star>(
    COLLECTIONS.STARS,
    (star) => star.userId === userId
  );

  const labs: Lab[] = [];
  for (const star of stars) {
    const lab = await getLabById(star.labId);
    if (lab) {
      labs.push(lab);
    }
  }

  return labs;
}

/**
 * Get users who starred a lab
 */
export async function getLabStargazers(labId: string): Promise<Star[]> {
  return findMany<Star>(COLLECTIONS.STARS, (star) => star.labId === labId);
}

// ==================== FORK OPERATIONS ====================

/**
 * Fork a lab
 */
export async function forkLab(
  sourceLabId: string,
  newOwner: LabOwner,
  newSlug?: string
): Promise<Lab> {
  // Get source lab
  const sourceLab = await getLabById(sourceLabId);
  if (!sourceLab) {
    throw new Error("Source lab not found");
  }

  // Check visibility
  if (sourceLab.visibility === "private" && sourceLab.owner.id !== newOwner.id) {
    throw new Error("Cannot fork private lab");
  }

  // Generate new slug if not provided
  const slug = newSlug || `${sourceLab.slug}-fork`;

  // Create fork info
  const forkInfo: ForkInfo = {
    sourceLabId: sourceLab.id,
    sourceSlug: sourceLab.slug,
    sourceOwner: sourceLab.owner.username,
    forkedAt: new Date().toISOString(),
  };

  // Create new lab
  const input: CreateLabInput = {
    name: sourceLab.name,
    slug,
    description: sourceLab.description,
    domainSlug: sourceLab.domainSlug,
    visibility: "public",
    tags: [...sourceLab.tags],
    primaryColor: sourceLab.primaryColor,
    readme: sourceLab.readme,
  };

  const newLab = await createLab(input, newOwner);

  // Add fork info to the new lab
  const { updateOne } = await import("@/lib/db/json-store");
  await updateOne<Lab>(COLLECTIONS.LABS, (lab) => lab.id === newLab.id, {
    forkedFrom: forkInfo,
  });

  // Create fork record
  const fork: Fork = {
    id: generateLabId().replace("lab_", "fork_"),
    sourceLabId: sourceLab.id,
    forkedLabId: newLab.id,
    userId: newOwner.id,
    createdAt: new Date().toISOString(),
  };
  await insertOne(COLLECTIONS.FORKS, fork);

  // Update source lab fork count
  await updateLabStats(sourceLabId, { forks: sourceLab.stats.forks + 1 });

  // Return the new lab with fork info
  return {
    ...newLab,
    forkedFrom: forkInfo,
  };
}

/**
 * Get fork count for a lab
 */
export async function getForkCount(labId: string): Promise<number> {
  return count<Fork>(COLLECTIONS.FORKS, (fork) => fork.sourceLabId === labId);
}

/**
 * Get forks of a lab
 */
export async function getLabForks(labId: string): Promise<Lab[]> {
  const forks = await findMany<Fork>(
    COLLECTIONS.FORKS,
    (fork) => fork.sourceLabId === labId
  );

  const labs: Lab[] = [];
  for (const fork of forks) {
    const lab = await getLabById(fork.forkedLabId);
    if (lab) {
      labs.push(lab);
    }
  }

  return labs;
}

/**
 * Get lab lineage (chain of forks back to original)
 */
export async function getLabLineage(labId: string): Promise<Lab[]> {
  const lineage: Lab[] = [];
  let currentLab = await getLabById(labId);

  while (currentLab?.forkedFrom) {
    const parentLab = await getLabById(currentLab.forkedFrom.sourceLabId);
    if (parentLab) {
      lineage.push(parentLab);
      currentLab = parentLab;
    } else {
      break;
    }
  }

  return lineage;
}

/**
 * Check if lab is a fork
 */
export async function isLabFork(labId: string): Promise<boolean> {
  const lab = await getLabById(labId);
  return !!lab?.forkedFrom;
}

/**
 * Get original lab (root of fork chain)
 */
export async function getOriginalLab(labId: string): Promise<Lab | null> {
  const lineage = await getLabLineage(labId);
  return lineage.length > 0 ? lineage[lineage.length - 1] : null;
}

// ==================== COMBINED SOCIAL STATS ====================

/**
 * Get social stats for a lab
 */
export interface LabSocialStats {
  stars: number;
  forks: number;
  isStarred: boolean;
  isFork: boolean;
  forkedFrom?: ForkInfo;
}

export async function getLabSocialStats(
  labId: string,
  userId?: string
): Promise<LabSocialStats> {
  const lab = await getLabById(labId);
  const stars = await getStarCount(labId);
  const forks = await getForkCount(labId);
  const isStarred = userId ? await isLabStarred(userId, labId) : false;

  return {
    stars,
    forks,
    isStarred,
    isFork: !!lab?.forkedFrom,
    forkedFrom: lab?.forkedFrom,
  };
}

/**
 * Toggle star (star if not starred, unstar if starred)
 */
export async function toggleStar(
  userId: string,
  labId: string
): Promise<{ starred: boolean; count: number }> {
  const isStarred = await isLabStarred(userId, labId);

  if (isStarred) {
    await unstarLab(userId, labId);
  } else {
    await starLab(userId, labId);
  }

  const count = await getStarCount(labId);
  return {
    starred: !isStarred,
    count,
  };
}
