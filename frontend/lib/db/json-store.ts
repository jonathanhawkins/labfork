/**
 * JSON File-based Storage
 *
 * Simple file-based storage for labs, stars, and forks.
 * Uses JSON files in .data directory for persistence.
 * This is suitable for development and small-scale usage.
 */

import { promises as fs } from "fs";
import path from "path";

/**
 * Data directory path
 */
const DATA_DIR = path.join(process.cwd(), ".data");

/**
 * Ensure data directory exists
 */
async function ensureDataDir(): Promise<void> {
  try {
    await fs.mkdir(DATA_DIR, { recursive: true });
  } catch (error) {
    // Directory may already exist
  }
}

/**
 * Get file path for a collection
 */
function getFilePath(collection: string): string {
  return path.join(DATA_DIR, `${collection}.json`);
}

/**
 * Read a collection from file
 */
export async function readCollection<T>(collection: string): Promise<T[]> {
  await ensureDataDir();
  const filePath = getFilePath(collection);

  try {
    const data = await fs.readFile(filePath, "utf-8");
    return JSON.parse(data);
  } catch (error) {
    // File doesn't exist or is invalid, return empty array
    return [];
  }
}

/**
 * Write a collection to file
 */
export async function writeCollection<T>(
  collection: string,
  data: T[]
): Promise<void> {
  await ensureDataDir();
  const filePath = getFilePath(collection);
  await fs.writeFile(filePath, JSON.stringify(data, null, 2));
}

/**
 * Find one item in a collection
 */
export async function findOne<T extends Record<string, unknown>>(
  collection: string,
  predicate: (item: T) => boolean
): Promise<T | null> {
  const items = await readCollection<T>(collection);
  return items.find(predicate) || null;
}

/**
 * Find all items matching a predicate
 */
export async function findMany<T extends Record<string, unknown>>(
  collection: string,
  predicate: (item: T) => boolean
): Promise<T[]> {
  const items = await readCollection<T>(collection);
  return items.filter(predicate);
}

/**
 * Insert one item
 */
export async function insertOne<T extends Record<string, unknown>>(
  collection: string,
  item: T
): Promise<T> {
  const items = await readCollection<T>(collection);
  items.push(item);
  await writeCollection(collection, items);
  return item;
}

/**
 * Insert many items
 */
export async function insertMany<T extends Record<string, unknown>>(
  collection: string,
  newItems: T[]
): Promise<T[]> {
  const items = await readCollection<T>(collection);
  items.push(...newItems);
  await writeCollection(collection, items);
  return newItems;
}

/**
 * Update one item
 */
export async function updateOne<T extends Record<string, unknown>>(
  collection: string,
  predicate: (item: T) => boolean,
  updates: Partial<T>
): Promise<T | null> {
  const items = await readCollection<T>(collection);
  const index = items.findIndex(predicate);

  if (index === -1) return null;

  items[index] = { ...items[index], ...updates };
  await writeCollection(collection, items);
  return items[index];
}

/**
 * Update many items
 */
export async function updateMany<T extends Record<string, unknown>>(
  collection: string,
  predicate: (item: T) => boolean,
  updates: Partial<T>
): Promise<number> {
  const items = await readCollection<T>(collection);
  let count = 0;

  for (let i = 0; i < items.length; i++) {
    if (predicate(items[i])) {
      items[i] = { ...items[i], ...updates };
      count++;
    }
  }

  if (count > 0) {
    await writeCollection(collection, items);
  }

  return count;
}

/**
 * Delete one item
 */
export async function deleteOne<T extends Record<string, unknown>>(
  collection: string,
  predicate: (item: T) => boolean
): Promise<boolean> {
  const items = await readCollection<T>(collection);
  const index = items.findIndex(predicate);

  if (index === -1) return false;

  items.splice(index, 1);
  await writeCollection(collection, items);
  return true;
}

/**
 * Delete many items
 */
export async function deleteMany<T extends Record<string, unknown>>(
  collection: string,
  predicate: (item: T) => boolean
): Promise<number> {
  const items = await readCollection<T>(collection);
  const originalLength = items.length;
  const filtered = items.filter((item) => !predicate(item));

  if (filtered.length !== originalLength) {
    await writeCollection(collection, filtered);
  }

  return originalLength - filtered.length;
}

/**
 * Count items matching a predicate
 */
export async function count<T extends Record<string, unknown>>(
  collection: string,
  predicate?: (item: T) => boolean
): Promise<number> {
  const items = await readCollection<T>(collection);
  if (!predicate) return items.length;
  return items.filter(predicate).length;
}

/**
 * Check if item exists
 */
export async function exists<T extends Record<string, unknown>>(
  collection: string,
  predicate: (item: T) => boolean
): Promise<boolean> {
  const items = await readCollection<T>(collection);
  return items.some(predicate);
}

/**
 * Get all items sorted and paginated
 */
export async function findPaginated<T extends Record<string, unknown>>(
  collection: string,
  options: {
    filter?: (item: T) => boolean;
    sort?: (a: T, b: T) => number;
    page?: number;
    limit?: number;
  }
): Promise<{
  items: T[];
  total: number;
  page: number;
  totalPages: number;
  hasMore: boolean;
}> {
  let items = await readCollection<T>(collection);

  // Filter
  if (options.filter) {
    items = items.filter(options.filter);
  }

  const total = items.length;

  // Sort
  if (options.sort) {
    items.sort(options.sort);
  }

  // Paginate
  const page = options.page || 1;
  const limit = options.limit || 20;
  const offset = (page - 1) * limit;
  const paginatedItems = items.slice(offset, offset + limit);
  const totalPages = Math.ceil(total / limit);

  return {
    items: paginatedItems,
    total,
    page,
    totalPages,
    hasMore: page < totalPages,
  };
}

/**
 * Collection names
 */
export const COLLECTIONS = {
  LABS: "labs",
  STARS: "stars",
  FORKS: "forks",
  USERS: "users",
  ACTIVITIES: "lab_activities",
  PAPERS: "papers",
} as const;

/**
 * Initialize database with seed data
 */
export async function initializeDatabase(): Promise<void> {
  await ensureDataDir();

  // Check if labs collection exists, if not create with seed data
  const labs = await readCollection(COLLECTIONS.LABS);
  if (labs.length === 0) {
    // Will be populated when labs are created
    await writeCollection(COLLECTIONS.LABS, []);
  }

  const stars = await readCollection(COLLECTIONS.STARS);
  if (stars.length === 0) {
    await writeCollection(COLLECTIONS.STARS, []);
  }

  const forks = await readCollection(COLLECTIONS.FORKS);
  if (forks.length === 0) {
    await writeCollection(COLLECTIONS.FORKS, []);
  }
}
