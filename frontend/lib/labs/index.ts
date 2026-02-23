/**
 * Labs Module Index
 *
 * Exports all lab-related types, utilities, and services.
 */

// Types
export type {
  Lab,
  LabVisibility,
  LabStatus,
  LabStats,
  LabOwner,
  LabActivity,
  LabResult,
  ForkInfo,
  Star,
  Fork,
  CreateLabInput,
  UpdateLabInput,
  LabListOptions,
  LabListResult,
} from "./types";

export {
  DEFAULT_LAB_STATS,
  generateLabId,
  isValidLabSlug,
  slugify,
  getLabPath,
  getLabApiPath,
  formatLabStats,
  canEditLab,
  canViewLab,
  isLab,
} from "./types";

// Repository
export {
  createLab,
  getLabById,
  getLabBySlug,
  updateLab,
  updateLabStats,
  updateLabActivity,
  deleteLab,
  listLabs,
  getLabsByUser,
  getPublicLabs,
  getFeaturedLabs,
  getRecentLabs,
  getActiveLabs,
  searchLabs,
  getLabsByDomain,
  isSlugAvailable,
  getUserLabCount,
  getTotalLabCount,
  incrementLabViews,
  setLabViewers,
} from "./repository";

// Social
export {
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
} from "./social";

export type { LabSocialStats } from "./social";

// Search and Filter
export {
  searchLabsInMemory,
  filterByDomain,
  filterByVisibility,
  filterByFeatured,
  filterByTags,
  filterByOwner,
  filterByStatus,
  sortLabs,
  paginateLabs,
  applyLabFilters,
  getTrendingLabs,
  getRelatedLabs,
  parseSearchQuery,
  highlightMatches,
  getSearchSuggestions,
  calculateRelevanceScore,
} from "./search";

export type { SortField, SortDirection } from "./search";
