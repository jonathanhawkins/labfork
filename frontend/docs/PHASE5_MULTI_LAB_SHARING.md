# Phase 5: Multi-Lab & Sharing

Complete documentation for the Multi-Lab and Sharing system implemented in Phase 5.

## Overview

Phase 5 introduces a complete lab sharing ecosystem that allows users to:
- Create and manage their own labs
- Discover public labs from the community
- Star and fork labs
- Share labs via links, embeds, and social media
- View live 3D lab visualizations

## Architecture

```
/lib/labs/
  types.ts      - Type definitions for labs, stars, forks
  repository.ts - CRUD operations for labs
  social.ts     - Star/fork operations
  search.ts     - Search and filter utilities
  index.ts      - Module exports

/lib/db/
  json-store.ts - JSON file-based storage layer

/lib/auth/
  mock-user.ts  - Mock authentication for development

/app/api/labs/
  route.ts              - List/create labs
  [id]/
    route.ts           - Get/update/delete single lab
    star/route.ts      - Star operations
    fork/route.ts      - Fork operations
    stats/route.ts     - Lab statistics
    og-image/route.tsx - Social preview image generation
  user/[username]/route.ts - User's labs

/app/explore/
  page.tsx - Browse and discover public labs

/app/labs/
  [username]/
    page.tsx          - User's lab listing
    [slug]/
      page.tsx        - Full lab portal
      layout.tsx      - Meta tags for social sharing

/components/labs/
  LabCard.tsx       - Card display for listings
  LabHeader.tsx     - Lab page header with actions
  StarButton.tsx    - Star/unstar with optimistic updates
  ForkDialog.tsx    - Fork modal with customization
  ShareDialog.tsx   - Share via link, embed, social
  LiveLabViewer.tsx - Embedded 3D lab viewer
```

## Data Models

### Lab

```typescript
interface Lab {
  id: string;              // Unique lab ID (lab_xxx)
  slug: string;            // URL-safe slug (unique per user)
  name: string;            // Display name
  description: string;     // Short description
  readme?: string;         // Long description/README
  domainSlug: string;      // Domain category
  domainName: string;      // Domain display name
  owner: LabOwner;         // Owner info
  visibility: "public" | "private" | "unlisted";
  status: "active" | "archived" | "suspended";
  stats: LabStats;         // Counts (stars, forks, tasks, etc.)
  forkedFrom?: ForkInfo;   // Fork source info
  tags: string[];          // Discovery tags
  isFeatured?: boolean;    // Featured status
  createdAt: string;
  updatedAt: string;
  lastActivityAt: string;
}
```

### LabStats

```typescript
interface LabStats {
  stars: number;       // Star count
  forks: number;       // Fork count
  tasks: number;       // Active tasks
  papers: number;      // Research papers
  experiments: number; // Experiments run
  viewers: number;     // Live viewers
}
```

### Star

```typescript
interface Star {
  userId: string;   // User who starred
  labId: string;    // Lab that was starred
  createdAt: string;
}
```

### Fork

```typescript
interface Fork {
  id: string;          // Fork record ID
  sourceLabId: string; // Original lab
  forkedLabId: string; // Forked lab
  userId: string;      // User who forked
  createdAt: string;
}
```

## API Reference

### List Labs
```
GET /api/labs?visibility=public&domain=voice-clone&search=prosody&sortBy=stars&page=1&limit=12
```

Query parameters:
- `visibility`: Filter by visibility (public, private, unlisted)
- `domain`: Filter by domain slug
- `search`: Text search (name, description, tags)
- `tags`: Filter by tags (comma-separated)
- `sortBy`: Sort field (stars, forks, activity, created, name)
- `sortDir`: Sort direction (asc, desc)
- `page`: Page number (1-based)
- `limit`: Items per page

Response:
```json
{
  "success": true,
  "labs": [...],
  "total": 42,
  "page": 1,
  "totalPages": 4,
  "hasMore": true
}
```

### Create Lab
```
POST /api/labs
Content-Type: application/json

{
  "name": "My Lab",
  "slug": "my-lab",
  "description": "Research on voice cloning",
  "domainSlug": "voice-clone",
  "visibility": "public",
  "tags": ["tts", "voice"]
}
```

### Get Lab
```
GET /api/labs/{id}
```

Response includes social stats:
```json
{
  "success": true,
  "lab": {...},
  "social": {
    "stars": 42,
    "forks": 10,
    "isStarred": false,
    "isFork": false
  }
}
```

### Update Lab
```
PATCH /api/labs/{id}
Content-Type: application/json

{
  "name": "Updated Name",
  "description": "New description",
  "visibility": "private"
}
```

### Delete Lab
```
DELETE /api/labs/{id}
```

### Star Operations
```
GET /api/labs/{id}/star    # Check star status
POST /api/labs/{id}/star   # Star/toggle star
DELETE /api/labs/{id}/star # Unstar
```

### Fork Operations
```
GET /api/labs/{id}/fork    # Get fork info and lineage
POST /api/labs/{id}/fork   # Fork the lab
Content-Type: application/json

{
  "slug": "my-fork",
  "name": "My Fork"
}
```

### Lab Stats
```
GET /api/labs/{id}/stats
```

### OG Image
```
GET /api/labs/{id}/og-image
```
Returns a dynamically generated 1200x630 PNG image for social sharing.

## Components

### LabCard

Displays a lab in card format for grids and listings.

```tsx
<LabCard
  lab={lab}
  compact={false}
  showStar={true}
  isStarred={false}
  onStarClick={() => {}}
  onClick={() => {}}
  isSelected={false}
/>
```

### LabHeader

Full header for lab pages with stats and actions.

```tsx
<LabHeader
  lab={lab}
  isStarred={false}
  isOwner={true}
  onStarToggle={(starred, count) => {}}
  onForkSuccess={(forkedLab) => {}}
  onShareClick={() => {}}
  onSettingsClick={() => {}}
/>
```

### StarButton

Optimistic star/unstar button.

```tsx
<StarButton
  labId="lab-1"
  initialCount={42}
  initialStarred={false}
  size="md"
  showCount={true}
  onToggle={(starred, count) => {}}
/>
```

### ForkDialog

Modal for forking a lab.

```tsx
<ForkDialog
  lab={lab}
  isOpen={true}
  onClose={() => {}}
  onSuccess={(forkedLab) => {}}
/>
```

### ShareDialog

Modal for sharing a lab.

```tsx
<ShareDialog
  lab={lab}
  isOpen={true}
  onClose={() => {}}
/>
```

### LiveLabViewer

Embedded 3D lab visualization.

```tsx
<LiveLabViewer
  lab={lab}
  readOnly={true}
  refreshInterval={5000}
  showViewers={true}
  showActivity={true}
  allowFullscreen={true}
  compact={false}
/>
```

## Search and Filter Utilities

```typescript
import {
  searchLabsInMemory,
  filterByDomain,
  filterByVisibility,
  filterByFeatured,
  filterByTags,
  sortLabs,
  paginateLabs,
  applyLabFilters,
  getTrendingLabs,
  getRelatedLabs,
  parseSearchQuery,
  highlightMatches,
  getSearchSuggestions,
  calculateRelevanceScore,
} from "@/lib/labs/search";

// Apply all filters at once
const result = applyLabFilters(labs, {
  domain: "voice-clone",
  visibility: "public",
  search: "prosody",
  sortBy: "stars",
  page: 1,
  limit: 12,
});

// Get trending labs
const trending = getTrendingLabs(labs, 10);

// Get related labs
const related = getRelatedLabs(currentLab, allLabs, 5);

// Parse advanced search query
const { text, filters } = parseSearchQuery("prosody domain:voice-clone tag:tts");
```

## Database Layer

The system uses a JSON file-based storage layer for simplicity:

```typescript
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
} from "@/lib/db/json-store";

// Find a lab
const lab = await findOne<Lab>(COLLECTIONS.LABS, (l) => l.id === labId);

// Insert a star
await insertOne(COLLECTIONS.STARS, { userId, labId, createdAt: new Date().toISOString() });

// Count stars
const starCount = await count<Star>(COLLECTIONS.STARS, (s) => s.labId === labId);
```

Data is stored in `.data/` directory:
- `.data/labs.json`
- `.data/stars.json`
- `.data/forks.json`
- `.data/users.json`
- `.data/activities.json`

## Mock Authentication

For development, a mock auth system is provided:

```typescript
import { getCurrentUser, setCurrentUser, getServerUser } from "@/lib/auth/mock-user";

// Get current user (client-side)
const user = getCurrentUser();

// Set current user (for testing)
setCurrentUser("researcher");

// Get user on server
const serverUser = await getServerUser(headers);
```

## Pages

### /explore

Browse and discover public labs with:
- Search bar with text search
- Domain filters
- Sort options (popular, trending, recent, active)
- Featured labs section
- Paginated grid of lab cards

### /labs/[username]

User's lab listing showing:
- User profile header with stats
- Grid of user's public labs

### /labs/[username]/[slug]

Full lab portal with:
- Lab header with stats and actions (star, fork, share)
- Tab navigation (Overview, Tasks, Activity, Settings)
- 3D lab viewer placeholder
- Description/README
- Quick stats sidebar
- Activity feed
- Settings (owner only)

## Testing

### Unit Tests

```bash
npm test -- --run __tests__/lib/labs/
npm test -- --run __tests__/components/labs/
```

Test coverage:
- Lab types and utilities (35 tests)
- Lab components (45 tests)
- Social services (40+ tests)

### Integration Tests

```bash
npm test -- --run __tests__/integration/lab-flow.test.ts
```

Tests the complete user journey:
1. Create lab
2. View in explore
3. Star
4. Fork
5. Share

## Build

```bash
npm run build
```

All Phase 5 pages and API routes are included in the production build:
- /explore - Static
- /labs/[username] - Dynamic
- /labs/[username]/[slug] - Dynamic
- All API routes - Serverless functions

## Summary

Phase 5 delivers a complete lab sharing ecosystem with:
- 22 completed tasks
- 747 passing tests
- Full API for lab CRUD, stars, forks
- Rich UI components for discovery and sharing
- Search and filter utilities
- Social meta tags and OG images
- Mock auth for development

The system is ready for Phase 6: Social Layer (Result Cards, Comments, Feeds).
