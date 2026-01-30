# Phase 6: Social Layer

Complete documentation for the Social Layer system implemented in Phase 6.

## Overview

Phase 6 introduces a comprehensive social layer that enables community interaction and engagement:
- **Results**: Share research findings with metrics and media
- **Comments**: Threaded discussions with reactions and mentions
- **Activity Feed**: Track actions across labs
- **Suggestions**: Community-driven research direction voting
- **Notifications**: Real-time alerts for engagement

## Architecture

```
/lib/social/
  results/
    types.ts      - Result, metric, media types
    repository.ts - CRUD for results, likes, saves
    index.ts      - Module exports

  comments/
    types.ts      - Comment, reaction, mention types
    repository.ts - CRUD for comments, reactions
    index.ts      - Module exports

  activity/
    types.ts      - Activity types and utilities
    service.ts    - Activity logging and aggregation
    index.ts      - Module exports

  suggestions/
    types.ts      - Suggestion, vote types
    repository.ts - CRUD for suggestions, votes
    index.ts      - Module exports

  notifications/
    types.ts      - Notification types
    service.ts    - Notification creation and delivery
    index.ts      - Module exports

/app/api/
  results/
    route.ts                - List/create results
    [id]/
      route.ts             - Get/update/delete result
      like/route.ts        - Like operations
      save/route.ts        - Save/bookmark operations
      publish/route.ts     - Publish draft results
    trending/route.ts      - Trending results
    featured/route.ts      - Featured results

  comments/
    route.ts              - List/create comments
    [id]/
      route.ts           - Get/update/delete comment
      reactions/route.ts - Reaction operations

  activity/
    route.ts             - Activity feed

  social/suggestions/
    route.ts             - List/create suggestions
    [id]/
      route.ts          - Get/update/delete suggestion
      vote/route.ts     - Vote operations

  notifications/
    route.ts             - List/mark-read notifications
    [id]/route.ts        - Single notification operations

/app/feed/
  page.tsx              - Activity feed page

/components/social/
  ResultCard.tsx         - Result display card
  CreateResultDialog.tsx - Result creation modal
  CommentThread.tsx      - Threaded comment display
  CommentForm.tsx        - Comment input
  ActivityFeed.tsx       - Feed component
  FeedItem.tsx          - Single activity item
  SuggestionCard.tsx    - Suggestion display
  SuggestionForm.tsx    - Suggestion creation
  NotificationBell.tsx  - Notification dropdown
  index.ts              - Module exports
```

## Data Models

### Result

```typescript
interface Result {
  id: string;                 // Unique ID (res_xxx)
  type: ResultType;           // model, demo, finding, comparison, dataset, paper
  title: string;              // Display title
  description: string;        // Short description
  content?: string;           // Full markdown content
  visibility: ResultVisibility; // public, private, unlisted
  status: ResultStatus;       // draft, published, archived
  author: ResultAuthor;       // Author info
  labId: string;              // Parent lab
  taskId?: string;            // Related task (optional)
  media: ResultMedia[];       // Attached media
  metrics?: ResultMetrics;    // Performance metrics
  metadata?: TypeMetadata;    // Type-specific data
  tags: string[];             // Discovery tags
  stats: ResultStats;         // Engagement counts
  isFeatured?: boolean;       // Featured status
  isPinned?: boolean;         // Pinned to top
  createdAt: string;
  updatedAt: string;
  publishedAt?: string;
}
```

### ResultMetric

```typescript
interface ResultMetric {
  name: string;          // e.g., "MOS", "WER", "RTF"
  value: number;         // Metric value
  unit?: string;         // e.g., "%", "ms", "score"
  higherIsBetter?: boolean;
  baseline?: number;     // Comparison baseline
  description?: string;
}
```

### ResultMedia

```typescript
interface ResultMedia {
  id: string;            // Unique media ID (med_xxx)
  type: MediaType;       // image, audio, video, chart, code, file
  url: string;           // Media URL
  thumbnailUrl?: string; // Preview thumbnail
  alt?: string;          // Alt text
  filename?: string;
  size?: number;         // Bytes
  mimeType?: string;
  duration?: number;     // For audio/video
  width?: number;        // For images/video
  height?: number;
  order: number;         // Display order
}
```

### Comment

```typescript
interface Comment {
  id: string;                    // Unique ID (cmt_xxx)
  entityType: CommentableEntity; // result, lab, task, paper
  entityId: string;              // Entity ID
  parentId?: string;             // Parent comment (for replies)
  rootId?: string;               // Root of thread
  author: CommentAuthor;         // Author info
  content: string;               // Markdown content
  contentPlain: string;          // Plain text for search
  mentions: CommentMention[];    // @mentions
  status: CommentStatus;         // active, deleted, hidden
  reactionCounts: ReactionCounts;
  replyCount: number;
  depth: number;                 // Thread depth (0 = top-level)
  createdAt: string;
  editedAt?: string;
  isEdited: boolean;
}
```

### ReactionCounts

```typescript
interface ReactionCounts {
  like: number;
  love: number;
  insightful: number;
  celebrate: number;
  curious: number;
  disagree: number;
}
```

### Activity

```typescript
interface Activity {
  id: string;                    // Unique ID (act_xxx)
  type: ActivityType;            // result_created, result_liked, etc.
  actor: ActivityActor;          // Who performed action
  target: ActivityTarget;        // What was affected
  labId?: string;                // Related lab
  metadata?: Record<string, unknown>;
  isPublic: boolean;
  createdAt: string;
}
```

### Suggestion

```typescript
interface Suggestion {
  id: string;                    // Unique ID (sug_xxx)
  labId: string;                 // Parent lab
  taskId?: string;               // Related task
  resultId?: string;             // Related result
  author: SuggestionAuthor;
  title: string;
  description: string;           // Markdown
  category: SuggestionCategory;  // research_direction, improvement, etc.
  status: SuggestionStatus;      // open, under_review, planned, etc.
  priority: SuggestionPriority;  // low, medium, high, critical
  tags: string[];
  stats: SuggestionStats;        // upvotes, downvotes, comments
  statusHistory: StatusChange[];
  isPinned?: boolean;
  createdAt: string;
  updatedAt: string;
  resolvedAt?: string;
}
```

### Notification

```typescript
interface Notification {
  id: string;                    // Unique ID (notif_xxx)
  userId: string;                // Recipient
  type: NotificationType;        // result_liked, comment_received, etc.
  title: string;                 // Short title
  message?: string;              // Longer message
  actorId?: string;              // Who triggered it
  actorUsername?: string;
  actorAvatar?: string;
  targetType?: string;           // result, comment, etc.
  targetId?: string;
  targetTitle?: string;
  targetUrl?: string;
  activityId?: string;           // Related activity
  isRead: boolean;
  readAt?: string;
  createdAt: string;
}
```

## API Reference

### Results API

#### List Results
```
GET /api/results?labId=xxx&type=finding&status=published&sortBy=likes&page=1&limit=12
```

Query parameters:
- `labId`: Filter by lab
- `authorId`: Filter by author
- `type`: Filter by result type
- `visibility`: Filter by visibility
- `status`: Filter by status
- `tags`: Filter by tags (comma-separated)
- `search`: Text search
- `sortBy`: Sort field (likes, comments, views, created, updated)
- `sortDir`: Sort direction (asc, desc)
- `page`: Page number (1-based)
- `limit`: Items per page

Response:
```json
{
  "success": true,
  "results": [...],
  "total": 42,
  "page": 1,
  "totalPages": 4,
  "hasMore": true
}
```

#### Create Result
```
POST /api/results
Content-Type: application/json

{
  "type": "finding",
  "title": "Improved Prosody Control",
  "description": "Novel approach to emotion transfer",
  "content": "## Summary\n\n...",
  "labId": "lab-1",
  "visibility": "public",
  "tags": ["prosody", "emotion"]
}
```

#### Get Result
```
GET /api/results/{id}
```

#### Update Result
```
PATCH /api/results/{id}
Content-Type: application/json

{
  "title": "Updated Title",
  "status": "published",
  "metrics": {
    "primary": [
      { "name": "MOS", "value": 4.2, "higherIsBetter": true }
    ]
  }
}
```

#### Delete Result
```
DELETE /api/results/{id}
```

#### Like Operations
```
GET /api/results/{id}/like    # Check like status
POST /api/results/{id}/like   # Like result
DELETE /api/results/{id}/like # Unlike result
```

#### Save Operations
```
GET /api/results/{id}/save    # Check save status
POST /api/results/{id}/save   # Save result
DELETE /api/results/{id}/save # Unsave result
```

#### Publish Result
```
POST /api/results/{id}/publish
```

#### Trending Results
```
GET /api/results/trending?limit=10
```

#### Featured Results
```
GET /api/results/featured?limit=5
```

### Comments API

#### List Comments
```
GET /api/comments?entityType=result&entityId=xxx&topLevelOnly=true&includeReplies=true&maxDepth=3
```

Query parameters:
- `entityType`: Entity type (result, lab, task, paper)
- `entityId`: Entity ID
- `topLevelOnly`: Only top-level comments
- `parentId`: Replies to specific comment
- `includeReplies`: Include nested replies
- `maxDepth`: Max reply depth
- `sortBy`: Sort order (newest, oldest, popular)
- `page`: Page number
- `limit`: Items per page

#### Create Comment
```
POST /api/comments
Content-Type: application/json

{
  "entityType": "result",
  "entityId": "res_123",
  "content": "Great work! @alice mentioned something similar.",
  "parentId": "cmt_parent" // Optional, for replies
}
```

#### Update Comment
```
PATCH /api/comments/{id}
Content-Type: application/json

{
  "content": "Updated content"
}
```

#### Delete Comment
```
DELETE /api/comments/{id}
```

#### Reaction Operations
```
GET /api/comments/{id}/reactions     # Get reaction counts
POST /api/comments/{id}/reactions    # Add/change reaction
Content-Type: application/json
{ "type": "insightful" }

DELETE /api/comments/{id}/reactions  # Remove reaction
```

### Activity API

#### Get Activity Feed
```
GET /api/activity?labId=xxx&userId=xxx&types=result_liked,comment_created&aggregate=true&page=1&limit=20
```

Query parameters:
- `labId`: Filter by lab
- `userId`: Filter by user
- `types`: Activity types (comma-separated)
- `aggregate`: Aggregate similar activities
- `publicOnly`: Only public activities
- `page`: Page number
- `limit`: Items per page

Response:
```json
{
  "success": true,
  "activities": [
    {
      "id": "act_123",
      "type": "result_liked",
      "actor": { "id": "user-1", "displayName": "Alice" },
      "target": { "type": "result", "id": "res_123", "title": "..." },
      "otherActors": [...], // If aggregated
      "totalCount": 5,      // If aggregated
      "createdAt": "2024-01-01T00:00:00Z"
    }
  ],
  "total": 100,
  "page": 1,
  "hasMore": true
}
```

### Suggestions API

#### List Suggestions
```
GET /api/social/suggestions?labId=xxx&category=feature_request&status=open&sortBy=votes
```

Query parameters:
- `labId`: Filter by lab
- `authorId`: Filter by author
- `category`: Filter by category
- `status`: Filter by status (can be comma-separated)
- `priority`: Filter by priority
- `search`: Text search
- `sortBy`: Sort field (votes, comments, newest, oldest, priority)
- `page`: Page number
- `limit`: Items per page

#### Create Suggestion
```
POST /api/social/suggestions
Content-Type: application/json

{
  "labId": "lab-1",
  "title": "Add multi-speaker support",
  "description": "Would be great to support multiple speakers...",
  "category": "feature_request",
  "priority": "medium",
  "tags": ["multi-speaker"]
}
```

#### Update Suggestion
```
PATCH /api/social/suggestions/{id}
Content-Type: application/json

{
  "status": "planned",
  "statusReason": "Scheduled for Q2"
}
```

#### Delete Suggestion
```
DELETE /api/social/suggestions/{id}
```

#### Vote Operations
```
GET /api/social/suggestions/{id}/vote    # Get user's vote
POST /api/social/suggestions/{id}/vote   # Vote
Content-Type: application/json
{ "vote": 1 }  // 1 = upvote, -1 = downvote

DELETE /api/social/suggestions/{id}/vote # Remove vote
```

### Notifications API

#### List Notifications
```
GET /api/notifications?unreadOnly=true&types=result_liked,comment_received&page=1&limit=20
```

Query parameters:
- `unreadOnly`: Only unread notifications
- `types`: Notification types (comma-separated)
- `countOnly`: Return only unread count
- `page`: Page number
- `limit`: Items per page

#### Mark Notification as Read
```
PATCH /api/notifications/{id}
Content-Type: application/json

{ "isRead": true }
```

#### Mark All as Read
```
PATCH /api/notifications
Content-Type: application/json

{ "markAllRead": true }
```

#### Delete Notification
```
DELETE /api/notifications/{id}
```

## Components

### ResultCard

Displays a result with media preview and engagement stats.

```tsx
<ResultCard
  result={result}
  isLiked={false}
  isSaved={false}
  onLike={() => {}}
  onSave={() => {}}
  onClick={() => {}}
/>
```

### CreateResultDialog

Modal for creating new results.

```tsx
<CreateResultDialog
  labId="lab-1"
  isOpen={true}
  onClose={() => {}}
  onSuccess={(result) => {}}
/>
```

### CommentThread

Displays a threaded comment with nested replies.

```tsx
<CommentThread
  comment={commentWithReplies}
  currentUserId="user-1"
  onReply={(parentId, content) => {}}
  onEdit={(commentId, content) => {}}
  onDelete={(commentId) => {}}
  onReact={(commentId, type) => {}}
/>
```

### CommentForm

Comment input with mention support.

```tsx
<CommentForm
  entityType="result"
  entityId="res_123"
  parentId={null}
  placeholder="Add a comment..."
  onSubmit={async (content) => {}}
  onCancel={() => {}}
/>
```

### ActivityFeed

Live-updating activity feed with grouping.

```tsx
<ActivityFeed
  labId="lab-1"
  userId={null}
  limit={20}
  refreshInterval={30000}
  showFilters={true}
  aggregate={true}
/>
```

### FeedItem

Single activity item display.

```tsx
<FeedItem
  activity={activity}
  onClick={() => {}}
/>
```

### SuggestionCard

Displays a suggestion with voting.

```tsx
<SuggestionCard
  suggestion={suggestion}
  userVote={1}
  onVote={(vote) => {}}
  onClick={() => {}}
/>
```

### SuggestionForm

Suggestion creation form.

```tsx
<SuggestionForm
  labId="lab-1"
  onSubmit={async (data) => {}}
  onCancel={() => {}}
/>
```

### NotificationBell

Notification dropdown with badge.

```tsx
<NotificationBell
  userId="user-1"
  refreshInterval={30000}
  maxNotifications={10}
/>
```

## Utility Functions

### Result Utilities

```typescript
import {
  generateResultId,
  generateMediaId,
  formatMetricValue,
  calculateImprovement,
  getResultPath,
  canEditResult,
  canViewResult,
  isResult,
  isModelMetadata,
  isDemoMetadata,
  isFindingMetadata,
  isComparisonMetadata,
} from "@/lib/social/results/types";

// Format metric for display
formatMetricValue({ name: "MOS", value: 4.2 }); // "4.20"
formatMetricValue({ name: "acc", value: 0.95, unit: "%" }); // "95.0%"
formatMetricValue({ name: "params", value: 1500000 }); // "1.5M"

// Calculate improvement over baseline
calculateImprovement({ name: "MOS", value: 4.5, baseline: 4.0, higherIsBetter: true }); // 12.5
```

### Comment Utilities

```typescript
import {
  generateCommentId,
  extractMentions,
  stripMarkdown,
  getTotalReactions,
  canEditComment,
  canDeleteComment,
  isComment,
  MAX_COMMENT_LENGTH,
  MAX_REPLY_DEPTH,
} from "@/lib/social/comments/types";

// Extract @mentions from content
extractMentions("Thanks @alice and @bob!");
// [{ username: "alice", startIndex: 7, ... }, { username: "bob", startIndex: 19, ... }]

// Strip markdown for plain text
stripMarkdown("**Bold** and [link](url)"); // "Bold and link"
```

### Activity Utilities

```typescript
import {
  generateActivityId,
  getActivityDescription,
  getAggregatedDescription,
  formatActivityTime,
  groupActivitiesByDate,
  getDateGroupLabel,
  shouldNotify,
  isActivity,
} from "@/lib/social/activity/types";

// Format activity time
formatActivityTime(new Date().toISOString()); // "just now"
formatActivityTime(tenMinutesAgo); // "10m ago"
formatActivityTime(threeDaysAgo); // "3d ago"

// Group by date
const groups = groupActivitiesByDate(activities);
// Map { "2024-01-15" => [...], "2024-01-14" => [...] }

// Get group label
getDateGroupLabel("2024-01-15"); // "Today" or "Yesterday" or "Jan 15"
```

### Suggestion Utilities

```typescript
import {
  generateSuggestionId,
  getNetVotes,
  isClosedStatus,
  canEditSuggestion,
  isSuggestion,
} from "@/lib/social/suggestions/types";

// Get net vote score
getNetVotes({ upvotes: 10, downvotes: 3, comments: 0 }); // 7

// Check if status is closed
isClosedStatus("completed"); // true
isClosedStatus("open"); // false
```

### Notification Utilities

```typescript
import {
  generateNotificationId,
  getNotificationUrl,
  isNotification,
  NOTIFICATION_TYPE_ICONS,
  NOTIFICATION_TYPE_COLORS,
} from "@/lib/social/notifications/types";
```

## Constants

### Result Types
```typescript
const RESULT_TYPE_LABELS = {
  model: "Model",
  demo: "Demo",
  finding: "Finding",
  comparison: "Comparison",
  dataset: "Dataset",
  paper: "Paper",
};

const RESULT_TYPE_ICONS = {
  model: "Box",
  demo: "Play",
  finding: "Lightbulb",
  comparison: "GitCompare",
  dataset: "Database",
  paper: "FileText",
};
```

### Reaction Types
```typescript
const REACTION_LABELS = {
  like: "Like",
  love: "Love",
  insightful: "Insightful",
  celebrate: "Celebrate",
  curious: "Curious",
  disagree: "Disagree",
};

const REACTION_EMOJIS = {
  like: "👍",
  love: "❤️",
  insightful: "💡",
  celebrate: "🎉",
  curious: "🤔",
  disagree: "👎",
};
```

### Suggestion Categories
```typescript
const CATEGORY_LABELS = {
  research_direction: "Research Direction",
  improvement: "Improvement",
  bug_report: "Bug Report",
  feature_request: "Feature Request",
  question: "Question",
  collaboration: "Collaboration",
};
```

### Suggestion Statuses
```typescript
const STATUS_LABELS = {
  open: "Open",
  under_review: "Under Review",
  planned: "Planned",
  in_progress: "In Progress",
  completed: "Completed",
  declined: "Declined",
  duplicate: "Duplicate",
};

const STATUS_COLORS = {
  open: "text-blue-400 bg-blue-500/10",
  under_review: "text-yellow-400 bg-yellow-500/10",
  planned: "text-purple-400 bg-purple-500/10",
  in_progress: "text-orange-400 bg-orange-500/10",
  completed: "text-green-400 bg-green-500/10",
  declined: "text-red-400 bg-red-500/10",
  duplicate: "text-foreground-muted bg-foreground-muted/10",
};
```

## Pages

### /feed

Activity feed page showing:
- Combined feed from followed labs
- Filter by activity type
- Infinite scroll pagination
- Date grouping
- Aggregated similar activities

## Testing

### Unit Tests

```bash
npm test -- --run __tests__/lib/social/results.test.ts     # 30 tests
npm test -- --run __tests__/lib/social/comments.test.ts    # 28 tests
npm test -- --run __tests__/lib/social/activity.test.ts    # 18 tests
npm test -- --run __tests__/lib/social/suggestions.test.ts # 10 tests
```

Total: 86 unit tests

### Integration Tests

```bash
npm test -- --run __tests__/integration/social-flow.test.ts
```

42 integration tests covering:
- Result creation and publishing
- Result discovery and filtering
- Like/save engagement
- Comment threads and reactions
- Activity feed
- Suggestion voting and status updates
- Notification management
- Error handling

## Build

```bash
npm run build
```

All Phase 6 pages and API routes are included in the production build:
- /feed - Activity feed page
- All /api/results/* routes
- All /api/comments/* routes
- /api/activity
- All /api/social/suggestions/* routes
- All /api/notifications/* routes

## Summary

Phase 6 delivers a complete social layer with:
- **31 completed tasks**
- **128 passing tests** (86 unit + 42 integration)
- **5 interconnected modules** (results, comments, activity, suggestions, notifications)
- **Full REST API** for all social features
- **Rich UI components** for engagement
- **Real-time notifications** with polling
- **Threaded comments** with reactions
- **Community suggestions** with voting

The social layer enables researchers to:
1. Share findings with metrics and media
2. Engage through likes, saves, and comments
3. Discuss with threaded conversations
4. Track activity across labs
5. Suggest improvements and new directions
6. Stay informed with notifications

The system is ready for Phase 7: Real-time Features (WebSocket updates, presence, live collaboration).
