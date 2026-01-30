# Contributor Profile & Leaderboard System

This document explains the user account and contribution tracking system for the LabFork distributed compute network.

## Overview

The contributor system tracks user contributions across the distributed compute network, awarding credits, badges, and ranks based on participation. It provides profile management, leaderboards, and global statistics.

## Architecture

### Database Tables

**contributor_profiles**
- `user_id` (string, primary key)
- `display_name` (string)
- `avatar_url` (string, nullable)
- `bio` (string, nullable)
- `badges` (json array)
- `joined_at` (timestamp)
- `updated_at` (timestamp)

This table stores user profile information separate from the stats, which are calculated from existing tables (`user_credits`, `devices`).

### Type Definitions

**ContributorProfile** (`/lib/compute/user-types.ts`)
```typescript
interface ContributorProfile {
  userId: string;
  displayName: string;
  avatarUrl?: string;
  bio?: string;
  rank: ContributorRank;
  totalCreditsEarned: number;
  totalTasksCompleted: number;
  totalComputeTime: number;
  devices: string[];
  badges: Badge[];
  joinedAt: string;
}
```

**ContributorRank**
- `novice`: < 10 tasks
- `contributor`: 10-99 tasks
- `expert`: 100-999 tasks
- `legend`: 1000+ tasks

**Badge**
```typescript
interface Badge {
  id: string;
  name: string;
  description: string;
  icon: string;
  earnedAt: string;
}
```

## Badge System

Badges are automatically awarded when conditions are met:

| Badge | Condition | Icon |
|-------|-----------|------|
| First Contribution | Complete 1 task | 🌟 |
| Century | Complete 100 tasks | 💯 |
| Millionaire | Earn 1,000 credits | 💰 |
| Legend | Complete 1,000 tasks | 🏆 |
| Power Contributor | Register a power-tier device | ⚡ |
| Week Warrior | Contribute every day for a week | 🔥 |

Badges are checked and awarded automatically when fetching a profile.

## API Endpoints

### GET /api/contributor/[userId]
Get contributor profile by user ID.

**Response:**
```json
{
  "userId": "user123",
  "displayName": "Contributor abc123",
  "rank": "contributor",
  "totalCreditsEarned": 250,
  "totalTasksCompleted": 42,
  "totalComputeTime": 3600,
  "devices": ["dev_xyz"],
  "badges": [
    {
      "id": "first_task",
      "name": "First Contribution",
      "description": "Completed your first task",
      "icon": "🌟",
      "earnedAt": "2026-01-30T12:00:00Z"
    }
  ],
  "joinedAt": "2026-01-30T12:00:00Z"
}
```

### PATCH /api/contributor/[userId]
Update contributor profile.

**Request body:**
```json
{
  "displayName": "My New Name",
  "bio": "AI researcher contributing compute power",
  "avatarUrl": "https://example.com/avatar.jpg"
}
```

**Validation:**
- `displayName`: Max 50 characters
- `bio`: Max 500 characters

### GET /api/contributor/leaderboard
Get top contributors ranked by credits earned.

**Query params:**
- `limit` (optional): Number of contributors (1-100, default 10)

**Response:**
```json
[
  {
    "userId": "user123",
    "displayName": "Top Contributor",
    "rank": "legend",
    "totalCreditsEarned": 50000,
    "totalTasksCompleted": 5000,
    "badges": [...],
    ...
  },
  ...
]
```

### GET /api/contributor/stats
Get global contribution statistics.

**Response:**
```json
{
  "totalContributors": 150,
  "totalCreditsEarned": 100000,
  "totalTasksCompleted": 10000
}
```

## React Hooks

### useContributor(userId)
Fetch and display a contributor profile.

```typescript
import { useContributor } from "@/lib/compute/useContributor";

function ProfilePage({ userId }) {
  const { profile, isLoading, error, refresh } = useContributor(userId);

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div>
      <h1>{profile.displayName}</h1>
      <p>Rank: {profile.rank}</p>
      <p>Credits: {profile.totalCreditsEarned}</p>
    </div>
  );
}
```

### useLeaderboard(limit)
Fetch top contributors.

```typescript
import { useLeaderboard } from "@/lib/compute/useContributor";

function LeaderboardPage() {
  const { leaders, isLoading, error, refresh } = useLeaderboard(10);

  return (
    <div>
      {leaders.map((contributor, index) => (
        <div key={contributor.userId}>
          #{index + 1} {contributor.displayName} - {contributor.totalCreditsEarned} credits
        </div>
      ))}
    </div>
  );
}
```

### useContributionStats()
Fetch global statistics.

```typescript
import { useContributionStats } from "@/lib/compute/useContributor";

function StatsPage() {
  const {
    totalContributors,
    totalCreditsEarned,
    totalTasksCompleted,
    isLoading,
    error,
    refresh
  } = useContributionStats();

  return (
    <div>
      <p>Contributors: {totalContributors}</p>
      <p>Credits Earned: {totalCreditsEarned}</p>
      <p>Tasks Completed: {totalTasksCompleted}</p>
    </div>
  );
}
```

## Components

### ContributorProfile
Display a contributor's profile with stats, badges, and devices.

```tsx
import { ContributorProfile } from "@/components/compute";

<ContributorProfile
  userId="user123"
  showBadges={true}
  showDevices={false}
/>
```

**Props:**
- `userId` (required): User ID to display
- `showBadges` (optional): Show badges section (default true)
- `showDevices` (optional): Show devices section (default false)

### Leaderboard
Display top contributors ranked by credits.

```tsx
import { Leaderboard } from "@/components/compute";

<Leaderboard limit={10} />
```

**Props:**
- `limit` (optional): Number of contributors to show (default 10)

## Database Setup

To set up the contributor_profiles table in Supabase:

```sql
-- Create contributor_profiles table
CREATE TABLE contributor_profiles (
  user_id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  avatar_url TEXT,
  bio TEXT,
  badges JSONB DEFAULT '[]'::jsonb,
  joined_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index on joined_at for sorting
CREATE INDEX idx_contributor_profiles_joined_at ON contributor_profiles(joined_at);

-- Enable Row Level Security
ALTER TABLE contributor_profiles ENABLE ROW LEVEL SECURITY;

-- Allow anyone to read profiles (public leaderboard)
CREATE POLICY "Profiles are viewable by everyone"
  ON contributor_profiles FOR SELECT
  USING (true);

-- Allow users to update their own profile
CREATE POLICY "Users can update own profile"
  ON contributor_profiles FOR UPDATE
  USING (auth.uid()::text = user_id);

-- Allow users to insert their own profile
CREATE POLICY "Users can insert own profile"
  ON contributor_profiles FOR INSERT
  WITH CHECK (auth.uid()::text = user_id);
```

## Integration with Device Registration

When a user registers a device, their contributor profile is automatically initialized:

```typescript
// In /app/api/compute/devices/route.ts
const device = orchestrator.registerDevice(body, userId);

// Initialize contributor profile
if (userId) {
  await initializeContributorProfile(userId);
}
```

## Graceful Degradation

The system handles missing Supabase configuration gracefully:
- All functions return `null` or empty arrays if Supabase is not configured
- Components display appropriate loading/error states
- No errors thrown if database is unavailable

## Future Enhancements

Potential improvements to the contributor system:

1. **Streak Tracking**: Track consecutive days of contribution
2. **Custom Badges**: Allow labs to award custom badges
3. **Social Features**: Following, activity feeds, comments
4. **OAuth Integration**: Google, GitHub, Discord login
5. **Profile Verification**: Verified contributor badges
6. **Team Formation**: Allow contributors to form teams
7. **Regional Leaderboards**: Rank by country/region
8. **Time-based Leaderboards**: Daily, weekly, monthly rankings
9. **Referral System**: Reward users for inviting others
10. **NFT Badges**: Mint badges as NFTs on blockchain

## Example Usage

### Display Current User Profile
```tsx
import { ContributorProfile } from "@/components/compute";

export default function MyProfilePage() {
  const userId = "current-user-id"; // Get from auth context

  return (
    <div className="container mx-auto p-8">
      <h1 className="text-3xl font-bold mb-8">My Profile</h1>
      <ContributorProfile
        userId={userId}
        showBadges={true}
        showDevices={true}
      />
    </div>
  );
}
```

### Display Leaderboard
```tsx
import { Leaderboard } from "@/components/compute";

export default function LeaderboardPage() {
  return (
    <div className="container mx-auto p-8">
      <h1 className="text-3xl font-bold mb-8">Top Contributors</h1>
      <Leaderboard limit={25} />
    </div>
  );
}
```

### Update Profile
```tsx
import { useState } from "react";

export default function EditProfilePage() {
  const [displayName, setDisplayName] = useState("");
  const [bio, setBio] = useState("");
  const [saving, setSaving] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);

    try {
      const response = await fetch("/api/contributor/user123", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ displayName, bio }),
      });

      if (response.ok) {
        alert("Profile updated!");
      }
    } catch (error) {
      alert("Failed to update profile");
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm font-medium mb-1">
          Display Name
        </label>
        <input
          type="text"
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
          maxLength={50}
          className="w-full px-3 py-2 border rounded-lg"
        />
      </div>

      <div>
        <label className="block text-sm font-medium mb-1">
          Bio
        </label>
        <textarea
          value={bio}
          onChange={(e) => setBio(e.target.value)}
          maxLength={500}
          rows={4}
          className="w-full px-3 py-2 border rounded-lg"
        />
      </div>

      <button
        type="submit"
        disabled={saving}
        className="px-4 py-2 bg-blue-600 text-white rounded-lg"
      >
        {saving ? "Saving..." : "Save Profile"}
      </button>
    </form>
  );
}
```
