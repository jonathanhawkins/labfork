# Contributor System Quick Start

This guide shows you how to quickly integrate the contributor profile and leaderboard system into your LabFork application.

## Prerequisites

1. **Supabase Setup**: The contributor system requires Supabase to be configured.
2. **Database Table**: The `contributor_profiles` table must be created in your Supabase database.

## Database Setup

Run this SQL in your Supabase SQL Editor:

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

-- Allow users to update their own profile (optional - requires auth)
CREATE POLICY "Users can update own profile"
  ON contributor_profiles FOR UPDATE
  USING (auth.uid()::text = user_id);

-- Allow users to insert their own profile (optional - requires auth)
CREATE POLICY "Users can insert own profile"
  ON contributor_profiles FOR INSERT
  WITH CHECK (auth.uid()::text = user_id);
```

## Quick Examples

### 1. Display User Profile

```tsx
import { ContributorProfile } from "@/components/compute";

export default function ProfilePage() {
  return (
    <div className="container mx-auto p-8">
      <ContributorProfile
        userId="user_abc123"
        showBadges={true}
        showDevices={true}
      />
    </div>
  );
}
```

### 2. Display Leaderboard

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

### 3. Show Global Stats

```tsx
import { useContributionStats } from "@/lib/compute/useContributor";

export default function StatsWidget() {
  const { totalContributors, totalCreditsEarned, totalTasksCompleted, isLoading } =
    useContributionStats();

  if (isLoading) return <div>Loading...</div>;

  return (
    <div className="grid grid-cols-3 gap-4">
      <div className="text-center">
        <div className="text-3xl font-bold">{totalContributors}</div>
        <div className="text-sm text-gray-600">Contributors</div>
      </div>
      <div className="text-center">
        <div className="text-3xl font-bold">{totalCreditsEarned.toLocaleString()}</div>
        <div className="text-sm text-gray-600">Credits Earned</div>
      </div>
      <div className="text-center">
        <div className="text-3xl font-bold">{totalTasksCompleted.toLocaleString()}</div>
        <div className="text-sm text-gray-600">Tasks Completed</div>
      </div>
    </div>
  );
}
```

### 4. Update User Profile

```tsx
import { useState } from "react";

export default function EditProfile({ userId }: { userId: string }) {
  const [displayName, setDisplayName] = useState("");
  const [bio, setBio] = useState("");
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      const response = await fetch(`/api/contributor/${userId}`, {
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
    <div className="space-y-4">
      <input
        type="text"
        value={displayName}
        onChange={(e) => setDisplayName(e.target.value)}
        placeholder="Display Name"
        maxLength={50}
        className="w-full px-3 py-2 border rounded"
      />
      <textarea
        value={bio}
        onChange={(e) => setBio(e.target.value)}
        placeholder="Bio"
        maxLength={500}
        className="w-full px-3 py-2 border rounded"
      />
      <button
        onClick={handleSave}
        disabled={saving}
        className="px-4 py-2 bg-blue-600 text-white rounded"
      >
        {saving ? "Saving..." : "Save"}
      </button>
    </div>
  );
}
```

## API Endpoints

The following API endpoints are available:

- `GET /api/contributor/[userId]` - Get contributor profile
- `PATCH /api/contributor/[userId]` - Update profile (displayName, bio, avatarUrl)
- `GET /api/contributor/leaderboard?limit=10` - Get top contributors
- `GET /api/contributor/stats` - Get global statistics

## Automatic Features

The contributor system automatically:

1. **Initializes profiles** when users register a device
2. **Awards badges** when conditions are met
3. **Calculates ranks** based on tasks completed
4. **Aggregates stats** from credits and devices tables

## Badge System

Badges are automatically awarded when users meet these conditions:

| Badge | Requirement |
|-------|-------------|
| First Contribution (🌟) | Complete 1 task |
| Century (💯) | Complete 100 tasks |
| Millionaire (💰) | Earn 1,000 credits |
| Legend (🏆) | Complete 1,000 tasks |
| Power Contributor (⚡) | Register a power-tier device |

## Rank System

Ranks are automatically assigned based on tasks completed:

- **Novice**: < 10 tasks
- **Contributor**: 10-99 tasks
- **Expert**: 100-999 tasks
- **Legend**: 1000+ tasks

## Graceful Degradation

If Supabase is not configured:
- All functions return `null` or empty arrays
- No errors are thrown
- Components display appropriate empty states

This allows the system to work without Supabase in development.

## Testing Without Supabase

For local development without Supabase:

1. The hooks will return empty data
2. Components will show loading/empty states
3. API routes will return 500 errors (gracefully handled)

To test with real data, configure Supabase environment variables:

```env
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_key
```

## Full Documentation

For complete documentation, see:
- `/components/compute/CONTRIBUTOR_SYSTEM.md` - Comprehensive system documentation
- `/lib/compute/user-types.ts` - Type definitions
- `/lib/compute/useContributor.ts` - React hooks
- `/lib/supabase/contributors.ts` - Database helpers
