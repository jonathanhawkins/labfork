# Client-Side Authentication Usage

This guide explains how to use proper Clerk authentication in client components.

## Overview

The authentication system has been updated to use Clerk's `useUser()` hook in client components instead of the mock user system. This ensures proper authentication state management and ownership detection.

## Components Updated

### Lab Detail Page (`/app/labs/[username]/[slug]/page.tsx`)

**Before:**
```typescript
import { getCurrentUser } from "@/lib/auth/mock-user";

// Inside component
const currentUser = getCurrentUser();
setIsOwner(currentUser?.id === detailData.lab.owner.id);
```

**After:**
```typescript
import { useUser } from "@clerk/nextjs";
import { getClientUser } from "@/lib/auth/client";

// Inside component
const { user: clerkUser, isLoaded: isUserLoaded } = useUser();
const currentUser = getClientUser(clerkUser);

// Wait for auth to load
useEffect(() => {
  if (isUserLoaded) {
    fetchLab();
  }
}, [isUserLoaded, fetchLab]);

// Show loading state
if (!isUserLoaded || isLoading) {
  return <Loader2 />;
}

// Use currentUser for ownership checks
setIsOwner(currentUser?.id === detailData.lab.owner.id);
```

### User Labs Page (`/app/labs/[username]/page.tsx`)

Same pattern as above - uses `useUser()` hook and `getClientUser()` helper.

## How It Works

### 1. Import Required Dependencies

```typescript
import { useUser } from "@clerk/nextjs";
import { getClientUser } from "@/lib/auth/client";
```

### 2. Get User Data

```typescript
const { user: clerkUser, isLoaded: isUserLoaded } = useUser();
const currentUser = getClientUser(clerkUser);
```

### 3. Wait for Auth to Load

```typescript
useEffect(() => {
  if (isUserLoaded) {
    // Fetch data that depends on auth
    fetchData();
  }
}, [isUserLoaded, fetchData]);
```

### 4. Handle Loading State

```typescript
if (!isUserLoaded || isLoading) {
  return <LoadingSpinner />;
}
```

## Authentication States

| State | `isUserLoaded` | `clerkUser` | `currentUser` | Meaning |
|-------|----------------|-------------|---------------|---------|
| Loading | `false` | `undefined` | `null` | Auth is loading |
| Authenticated (Clerk) | `true` | `{...}` | `{...}` | User is signed in via Clerk |
| Not authenticated (Clerk configured) | `true` | `null` | `null` | User is not signed in |
| Development (no Clerk) | `true` | `null` | `{...mock}` | Using mock user |

## The `getClientUser()` Helper

Located in `/lib/auth/client.ts`, this helper:

1. **Converts Clerk user to our User interface** - Maps Clerk's user object to our standardized format
2. **Falls back to mock user in development** - Returns mock user when Clerk is not configured
3. **Returns null in production** - Returns null when Clerk is configured but user is not authenticated

```typescript
export function getClientUser(clerkUser?: any): User | null {
  // If we have a Clerk user, convert it
  if (clerkUser) {
    return {
      id: clerkUser.id,
      username: clerkUser.username || ...,
      displayName: `${clerkUser.firstName} ${clerkUser.lastName}`,
      avatar: clerkUser.imageUrl,
      email: clerkUser.emailAddresses[0]?.emailAddress,
      createdAt: new Date(clerkUser.createdAt).toISOString(),
    };
  }

  // In development without Clerk, use mock user
  if (process.env.NODE_ENV === "development") {
    return getCurrentUser(); // Mock user from localStorage
  }

  // In production with Clerk configured, return null
  return null;
}
```

## Benefits

1. **Proper ownership detection** - `isOwner` now correctly compares real Clerk user IDs
2. **Loading states** - Waits for auth to load before fetching data
3. **Development convenience** - Still uses mock users in development
4. **Production ready** - Seamlessly works with Clerk in production
5. **Type safety** - All users conform to the same `User` interface

## Common Patterns

### Check if user owns a resource

```typescript
const { user: clerkUser, isLoaded: isUserLoaded } = useUser();
const currentUser = getClientUser(clerkUser);

const isOwner = currentUser?.id === resource.ownerId;
```

### Conditional rendering based on auth

```typescript
{currentUser ? (
  <button>Edit</button>
) : (
  <SignInButton>Sign in to edit</SignInButton>
)}
```

### Fetch user-specific data

```typescript
useEffect(() => {
  if (isUserLoaded && currentUser) {
    fetchUserData(currentUser.id);
  }
}, [isUserLoaded, currentUser]);
```

## Migration Checklist

When migrating a component from mock auth to Clerk:

- [ ] Import `useUser` from `@clerk/nextjs`
- [ ] Import `getClientUser` from `@/lib/auth/client`
- [ ] Replace `getCurrentUser()` calls with `useUser()` hook
- [ ] Convert Clerk user with `getClientUser(clerkUser)`
- [ ] Add `isUserLoaded` check to data fetching
- [ ] Update loading state to check both `isUserLoaded` and data loading
- [ ] Update `useCallback`/`useEffect` dependencies to include `currentUser`
- [ ] Test in both development (mock) and production (Clerk) modes

## Testing

### Development Mode (No Clerk Keys)

```bash
# No Clerk keys in .env.local
npm run dev
```

- Should use mock user "researcher" by default
- Owner checks work with mock user IDs
- Can switch users via developer tools

### Production Mode (With Clerk Keys)

```bash
# Add to .env.local
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
CLERK_SECRET_KEY=sk_test_...

npm run dev
```

- Should show Clerk sign-in UI
- Owner checks work with real Clerk user IDs
- Returns null when not authenticated

## Troubleshooting

### Issue: `isOwner` always false

**Cause:** Using `getCurrentUser()` instead of `useUser()` hook

**Fix:** Import and use `useUser()` hook as shown above

### Issue: Data fetches before auth loads

**Cause:** Missing `isUserLoaded` check in `useEffect`

**Fix:** Add dependency and guard:
```typescript
useEffect(() => {
  if (isUserLoaded) {
    fetchData();
  }
}, [isUserLoaded, fetchData]);
```

### Issue: "Cannot access X before initialization"

**Cause:** Using `currentUser` before declaring it

**Fix:** Declare `useUser()` and `getClientUser()` at the top of the component

## See Also

- [lib/auth/README.md](./README.md) - General authentication guide
- [lib/auth/server.ts](./server.ts) - Server-side authentication
- [lib/auth/client.ts](./client.ts) - Client-side authentication helpers
