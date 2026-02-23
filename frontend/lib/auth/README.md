# Authentication System

LabFork uses a flexible authentication system that supports both Clerk (production) and mock users (development).

## Overview

The authentication system automatically detects whether Clerk is configured and falls back to mock authentication for development convenience.

### Mock Authentication (Development)

When Clerk is not configured or `NODE_ENV=development`, the system uses mock users for testing:

- **Default User**: "Alex Researcher" (researcher)
- **Available Mock Users**: researcher, scientist, developer
- Stored in localStorage for browser persistence
- Always returns DEFAULT_USER for server-side calls

### Clerk Authentication (Production)

When Clerk environment variables are set, the system uses real JWT-based authentication:

- Validates users via Clerk's session tokens
- Converts Clerk user data to LabFork User interface
- Returns `null` for unauthenticated requests in production

## Setup

### 1. Install Clerk (Already Done)

```bash
npm install @clerk/nextjs
```

### 2. Configure Environment Variables

Add to `.env.local`:

```bash
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
CLERK_SECRET_KEY=sk_test_...
```

Get these from: https://dashboard.clerk.com/

### 3. Middleware (Already Configured)

The `middleware.ts` file automatically enables Clerk middleware when configured:

```typescript
// Clerk is used when both keys are set
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY && CLERK_SECRET_KEY
```

## Usage in API Routes

```typescript
import { getServerUser } from "@/lib/auth/mock-user";

export async function POST(request: NextRequest) {
  // Get current user (Clerk user or mock user)
  const user = await getServerUser();

  if (!user) {
    // Only happens in production with Clerk when not authenticated
    return NextResponse.json(
      { success: false, error: "Authentication required" },
      { status: 401 }
    );
  }

  // Use user.id, user.username, etc.
  const owner = userToLabOwner(user);
  // ...
}
```

## User Interface

Both Clerk and mock users conform to the same interface:

```typescript
interface User {
  id: string;          // Clerk user ID or mock ID
  username: string;    // Clerk username or mock username
  displayName: string; // Full name
  avatar?: string;     // Profile image URL
  email?: string;      // Email address
  bio?: string;        // User bio (mock only)
  createdAt: string;   // ISO timestamp
}
```

## Client-Side Authentication

### Mock User (Development)

```typescript
import {
  getCurrentUser,
  setCurrentUser,
  loginAs
} from "@/lib/auth/mock-user";

// Get current user
const user = getCurrentUser(); // Returns mock user

// Switch users (development only)
loginAs("scientist"); // Switch to different mock user
```

### Clerk User (Production)

Use Clerk's React hooks and components:

```typescript
import { useUser } from "@clerk/nextjs";
import { SignInButton, UserButton } from "@clerk/nextjs";

function MyComponent() {
  const { user, isLoaded, isSignedIn } = useUser();

  return (
    <div>
      {isSignedIn ? (
        <UserButton />
      ) : (
        <SignInButton mode="modal">Sign In</SignInButton>
      )}
    </div>
  );
}
```

## Behavior Matrix

| Environment | Clerk Keys Set | Behavior |
|-------------|----------------|----------|
| Development | No | Mock user (DEFAULT_USER) |
| Development | Yes | Clerk auth (optional) |
| Production | No | Mock user (DEFAULT_USER) |
| Production | Yes | Clerk auth (returns null if not authenticated) |

## Migration from Mock to Clerk

The system is designed for seamless migration:

1. **Development**: Works out of the box with mock users
2. **Staging**: Add Clerk keys to test real authentication
3. **Production**: Clerk keys required for real users

No code changes needed - just set environment variables.

## Security Notes

1. **Mock users are NOT secure** - Only use in development
2. **Server-side auth** - Always validate on server via `getServerUser()`
3. **Client state** - Client-side user can be manipulated; never trust for authorization
4. **API protection** - Middleware allows all requests; add route protection as needed

## Adding Route Protection

To require authentication on specific routes, update `middleware.ts`:

```typescript
export default clerkMiddleware((auth, req) => {
  if (isProtectedRoute(req)) {
    auth().protect(); // Require authentication
  }
});
```

## Testing

### Test with Mock Users

```bash
# No setup needed - works automatically
npm run dev
```

### Test with Clerk

```bash
# 1. Create a Clerk application at dashboard.clerk.com
# 2. Add keys to .env.local
# 3. Start dev server
npm run dev

# 4. Sign in with a test user
# Mock users are bypassed when Clerk is configured
```

## Troubleshooting

### "Clerk not configured" message

This is expected in development when Clerk keys aren't set. The app falls back to mock users.

### Returns null in production

Check that both Clerk environment variables are set:
- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`
- `CLERK_SECRET_KEY`

### User data mismatch

Clerk users have different IDs than mock users. Database relationships need to use Clerk user IDs in production.

## Future Enhancements

- [ ] Add Clerk webhook handlers for user lifecycle events
- [ ] Sync Clerk users to local database for relationships
- [ ] Add role-based access control (RBAC)
- [ ] Add organization support for team labs
