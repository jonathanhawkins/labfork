# Testing Supabase Graceful Degradation

## Quick Test Procedure

### 1. Verify Current Configuration Status

The current `.env.local` has placeholder values:
```bash
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```

These will be detected as invalid, so `isSupabaseConfigured` will be `false`.

### 2. Test API Routes Return Mock Data

With the current placeholder configuration, test that API routes return graceful responses:

#### Test Credits Balance
```bash
curl http://localhost:3003/api/credits?userId=test123
```

**Expected Response** (200 OK):
```json
{
  "userId": "",
  "balance": 0,
  "totalEarned": 0,
  "totalSpent": 0,
  "updatedAt": "2026-02-03T...",
  "message": "Credits system not configured"
}
```

#### Test Earn Credits
```bash
curl -X POST http://localhost:3003/api/credits/earn \
  -H "Content-Type: application/json" \
  -d '{"userId":"test123","amount":10,"taskId":"task1","description":"test"}'
```

**Expected Response** (200 OK):
```json
{
  "success": false,
  "message": "Credits system not configured"
}
```

#### Test Contributor Stats
```bash
curl http://localhost:3003/api/contributor/stats
```

**Expected Response** (200 OK):
```json
{
  "totalContributors": 0,
  "totalCreditsEarned": 0,
  "totalTasksCompleted": 0,
  "message": "Credits system not configured"
}
```

#### Test Contributor Profile
```bash
curl http://localhost:3003/api/contributor/test123
```

**Expected Response** (200 OK):
```json
{
  "userId": "test123",
  "displayName": "Anonymous",
  "rank": "Newcomer",
  "totalCreditsEarned": 0,
  "totalTasksCompleted": 0,
  "totalComputeTime": 0,
  "devices": [],
  "badges": [],
  "joinedAt": "2026-02-03T...",
  "message": "Credits system not configured"
}
```

### 3. Verify No Console Errors

In development mode:
- No console errors should appear
- Server warnings should be silenced (to avoid log spam)

In production mode:
- One warning on startup about Supabase not being configured
- No runtime errors

### 4. Test TypeScript Compilation

```bash
npx tsc --noEmit
```

Should complete without errors.

### 5. Verify Build Works

```bash
npm run build
```

Should complete successfully without build errors.

## Testing With Real Supabase

To test with real Supabase credentials:

1. Update `.env.local`:
```bash
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGc...your-actual-key
SUPABASE_SERVICE_ROLE_KEY=eyJhbGc...your-actual-service-key
```

2. Restart dev server:
```bash
npm run dev -- -p 3003
```

3. Test same endpoints - they should now:
   - Return real data from Supabase
   - No "Credits system not configured" messages
   - Actually read/write to database

## Automated Test Script

You can create a simple test script to verify all endpoints:

```bash
#!/bin/bash

echo "Testing Supabase Graceful Degradation..."
echo ""

BASE_URL="http://localhost:3003/api"

echo "1. Testing Credits Balance..."
curl -s "$BASE_URL/credits?userId=test123" | jq .
echo ""

echo "2. Testing Earn Credits..."
curl -s -X POST "$BASE_URL/credits/earn" \
  -H "Content-Type: application/json" \
  -d '{"userId":"test123","amount":10,"taskId":"task1","description":"test"}' | jq .
echo ""

echo "3. Testing Contributor Stats..."
curl -s "$BASE_URL/contributor/stats" | jq .
echo ""

echo "4. Testing Contributor Profile..."
curl -s "$BASE_URL/contributor/test123" | jq .
echo ""

echo "All tests complete!"
```

Save as `test-graceful-degradation.sh` and run:
```bash
chmod +x test-graceful-degradation.sh
./test-graceful-degradation.sh
```

## Success Criteria

✅ App builds without errors
✅ TypeScript compilation passes
✅ API routes return 200 OK (not 500 errors)
✅ Responses include "Credits system not configured" message
✅ Mock data structure matches expected schema
✅ No console errors in development
✅ When real Supabase credentials are added, full functionality works

## Troubleshooting

If tests fail:

1. **Check environment variables are loaded:**
   ```bash
   # In your Next.js app, add a test endpoint
   echo $NEXT_PUBLIC_SUPABASE_URL
   ```

2. **Verify isSupabaseConfigured is being imported correctly:**
   Check import statements in API routes

3. **Check console for TypeScript errors:**
   ```bash
   npx tsc --noEmit --pretty
   ```

4. **Restart dev server after any changes:**
   Environment variable changes require server restart
