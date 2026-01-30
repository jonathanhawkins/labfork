# Supabase Setup Guide

This guide walks you through setting up Supabase for the LabFork distributed compute network credit system.

## Prerequisites

- A Supabase account (sign up at https://supabase.com)

## Setup Steps

### 1. Create a New Supabase Project

1. Go to https://app.supabase.com
2. Click "New project"
3. Choose your organization
4. Set project name (e.g., "labfork-compute")
5. Generate a strong database password
6. Select a region close to your users
7. Click "Create new project"

### 2. Run the Migration

1. In your Supabase dashboard, go to the SQL Editor
2. Copy the contents of `migrations/20260130_create_credit_system.sql`
3. Paste into the SQL Editor
4. Click "Run" to execute the migration

This will create:
- `user_credits` table
- `credit_transactions` table
- `devices` table
- Indexes for performance
- Row Level Security (RLS) policies
- Triggers for automatic updates

### 3. Get Your API Keys

1. In your Supabase dashboard, go to Project Settings > API
2. Copy the following values:
   - **Project URL**: Your `NEXT_PUBLIC_SUPABASE_URL`
   - **anon public**: Your `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - **service_role**: Your `SUPABASE_SERVICE_ROLE_KEY` (keep this secret!)

### 4. Configure Environment Variables

Add these to your `frontend/.env.local`:

```bash
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key_here
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here
```

### 5. Verify Setup

Test the API endpoints:

```bash
# Get user credits (should return zero balance for new user)
curl "http://localhost:3003/api/credits?userId=test_user_123"

# Get transaction history (should return empty array)
curl "http://localhost:3003/api/credits/history?userId=test_user_123"
```

## Database Schema

### user_credits
Stores user credit balances and lifetime statistics.

| Column | Type | Description |
|--------|------|-------------|
| user_id | TEXT | Primary key, user identifier |
| balance | DECIMAL | Current available credits |
| total_earned | DECIMAL | Lifetime credits earned |
| total_spent | DECIMAL | Lifetime credits spent |
| updated_at | TIMESTAMPTZ | Last update timestamp |

### credit_transactions
Records all credit transactions.

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT | Primary key, transaction ID |
| user_id | TEXT | User identifier |
| amount | DECIMAL | Credit amount (positive=earn, negative=spend) |
| type | TEXT | Transaction type: earn, spend, bonus, refund |
| task_id | TEXT | Related task ID (optional) |
| description | TEXT | Human-readable description |
| created_at | TIMESTAMPTZ | Transaction timestamp |

### devices
Stores registered compute devices.

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT | Primary key, device ID |
| user_id | TEXT | Owner user ID (optional) |
| name | TEXT | Display name |
| tier | TEXT | Device tier: power, standard, crowd |
| capabilities | JSONB | Hardware capabilities |
| availability | JSONB | Availability preferences |
| status | TEXT | Current status: online, busy, offline, paused |
| current_task_id | TEXT | Currently assigned task (optional) |
| stats | JSONB | Lifetime statistics |
| registered_at | TIMESTAMPTZ | Registration timestamp |
| last_seen | TIMESTAMPTZ | Last heartbeat timestamp |

## Security

### Row Level Security (RLS)

All tables have RLS enabled with policies:
- Users can only view/update their own records
- Service role (backend) can perform all operations
- Public read access for network stats (online devices)

### API Keys

- **Anon Key**: Safe for client-side use, respects RLS policies
- **Service Role Key**: Full database access, NEVER expose to clients

## Troubleshooting

### Migration Fails

If the migration fails, check:
1. You have a fresh project (no conflicting tables)
2. The SQL was copied completely
3. Your database is online and accessible

### RLS Blocks Queries

If queries are blocked:
1. Check you're using the correct API key
2. Verify RLS policies match your use case
3. Use service role key for backend operations

### Environment Variables Not Working

1. Restart your Next.js dev server after changing .env.local
2. Verify variable names match exactly (case-sensitive)
3. Check for typos in the Supabase URL

## Next Steps

1. Implement authentication (Supabase Auth recommended)
2. Connect real user sessions to `userId` parameters
3. Set up monitoring and alerts
4. Configure backups in Supabase dashboard
5. Review and adjust RLS policies for your security needs
