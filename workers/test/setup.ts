import { env } from 'cloudflare:test';

// Apply D1 schema before all tests
// TEST_SCHEMA_SQL is injected by vitest.config.ts (read from migrations/0001_initial.sql in Node context)
const schemaSQL = (env as Record<string, unknown>).TEST_SCHEMA_SQL as string;

// Use batch to execute all statements
// Split carefully: remove comments, split on semicolons, filter empties
const lines = schemaSQL
  .split('\n')
  .filter((line) => !line.trim().startsWith('--'))
  .join('\n');

const statements = lines
  .split(';')
  .map((s) => s.trim())
  .filter((s) => s.length > 0);

// Use D1 batch API for reliability
const batch = statements.map((stmt) => env.DB.prepare(stmt));
await env.DB.batch(batch);
