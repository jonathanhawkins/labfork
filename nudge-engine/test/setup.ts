import { env } from 'cloudflare:test';

// Apply D1 schema before all tests
const schemaSQL = (env as unknown as Record<string, unknown>).TEST_SCHEMA_SQL as string;

// Remove comments, split on semicolons, filter empties
const lines = schemaSQL
  .split('\n')
  .filter((line) => !line.trim().startsWith('--'))
  .join('\n');

const statements = lines
  .split(';')
  .map((s) => s.trim())
  .filter((s) => s.length > 0);

const batch = statements.map((stmt) => env.DB.prepare(stmt));
await env.DB.batch(batch);
