import { defineWorkersConfig } from '@cloudflare/vitest-pool-workers/config';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const schemaSQL = readFileSync(path.join(__dirname, 'schema.sql'), 'utf-8');

export default defineWorkersConfig({
  test: {
    include: ['test/**/*.test.ts'],
    setupFiles: ['./test/setup.ts'],
    poolOptions: {
      workers: {
        wrangler: { configPath: './wrangler.toml' },
        miniflare: {
          d1Databases: ['DB'],
          bindings: {
            TEST_SCHEMA_SQL: schemaSQL,
            ADMIN_API_KEY: 'test-admin-key-for-vitest',
          },
        },
      },
    },
  },
});
