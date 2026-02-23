# Nudge Engine

Persistent autonomous task manager for AI coding agents. Cloudflare Worker with D1 database.

## Build & Test

```bash
npm install
npm test             # vitest — 12 tests
npm run typecheck    # tsc --noEmit
npm run dev          # local dev server at :8787
npm run deploy       # deploy to Cloudflare Workers
```

## Architecture

- `src/index.ts` — Entry point. Hono app + cron handler.
- `src/api.ts` — HTTP endpoints. 4 core protocol + 7 admin/public.
- `src/cron.ts` — The nudge loop: observe, decide, nudge, cleanup. Runs every 15 min.
- `src/observers.ts` — Read-only state queries. No side effects.
- `src/rules.ts` — Heuristic task creation. Pure functions. No LLM, no API calls.
- `src/utils.ts` — Shared helpers.
- `schema.sql` — D1 schema. 4 tables: workers, tasks, observations, work_log.

## Protocol

Three endpoints. That's it.

- `POST /register` — Register a worker. Returns `{id, token}`.
- `POST /poll` — Heartbeat + get next task. Returns `{task}` or `{task: null}`.
- `POST /report` — Report result. Body: `{taskId, success, result?, error?}`.

## Conventions

- All IDs are prefixed: `w_` (workers), `t_` (tasks), `obs_` (observations), `log_` (work_log).
- Timestamps are ISO 8601 strings.
- JSON columns (context, constraints, result) are stored as TEXT in D1.
- Auth uses bearer tokens. Workers get tokens at registration. Admin uses `ADMIN_API_KEY` secret.
- Rules in `rules.ts` are pure functions. Add new rules to the `RULES` array and redeploy.

## Boundaries

- Never add an LLM call to the cron loop. Rules must be deterministic.
- Never add external API calls to observers. They read D1 only.
- Keep the protocol stable. Workers should not need to update when the engine updates.
