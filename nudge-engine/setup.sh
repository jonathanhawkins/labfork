#!/bin/bash
#
# One-line setup for Nudge Engine.
#
# Usage:
#   ./setup.sh
#
# What it does:
#   1. Installs dependencies
#   2. Creates the D1 database
#   3. Updates wrangler.toml with the database ID
#   4. Applies the schema
#   5. Prompts for admin API key
#   6. Deploys
#

set -euo pipefail

log() { echo ""; echo "==> $*"; }

log "Installing dependencies..."
npm install

log "Creating D1 database..."
DB_OUTPUT=$(npx wrangler d1 create nudge-engine-db 2>&1) || {
  if echo "$DB_OUTPUT" | grep -q "already exists"; then
    echo "  Database already exists, skipping."
  else
    echo "  Error creating database:"
    echo "$DB_OUTPUT"
    exit 1
  fi
}

# Extract database ID (works on both macOS and Linux)
DB_ID=$(echo "$DB_OUTPUT" | grep -o '[0-9a-f]\{8\}-[0-9a-f]\{4\}-[0-9a-f]\{4\}-[0-9a-f]\{4\}-[0-9a-f]\{12\}' | head -1 || echo "")

if [ -z "$DB_ID" ]; then
  echo "  Could not extract database ID. Please update wrangler.toml manually."
  echo "  Output was: $DB_OUTPUT"
else
  echo "  Database ID: $DB_ID"

  # Update wrangler.toml
  if grep -q "00000000-0000-0000-0000-000000000000" wrangler.toml; then
    sed -i.bak "s/00000000-0000-0000-0000-000000000000/$DB_ID/" wrangler.toml
    rm -f wrangler.toml.bak
    echo "  Updated wrangler.toml"
  fi
fi

log "Applying database schema..."
npx wrangler d1 execute nudge-engine-db --remote --file=schema.sql

log "Setting admin API key..."
echo "  Enter an admin API key (or press Enter to skip):"
read -r -s ADMIN_KEY
if [ -n "$ADMIN_KEY" ]; then
  echo "$ADMIN_KEY" | npx wrangler secret put ADMIN_API_KEY
  echo "  Admin key set."
else
  echo "  Skipped. You can set one later: npx wrangler secret put ADMIN_API_KEY"
fi

log "Making example scripts executable..."
chmod +x examples/*.sh setup.sh 2>/dev/null || true

log "Deploying..."
DEPLOY_OUTPUT=$(npx wrangler deploy 2>&1)
echo "$DEPLOY_OUTPUT"

WORKER_URL=$(echo "$DEPLOY_OUTPUT" | grep -o 'https://[^ ]*' | head -1 || echo "https://nudge-engine.<your-subdomain>.workers.dev")

log "Done!"
echo ""
echo "Your Nudge Engine is live."
echo ""
echo "Next steps:"
echo "  1. Register a worker:"
echo "     curl -s -X POST $WORKER_URL/register \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"name\": \"my-agent\", \"type\": \"claude-code\"}'"
echo ""
echo "  2. Run a worker:  ./examples/claude-code-worker.sh"
echo "  3. tmux session:  ./examples/tmux-session.sh"
echo "  4. Check stats:   curl -s $WORKER_URL/stats | jq ."
echo ""
