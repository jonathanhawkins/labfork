#!/bin/bash
#
# Git observer — reports CI status, stale PRs, and new issues to nudge-engine.
#
# Run this on a cron (e.g., every 15 minutes alongside the engine's cron)
# or just call it manually whenever you want to feed state into the engine.
#
# Prerequisites:
#   - GitHub CLI: brew install gh
#   - Authenticated: gh auth login
#
# Usage:
#   ENGINE=https://your-engine.workers.dev TOKEN=abc123 REPO=owner/repo ./git-observer.sh

set -euo pipefail

ENGINE="${ENGINE:?Set ENGINE=https://your-engine.workers.dev}"
TOKEN="${TOKEN:?Set TOKEN=your-worker-token}"
REPO="${REPO:?Set REPO=owner/repo}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "Observing $REPO..."

# Check CI status on default branch
CI_FAILING=false
LATEST_RUN=$(gh run list --repo "$REPO" --branch main --limit 1 --json conclusion -q '.[0].conclusion' 2>/dev/null || echo "")
if [ "$LATEST_RUN" = "failure" ]; then
  CI_FAILING=true
fi

# Find stale PRs (open > 3 days, no review activity)
STALE_PRS=$(gh pr list --repo "$REPO" --state open --json number,title,updatedAt \
  --jq "[.[] | select(.updatedAt < \"$(date -u -v-3d +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d '3 days ago' +%Y-%m-%dT%H:%M:%SZ)\") | {number, title, staleDays: 3}]" 2>/dev/null || echo "[]")

# Find new issues (created in last 24h, no labels)
NEW_ISSUES=$(gh issue list --repo "$REPO" --state open --label "" --json number,title,createdAt \
  --jq "[.[] | select(.createdAt > \"$(date -u -v-1d +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d '1 day ago' +%Y-%m-%dT%H:%M:%SZ)\") | {id: .number, title}]" 2>/dev/null || echo "[]")

# Post git observation
curl -sf -X POST "$ENGINE/observe" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"type\": \"git\",
    \"data\": {
      \"repo\": \"$REPO\",
      \"failingCI\": $CI_FAILING,
      \"stalePRs\": $STALE_PRS
    }
  }" > /dev/null

log "Git observation posted (CI failing: $CI_FAILING, stale PRs: $(echo "$STALE_PRS" | jq length))"

# Post issues observation (if any)
ISSUE_COUNT=$(echo "$NEW_ISSUES" | jq length)
if [ "$ISSUE_COUNT" -gt 0 ]; then
  curl -sf -X POST "$ENGINE/observe" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{
      \"type\": \"issues\",
      \"data\": {
        \"repo\": \"$REPO\",
        \"newIssues\": $NEW_ISSUES
      }
    }" > /dev/null

  log "Issues observation posted ($ISSUE_COUNT new)"
fi

log "Done"
