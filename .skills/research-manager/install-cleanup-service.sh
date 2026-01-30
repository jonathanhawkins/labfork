#!/bin/bash
# Install the periodic agent cleanup service on macOS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLIST_NAME="com.voice-clone.agent-cleanup.plist"
PLIST_SRC="$SCRIPT_DIR/$PLIST_NAME"
PLIST_DEST="$HOME/Library/LaunchAgents/$PLIST_NAME"

echo "Installing agent cleanup service..."

# Check if source plist exists
if [ ! -f "$PLIST_SRC" ]; then
    echo "Error: $PLIST_SRC not found"
    exit 1
fi

# Create LaunchAgents directory if needed
mkdir -p "$HOME/Library/LaunchAgents"

# Unload existing service if present
launchctl unload "$PLIST_DEST" 2>/dev/null

# Copy plist
cp "$PLIST_SRC" "$PLIST_DEST"

# Load service
launchctl load "$PLIST_DEST"

echo "✓ Installed and loaded $PLIST_NAME"
echo ""
echo "The cleanup service will run every hour."
echo ""
echo "Commands:"
echo "  Check status:  launchctl list | grep voice-clone"
echo "  Run now:       launchctl start $PLIST_NAME"
echo "  View logs:     tail -f $SCRIPT_DIR/state/cleanup.log"
echo "  Uninstall:     launchctl unload $PLIST_DEST && rm $PLIST_DEST"
