#!/bin/bash
# Install OpenAI Codex CLI on Linux (4090 machine)
# Based on the official releases from https://github.com/openai/codex

set -e

echo "Installing OpenAI Codex CLI for Linux..."

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    BINARY_NAME="codex-x86_64-unknown-linux-musl"
elif [ "$ARCH" = "aarch64" ]; then
    BINARY_NAME="codex-aarch64-unknown-linux-musl"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

# Get latest version
echo "Fetching latest version..."
LATEST_VERSION=$(curl -fsSL https://api.github.com/repos/openai/codex/releases/latest | grep '"tag_name"' | head -1 | sed -E 's/.*"(rust-)?v([^"]+)".*/\2/')

if [ -z "$LATEST_VERSION" ]; then
    echo "Failed to get latest version"
    exit 1
fi

echo "Latest version: v$LATEST_VERSION"

# Download URL
DOWNLOAD_URL="https://github.com/openai/codex/releases/download/v${LATEST_VERSION}/${BINARY_NAME}"

echo "Downloading from: $DOWNLOAD_URL"

# Download to temp
cd /tmp
curl -fsSL "$DOWNLOAD_URL" -o codex-binary

# Make executable
chmod +x codex-binary

# Install to ~/bin (or /usr/local/bin if sudo)
INSTALL_DIR="$HOME/bin"
mkdir -p "$INSTALL_DIR"
mv codex-binary "$INSTALL_DIR/codex"

echo ""
echo "✓ Codex installed to: $INSTALL_DIR/codex"
echo ""
echo "Add to PATH if not already:"
echo "  echo 'export PATH=\"\$HOME/bin:\$PATH\"' >> ~/.bashrc"
echo "  source ~/.bashrc"
echo ""
echo "Test it:"
echo "  codex --version"
echo ""
echo "Login:"
echo "  codex login"
