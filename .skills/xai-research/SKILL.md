---
name: xai-research
description: Research topics using xAI Grok with live X (Twitter) and web search
metadata:
  tags: xai, grok, x-search, twitter, web-search, research, real-time
---

# xAI Research Skill

**Research anything using Grok's Live Search with real-time X and web data.**

This skill uses xAI's Grok API to search:
- **X (Twitter)** - Real-time posts, threads, trending topics
- **Web** - Live web search and page browsing

## Setup

### 1. Get your xAI API Key

1. Create account at [x.ai](https://x.ai/api)
2. Go to [API Keys Page](https://console.x.ai/team/default/api-keys)
3. Create a new key (starts with `xai-`)

### 2. Set Environment Variables

**Local development:**
```bash
# Add to .env or export directly
export XAI_API_KEY="xai-YOUR_API_KEY_HERE"
```

**Vercel:**
1. Go to Vercel Dashboard → Project → Settings → Environment Variables
2. Add `XAI_API_KEY` with your key

## Quick Usage

```bash
# Search X for trending AI topics
~/bin/xai-search --x "What's trending in AI right now?"

# Search web for documentation
~/bin/xai-search --web "Next.js 15 new features"

# Search both X and web
~/bin/xai-search "What are people saying about Grok 4?"

# Search specific X handles
~/bin/xai-search --x --handles "@elonmusk,@xai" "latest announcements"

# Date-filtered X search
~/bin/xai-search --x --from "2026-01-01" "voice cloning breakthroughs"
```

## API Endpoints

The skill uses xAI's `/v1/responses` endpoint with Live Search tools:

### Available Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web and browse pages |
| `x_search` | Search X posts, users, threads |

### X Search Filters

- `allowed_x_handles` - Only search these accounts (max 10)
- `excluded_x_handles` - Exclude these accounts (max 10)
- `from_date` / `to_date` - Date range (ISO8601)
- `enable_image_understanding` - Analyze images in posts
- `enable_video_understanding` - Analyze videos in posts

### Web Search Filters

- `allowed_domains` - Only these domains (max 5)
- `excluded_domains` - Exclude these domains (max 5)
- `user_location_*` - Location-based results

## Python Example

```python
import os
import requests

XAI_API_KEY = os.getenv("XAI_API_KEY")

def search_x(query: str, handles: list[str] = None) -> dict:
    """Search X (Twitter) using Grok Live Search."""

    tools = [{
        "type": "x_search",
        "x_search": {
            "enable_image_understanding": True
        }
    }]

    if handles:
        tools[0]["x_search"]["allowed_x_handles"] = handles

    response = requests.post(
        "https://api.x.ai/v1/responses",
        headers={
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "grok-4-1-fast",
            "input": [{"role": "user", "content": query}],
            "tools": tools
        }
    )

    return response.json()

def search_web(query: str, domains: list[str] = None) -> dict:
    """Search the web using Grok Live Search."""

    tools = [{"type": "web_search"}]

    if domains:
        tools[0]["web_search"] = {"allowed_domains": domains}

    response = requests.post(
        "https://api.x.ai/v1/responses",
        headers={
            "Authorization": f"Bearer {XAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "grok-4-1-fast",
            "input": [{"role": "user", "content": query}],
            "tools": tools
        }
    )

    return response.json()

# Example usage
result = search_x("What's the latest on voice AI?")
print(result["output"]["content"])

# With citations
for citation in result.get("citations", []):
    print(f"- {citation['title']}: {citation['url']}")
```

## Models

| Model | Best For | Context |
|-------|----------|---------|
| `grok-4-1-fast` | Agentic search, tool calling | 2M tokens |
| `grok-4` | Advanced reasoning | 256K tokens |
| `grok-3` | General use | 128K tokens |

## Pricing

Live Search is currently **FREE in beta**!

Normal pricing (when beta ends):
- Token usage: varies by model
- Tool invocations: priced per call

## Research Ideas

### Voice Cloning Domain
```bash
# Latest research papers
~/bin/xai-search --web "voice cloning research papers 2026 arxiv"

# What practitioners are saying
~/bin/xai-search --x "voice cloning TTS" --from "2026-01-01"

# Competitor analysis
~/bin/xai-search --x --handles "@elevenlabs,@play_ht,@resaboratory" "new features"
```

### AI/ML Trends
```bash
# Trending ML topics
~/bin/xai-search --x "machine learning trending"

# What's hot on HuggingFace
~/bin/xai-search --x --handles "@huggingface" "new model release"

# Latest from AI researchers
~/bin/xai-search --x --handles "@kaborahinov,@ylaborinov" "research"
```

### Industry News
```bash
# Breaking AI news
~/bin/xai-search "breaking AI news today"

# Funding announcements
~/bin/xai-search --x "AI startup funding raised"

# Product launches
~/bin/xai-search --x "just launched AI product"
```

## Links

- [xAI Documentation](https://docs.x.ai/docs/overview)
- [Live Search Guide](https://docs.x.ai/docs/guides/live-search)
- [Models & Pricing](https://docs.x.ai/docs/models)
- [xAI Console](https://console.x.ai)
