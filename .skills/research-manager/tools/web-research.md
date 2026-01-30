---
name: web-research
description: Web search, arxiv papers, and GitHub repository search
metadata:
  tags: search, web, papers, arxiv, github, research
---

# Web Research Tools

Tools for searching the web, finding academic papers, and discovering GitHub repositories.

## Web Search

General web search with optional focus on academic papers.

```bash
# General web search
.skills/research-manager/rm search prosody conditioning TTS

# Focus on papers/research sites
.skills/research-manager/rm search --papers voice cloning neural networks

# Limit results
.skills/research-manager/rm search --max 10 emotion recognition speech
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--papers`, `-p` | Focus search on arxiv, paperswithcode, scholar |
| `--max`, `-m` | Maximum number of results (default: 5) |

## ArXiv Paper Search

Search arxiv.org directly for academic papers.

```bash
# Search for papers
.skills/research-manager/rm papers prosody voice cloning

# More results
.skills/research-manager/rm papers --max 10 text to speech neural
```

### Output

Returns for each paper:
- Title
- Authors
- Publication date
- ArXiv link
- Abstract summary

## GitHub Repository Search

Search GitHub for relevant repositories.

```bash
# Basic search
.skills/research-manager/rm github voice cloning

# Filter by language
.skills/research-manager/rm github --language python TTS

# Sort by different criteria
.skills/research-manager/rm github --sort updated prosody

# More results
.skills/research-manager/rm github --max 20 speech synthesis
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--language`, `-l` | Filter by programming language |
| `--sort`, `-s` | Sort by: stars, forks, updated (default: stars) |
| `--max`, `-m` | Maximum results (default: 10) |

## URL Fetching

Fetch and display content from any URL.

```bash
# Fetch and display cleaned text
.skills/research-manager/rm fetch https://arxiv.org/abs/2301.12345

# Get raw HTML
.skills/research-manager/rm fetch --raw https://example.com
```

## Research Workflow Example

```bash
# 1. Search for recent papers on your topic
.skills/research-manager/rm papers prosody conditioning voice synthesis

# 2. Find relevant GitHub implementations
.skills/research-manager/rm github --language python prosody TTS

# 3. Spawn an agent to do deep research
.skills/research-manager/rm spawn --type codex --name "prosody-research" \
  --task "Research the latest approaches to prosody conditioning in TTS.
  Focus on papers from 2023-2024. Summarize:
  1. Key techniques being used
  2. State of the art results
  3. Open source implementations
  4. Recommendations for our voice cloning project"

# 4. Wait and review
.skills/research-manager/rm wait --agent "prosody-research"
.skills/research-manager/rm read --name "prosody-research"
```

## Tips

- Use `--papers` flag when you specifically want academic sources
- GitHub search is great for finding implementations of papers
- Spawn a codex agent for deeper research that requires reasoning
- Combine multiple searches to get comprehensive coverage
