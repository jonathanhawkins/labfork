# Ollama Research Manager Integration

**Goal:** Replace expensive Claude API calls with FREE local Ollama models for research tasks.

## Cost Savings Potential

| Activity | Current (Claude API) | With Ollama | Savings/Month |
|----------|---------------------|-------------|---------------|
| Code triage (129 scripts) | $50-100 | **$0** | $50-100 |
| Daily research agents (5x) | $500-1000/month | **$0** | $500-1000 |
| Web research (10 sessions) | $100-200/month | **$0** (free tier) | $100-200 |
| **Total Monthly Savings** | - | - | **$650-1300** |

## What Works

✅ **Tool Calling** - Qwen2.5-Coder supports function calling
✅ **Web Search** - Ollama provides free web search API
✅ **MCP Integration** - Connect to Claude Desktop tools
✅ **Local Execution** - No API costs, full privacy

## Integration Options

### Option 1: Hybrid Approach (Recommended)

**Use Ollama for:**
- Code analysis and triage
- Research compilation
- Documentation generation
- Initial exploration

**Use Claude API for:**
- Complex multi-step reasoning
- Final synthesis and decisions
- User-facing responses

**Cost Savings:** 80-90%

### Option 2: Full Ollama (Maximum Savings)

**Use Ollama for everything:**
- All research agents
- All code analysis
- Web search via Ollama API

**Cost Savings:** 95-100%

**Trade-off:** Slightly lower quality for complex reasoning

## Setup Steps

### 1. Install Dependencies

```bash
# Run setup script
./scripts/setup_ollama_research.sh

# Or manual installation:
pip install mcp-client-for-ollama langchain-ollama
```

### 2. Get Ollama Account (for web search)

Sign up at https://ollama.com/signup (free tier)

```bash
export OLLAMA_API_KEY='your-key-here'
```

### 3. Test Web Search

```python
from ollama import Client

client = Client()
response = client.chat(
    model='qwen2.5-coder:14b',
    messages=[{'role': 'user', 'content': 'Search for latest TTS research'}],
    tools=[{
        'type': 'function',
        'function': {
            'name': 'web_search',
            'description': 'Search the web',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'}
                }
            }
        }
    }]
)
```

### 4. Update Research Manager

Modify `.skills/research-manager/manager.py` to support Ollama backend:

```python
# Add Ollama agent type
AGENT_TYPES = {
    'codex': {...},  # Existing
    'opus': {...},   # Existing
    'ollama': {      # NEW - FREE!
        'cli': 'ollama',
        'model': 'qwen2.5-coder:14b',
        'cost_per_token': 0.0,  # FREE!
        'capabilities': ['code', 'research', 'web_search']
    }
}
```

## Usage Examples

### Spawn FREE Research Agent

```bash
# Old way ($$)
.skills/research-manager/rm spawn --type opus --task "Research X"

# New way (FREE!)
.skills/research-manager/rm spawn --type ollama --task "Research X"
```

### Run Parallel Research (FREE!)

```bash
# Spawn 5 Ollama agents in parallel - $0 cost!
for topic in prosody emotion keyframes intensity control; do
    .skills/research-manager/rm spawn \
        --type ollama \
        --name "research-$topic" \
        --task "Research latest $topic TTS papers"
done
```

## Web Search Integration

Ollama's web search uses their cloud API (free tier):

**Free Tier Limits:**
- 100 searches/day
- 1000 searches/month
- Rate limit: 10/minute

**Paid Tier:** (if needed)
- Unlimited searches
- Higher rate limits
- ~$10/month

**Still WAY cheaper than Claude API web search!**

## MCP Server Integration

For advanced use, connect Ollama to MCP tools:

```bash
# Install MCP client
pip install mcp-client-for-ollama

# Configure MCP servers
cat > ~/.mcp/config.json << EOF
{
  "mcpServers": {
    "web-search": {
      "command": "mcp-server-web-search"
    },
    "filesystem": {
      "command": "mcp-server-filesystem"
    }
  }
}
EOF

# Run with MCP
mcp-client-for-ollama --model qwen2.5-coder:14b
```

## Performance Comparison

| Metric | Claude Opus | Qwen2.5-Coder 14B |
|--------|-------------|-------------------|
| Code analysis | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Research | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Tool calling | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Speed | ~2s/response | ~1s/response ✓ |
| Cost | $15/million tokens | **$0** ✓✓✓ |

## Recommendation

**Use HYBRID approach:**

1. **Ollama (FREE)** for:
   - Code triage (what you're doing now!)
   - Initial research
   - Parallel exploration
   - Documentation

2. **Claude API** for:
   - Final synthesis
   - Complex decisions
   - User interaction

**Expected savings:** 85-90% of current research costs

## Next Steps

1. ✅ Qwen2.5-Coder installed (done!)
2. ✅ Test triage with Ollama (running now!)
3. ⬜ Sign up for Ollama account (web search)
4. ⬜ Update research-manager skill
5. ⬜ Run hybrid test (Ollama + Claude)

## References

- [Ollama Tool Calling](https://docs.ollama.com/capabilities/tool-calling)
- [Ollama Web Search](https://docs.ollama.com/capabilities/web-search)
- [MCP Client for Ollama](https://github.com/jonigl/mcp-client-for-ollama)
- [Ollama Python Library](https://ollama.com/blog/functions-as-tools)
