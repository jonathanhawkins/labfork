---
name: ollama-research
description: FREE local AI research using Ollama (Qwen2.5-Coder) - saves $650-1300/month vs Claude API
metadata:
  tags: ollama, qwen, local-ai, free, research, web-search
---

# Ollama Research Skill

**Use local AI for research tasks - 100% FREE!**

This skill uses Ollama with Qwen2.5-Coder:14b to perform research tasks locally without API costs.

## When to Use

Use this skill instead of expensive Claude API agents for:
- ✅ Code analysis and triage
- ✅ Research compilation
- ✅ Documentation generation
- ✅ Paper summaries
- ✅ Initial exploration

**Cost:** $0 (vs $50-200 per task with Claude API)

## Prerequisites

```bash
# 1. Install Ollama (if not already)
brew install ollama

# 2. Pull Qwen2.5-Coder
ollama pull qwen2.5-coder:14b

# 3. (Optional) Sign up for Ollama account for web search
# https://ollama.com/signup
```

## Commands

### Analyze Code

Triage and rank code files by relevance:

```bash
# Analyze all training scripts
python3 scripts/ollama_research_triage.py

# Custom analysis
ollama run qwen2.5-coder:14b "Analyze training/train_*.py and rank by prosody relevance"
```

### Research Papers

```bash
# Note: Requires Ollama web search API key
export OLLAMA_API_KEY='your-key'

ollama run qwen2.5-coder:14b "Search for latest prosody TTS papers and summarize top 5"
```

### Compare Approaches

```bash
ollama run qwen2.5-coder:14b "Compare LoRA vs full fine-tuning for small datasets"
```

### Generate Documentation

```bash
ollama run qwen2.5-coder:14b "Write a README for training/train_lora_deepseek.py"
```

## Web Search Integration

To enable web search (FREE tier - 100 searches/day):

1. Sign up: https://ollama.com/signup
2. Get API key from dashboard
3. Export key:
   ```bash
   export OLLAMA_API_KEY='your-key-here'
   ```

4. Use with tools:
   ```python
   from ollama import Client

   client = Client(api_key=os.getenv('OLLAMA_API_KEY'))
   response = client.chat(
       model='qwen2.5-coder:14b',
       messages=[{'role': 'user', 'content': 'Search TTS papers 2026'}],
       tools=[{
           'type': 'function',
           'function': {
               'name': 'web_search',
               'description': 'Search the web',
               'parameters': {
                   'type': 'object',
                   'properties': {'query': {'type': 'string'}}
               }
           }
       }]
   )
   ```

## Cost Comparison

| Task | Claude API | Ollama | Savings |
|------|------------|--------|---------|
| Triage 129 scripts | $50-100 | $0 | 100% |
| 10 research sessions | $100-200 | $0 | 100% |
| 5 parallel agents | $250-500 | $0 | 100% |
| **Monthly research** | **$650-1300** | **$0** | **100%** |

## Example: Triage Research Implementations

The script `scripts/ollama_research_triage.py` shows a real example:

```python
def ask_ollama(prompt, model="qwen2.5-coder:14b"):
    """Ask Ollama a question - FREE!"""
    result = subprocess.run(
        ['ollama', 'run', model, prompt],
        capture_output=True,
        text=True,
        timeout=60
    )
    return result.stdout.strip()

# Analyze script
analysis = ask_ollama(f"""
Analyze this training script:
1. What technique does it implement?
2. Relevance for keyframe prosody control?
3. Score 1-10

{code_sample}

Return JSON: {{"technique": "...", "score": 1-10, "reason": "..."}}
""")
```

## When to Use Claude API Instead

Use Claude API ($$) for:
- Complex multi-step reasoning
- Final synthesis and decisions
- User-facing responses
- Critical production tasks

**Strategy:** Use Ollama for 90% of research (FREE), Claude for 10% of final synthesis ($$).

## Tips

1. **Parallel processing:** Ollama is fast locally, you can run multiple queries in parallel
2. **Context management:** Qwen2.5-Coder has 128K context window
3. **Tool calling:** Supports function calling for structured outputs
4. **No rate limits:** It's your local machine!

## Troubleshooting

**Ollama not running:**
```bash
ollama serve
```

**Model not found:**
```bash
ollama pull qwen2.5-coder:14b
```

**Slow responses:**
- Check RAM (needs ~16GB)
- Try smaller model: `qwen2.5-coder:7b`
- Check CPU/GPU utilization

## References

- [Ollama Documentation](https://docs.ollama.com)
- [Qwen2.5-Coder](https://ollama.com/library/qwen2.5-coder)
- [Tool Calling Guide](https://docs.ollama.com/capabilities/tool-calling)
- [Web Search API](https://docs.ollama.com/capabilities/web-search)
