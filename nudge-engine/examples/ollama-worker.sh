#!/bin/bash
#
# Nudge Engine hybrid worker — Ollama + Codex.
#
# Routes tasks to the right tool:
#   - Monitoring tasks  → gather real system data + Ollama analysis
#   - Advisory tasks    → Ollama API (fast, lightweight)
#   - Implementation    → Codex CLI (can read/write files, run commands)
#
# Codex uses Ollama's OpenAI-compatible API as its backend, so it's
# still free and local — but gains agentic file I/O and shell access.
#
# Prerequisites:
#   - Ollama running: ollama serve
#   - A model pulled: ollama pull qwen3:8b
#   - Codex CLI installed (optional, for implementation tasks)
#
# Usage:
#   ENGINE=https://your-engine.workers.dev TOKEN=abc123 ./ollama-worker.sh
#   ENGINE=https://your-engine.workers.dev TOKEN=abc123 MODEL=llama3.3 ./ollama-worker.sh

set -euo pipefail

ENGINE="${ENGINE:?Set ENGINE=https://your-engine.workers.dev}"
TOKEN="${TOKEN:?Set TOKEN=your-worker-token}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
MODEL="${MODEL:-qwen3:8b}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-180}"
CODEX_TIMEOUT="${CODEX_TIMEOUT:-600}"
REPO_DIR="${REPO_DIR:-$HOME/dev/voice-clone-pipeline}"
CODEX_SANDBOX="${CODEX_SANDBOX:-danger-full-access}"
MAX_RETRIES="${MAX_RETRIES:-3}"
DECOMPOSE_MODEL="${DECOMPOSE_MODEL:-$MODEL}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ==========================================================================
# Codebase map: injected into decompose prompts so subtasks reference REAL
# files, modules, and available packages instead of inventing toy utilities.
# ==========================================================================
read -r -d '' CODEBASE_MAP << 'CMAP' || true
## Repository: ~/dev/voice-clone-pipeline

### Available Python Packages
numpy 1.26, scipy 1.15, librosa 0.11, soundfile 0.12, torch 2.10+cu128 (CUDA), wave, json, os, sys, pathlib

### Backend Modules (backend/)
- prosody_analyzer.py — Classes: AcousticAnalyzer, RhythmAnalyzer, SemanticAnalyzer, ContourExtractor, CompleteProsodyAnalyzer
- main.py — FastAPI endpoints: /upload, /process/{id}, /transcribe/{id}, /analyze/{id}, /generate, /generate-pocket-tts, /predict-prosody
- data_augmentation.py, cwt_pitch.py, differentiable_pitch.py, emotion_style_tokens.py, cross_modal_attention_prosody.py

### Inference Engines (inference/)
- generate.py — class VoiceGenerator
- pocket_tts.py — Zero-shot voice cloning (100M params, CPU)
- generate_with_*.py — 10+ emotion/style TTS variants (chatterbox, emoknob, emorl, etc.)
- audio_processor.py, evaluate.py, analyze_emotions.py

### Training (training/)
- train_deepseek.py — Full fine-tuning: MultiHeadLatentAttention, MultiTokenPredictionHead, DeepSeekEnhancedTrainer
- train_lora_deepseek.py — LoRA training: LoRATrainer, AugmentedCSMDataset

### Data Paths
- data/raw/, data/processed/, data/labeled/, data/splits/ — Voice clone pipeline
- data/voice_samples/, data/audio_datasets/ — Training audio
- models/csm-1b, models/whisper, models/qwen2-audio, models/prosody_encoder_ravdess

### Research Directories (research/)
26 dirs: VAE prosody disentanglement, CWT pitch, emotion DPO, cross-modal attention, differentiable pitch contour, dual-codebook VQVAE, etc.

### Rules for Subtasks
- EXTEND existing modules, don't create standalone toy files
- Use available packages (numpy, scipy, librosa, torch)
- Reference real data paths (data/voice_samples/, data/labeled/)
- Build ON TOP of existing classes (e.g., add methods to AcousticAnalyzer)
- Write tests that verify against real audio properties (sample rate, duration, spectral features)
CMAP

# Check if codex CLI is available
CODEX_BIN=$(which codex 2>/dev/null || echo "")
if [ -n "$CODEX_BIN" ]; then
  CODEX_AVAILABLE=true
  log "Hybrid worker starting (Ollama + Codex)"
else
  CODEX_AVAILABLE=false
  log "Ollama-only worker starting (codex not found)"
fi
log "  Engine:   $ENGINE"
log "  Model:    $MODEL"
log "  Ollama:   $OLLAMA_HOST"
log "  Timeouts: ollama=${OLLAMA_TIMEOUT}s codex=${CODEX_TIMEOUT}s"
if [ "$CODEX_AVAILABLE" = true ]; then
  log "  Codex:    $CODEX_BIN (sandbox=$CODEX_SANDBOX, repo=$REPO_DIR)"
fi

# Check Ollama is running
if ! curl -sf "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
  echo "Error: Ollama not running at $OLLAMA_HOST"
  echo "Start it with: ollama serve"
  exit 1
fi

# DNS workaround: resolve ENGINE hostname via Google DNS if system DNS is broken.
# WSL2 DNS resolver can die randomly. This ensures the worker can always reach the engine.
ENGINE_HOST=$(echo "$ENGINE" | sed 's|https://||;s|/.*||')
ENGINE_IP=$(dig +short "$ENGINE_HOST" @8.8.8.8 2>/dev/null | head -1)
CURL_RESOLVE=""
if [ -n "$ENGINE_IP" ]; then
  CURL_RESOLVE="--resolve ${ENGINE_HOST}:443:${ENGINE_IP}"
  log "  DNS fix: $ENGINE_HOST → $ENGINE_IP (via 8.8.8.8)"
fi
# Wrapper: use resolved IP for all engine requests
ecurl() { curl $CURL_RESOLVE "$@"; }

# ==========================================================================
# Self-check: gather REAL system data for monitoring tasks
# ==========================================================================
gather_system_data() {
  local data=""

  # Engine stats
  local stats
  stats=$(ecurl -sf "$ENGINE/stats" 2>/dev/null || echo '{"error":"unreachable"}')
  data+="=== Engine Stats ===
$stats

"

  # Recent tasks
  local tasks
  tasks=$(ecurl -sf "$ENGINE/tasks?limit=10" 2>/dev/null || echo '{"error":"unreachable"}')
  data+="=== Recent Tasks (last 10) ===
$tasks

"

  # Worker info
  local workers
  workers=$(ecurl -sf "$ENGINE/workers" 2>/dev/null || echo '{"error":"unreachable"}')
  data+="=== Workers ===
$workers

"

  # Recent logs
  local logs
  logs=$(ecurl -sf "$ENGINE/log?limit=10" -H "Authorization: Bearer $TOKEN" 2>/dev/null || echo '{"error":"unreachable"}')
  data+="=== Recent Work Log ===
$logs

"

  # Local process check
  local procs
  procs=$(ps aux | grep -E 'ollama|nudge' | grep -v grep || echo "no matching processes")
  data+="=== Local Processes ===
$procs

"

  # Ollama health
  local ollama_models
  ollama_models=$(curl -sf "$OLLAMA_HOST/api/tags" 2>/dev/null | jq -r '.models[].name' 2>/dev/null || echo "ollama unreachable")
  data+="=== Ollama Models ===
$ollama_models

"

  # Disk and memory
  local resources
  resources=$(free -h 2>/dev/null | head -2 || echo "free not available")
  resources+="
$(df -h / 2>/dev/null | tail -1 || echo "df not available")"
  data+="=== System Resources ===
$resources

"

  # GPU
  local gpu
  gpu=$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available")
  data+="=== GPU ===
$gpu
"

  echo "$data"
}

# ==========================================================================
# Complexity heuristic: should this goal be decomposed?
# ==========================================================================
# Returns 0 (true) if goal is complex and should be decomposed first.
# Returns 1 (false) if goal is simple enough for direct codex execution.
is_complex_goal() {
  local desc="$1"
  local word_count
  word_count=$(echo "$desc" | wc -w | tr -d ' ')

  # Long descriptions (>50 words) suggest complex multi-step work
  if [ "$word_count" -gt 50 ]; then
    return 0
  fi

  # Multiple sentences with "and" or enumeration suggest multi-step
  local and_count
  and_count=$(echo "$desc" | grep -oi ' and ' | wc -l | tr -d ' ')
  if [ "$and_count" -ge 2 ]; then
    return 0
  fi

  # Numbered steps or bullet points
  if echo "$desc" | grep -qE '^\s*[0-9]+\.|^\s*[-*]' ; then
    return 0
  fi

  # Keywords that suggest multi-component work
  if echo "$desc" | grep -qiE 'system|pipeline|framework|architecture|module.*module|endpoint.*endpoint|component.*component|integration|full|complete|comprehensive'; then
    return 0
  fi

  # Multiple file types mentioned
  local file_types
  file_types=$(echo "$desc" | grep -oiE '\.(py|ts|tsx|js|jsx|sql|yaml|json|sh)' | wc -l | tr -d ' ')
  if [ "$file_types" -ge 2 ]; then
    return 0
  fi

  # Simple enough for direct execution
  return 1
}

# Disable set -e for the main loop — a daemon should never die from a stray exit code.
set +e

while true; do
  RESPONSE=$(ecurl -sf -X POST "$ENGINE/poll" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" 2>/dev/null || echo '{"task":null}')

  TASK_ID=$(echo "$RESPONSE" | jq -r '.task.id // empty')

  if [ -z "$TASK_ID" ]; then
    sleep "$POLL_INTERVAL"
    continue
  fi

  ACTION=$(echo "$RESPONSE" | jq -r '.task.action')
  DESC=$(echo "$RESPONSE" | jq -r '.task.description')
  CONTEXT=$(echo "$RESPONSE" | jq -c '.task.context // {}')
  PARENT_TASK_ID=$(echo "$RESPONSE" | jq -r '.task.parent_task_id // empty')

  log "Task: $ACTION ($TASK_ID)${PARENT_TASK_ID:+ [subtask of $PARENT_TASK_ID]}"

  # ========================================================================
  # Route task to the right execution mode
  # ========================================================================
  TASK_MODE="ollama"   # default: advisory via Ollama API

  case "$ACTION" in
    check-health|goal-verify-*|goal-check-*|goal-review-*)
      TASK_MODE="monitor"
      ;;
    goal-*)
      # Goal tasks: simple ones go to codex, complex ones get decomposed first.
      # NEVER decompose subtasks (tasks that already have a parent) — they should
      # already be small enough for codex. This prevents cascading decomposition.
      if [ "$CODEX_AVAILABLE" = true ]; then
        if [ -z "$PARENT_TASK_ID" ] && is_complex_goal "$DESC"; then
          TASK_MODE="decompose"
        else
          TASK_MODE="codex"
        fi
      fi
      ;;
    implement-*|code-*|build-*|fix-code-*|write-*)
      if [ "$CODEX_AVAILABLE" = true ]; then
        TASK_MODE="codex"
      fi
      ;;
  esac

  log "  Mode: $TASK_MODE"

  OUTPUT=""
  TASK_SUCCESS=true
  TASK_ERROR=""

  # ========================================================================
  # MODE: monitor — gather real system data, then Ollama analysis
  # ========================================================================
  if [ "$TASK_MODE" = "monitor" ]; then
    log "  Gathering real system data for monitoring task..."
    SYSTEM_DATA=$(gather_system_data)

    PROMPT="You are a system health monitor analyzing REAL data from the nudge engine.

Task: $ACTION

$DESC

Here is the ACTUAL system data gathered just now:

$SYSTEM_DATA

Based on this REAL data, provide your analysis. Be specific — reference actual numbers, task IDs, timestamps from the data above.
Flag any issues: failed tasks, offline workers, stuck tasks, crash patterns, resource problems.
End with a one-line SUMMARY: of the system health status."

    OLLAMA_RESPONSE=$(curl -sf --max-time "$OLLAMA_TIMEOUT" -X POST "$OLLAMA_HOST/api/generate" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"$MODEL\",
        \"prompt\": $(printf '%s' "$PROMPT" | jq -Rs .),
        \"stream\": false
      }" 2>/dev/null) || true

    OUTPUT=$(echo "$OLLAMA_RESPONSE" | jq -r '.response // empty' 2>/dev/null)

  # ========================================================================
  # MODE: codex — implementation via Codex CLI (agentic file I/O)
  # ========================================================================
  elif [ "$TASK_MODE" = "codex" ]; then
    log "  Running Codex for implementation task..."

    # Codex uses Ollama via OpenAI-compatible API
    export OPENAI_API_KEY="ollama"
    export OPENAI_BASE_URL="$OLLAMA_HOST/v1"

    RETRY_CONTEXT=""
    CODEX_ATTEMPT=0
    # Track ALL test files across retry attempts (not just per-attempt)
    ALL_PY_TESTS_ACCUMULATED=""
    ALL_TS_TESTS_ACCUMULATED=""

    while [ "$CODEX_ATTEMPT" -lt "$MAX_RETRIES" ]; do
      CODEX_ATTEMPT=$((CODEX_ATTEMPT + 1))
      TASK_SUCCESS=true
      TASK_ERROR=""
      VALIDATION_ERRORS=""

      if [ "$CODEX_ATTEMPT" -gt 1 ]; then
        log "  Retry $CODEX_ATTEMPT/$MAX_RETRIES (with error context)..."
      fi

      CODEX_PROMPT="$DESC

Packages: numpy, scipy, librosa, soundfile, torch, wave.
Repo: backend/, inference/, training/, scripts/, data/voice_samples/.

Create the .py file and test_*.py file. Run pytest. Fix until green.
${RETRY_CONTEXT}"

      # Snapshot git state before codex runs (for success detection + scoped validation)
      PRE_DIFF=$(cd "$REPO_DIR" && git diff --stat HEAD 2>/dev/null || echo "")
      PRE_UNTRACKED=$(cd "$REPO_DIR" && git ls-files --others --exclude-standard 2>/dev/null | wc -l || echo "0")
      PRE_CHANGED_FILES=$(cd "$REPO_DIR" && git diff --name-only HEAD 2>/dev/null || echo "")
      PRE_UNTRACKED_FILES=$(cd "$REPO_DIR" && git ls-files --others --exclude-standard 2>/dev/null || echo "")

      RAW_OUTPUT=$(cd "$REPO_DIR" && timeout "$CODEX_TIMEOUT" codex exec \
        --model "$MODEL" \
        --sandbox "$CODEX_SANDBOX" \
        "$CODEX_PROMPT" 2>&1) || true

      # Filter noise: strip model list errors and session metadata
      OUTPUT=$(echo "$RAW_OUTPUT" | sed -n '/^user$/,$p' | tail -n +2)
      if [ -z "$OUTPUT" ]; then
        OUTPUT=$(echo "$RAW_OUTPUT" | grep -v '^[0-9T:.\-Z]* ERROR\|^OpenAI Codex\|^--------\|^workdir:\|^model:\|^provider:\|^approval:\|^sandbox:\|^reasoning\|^session id:')
      fi

      # Success detection: did codex actually change/create files?
      POST_DIFF=$(cd "$REPO_DIR" && git diff --stat HEAD 2>/dev/null || echo "")
      POST_UNTRACKED=$(cd "$REPO_DIR" && git ls-files --others --exclude-standard 2>/dev/null | wc -l || echo "0")

      if [ -z "$OUTPUT" ]; then
        TASK_SUCCESS=false
        TASK_ERROR="Codex produced no output (timeout: ${CODEX_TIMEOUT}s)"
        break  # No output = no point retrying
      elif [ "$PRE_DIFF" = "$POST_DIFF" ] && [ "$PRE_UNTRACKED" = "$POST_UNTRACKED" ]; then
        # No files created/modified — this is a failure, not a warning.
        # Retry with a more forceful prompt that demands file creation.
        log "  No file changes (attempt $CODEX_ATTEMPT/$MAX_RETRIES)"

        if [ "$CODEX_ATTEMPT" -lt "$MAX_RETRIES" ]; then
          RETRY_CONTEXT="

YOU DID NOT CREATE ANY FILES. Your previous attempt only produced text.
You MUST use the write tool to create actual files on disk.
Do NOT explain what to do. DO it. Create the .py file and test file NOW."
          continue  # Retry with forceful prompt
        else
          TASK_SUCCESS=false
          TASK_ERROR="No files created after $MAX_RETRIES attempts (advisory-only responses)"
          break
        fi
      else
        FILES_CHANGED=$(($(echo "$POST_DIFF" | wc -l) - $(echo "$PRE_DIFF" | wc -l)))
        NEW_FILES=$((POST_UNTRACKED - PRE_UNTRACKED))
        log "  Codex result: ${FILES_CHANGED} files modified, ${NEW_FILES} new files"

        # Catch the case where git state changed (e.g., codex ran ls/cat) but no actual files created
        if [ "$FILES_CHANGED" -le 0 ] && [ "$NEW_FILES" -le 0 ]; then
          log "  No useful file changes (attempt $CODEX_ATTEMPT/$MAX_RETRIES)"
          if [ "$CODEX_ATTEMPT" -lt "$MAX_RETRIES" ]; then
            RETRY_CONTEXT="

YOU DID NOT CREATE ANY FILES. You ran commands but did not write code.
You MUST create the .py file and test_*.py file. Use the write tool NOW."
            continue
          else
            TASK_SUCCESS=false
            TASK_ERROR="No files created after $MAX_RETRIES attempts"
            break
          fi
        fi
      fi

      # ==== Post-task validation: RUN ACTUAL TESTS ====
      if [ "$TASK_SUCCESS" = true ]; then
        log "  Validating: running tests..."

        # Get current state and diff against pre-codex snapshot
        POST_CHANGED_FILES=$(cd "$REPO_DIR" && git diff --name-only HEAD 2>/dev/null || echo "")
        POST_UNTRACKED_FILES=$(cd "$REPO_DIR" && git ls-files --others --exclude-standard 2>/dev/null || echo "")

        # Only validate files created/modified by THIS codex run
        CODEX_MODIFIED=$(comm -13 <(echo "$PRE_CHANGED_FILES" | sort) <(echo "$POST_CHANGED_FILES" | sort) 2>/dev/null || echo "")
        CODEX_CREATED=$(comm -13 <(echo "$PRE_UNTRACKED_FILES" | sort) <(echo "$POST_UNTRACKED_FILES" | sort) 2>/dev/null || echo "")
        CODEX_ALL_FILES=$(echo -e "${CODEX_MODIFIED}\n${CODEX_CREATED}" | sort -u | grep -v '^$' || echo "")

        # Find test files created by codex (accumulate across retries)
        NEW_PY_TESTS=$(echo "$CODEX_ALL_FILES" | grep -E 'test_.*\.py$' || echo "")
        NEW_TS_TESTS=$(echo "$CODEX_ALL_FILES" | grep -E '\.test\.(ts|tsx)$' || echo "")
        ALL_PY_TESTS_ACCUMULATED=$(echo -e "${ALL_PY_TESTS_ACCUMULATED}\n${NEW_PY_TESTS}" | sort -u | grep -v '^$' || echo "")
        ALL_TS_TESTS_ACCUMULATED=$(echo -e "${ALL_TS_TESTS_ACCUMULATED}\n${NEW_TS_TESTS}" | sort -u | grep -v '^$' || echo "")
        PY_TESTS="$ALL_PY_TESTS_ACCUMULATED"
        TS_TESTS="$ALL_TS_TESTS_ACCUMULATED"

        TESTS_RUN=0
        TESTS_PASSED=0

        # --- Run Python tests ---
        if [ -n "$PY_TESTS" ]; then
          while IFS= read -r testfile; do
            [ -z "$testfile" ] && continue
            log "    pytest: $testfile"
            TESTS_RUN=$((TESTS_RUN + 1))
            TEST_DIR=$(dirname "$testfile")
            TEST_OUT=$(cd "$REPO_DIR" && PYTHONPATH="${TEST_DIR}:${PYTHONPATH:-}" timeout 120 python3 -m pytest "$testfile" -v --tb=short 2>&1) || true
            TEST_EXIT=$?

            # Extract pass/fail counts from pytest output
            PASS_COUNT=$(echo "$TEST_OUT" | grep -oE '[0-9]+ passed' | head -1 | grep -oE '[0-9]+' | head -1)
            [ -z "$PASS_COUNT" ] && PASS_COUNT=0
            FAIL_COUNT=$(echo "$TEST_OUT" | grep -oE '[0-9]+ failed' | head -1 | grep -oE '[0-9]+' | head -1)
            [ -z "$FAIL_COUNT" ] && FAIL_COUNT=0

            if echo "$TEST_OUT" | grep -q "passed" && [ "$FAIL_COUNT" = "0" ] && [ "$PASS_COUNT" != "0" ] && ! echo "$TEST_OUT" | grep -qi "error collecting"; then
              log "    PASSED ($PASS_COUNT tests)"
              TESTS_PASSED=$((TESTS_PASSED + 1))
            else
              # Extract failure details for retry context (include import errors, empty files)
              FAIL_DETAIL=$(echo "$TEST_OUT" | grep -A5 "FAILED\|ERROR\|AssertionError\|ImportError\|ModuleNotFoundError\|no tests ran\|collected 0" | head -30)
              if echo "$TEST_OUT" | grep -qi "no tests ran\|collected 0"; then
                FAIL_DETAIL+="\n\nHINT: The test file is EMPTY or contains no test functions. Write actual test functions (def test_xxx) in the test file.\n"
              fi
              ERROR_COUNT=$(echo "$TEST_OUT" | grep -oE '[0-9]+ error' | head -1 | grep -oE '[0-9]+' | head -1)
              [ -z "$ERROR_COUNT" ] && ERROR_COUNT=0
              VALIDATION_ERRORS+="pytest $testfile FAILED ($FAIL_COUNT failures, $ERROR_COUNT errors):\n$FAIL_DETAIL\n\nHINT: If ImportError, the test imports from a local module. Make sure the module exists in the same directory as the test file. Use relative imports or ensure sys.path includes the test directory.\n\n"
              log "    FAILED ($FAIL_COUNT failures, $ERROR_COUNT errors)"
            fi
          done <<< "$PY_TESTS"
        fi

        # --- Run TypeScript tests ---
        if [ -n "$TS_TESTS" ]; then
          while IFS= read -r testfile; do
            [ -z "$testfile" ] && continue
            log "    vitest: $testfile"
            TESTS_RUN=$((TESTS_RUN + 1))
            TEST_OUT=$(cd "$REPO_DIR/frontend" && timeout 120 npx vitest run "../$testfile" --reporter=verbose 2>&1) || true

            if echo "$TEST_OUT" | grep -q "Tests.*passed" && ! echo "$TEST_OUT" | grep -q "Tests.*failed"; then
              TS_PASS=$(echo "$TEST_OUT" | grep -oP '\d+(?= passed)' || echo "0")
              log "    PASSED ($TS_PASS tests)"
              TESTS_PASSED=$((TESTS_PASSED + 1))
            else
              FAIL_DETAIL=$(echo "$TEST_OUT" | grep -A5 "FAIL\|Error\|expect\|AssertionError" | head -20)
              VALIDATION_ERRORS+="vitest $testfile FAILED:\n$FAIL_DETAIL\n\n"
              log "    FAILED"
            fi
          done <<< "$TS_TESTS"
        fi

        # --- Fallback: syntax check if no test files found ---
        if [ "$TESTS_RUN" -eq 0 ]; then
          ALL_PY=$(echo "$CODEX_ALL_FILES" | grep -E '\.py$' || echo "")
          ALL_TS=$(echo "$CODEX_ALL_FILES" | grep -E '\.(ts|tsx)$' || echo "")
          SYNTAX_CHECKED=0

          if [ -n "$ALL_PY" ]; then
            while IFS= read -r pyfile; do
              [ -z "$pyfile" ] && continue
              ERR=$(cd "$REPO_DIR" && python3 -m py_compile "$pyfile" 2>&1) || true
              if [ -n "$ERR" ]; then
                VALIDATION_ERRORS+="Syntax error in $pyfile: $ERR\n"
              fi
              SYNTAX_CHECKED=$((SYNTAX_CHECKED + 1))
            done <<< "$ALL_PY"
          fi

          if [ "$SYNTAX_CHECKED" -gt 0 ] || [ -n "$ALL_TS" ]; then
            log "  No test files found — syntax-only validation ($SYNTAX_CHECKED files)"
          fi
        fi

        # --- Verdict ---
        if [ -z "$VALIDATION_ERRORS" ]; then
          if [ "$TESTS_RUN" -gt 0 ]; then
            log "  ALL TESTS PASSED ($TESTS_PASSED/$TESTS_RUN test files)"
          fi
          break  # Success — exit retry loop
        fi

        # Validation failed — prepare for retry
        log "  TESTS FAILED (attempt $CODEX_ATTEMPT/$MAX_RETRIES)"

        if [ "$CODEX_ATTEMPT" -lt "$MAX_RETRIES" ]; then
          RETRY_CONTEXT="

IMPORTANT: Your tests are FAILING. Fix the implementation to make them pass.

Test failures:
$(echo -e "$VALIDATION_ERRORS")

Rules:
- Do NOT modify the test files. The tests define the correct behavior.
- Fix the IMPLEMENTATION code to make the tests pass.
- Run the tests again after fixing to verify."
          log "  Retrying with test failure context..."
        else
          TASK_SUCCESS=false
          TASK_ERROR="Tests still failing after $MAX_RETRIES attempts ($TESTS_PASSED/$TESTS_RUN passed)"
          OUTPUT+="

=== TEST FAILURES (final attempt) ===
$(echo -e "$VALIDATION_ERRORS")"
        fi
      else
        break  # Non-validation failure, don't retry
      fi
    done

  # ========================================================================
  # MODE: decompose — break complex goals into concrete subtasks
  # ========================================================================
  elif [ "$TASK_MODE" = "decompose" ]; then
    log "  Decomposing complex goal into subtasks..."

    # The decomposition prompt encodes the constraints of qwen3-coder AND
    # provides codebase context so subtasks reference REAL modules and files.
    PROMPT="You are a task decomposer for a voice-cloning research platform. Break complex goals into small subtasks that EXTEND the existing codebase.

CRITICAL CONSTRAINTS — each subtask is executed by a small AI model (30B params):
- ONE function per task, ONE file created or modified
- Available packages: numpy, scipy, librosa, soundfile, torch (CUDA), wave, json — NO pip install
- Exact test values must be provided (the model cannot infer what to test)
- Max 10 minutes execution time
- Python or TypeScript only

$CODEBASE_MAP

GOAL TO DECOMPOSE:
$DESC

CONTEXT:
$CONTEXT

Break this into 3-7 subtasks. For EACH subtask, output this EXACT format:

SUBTASK: <action-slug> | <one-sentence what to build> | <capability> | <full description with test spec>

The <full description> MUST include:
1. The EXACT function signature (e.g., def analyze_pitch(wav_path: str) -> dict)
2. What it does in 1-2 sentences, referencing existing modules where relevant
3. EXACT test cases with real values:
   - assert abs(analyze_pitch('data/voice_samples/session/calm_1.wav')['mean_f0'] - 120.0) < 50.0
   - assert len(result['contour']) > 0
4. The filepath: EXTEND existing files when possible (e.g., backend/prosody_analyzer.py), or create in the right directory (scripts/, backend/, inference/)

Rules for action-slug:
- kebab-case, prefixed with goal-implement-, goal-create-, goal-build-
- Max 40 chars after goal- prefix

Rules for capability:
- 'code' for implementation tasks (most subtasks)
- 'monitor' for verification/review tasks (final subtask if needed)

RESEARCH QUALITY RULES:
- Subtasks must DO something meaningful: analyze real audio, compute spectral features, extract prosody patterns
- Use librosa for audio I/O and features, scipy for signal processing, numpy for arrays, torch for ML
- Reference real data paths: data/voice_samples/, data/labeled/, data/audio_datasets/
- NEVER create toy string/math utilities — every function should process audio, speech, or prosody data
- Tests should verify against real audio properties (sample rates, frequency ranges, array shapes)
- Build ON existing infrastructure: extend AcousticAnalyzer, add to inference pipeline, etc.

Output ONLY SUBTASK lines, nothing else. No explanation, no preamble."

    OLLAMA_RESPONSE=$(curl -sf --max-time "$OLLAMA_TIMEOUT" -X POST "$OLLAMA_HOST/api/generate" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"$DECOMPOSE_MODEL\",
        \"prompt\": $(printf '%s' "$PROMPT" | jq -Rs .),
        \"stream\": false
      }" 2>/dev/null) || true

    DECOMP_OUTPUT=$(echo "$OLLAMA_RESPONSE" | jq -r '.response // empty' 2>/dev/null)

    if [ -z "$DECOMP_OUTPUT" ]; then
      TASK_SUCCESS=false
      TASK_ERROR="Ollama returned no decomposition (model: $DECOMPOSE_MODEL)"
    else
      # Parse subtasks and create them via the API with parent_task_id linking
      SUBTASKS_CREATED=0
      SUBTASK_SUMMARY=""

      while IFS= read -r line; do
        # Match lines like: SUBTASK: action-slug | short desc | capability | full description
        if echo "$line" | grep -qi "^SUBTASK:"; then
          SUB_ACTION=$(echo "$line" | sed 's/^SUBTASK:[[:space:]]*//' | cut -d'|' -f1 | tr -d ' ')
          SUB_SHORT=$(echo "$line" | cut -d'|' -f2 | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//')
          SUB_CAP=$(echo "$line" | cut -d'|' -f3 | tr -d ' ' | tr '[:upper:]' '[:lower:]')
          SUB_FULL=$(echo "$line" | cut -d'|' -f4- | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//')

          # Use full description if available, otherwise fall back to short
          SUB_DESC="${SUB_FULL:-$SUB_SHORT}"

          # Validate capability
          case "$SUB_CAP" in
            code|monitor|advisory) ;;
            *) SUB_CAP="code" ;;
          esac

          # Sanitize action slug
          SUB_ACTION=$(echo "$SUB_ACTION" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]/-/g' | sed 's/--*/-/g' | sed 's/^-//;s/-$//')
          # Ensure goal- prefix
          if ! echo "$SUB_ACTION" | grep -q '^goal-'; then
            SUB_ACTION="goal-$SUB_ACTION"
          fi

          if [ -n "$SUB_ACTION" ] && [ -n "$SUB_DESC" ]; then
            # Create the subtask with parent_task_id linking
            CREATE_RESP=$(ecurl -sf -X POST "$ENGINE/tasks" \
              -H "Authorization: Bearer $TOKEN" \
              -H "Content-Type: application/json" \
              -d "{
                \"action\": $(printf '%s' "$SUB_ACTION" | jq -Rs .),
                \"description\": $(printf '%s' "$SUB_DESC" | jq -Rs .),
                \"priority\": 6,
                \"required_capability\": $(printf '%s' "$SUB_CAP" | jq -Rs .),
                \"parent_task_id\": \"$TASK_ID\"
              }" 2>/dev/null) || true

            if echo "$CREATE_RESP" | jq -e '.success' > /dev/null 2>&1; then
              SUBTASKS_CREATED=$((SUBTASKS_CREATED + 1))
              SUBTASK_SUMMARY+="- $SUB_ACTION: $SUB_SHORT\n"
              log "  Created subtask $SUBTASKS_CREATED: $SUB_ACTION"
            else
              log "  Failed to create subtask: $SUB_ACTION"
            fi
          fi
        fi
      done <<< "$DECOMP_OUTPUT"

      if [ "$SUBTASKS_CREATED" -eq 0 ]; then
        # Fallback: if decomposition failed to produce parseable subtasks,
        # downgrade to codex mode and let it try directly
        log "  Decomposition produced 0 parseable subtasks, falling back to codex"
        TASK_SUCCESS=false
        TASK_ERROR="Decomposition failed: no parseable subtasks from model output"
        OUTPUT="$DECOMP_OUTPUT"
      else
        OUTPUT="Decomposed into $SUBTASKS_CREATED subtasks:\n$(echo -e "$SUBTASK_SUMMARY")\n\nRaw decomposition:\n$DECOMP_OUTPUT"
        log "  Successfully decomposed into $SUBTASKS_CREATED subtasks"
      fi
    fi

  # ========================================================================
  # MODE: ollama — advisory response via Ollama API
  # ========================================================================
  else
    PROMPT="You are an advisory assistant. You can ONLY provide analysis and recommendations — you CANNOT run commands, create files, or make changes.

Task: $ACTION

$DESC

Context: $CONTEXT

IMPORTANT: Do NOT claim to have run any commands, installed anything, created files, or made any changes. You have NO ability to do those things. You can only analyze and recommend.

Provide a concise analysis and actionable recommendations in under 500 words. Be clear about what SHOULD be done (by a human or agentic tool), not what you did.

End with a one-line SUMMARY: of your recommendation."

    OLLAMA_RESPONSE=$(curl -sf --max-time "$OLLAMA_TIMEOUT" -X POST "$OLLAMA_HOST/api/generate" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"$MODEL\",
        \"prompt\": $(printf '%s' "$PROMPT" | jq -Rs .),
        \"stream\": false
      }" 2>/dev/null) || true

    OUTPUT=$(echo "$OLLAMA_RESPONSE" | jq -r '.response // empty' 2>/dev/null)
  fi

  # ========================================================================
  # Report result
  # ========================================================================
  if [ -z "$OUTPUT" ] && [ "$TASK_SUCCESS" = true ]; then
    TASK_SUCCESS=false
    TASK_ERROR="No response from $TASK_MODE (model: $MODEL)"
  fi

  if [ "$TASK_SUCCESS" = false ]; then
    log "Failed: $TASK_ID ($TASK_ERROR)"

    # Include truncated output for diagnostics on failure
    DIAG_OUTPUT=""
    if [ -n "$OUTPUT" ]; then
      DIAG_OUTPUT=$(echo "$OUTPUT" | tail -30 | head -c 2000)
    fi

    ecurl -sf -X POST "$ENGINE/report" \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d "{
        \"taskId\": \"$TASK_ID\",
        \"success\": false,
        \"error\": $(printf '%s' "$TASK_ERROR" | jq -Rs .),
        \"result\": {
          \"mode\": \"$TASK_MODE\",
          \"diagnostics\": $(printf '%s' "$DIAG_OUTPUT" | jq -Rs .)
        }
      }" > /dev/null 2>&1 || true
  else
    # Extract SUMMARY and TEST_CMD from codex output
    SUMMARY=$(echo "$OUTPUT" | grep -i "SUMMARY:" | grep -v "tool_call\|</\|</" | tail -1 | sed 's/.*SUMMARY:[[:space:]]*//;s/\*\+//g;s/^[[:space:]]*//;s/[[:space:]]*$//')
    TEST_CMD=$(echo "$OUTPUT" | grep -i "TEST_CMD:" | grep -v "tool_call\|</\|</" | tail -1 | sed 's/.*TEST_CMD:[[:space:]]*//;s/\*\+//g;s/^[[:space:]]*//;s/[[:space:]]*$//')
    if [ -z "$SUMMARY" ]; then
      # Fallback: last meaningful line (skip XML tags, empty lines, token counts, markdown)
      SUMMARY=$(echo "$OUTPUT" | grep -v '^$\|^[[:space:]]*$\|</\|tool_call\|tokens used\|^---\|^\*\*$' | tail -3 | head -1 | sed 's/\*\+//g' | head -c 200)
    fi

    log "Done: $TASK_ID [$TASK_MODE] — $SUMMARY"

    # Build result metadata
    RESULT_META="\"mode\": \"$TASK_MODE\", \"summary\": $(printf '%s' "$SUMMARY" | jq -Rs .)"
    if [ "$TASK_MODE" = "codex" ]; then
      RESULT_META+=", \"attempts\": $CODEX_ATTEMPT"
      if [ "${TESTS_RUN:-0}" -gt 0 ]; then
        RESULT_META+=", \"testsRun\": $TESTS_RUN, \"testsPassed\": $TESTS_PASSED, \"testDriven\": true"
      else
        RESULT_META+=", \"syntaxOnly\": true"
      fi
      if [ -n "${TEST_CMD:-}" ]; then
        RESULT_META+=", \"testCmd\": $(printf '%s' "$TEST_CMD" | jq -Rs .)"
      fi
    fi
    RESULT_META+=", \"fullResponse\": $(echo "$OUTPUT" | head -100 | jq -Rs .)"

    ecurl -sf -X POST "$ENGINE/report" \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d "{
        \"taskId\": \"$TASK_ID\",
        \"success\": true,
        \"result\": { $RESULT_META }
      }" > /dev/null 2>&1 || true
  fi

  sleep 5
done
