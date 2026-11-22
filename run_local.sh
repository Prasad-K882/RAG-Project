#!/bin/bash
# PVGCOET Voice Assistant: start everything (no Docker).
# Usage: bash run_local.sh        (Ctrl+C to stop)

set -e
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJECT_DIR/venv"

LLM_MODEL="${OLLAMA_LLM_MODEL:-gemma2:2b}"
EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"
WHISPER="${WHISPER_MODEL_SIZE:-medium.en}"      # try small.en if it feels slow

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[PVG]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
die()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

cleanup() {
    log "Shutting down..."
    [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null
    [ -n "$QDRANT_PID" ]   && kill "$QDRANT_PID"   2>/dev/null
}
trap cleanup EXIT INT TERM

# 1. Python environment (created once, reused after that)
if [ ! -d "$VENV" ]; then
    log "Creating virtual environment (first run only)..."
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install -q --upgrade pip
    pip install -q -r "$PROJECT_DIR/backend/requirements.txt"
else
    source "$VENV/bin/activate"
fi

# Piper executable: use $PIPER_BIN if set, otherwise look in the usual places
if [ -z "$PIPER_BIN" ]; then
    for c in "$PROJECT_DIR/tts_models/piper/piper" "$PROJECT_DIR/tts_models/piper" \
             "$PROJECT_DIR/piper/piper" "$(command -v piper 2>/dev/null)"; do
        if [ -n "$c" ] && [ -f "$c" ]; then
            chmod +x "$c" 2>/dev/null
            PIPER_BIN="$c"; break
        fi
    done
    if [ -z "$PIPER_BIN" ]; then
        # Not found anywhere: download the standalone Piper program once (~26 MB, Linux only)
        case "$(uname -s)-$(uname -m)" in
            Linux-x86_64)  PIPER_ASSET="piper_linux_x86_64.tar.gz" ;;
            Linux-aarch64) PIPER_ASSET="piper_linux_aarch64.tar.gz" ;;
            *)             PIPER_ASSET="" ;;
        esac
        if [ -n "$PIPER_ASSET" ]; then
            log "Piper program not found. Downloading it (first run only)..."
            if curl -fL --progress-bar -o "$PROJECT_DIR/piper.tgz" \
                 "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/$PIPER_ASSET" \
               && tar -xzf "$PROJECT_DIR/piper.tgz" -C "$PROJECT_DIR"; then
                PIPER_BIN="$PROJECT_DIR/piper/piper"
            fi
            rm -f "$PROJECT_DIR/piper.tgz"
        fi
        [ -z "$PIPER_BIN" ] && { PIPER_BIN="piper"; warn "Could not set up Piper. Voice falls back to the browser's voice."; }
    fi
fi
log "Piper: $PIPER_BIN"
export PIPER_BIN
[ -f "$PROJECT_DIR/tts_models/en_US-amy-medium.onnx" ] || warn "tts_models/en_US-amy-medium.onnx is missing."

# 2. Ollama + check that your existing models are there (never downloads anything)
command -v ollama >/dev/null 2>&1 || die "Ollama not installed. See https://ollama.com"
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    log "Starting Ollama..."
    ollama serve > /tmp/ollama.log 2>&1 &
    for i in $(seq 1 20); do curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
fi
MODELS="$(ollama list)"
for m in "$LLM_MODEL" "$EMBED_MODEL"; do
    echo "$MODELS" | grep -q "$m" || die "Ollama model '$m' not found. Run: ollama list   (then set OLLAMA_LLM_MODEL=<your model> or: ollama pull $m)"
done
log "Ollama OK ($LLM_MODEL + $EMBED_MODEL)"

# 3. Qdrant (native binary, no Docker). Started from the project root so its data
#    lives in ./storage. If the binary is missing it is downloaded once (~70 MB).
QDRANT_BIN="$PROJECT_DIR/qdrant"
QDRANT_VERSION="v1.11.5"     # matches the qdrant-client version in requirements.txt
if curl -sf http://localhost:6333/healthz >/dev/null 2>&1; then
    log "Qdrant already running on :6333 (reusing it)."
else
    if [ ! -f "$QDRANT_BIN" ]; then
        case "$(uname -s)-$(uname -m)" in
            Linux-x86_64)  ASSET="qdrant-x86_64-unknown-linux-gnu.tar.gz" ;;
            Linux-aarch64) ASSET="qdrant-aarch64-unknown-linux-musl.tar.gz" ;;
            Darwin-x86_64) ASSET="qdrant-x86_64-apple-darwin.tar.gz" ;;
            Darwin-arm64)  ASSET="qdrant-aarch64-apple-darwin.tar.gz" ;;
            *) die "Unsupported system $(uname -s)-$(uname -m). Download Qdrant manually into $PROJECT_DIR" ;;
        esac
        log "Downloading Qdrant $QDRANT_VERSION (first run only)..."
        curl -fL --progress-bar -o "$PROJECT_DIR/qdrant.tgz" \
            "https://github.com/qdrant/qdrant/releases/download/$QDRANT_VERSION/$ASSET" \
            || die "Qdrant download failed. Check your internet and run again."
        tar -xzf "$PROJECT_DIR/qdrant.tgz" -C "$PROJECT_DIR" qdrant && rm -f "$PROJECT_DIR/qdrant.tgz"
        chmod +x "$QDRANT_BIN"
    fi
    log "Starting Qdrant..."
    cd "$PROJECT_DIR"
    "$QDRANT_BIN" > /tmp/qdrant.log 2>&1 &
    QDRANT_PID=$!
    for i in $(seq 1 30); do curl -sf http://localhost:6333/healthz >/dev/null 2>&1 && break; sleep 1; done
    curl -sf http://localhost:6333/healthz >/dev/null 2>&1 || die "Qdrant did not start. See /tmp/qdrant.log"
fi

# 4. Chunk + embed the .txt files in ./data into Qdrant (only does work if Qdrant is empty)
python3 "$PROJECT_DIR/scripts/ingest.py" --data_dir "$PROJECT_DIR/data"

# 5. Frontend
cd "$PROJECT_DIR/frontend"
python3 -m http.server 3000 > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!

# 6. Backend (runs in the foreground)
log ">>> Open http://localhost:3000 in Chrome <<<   (Ctrl+C to stop)"
cd "$PROJECT_DIR/backend"
OLLAMA_LLM_MODEL="$LLM_MODEL" OLLAMA_EMBED_MODEL="$EMBED_MODEL" WHISPER_MODEL_SIZE="$WHISPER" \
PIPER_MODEL_PATH="$PROJECT_DIR/tts_models/en_US-amy-medium.onnx" \
uvicorn main:app --host 0.0.0.0 --port 8000
