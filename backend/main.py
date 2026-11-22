"""
PVGCOET Voice Assistant: basic RAG pipeline (CPU only)

    mic audio --> Whisper (speech to text)
              --> Nomic embedding --> Qdrant search (top-K chunks)
              --> Gemma via Ollama (answer, streamed token by token)
              --> Piper (text to speech, one sentence at a time)
              --> browser plays audio

Run:  uvicorn main:app --port 8000      (see ../run_local.sh)
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import uuid
import wave
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import noisereduce as nr
import numpy as np
import ollama
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS (all overridable with environment variables)
# ─────────────────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium.en")
OLLAMA_LLM_MODEL   = os.getenv("OLLAMA_LLM_MODEL",   "gemma2:2b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_HOST        = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT        = int(os.getenv("OLLAMA_PORT", "11434"))
QDRANT_HOST        = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT        = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION  = "pvg_college"
PIPER_MODEL_PATH   = os.getenv("PIPER_MODEL_PATH", "./tts_models/en_US-amy-medium.onnx")
PIPER_BIN          = os.getenv("PIPER_BIN", "piper")   # path to the piper executable

TOP_K        = 5      # how many chunks to retrieve from Qdrant
MIN_SCORE    = 0.45   # ignore chunks less similar than this (cosine similarity)
MEMORY_TURNS = 4      # remember the last 4 question/answer pairs per session

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

SYSTEM_PROMPT = """You are a helpful voice assistant for Pune Vidyarthi Griha's College of Engineering (PVGCOET), Pune, India.
You help students, faculty, visitors, and admin staff with:
- Admissions and fees
- Faculty and department information
- Campus facilities and hostel
- Placements

Answer concisely in 2-3 sentences, your response will be spoken aloud.
Answer from the provided context. If unsure, say so politely. Do NOT fabricate information."""


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS ONCE AT STARTUP
# ─────────────────────────────────────────────────────────────────────────────
log.info("Loading Faster-Whisper (%s) on CPU...", WHISPER_MODEL_SIZE)
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device="cpu",
    compute_type="int8",                       # 8-bit weights: fast on CPU
    cpu_threads=min(6, os.cpu_count() or 4),
    num_workers=1,
)
log.info("Whisper ready.")

qdrant        = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
ollama_client = ollama.Client(host=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")

# chat memory: {session_id: last N messages}
sessions: Dict[str, deque] = {}

if not Path(PIPER_MODEL_PATH).exists():
    log.warning("Piper voice model not found at %s, the browser will read answers instead.", PIPER_MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up so the first real question isn't slow (models get loaded into RAM)
    log.info("Warming up LLM and embedding model...")
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, lambda: ollama_client.chat(
            model=OLLAMA_LLM_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            options={"num_predict": 1},
        ))
        await loop.run_in_executor(None, lambda: ollama_client.embeddings(
            model=OLLAMA_EMBED_MODEL, prompt="test"
        ))
        log.info("Warmup complete.")
    except Exception as e:
        log.warning("Warmup failed (first query will be slower): %s", e)
    yield


app = FastAPI(title="PVGCOET Voice Assistant", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: SPEECH TO TEXT
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_audio(pcm_bytes: bytes, sr: int = 16000) -> bytes:
    """Remove background noise, then normalise the volume."""
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    try:
        arr = nr.reduce_noise(y=arr, sr=sr, stationary=True, prop_decrease=0.75)
    except Exception:
        pass
    peak = np.max(np.abs(arr))
    if peak > 0.01:
        arr = arr / peak * 0.95
    return (arr * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()


def transcribe_audio(pcm_bytes: bytes, sr: int = 16000) -> str:
    """Raw 16-bit mono audio from the browser -> text."""
    clean = preprocess_audio(pcm_bytes, sr)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(clean)
    try:
        segs, info = whisper_model.transcribe(
            tmp,
            language="en",
            beam_size=3,
            vad_filter=True,   # skip silence
            vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 200, "threshold": 0.35},
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
        )
        text = " ".join(s.text.strip() for s in segs).strip()
        # Whisper hallucinates these on silence/noise, so treat them as "heard nothing"
        if text.lower().strip(" .") in {"thank you", "you", "", "...", "hmm", "uh", "um"}:
            return ""
        log.info("STT: '%s' (%.1fs audio)", text, info.duration)
        return text
    finally:
        os.unlink(tmp)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: RETRIEVAL (the "R" in RAG)
# ─────────────────────────────────────────────────────────────────────────────
def retrieve_context(query: str) -> str:
    """Question -> embedding vector -> most similar chunks stored in Qdrant."""
    resp = ollama_client.embeddings(model=OLLAMA_EMBED_MODEL, prompt=query)
    vec = resp["embedding"] if isinstance(resp, dict) else resp.embedding

    try:
        hits = qdrant.query_points(
            collection_name=QDRANT_COLLECTION, query=vec, limit=TOP_K,
            with_payload=True, score_threshold=MIN_SCORE,
        ).points
    except AttributeError:   # older qdrant-client versions
        hits = qdrant.search(
            collection_name=QDRANT_COLLECTION, query_vector=vec, limit=TOP_K,
            with_payload=True, score_threshold=MIN_SCORE,
        )

    if hits:
        log.info("RAG: %d chunks (scores: %s)", len(hits), [round(h.score, 3) for h in hits])
    else:
        log.warning("RAG: nothing above %.2f for '%s'", MIN_SCORE, query[:50])

    return "\n\n---\n\n".join(h.payload.get("text", "") for h in hits)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: TEXT TO SPEECH
# ─────────────────────────────────────────────────────────────────────────────
def text_to_speech(text: str) -> bytes:
    """Sentence -> WAV bytes using the Piper command-line tool. Empty = failed."""
    text = text.strip()
    if not text or not Path(PIPER_MODEL_PATH).exists():
        return b""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out = f.name
    try:
        r = subprocess.run(
            [PIPER_BIN, "--model", PIPER_MODEL_PATH, "--output_file", out],
            input=text.encode(), capture_output=True, timeout=30,
        )
        if r.returncode != 0:
            log.warning("Piper error: %s", r.stderr.decode()[:200])
            return b""
        with open(out, "rb") as f:
            return f.read()
    except FileNotFoundError:
        log.warning("Piper executable not found (%s). Set PIPER_BIN to its full path.", PIPER_BIN)
        return b""
    except Exception as e:
        log.warning("Piper exception: %s", e)
        return b""
    finally:
        if os.path.exists(out):
            os.unlink(out)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: GENERATION (the "G" in RAG), streamed, with TTS running in parallel
# ─────────────────────────────────────────────────────────────────────────────
async def run_pipeline(ws: WebSocket, question: str, session_id: str):
    loop = asyncio.get_running_loop()

    # --- retrieve ---
    await ws.send_json({"type": "status", "message": "Searching knowledge base..."})
    context = await loop.run_in_executor(None, retrieve_context, question)
    await ws.send_json({"type": "rag_done"})

    # --- build the prompt: system + chat history + (context + question) ---
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(sessions[session_id])

    if context:
        user_content = (
            f"Context from college knowledge base:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer using only the context above. Be concise, 2-3 sentences."
        )
    else:
        user_content = (
            f"Question: {question}\n\n"
            f"No relevant information was found in the college knowledge base. "
            f"Politely tell the user you don't have that information and suggest they "
            f"contact the college at 020-24228258 or enquiry@pvgcoet.ac.in"
        )
    messages.append({"role": "user", "content": user_content})

    await ws.send_json({"type": "status", "message": "Generating response..."})

    # --- stream tokens from Ollama on a background thread into a queue ---
    token_queue: asyncio.Queue = asyncio.Queue()

    def stream_llm():
        try:
            for chunk in ollama_client.chat(
                model=OLLAMA_LLM_MODEL, messages=messages, stream=True,
                options={"temperature": 0.3, "num_predict": 100, "num_ctx": 2048},
            ):
                if hasattr(chunk, "message"):
                    token, done = chunk.message.content or "", chunk.done
                else:
                    token, done = chunk.get("message", {}).get("content", ""), chunk.get("done", False)
                asyncio.run_coroutine_threadsafe(token_queue.put((token, done)), loop)
                if done:
                    break
        except Exception as e:
            log.error("LLM stream error: %s", e)
            asyncio.run_coroutine_threadsafe(token_queue.put(("__ERROR__", True)), loop)

    threading.Thread(target=stream_llm, daemon=True).start()

    async def speak(sentence: str):
        """Turn one sentence into audio and send it to the browser."""
        audio = await loop.run_in_executor(None, text_to_speech, sentence)
        if audio:
            await ws.send_json({"type": "tts_chunk_start"})
            await ws.send_bytes(audio)
            await ws.send_json({"type": "tts_chunk_done"})
        else:
            await ws.send_json({"type": "tts_fallback", "text": sentence})   # browser voice

    # --- as tokens arrive: show them, and speak each finished sentence right away ---
    full_response = ""
    sentence_buf = ""
    tts_tasks: List[asyncio.Task] = []

    while True:
        token, done = await token_queue.get()
        if token == "__ERROR__":
            await ws.send_json({"type": "error", "message": "LLM error occurred."})
            break
        if token:
            full_response += token
            sentence_buf += token
            await ws.send_json({"type": "token", "text": token})

            parts = SENTENCE_SPLIT.split(sentence_buf)
            if len(parts) > 1:                       # at least one full sentence is ready
                for sentence in parts[:-1]:
                    if sentence.strip():
                        tts_tasks.append(asyncio.create_task(speak(sentence)))
                sentence_buf = parts[-1]             # keep the unfinished part
        if done:
            break

    if sentence_buf.strip():
        tts_tasks.append(asyncio.create_task(speak(sentence_buf)))
    if tts_tasks:
        await asyncio.gather(*tts_tasks)

    answer = full_response.strip()
    sessions[session_id].append({"role": "user", "content": question})
    sessions[session_id].append({"role": "assistant", "content": answer})

    await ws.send_json({"type": "response", "text": answer})
    await ws.send_json({"type": "audio_done"})
    log.info("Answer: '%s'", answer[:80])


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET: the browser talks to the backend through this
# ─────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket):
    await websocket.accept()
    sid = str(uuid.uuid4())
    sessions[sid] = deque(maxlen=MEMORY_TURNS * 2)
    log.info("Session connected: %s", sid)

    try:
        await websocket.send_json({"type": "session", "session_id": sid})
        while True:
            msg = await websocket.receive()

            if msg.get("bytes"):                                    # voice input
                await websocket.send_json({"type": "status", "message": "Transcribing..."})
                text = await asyncio.get_running_loop().run_in_executor(None, transcribe_audio, msg["bytes"])
                if not text:
                    await websocket.send_json({"type": "error", "message": "Could not understand. Please try again."})
                    continue
                await websocket.send_json({"type": "transcript", "text": text})
                await run_pipeline(websocket, text, sid)

            elif msg.get("text"):                                   # typed input / controls
                data = json.loads(msg["text"])
                kind = data.get("type")
                if kind == "ping":
                    await websocket.send_json({"type": "pong"})
                elif kind == "clear_history":
                    sessions[sid].clear()
                    await websocket.send_json({"type": "history_cleared"})
                elif kind == "text_query" and data.get("text", "").strip():
                    q = data["text"].strip()
                    await websocket.send_json({"type": "transcript", "text": q})
                    await run_pipeline(websocket, q, sid)

    except WebSocketDisconnect:
        log.info("Session disconnected: %s", sid)
    except RuntimeError as e:
        if "disconnect" not in str(e).lower():
            log.exception("Runtime error in session %s: %s", sid, e)
    except Exception as e:
        log.exception("Session error %s: %s", sid, e)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        sessions.pop(sid, None)


@app.get("/health")
async def health():
    return {"status": "ok", "llm": OLLAMA_LLM_MODEL, "stt": WHISPER_MODEL_SIZE}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
