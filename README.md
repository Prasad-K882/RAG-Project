# PVGCOET Voice Assistant (basic RAG, CPU only)

Ask a question by voice or text, get a spoken answer based on the college's own documents.

## How it works

```
OFFLINE (once):   data/*.txt -> split into ~400 char chunks -> Nomic embeddings -> Qdrant
                  (scripts/ingest.py)

ONLINE (per question):
  mic -> Whisper -> text
      -> embed question -> Qdrant top-5 similar chunks      (Retrieval)
      -> prompt = system rules + chat history + chunks + question
      -> Gemma (Ollama) streams the answer                   (Generation)
      -> each finished sentence -> Piper -> audio -> browser
```

| Concept | Where |
|---|---|
| Chunking | `scripts/ingest.py` -> `chunk_text()` |
| Embeddings + vector DB | `ingest.py` (write) / `backend/main.py` -> `retrieve_context()` (read) |
| Retrieval threshold | `TOP_K`, `MIN_SCORE` in `main.py` |
| Prompt building + LLM | `main.py` -> `run_pipeline()` |
| Speech to text | `main.py` -> `transcribe_audio()` |
| Text to speech | `main.py` -> `text_to_speech()` |
| Chat memory | `sessions` dict in `main.py` (last 4 turns) |
| Browser <-> server | WebSocket `/ws/voice` |

## Run

```bash
bash run_local.sh      # starts Ollama check, Qdrant, ingestion, frontend, backend
# open http://localhost:3000 in Chrome
bash stop_local.sh     # stop everything
```

First run: creates `venv/`, downloads the Qdrant binary, and embeds everything in `data/` into Qdrant
(a minute or two). After that, ingestion is skipped automatically because Qdrant already has vectors.
After editing files in `data/`:  `python scripts/ingest.py --data_dir ./data --reset`  (wipes and rebuilds the vectors)

## Config (environment variables, all optional)

| Variable | Default |
|---|---|
| `OLLAMA_LLM_MODEL` | `gemma2:2b` |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` |
| `WHISPER_MODEL_SIZE` | `medium.en` (use `small.en` for speed) |
