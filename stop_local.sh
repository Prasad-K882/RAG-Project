#!/bin/bash
# Stops the backend, frontend server and Qdrant. Leaves Ollama alone.
# Your vectors stay in ./storage.
pkill -f "uvicorn main:app"   && echo "backend stopped"  || echo "backend was not running"
pkill -f "http.server 3000"   && echo "frontend stopped" || echo "frontend was not running"
pkill -x qdrant               && echo "qdrant stopped"   || echo "qdrant was not running"
