#!/usr/bin/env python3
"""
Borealis 4B LoRA Demo Server

Lightweight FastAPI server that:
1. Downloads LoRA adapters from HuggingFace
2. Proxies to a vLLM backend with dynamic LoRA loading/unloading
3. Serves a simple frontend for toggling adapters and chatting

Usage:
    # Terminal 1: Start vLLM backend
    VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 vllm serve NbAiLab/borealis-4b-instruct-preview \
        --enable-lora --max-loras 7 --max-lora-rank 64 --max-cpu-loras 7 \
        --dtype bfloat16 --port 8000

    # Terminal 2: Start this demo server
    python server.py [--vllm-url http://localhost:8000] [--port 8080]
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_REPO = "NbAiLab/borealis-4b-alignment-loras"
BASE_MODEL = "NbAiLab/borealis-4b-instruct-preview"

ADAPTERS = {
    "alignment_consolidated": "All alignment dimensions merged",
    "alignment_ethics": "Ethical reasoning and safety",
    "alignment_factuality": "Factual accuracy and grounding",
    "alignment_fictional_fidelity": "Creative writing consistency",
    "alignment_language": "Norwegian language proficiency",
    "alignment_personality": "Consistent persona and tone",
    "alignment_prompt_requirements": "Instruction following",
}

app = FastAPI(title="Borealis LoRA Demo")

# Global state
vllm_url: str = "http://localhost:8000"
adapter_paths: dict[str, str] = {}
loaded_adapters: set[str] = set()


def download_adapters(cache_dir: str | None = None) -> dict[str, str]:
    """Download all adapter subfolders and return name -> local_path mapping."""
    paths = {}
    for name in ADAPTERS:
        logger.info(f"Downloading adapter: {name}")
        local = snapshot_download(
            HF_REPO,
            allow_patterns=[f"{name}/*"],
            cache_dir=cache_dir,
            token=os.environ.get("HF_TOKEN"),
        )
        paths[name] = str(Path(local) / name)
        logger.info(f"  -> {paths[name]}")
    return paths


async def vllm_request(method: str, path: str, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(timeout=120) as client:
        return await client.request(method, f"{vllm_url}{path}", **kwargs)


# --- API routes ---


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "index.html").read_text()


@app.get("/api/adapters")
async def list_adapters():
    return {
        name: {
            "description": desc,
            "loaded": name in loaded_adapters,
        }
        for name, desc in ADAPTERS.items()
    }


@app.post("/api/adapters/{name}/load")
async def load_adapter(name: str):
    if name not in ADAPTERS:
        return JSONResponse({"error": f"Unknown adapter: {name}"}, status_code=404)
    if name in loaded_adapters:
        return {"status": "already_loaded"}

    path = adapter_paths.get(name)
    if not path:
        return JSONResponse({"error": "Adapter not downloaded"}, status_code=500)

    resp = await vllm_request(
        "POST",
        "/v1/load_lora_adapter",
        json={"lora_name": name, "lora_path": path},
    )
    if resp.status_code == 200:
        loaded_adapters.add(name)
        return {"status": "loaded"}
    return JSONResponse({"error": resp.text}, status_code=resp.status_code)


@app.post("/api/adapters/{name}/unload")
async def unload_adapter(name: str):
    if name not in loaded_adapters:
        return {"status": "not_loaded"}

    resp = await vllm_request(
        "POST",
        "/v1/unload_lora_adapter",
        json={"lora_name": name},
    )
    if resp.status_code == 200:
        loaded_adapters.discard(name)
        return {"status": "unloaded"}
    return JSONResponse({"error": resp.text}, status_code=resp.status_code)


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    adapter = body.get("adapter")  # None means base model
    stream = body.get("stream", True)

    model = adapter if adapter and adapter in loaded_adapters else BASE_MODEL

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": body.get("max_tokens", 1024),
        "temperature": body.get("temperature", 0.7),
        "stream": stream,
    }

    if stream:

        async def event_stream():
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST", f"{vllm_url}/v1/chat/completions", json=payload
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line:
                            yield line + "\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        resp = await vllm_request("POST", "/v1/chat/completions", json=payload)
        return JSONResponse(resp.json())


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Borealis LoRA Demo")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--adapter-cache", default=None, help="HF cache dir")
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip adapter download"
    )
    args = parser.parse_args()

    vllm_url = args.vllm_url

    if not args.skip_download:
        logger.info("Downloading LoRA adapters...")
        adapter_paths = download_adapters(args.adapter_cache)
        logger.info("All adapters downloaded.")
    else:
        logger.warning("Skipping download - adapters must be loaded manually")

    uvicorn.run(app, host=args.host, port=args.port)
