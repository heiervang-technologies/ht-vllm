# Borealis 4B LoRA Adapter Demo

Interactive frontend for toggling [NbAiLab/borealis-4b-alignment-loras](https://huggingface.co/NbAiLab/borealis-4b-alignment-loras) on/off against the [borealis-4b-instruct-preview](https://huggingface.co/NbAiLab/borealis-4b-instruct-preview) base model, served via vLLM.

## Setup

```bash
# Requires HF_TOKEN for the private LoRA repo
export HF_TOKEN=<your-token>

# Terminal 1: Start vLLM backend with LoRA support
VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 vllm serve NbAiLab/borealis-4b-instruct-preview \
    --enable-lora \
    --max-loras 7 \
    --max-lora-rank 64 \
    --max-cpu-loras 7 \
    --dtype bfloat16 \
    --port 8000

# Terminal 2: Start the demo server (downloads adapters on first run)
pip install fastapi uvicorn httpx huggingface_hub
python server.py --port 8080
```

Then open http://localhost:8080

## Features

- **Load/Unload**: Toggle any adapter on or off at runtime
- **Live Chat**: Chat with the base model or any loaded adapter
- **Compare Mode**: Side-by-side responses from base model vs adapter
- **Streaming**: Real-time token streaming

## Available Adapters

| Adapter | Description |
|---------|-------------|
| Consolidated | All alignment dimensions merged |
| Ethics | Ethical reasoning and safety |
| Factuality | Factual accuracy and grounding |
| Fictional Fidelity | Creative writing consistency |
| Language | Norwegian language proficiency |
| Personality | Consistent persona and tone |
| Prompt Requirements | Instruction following |
