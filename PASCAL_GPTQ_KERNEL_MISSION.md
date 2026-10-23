# Mission: correct + fast INT4 GPTQ on Pascal (sm_61) for the HaiOS gem fleet

You are a CUDA kernel engineer. Your job is to make vLLM's **4-bit `gptq_gemm`
kernel correct AND faster than fp16** on the **Quadro P5200 (Pascal, sm_61)**,
in this fork (`heiervang-technologies/ht-vllm`), and open a PR with benchmarks.

This is integral to HaiOS: every Pascal "gem" runs Qwen3-ASR for the `ears`
speech-to-text daemon. Decode is **memory-bandwidth bound** (reading fp16 weights
every token), so INT4 weights are the win — *if* the kernel is correct and fast.

## What we already proved (don't re-derive)

- On sm_61, vLLM's supported quant kernels are only `gptq` (classic) and `gguf`.
  `gptq_marlin`, `awq`, `awq_marlin`, `fp8`, `bitsandbytes`, `compressed-tensors`
  all require sm_75/sm_80+ (checked via `get_min_capability()`).
- The classic 4-bit path is broken: vLLM prints **"the 4-bit gptq_gemm kernel for
  GPTQ is buggy. Please switch to gptq_marlin"** (source:
  `vllm/model_executor/layers/quantization/gptq.py:101`).
- Empirically on the P5200 with `Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4`:
  - **Correctness is wrong**: coherent text but "2 plus 2 equals 10".
  - **Speed 16.7 tok/s — SLOWER than fp16** (~38 tok/s for the larger 1.7B). The
    classic dequant runs on CUDA cores; no fast Pascal path.
  - VRAM win is real: weights loaded in **1.1 GB vs 3.4 GB** fp16.

## Objective (definition of done)

1. **Correct**: the 4-bit `gptq_gemm` path produces correct outputs on sm_61
   ("2 plus 2 equals 4", matches fp16 within quantization noise).
2. **Fast**: INT4 decode throughput on the P5200 is **meaningfully faster than the
   same model in fp16** (target: ≥1.5× fp16 tok/s; stretch: approach llama.cpp).
3. A **PR to `ht-vllm`** (base `ht`) with the kernel change, the guard in
   `gptq.py` relaxed/removed for the now-correct sm_61 path, and a benchmark table
   (fp16 vs int4: tok/s + a correctness check) in the description.

## Where to look

- `csrc/quantization/gptq/q_gemm.cu` — the GEMM + dequant entry points.
- `csrc/quantization/gptq/qdq_4.cuh` — 4-bit dequant (int4 → half). `qdq_2/3/8`
  are the other bit widths. `compat.cuh` holds arch-compat shims.
- The gate/warning: `vllm/model_executor/layers/quantization/gptq.py:~101`.

## The Pascal angle: DP4A

Pascal (sm_61) has **`__dp4a`** (8-bit 4-way integer dot-product) — the *same*
instruction llama.cpp forces via `GGML_CUDA_FORCE_MMQ` for its fast Pascal INT
matmul. sm_61 has **no** INT8 tensor cores and only 1/64-rate native fp16 ALU, so:
- The current "dequant int4→fp16, fp16 GEMM" path is doubly bad on Pascal (slow
  fp16 math + CUDA-core dequant).
- The promising direction: keep activations/accumulation in a form that lets the
  inner product run on **`__dp4a`** (int8×int8→int32), i.e. an INT8-activation
  MMQ-style path, or at minimum fix correctness then move the hot dequant off the
  slow fp16 ALU. Study llama.cpp's `mmq`/`mul_mat_q` DP4A path for the pattern
  (reference build recipe: `~/ht/hai-os` isn't it — see
  `~/ht/cloud/gems/docker/llama-pascal/Dockerfile.pascal` notes on `__dp4a`).
- First **fix correctness** (likely a dequant/packing or atomic/half-op bug that
  only manifests < sm_70), THEN optimize with DP4A.

## Build + test loop (the hard part — set this up first)

You must be able to rebuild ONLY the changed CUDA object fast (a full vLLM wheel
is ~40 min; you need incremental). Recommended:

- Use a CUDA **devel** container as the dev env (matches the shipping build):
  base `nvidia/cuda:12.6.3-devel-ubuntu22.04`, `uv venv --python 3.12`,
  `uv pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.11.0`,
  then the build deps from `~/ht/hai-os/containers/vllm-pascal/Dockerfile`
  (cmake, ninja, ccache, setuptools, etc.). Mount this repo at `/src/vllm` and
  `~/Models/huggingface` for weights.
- Build with `TORCH_CUDA_ARCH_LIST=6.1` and **ccache** so edits to `q_gemm.cu`
  recompile in seconds, not minutes. Incremental: `cmake --build build --target _C`
  (or `python setup.py build_ext --inplace`) — do NOT rebuild the whole wheel each
  iteration.
- Correctness+speed harness: serve the built vLLM with
  `Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4` (already at `~/Models/huggingface`),
  `--quantization gptq --dtype float16 --enforce-eager` (skip cudagraphs while
  iterating), hit `/v1/completions` with `"The capital of France is"` (expect
  Paris) and `"2 plus 2 equals"` (expect 4), and a 128-token `ignore_eos` run for
  tok/s. Compare against `Qwen/Qwen2.5-1.5B-Instruct` (fp16).

## Guardrails (do not break the live system)

- A production ASR server runs as docker container **`vllm-pascal`** on **port
  8000** — DO NOT stop, rebuild over, or starve it. Use other ports (8010+).
- The P5200 is shared with the desktop + that server (~8 GB used). Keep your test
  instances small (`--gpu-memory-utilization 0.25`, the 1.5B model) and clean them
  up (`docker rm -f`) between runs. Check `nvidia-smi` before launching.
- Work on a branch off `origin/ht` (e.g. `feat/pascal-gptq-dp4a`). Commit with
  `Co-authored-by: Agent <agent@example.com>`, NO Claude/Generated trailer in the
  commit. The PR body may end with the Claude Code trailer.
- If you conclude the kernel win isn't achievable in reasonable effort, say so
  explicitly with evidence (what you tried, the numbers) — a well-evidenced NO-GO
  is a valid, valuable result. Don't fake a speedup.

## Context / provenance

- Full plan + de-risk data: hai-os RFC **#136**.
- The container recipe that consumes this fork's build: hai-os
  `containers/vllm-pascal/` (PR #135, merged). sm_61 arch enablement: ht-vllm PR
  #6 (merged into `ht`).
- Report progress to the orchestrating agent via `director send <your-caller>`;
  the human owner is Markus. Ping when you have a correct kernel, and again when
  you have a speedup (or a NO-GO with numbers).
