# Expert Tools

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/hivellm/expert-tools/releases/tag/v0.1.0)
[![License](https://img.shields.io/badge/license-CC--BY--4.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-experimental-orange.svg)](README.md#quick-start)

[![Base Model](https://img.shields.io/badge/base%20model-Qwen3--0.6B-orange.svg)](README.md#features)
[![Adapter](https://img.shields.io/badge/adapter-DoRA%20r%3D16-blue.svg)](README.md#training--configuration)
[![Dataset](https://img.shields.io/badge/dataset-10k%20examples-brightgreen.svg)](README.md#dataset)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20CUDA-0078d4.svg)](README.md#features)

Tool orchestration expert that plans MCP/tool invocations and emits structured JSON control plans (`{"tool_call": {"name": ..., "arguments": ...}}`). Optimized for triaging Context7, Vectorizer, HTTP, and internal automation endpoints with explicit fallback reasoning.

**Version:** 0.1.0 (scaffold) | **Checkpoint:** TBD | **Dataset:** 8,500 train + 1,499 validation (English-only, 30 % synthetic failures)

**Status:** ⚠️ Experimental — dataset expansion e pipeline ainda em evolução  
**Dependência:** requer `expert-json` para serialização/validação JSON final (`manifest.constraints.requires`)

## Quick Start

```bash
# 1. Activate project environment
cd F:/Node/hivellm/expert
source .venv/bin/activate  # Windows: .venv/Scripts/activate

# 2. Enter expert workspace
cd experts/expert-tools

# 3. Construir dataset (gera train + validation + falhas sintéticas)
python preprocess.py --output datasets/train.jsonl --metadata datasets/train.metadata.json

# 3.1 (Opcional) Gerar gráficos de distribuição
python scripts/generate_distribution_charts.py  # arquivos em docs/

# 4. Train DoRA adapter (ensure expert-json is available for JSON validation)
../../cli/target/release/expert-cli train --manifest manifest.json

# 5. Run smoke tests
python tests/test_expert_tools.py

# 6. Package checkpoint (placeholder)
../../cli/target/release/expert-cli package --manifest manifest.json --output expert-tools-qwen3-06b.v0.1.0.expert
```

## Capabilities (Target)

- ✅ Structured tool-call JSON with validated schema (`tool_call`, `next_step`, `fallback`)
- ✅ MCP resource selection across Context7, Vectorizer, HTTP, and internal ops
- ✅ Multi-step planning hints (follow-up questions, recovery plans)
- ⚠️ Tool argument validation and schema enforcement (in progress)
- ⚠️ Failure narration when tools unavailable or unsuitable

## Works Best For ✅

- Choosing the right MCP endpoint when multiple tools are available
- Documentation lookups that require Context7 search + fallback summaries
- Embedding/vectorization instructions routed via Vectorizer MCP
- HTTP fetch plans with explicit failure paths
- Multi-step workflows that require “next action” guidance

## Known Limitations ⚠️

- JSON schema validation not yet enforced post-generation
- Sparse coverage for >2 step tool chains and browser automation
- Evaluation harness depends on locally trained adapter checkpoints
- Dataset still under construction (current placeholder ≪10k examples)

## Dataset Plan (English-first, 10k target)

| Source | Split | Count (snapshot) | Purpose |
|--------|-------|-----------------:|---------|
| BitAgent/tool_calling | train | 3,045 | Single-turn factual tool calls |
| jvhoffbauer/gsm8k-toolcalls | train | 1,861 | Raciocínio matemático + calculadora |
| roborovski/synthetic-tool-calls-v2-dpo-pairs | train | 612 | DPO (planos aceitos) |
| roborovski/synthetic-tool-calls-v2 | train | 147 | Single-turn sintéticos |
| interstellarninja/tool-calls-multiturn + sharegpt | train | 481 | Planejamento multi-turn |
| interstellarninja/tool-calls-dpo | train | 75 | Preferência (plan vs fallback) |
| llamafactory/glaive_toolcall (en/zh) | train | 216 | Intenções gerais |
| tejeshbhalla/tool_calling | train | 110 | Prompts numéricos/comerciais |
| Synthetic error traces (HiveLLM) | train | 1,953 | Fallbacks (timeout/auth/schema/rate-limit) |
| Layue13/mcp-tool-calling-dataset | train | 20 | MCP listagem |
| MCPToolBench/MCPToolBenchPP | eval | 500 | Benchmark prompts & regression set |
| quotientai/mcp-tool-use-eval | eval | 1,000 | Tool selection quality evaluation |
| qualifire/mcp-tool-use-quality-benchmark | eval | 5,000 | Error mining + QA labelling |
| alihmaou/Agents_MCP_Hackathon_Tools_List (+ enriched) | meta | 2,086 | Tool metadata lookup |

> **Snapshot (2025-11-08):** 8.5k treino + 1.5k validação (todo em inglês) com 30 % de falhas sintéticas. Meta: manter ~10k exemplos com ~1/3 multi-turn, ~1/3 alinhamento/falhas e ~1/3 tarefas factuais.

## Preprocessing Workflow (Planned)

```bash
python preprocess.py \
  --datasets-file configs/datasets.yaml \
  --output datasets/train.jsonl \
  --dedupe \
  --min-length 64 \
  --max-length 2048 \
  --schema configs/tool_schema.json \
  --include-eval datasets/eval.jsonl
```

Pipeline steps:
- Load Hugging Face datasets via `datasets` (cached locally)
- Normalize into ChatML with explicit `system` / `user` / `assistant` roles
- Enforce manifest-aligned tool names (`context7.search`, `vectorizer.lookup`, `http.get`, …)
- Inject fallback narratives when traces lack `next_step` hints
- Gerar exemplos de fallback (timeout/auth/schema/rate-limit) via `inject_error_examples`
- Deduplicate by (`user`, `tool_call.name`, `arguments`)
- Produzir `train.jsonl` e `validation.jsonl` mantendo mix de sucessos, fallbacks e negociações

## Training Configuration (manifest.json)

| Parameter | Value |
|-----------|-------|
| Base model | `F:/Node/hivellm/expert/models/Qwen3-0.6B` |
| Adapter | DoRA (`r=16`, `alpha=32`, dropout `0.1`) |
| Method | SFT (Unsloth accelerated) |
| Learning rate / epochs | `4.5e-4` / `2.5` |
| Warmup / Scheduler | 10 % warmup, cosine with restarts (3 ciclos) |
| Batch size / GA | `4` / `6` (effective 24) |
| Gradient checkpointing | `full` |
| Logging / Eval / Save | cada 10 | 100 | 100 steps |
| Rope scaling | NTK-by-parts (factor 8.0) |

## Evaluation Harness

- `tests/test_cases.json`: smoke tests for correct tool routing (Context7 search, Vectorizer lookup, no-tool fallback)
- `tests/test_expert_tools.py`: loads base + adapter, generates responses, extracts `tool_call.name`, asserts equality
- Planned additions: MCPToolBench regression suite, JSON schema validation, fallback quality scoring

## Roadmap

1. ✅ Inventory Hugging Face tool/MCP datasets
2. ⏳ Implement preprocessing pipeline with dataset registry + 10k merger
3. ⏳ Expand evaluation cases (Context7 docs, vectorizer embeddings, browser/tool combos)
4. ⏳ Train DoRA adapter and publish checksum
5. ⏳ Update docs/ with quality metrics and changelog entry
6. ⏳ Package `.expert` artifact with signed hash

## Repository Layout

```
expert-tools/
├── configs/                  # Dataset registry & schemas (planned)
├── datasets/
│   ├── train.jsonl           # Curated tool-calling corpus (10k target)
│   └── eval.jsonl            # Held-out regression prompts (planned)
├── manifest.json             # Training + runtime metadata
├── preprocess.py             # Dataset builder (pending implementation)
├── README.md                 # Documentation (this file)
├── tests/
│   ├── test_cases.json       # Expected tool routing behaviours
│   └── test_expert_tools.py  # Minimal evaluation harness
└── weights/                  # Adapter checkpoints (empty placeholder)
```

## Contribution Notes

- Keep dataset additions aligned with MCP/Tool schema in `manifest.json`
- Update `LICENSE` when introducing new upstream datasets
- Record benchmark deltas in `docs/CHANGELOG.md` prior to tagging releases
- Follow @hivellm/rulebook QA pipeline (tests → lint → coverage → docs update)

