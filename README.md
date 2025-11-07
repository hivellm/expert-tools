# Expert Tools

Reasoning-focused expert that plans and emits structured MCP/tool invocations. Target use case: orchestrating Context7, Vectorizer, HTTP, and internal automation endpoints with JSON tool-call outputs.

**Version:** 0.1.0 (scaffold) | **Checkpoint:** TBD | **Dataset:** Tool/MCP planning dialogues (in progress)

## Quick Start

```bash
# Navigate to expert directory
cd F:/Node/hivellm/expert/experts/expert-tools

# Populate dataset with curated tool-calling conversations (placeholder provided)
code datasets/train.jsonl

# Train (after configuring manifest + adapters)
../../cli/target/release/expert-cli train
```

## Capabilities (Planned)

- ✅ Structured tool-call JSON (name + arguments)
- ✅ MCP resource selection with fallback plan
- ✅ Multi-step planning notes (`next_step`, `fallback`)
- ⚠️ Error recovery narratives when tools unavailable
- ⚠️ Tool-safety verification (argument validation)

## Roadmap

1. Expand dataset to ~5k curated tool/MCP dialogues (Context7, Vectorizer, HTTP, custom ops)
2. Run Qwen-Max hyperparameter sweep to confirm adapter rank/alpha
3. Implement evaluation harness in `tests/test_expert_tools.py`
4. Package first `.expert` checkpoint with signed hash

## Repository Layout

```
expert-tools/
├── manifest.json             # Training + runtime metadata
├── README.md                 # Documentation (this file)
├── datasets/
│   └── train.jsonl           # Tool-calling training examples (placeholder)
├── tests/
│   ├── test_cases.json       # Expected tool routing test cases
│   └── test_expert_tools.py  # Validation harness (loads base + adapter)
└── weights/                  # Adapter checkpoints (empty placeholder)
```

## Next Actions

- Finalize tool catalog + schema definitions
- Generate balanced dataset covering success, fallback, and error states
- Update manifest with confirmed checkpoint path and SHA256
- Record benchmarking results in CHANGELOG once training completes

