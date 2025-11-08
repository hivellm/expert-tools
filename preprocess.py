#!/usr/bin/env python3
"""
Tool-calling dataset preprocessing for expert-tools.

Loads multiple Hugging Face corpora (Context7 / MCP / tool-call traces),
normalizes them into ChatML formatted SFT records, and exports a unified
training set that targets JSON tool-call plans.

Highlights:
- Dataset registry with dataset-specific converters
- Canonical ChatML format with expert-tools system prompt
- JSON validation / normalization of tool call payloads
- Deduplication + optional sampling to target ~10k examples
- Basic quality gates (min/max length, JSON sanity)
- Optional metadata export for auditing

Usage:
    # Default (sample to 10k) and save to datasets/train.jsonl
    python preprocess.py --output datasets/train.jsonl --target-count 10000

    # Use all available examples and dump stats only
    python preprocess.py --output datasets/train.jsonl --no-sample

    # Provide external dataset registry (YAML / JSON)
    python preprocess.py --config configs/datasets.yaml --output datasets/train.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """Role: expert-tools planner
Capabilities:
- select appropriate MCP/tool endpoints (Context7, Vectorizer, HTTP, automation)
- emit structured JSON control plans with `tool_calls`, `next_step`, `fallback`
- validate arguments and decline when no safe tool applies

Response Format (JSON):
{
  "tool_calls": [
    {"name": "<tool_name>", "arguments": {...}}
  ],
  "next_step": "<follow-up hint or ''>",
  "fallback": "<fallback reasoning or ''>"
}

Notes:
- Prefer concise plans (max 3 tool_calls).
- Populate `fallback` when declining or after final tool call.
- Never fabricate tool names that are not provided in context.
""".strip()

# Minimum / maximum ChatML length to keep example
MIN_TEXT_LENGTH = 64
MAX_TEXT_LENGTH = 8192
VALIDATION_SPLIT = 0.15
ERROR_INJECTION_RATIO = 0.30
ERROR_TYPES = ["timeout", "auth_error", "schema_mismatch", "rate_limit"]


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class NormalizedExample:
    """Container for a normalized training example."""

    text: str
    source: str
    meta: Dict[str, Any]


@dataclass
class DatasetSpec:
    """Configuration entry describing how to load + convert a dataset."""

    dataset_id: str
    split: str
    converter: Callable[[Dict[str, Any]], Optional[NormalizedExample]]
    weight: float = 1.0
    use_generator: bool = False
    description: str = ""


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def load_registry_from_file(path: Path) -> List[DatasetSpec]:
    """Load dataset registry from YAML/JSON file."""
    import yaml  # Lazy import to avoid hard dependency when not used

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    registry: List[DatasetSpec] = []
    for entry in raw:
        dataset_id = entry["dataset_id"]
        split = entry.get("split", "train")
        converter_name = entry["converter"]
        converter = CONVERTER_REGISTRY.get(converter_name)
        if converter is None:
            raise ValueError(f"Unknown converter: {converter_name}")
        registry.append(
            DatasetSpec(
                dataset_id=dataset_id,
                split=split,
                converter=converter,
                weight=float(entry.get("weight", 1.0)),
                use_generator=bool(entry.get("use_generator", False)),
                description=entry.get("description", ""),
            )
        )
    return registry


def build_chatml(system: str, user: str, assistant: str) -> str:
    """Compose ChatML string."""
    return (
        f"<|system|>\n{system.strip()}\n<|end|>\n"
        f"<|user|>\n{user.strip()}\n<|end|>\n"
        f"<|assistant|>\n{assistant.strip()}\n<|end|>"
    )


def safe_literal_eval(payload: str) -> Optional[Any]:
    """Parse Python-like literal safely."""
    try:
        return ast.literal_eval(payload)
    except Exception:
        return None


def loads_json_flex(payload: str) -> Optional[Any]:
    """Parse JSON or python dict string (fallback to literal)."""
    payload = payload.strip()
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return safe_literal_eval(payload)


def normalize_tool_call(name: str, arguments: Any) -> Optional[Dict[str, Any]]:
    """Normalize tool call entry to dict with JSON-safe args."""
    if not name:
        return None
    if isinstance(arguments, str):
        parsed = loads_json_flex(arguments)
    else:
        parsed = arguments
    if parsed is None:
        parsed = {}

    return {"name": name, "arguments": parsed}


CHATML_PATTERN = re.compile(
    r"<\|system\|>\s*\n(.*?)\n<\|end\|>\s*\n<\|user\|>\s*\n(.*?)\n<\|end\|>\s*\n<\|assistant\|>\s*\n(.*?)\n<\|end\|>\s*$",
    re.DOTALL,
)


def parse_chatml_sections(text: str) -> Optional[Tuple[str, str, str]]:
    """Extract system, user, assistant sections from ChatML."""
    match = CHATML_PATTERN.match(text.strip())
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3)


def _sanitize(obj: Any) -> Any:
    """Recursively sanitize objects so they become JSON serializable."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if obj is Ellipsis:
        return "..."
    if isinstance(obj, dict):
        return {str(key): _sanitize(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize(item) for item in obj]
    # Fallback: string representation
    return str(obj)


def parse_tool_call_blocks(text: str) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    Extract <tool_call> blocks from assistant text.

    Returns (prefix_text, tool_calls, suffix_text)
    """
    pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    tool_calls: List[Dict[str, Any]] = []
    last_end = 0
    for match in pattern.finditer(text):
        raw_payload = match.group(1)
        payload = loads_json_flex(raw_payload)
        if isinstance(payload, dict):
            name = payload.get("name")
            args = payload.get("arguments")
            normalized = normalize_tool_call(name, args)
            if normalized:
                tool_calls.append(normalized)
        last_end = match.end()

    prefix = text[: pattern.search(text).start()] if tool_calls else text
    suffix = text[last_end:].strip() if tool_calls else ""
    prefix = prefix.strip()
    return prefix, tool_calls, suffix


def make_plan_json(
    tool_calls: List[Dict[str, Any]],
    next_step: Optional[str] = None,
    fallback: Optional[str] = None,
) -> str:
    sanitized_calls = []
    for call in tool_calls:
        sanitized_calls.append(
            {
                "name": call.get("name", ""),
                "arguments": _sanitize(call.get("arguments", {})),
            }
        )
    payload = {
        "tool_calls": sanitized_calls,
        "next_step": str(next_step or "").strip(),
        "fallback": str(fallback or "").strip(),
    }
    return json.dumps(payload, ensure_ascii=False)


def validate_example(example: NormalizedExample) -> bool:
    """Quality gate for final ChatML text."""
    text_len = len(example.text)
    if text_len < MIN_TEXT_LENGTH or text_len > MAX_TEXT_LENGTH:
        return False
    # Ensure assistant block contains JSON braces (heuristic)
    if '"tool_calls"' not in example.text:
        return False
    return True


def inject_error_examples(
    base_examples: List[NormalizedExample], target_count: int
) -> List[NormalizedExample]:
    """Generate synthetic fallback/error trajectories."""
    generated: List[NormalizedExample] = []
    attempts = 0
    max_attempts = max(target_count * 4, 1)

    while len(generated) < target_count and attempts < max_attempts:
        attempts += 1
        base = random.choice(base_examples)
        parsed = parse_chatml_sections(base.text)
        if not parsed:
            continue
        system, user, _assistant = parsed
        error_type = random.choice(ERROR_TYPES)
        human_readable = error_type.replace("_", " ")

        assistant_json = make_plan_json(
            tool_calls=[],
            next_step=f"Escalate or retry after resolving {human_readable}.",
            fallback=f"Tool execution aborted due to {human_readable}.",
        )
        text = build_chatml(system, user, assistant_json)
        meta = {
            "dataset": "synthetic_error",
            "error_type": error_type,
            "base_source": base.source,
        }
        generated.append(
            NormalizedExample(
                text=text,
                source=f"synthetic:error_{error_type}",
                meta=meta,
            )
        )

    return generated


# -----------------------------------------------------------------------------
# Dataset-specific converters
# -----------------------------------------------------------------------------

def convert_interstellarninja_multiturn(example: Dict[str, Any]) -> Optional[NormalizedExample]:
    conversations = example.get("conversations") or []
    if not conversations:
        return None

    system_msgs = [m["value"] for m in conversations if m.get("from") == "system"]
    system = DEFAULT_SYSTEM_PROMPT
    if system_msgs:
        system += "\n\n--- Additional instructions ---\n" + system_msgs[0].strip()

    user_msg = next(
        (m["value"].strip() for m in conversations if m.get("from") in {"human", "user"}),
        None,
    )
    if not user_msg:
        return None

    assistant_msgs = [
        m["value"] for m in conversations if m.get("from") in {"assistant", "gpt"}
    ]
    if not assistant_msgs:
        return None

    raw_plan = None
    for msg in assistant_msgs:
        if "<tool_call>" in msg:
            raw_plan = msg
            break
    if raw_plan is None:
        raw_plan = assistant_msgs[0]

    prefix, calls, suffix = parse_tool_call_blocks(raw_plan)
    if not calls:
        return None

    assistant_json = make_plan_json(
        tool_calls=calls,
        next_step=prefix,
        fallback=suffix,
    )

    text = build_chatml(system, user_msg, assistant_json)
    meta = {
        "dataset": "interstellarninja/tool-calls-multiturn",
        "category": example.get("category"),
        "task": example.get("task"),
        "subcategory": example.get("subcategory"),
    }
    return NormalizedExample(text=text, source="interstellarninja/tool-calls-multiturn", meta=meta)


def convert_interstellarninja_dpo(example: Dict[str, Any]) -> Optional[NormalizedExample]:
    system_extra = example.get("system", "")
    user_msg = example.get("question", "").strip()
    chosen = example.get("chosen", "").strip()
    if not user_msg or not chosen:
        return None

    prefix, calls, suffix = parse_tool_call_blocks(chosen)
    if not calls:
        # If chosen is text (no tool), treat as fallback
        calls = []

    assistant_json = make_plan_json(
        tool_calls=calls,
        next_step=prefix,
        fallback=suffix or "No tool selected; respond in natural language.",
    )

    system = DEFAULT_SYSTEM_PROMPT
    if system_extra:
        system += "\n\n--- Additional instructions ---\n" + system_extra.strip()

    text = build_chatml(system, user_msg, assistant_json)
    meta = {"dataset": "interstellarninja/tool-calls-dpo"}
    return NormalizedExample(text=text, source="interstellarninja/tool-calls-dpo", meta=meta)


def convert_bitagent_toolcalling(example: Dict[str, Any]) -> Optional[NormalizedExample]:
    conversation = example.get("conversation")
    if not conversation:
        return None

    messages = loads_json_flex(conversation)
    if not isinstance(messages, list):
        return None

    system = DEFAULT_SYSTEM_PROMPT
    tools_listing = example.get("tools")
    if isinstance(tools_listing, str) and tools_listing.strip():
        system += "\n\n--- Available tools ---\n" + tools_listing.strip()

    user_msg = ""
    tool_calls: List[Dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "user" and not user_msg:
            user_msg = (content or "").strip()
        elif role in {"tool call", "tool_call"} and isinstance(content, dict):
            call = normalize_tool_call(content.get("name"), content.get("arguments"))
            if call:
                tool_calls.append(call)

    if not user_msg or not tool_calls:
        return None

    assistant_json = make_plan_json(
        tool_calls=tool_calls,
        next_step="Summarize tool response for the user.",
        fallback="",
    )

    text = build_chatml(system, user_msg, assistant_json)
    meta = {"dataset": "BitAgent/tool_calling"}
    return NormalizedExample(text=text, source="BitAgent/tool_calling", meta=meta)


def convert_roborovski_synthetic(example: Dict[str, Any]) -> Optional[NormalizedExample]:
    question = example.get("question", "").strip()
    tool_str = example.get("tool")
    if not question or not tool_str:
        return None

    tool_meta = loads_json_flex(tool_str)
    call = normalize_tool_call(
        name=(tool_meta or {}).get("name"),
        arguments=example.get("tool_call"),
    )
    if not call:
        return None

    assistant_json = make_plan_json(
        tool_calls=[call],
        next_step="Execute tool and relay concise result.",
        fallback=example.get("agent_output", ""),
    )

    system = DEFAULT_SYSTEM_PROMPT + "\n\nDataset hint: Synthetic tool-call trace."
    text = build_chatml(system, question, assistant_json)
    meta = {"dataset": "roborovski/synthetic-tool-calls-v2"}
    return NormalizedExample(text=text, source="roborovski/synthetic-tool-calls-v2", meta=meta)


def convert_roborovski_dpo(example: Dict[str, Any]) -> Optional[NormalizedExample]:
    question = example.get("question", "").strip()
    tool = loads_json_flex(example.get("tool", ""))
    call = normalize_tool_call(
        name=(tool or {}).get("name"),
        arguments=example.get("tool_call_accepted"),
    )
    if not question or not call:
        return None

    plan = make_plan_json(
        tool_calls=[call],
        next_step="Return tool result succinctly.",
        fallback="",
    )
    system = DEFAULT_SYSTEM_PROMPT + "\n\nDataset hint: DPO accepted tool trace."
    text = build_chatml(system, question, plan)
    meta = {"dataset": "roborovski/synthetic-tool-calls-v2-dpo-pairs"}
    return NormalizedExample(
        text=text,
        source="roborovski/synthetic-tool-calls-v2-dpo-pairs",
        meta=meta,
    )


def convert_llamafactory_toolcall(example: Dict[str, Any]) -> Optional[NormalizedExample]:
    conversations = example.get("conversations") or []
    if len(conversations) < 3:
        return None

    user = next((m["value"] for m in conversations if m["from"] == "human"), "").strip()
    func_call = next((m["value"] for m in conversations if m["from"] == "function_call"), "")
    if not user or not func_call:
        return None

    payload = loads_json_flex(func_call)
    if not isinstance(payload, dict) or "name" not in payload:
        return None

    call = normalize_tool_call(payload.get("name"), payload.get("arguments"))
    if not call:
        return None

    assistant_json = make_plan_json(
        tool_calls=[call],
        next_step="Present summarized tool result.",
        fallback="",
    )
    system = DEFAULT_SYSTEM_PROMPT + "\n\nDataset hint: LLaMA Factory tool-call dialogue."
    text = build_chatml(system, user, assistant_json)
    meta = {"dataset": "llamafactory/glaive_toolcall"}
    return NormalizedExample(text=text, source="llamafactory/glaive_toolcall", meta=meta)


def convert_tejeshbhalla_toolcall(example: Dict[str, Any]) -> Optional[NormalizedExample]:
    conversations = example.get("conversations") or []
    if len(conversations) < 3:
        return None

    user = conversations[0].get("value", "").strip()
    func_call = next((m["value"] for m in conversations if m["from"] == "function_call"), "")
    if not user or not func_call:
        return None

    payload = loads_json_flex(func_call)
    if not isinstance(payload, dict):
        return None

    call = normalize_tool_call(payload.get("name"), payload.get("arguments"))
    if not call:
        return None

    assistant_json = make_plan_json(
        tool_calls=[call],
        next_step="Summarize results and answer the user.",
        fallback="",
    )
    system = DEFAULT_SYSTEM_PROMPT + "\n\nDataset hint: Tool call with numeric reasoning."
    text = build_chatml(system, user, assistant_json)
    meta = {"dataset": "tejeshbhalla/tool_calling"}
    return NormalizedExample(text=text, source="tejeshbhalla/tool_calling", meta=meta)


def convert_smoltalk_toolcalling(example: Dict[str, Any]) -> Optional[NormalizedExample]:
    messages = example.get("messages") or []
    if len(messages) < 2:
        return None

    user = next((m["content"] for m in messages if m.get("role") == "user"), "").strip()
    assistant = next(
        (m["content"] for m in messages if m.get("role") == "assistant"), ""
    ).strip()
    if not user or not assistant:
        return None

    prefix, calls, suffix = parse_tool_call_blocks(assistant)
    if not calls:
        return None

    assistant_json = make_plan_json(
        tool_calls=calls,
        next_step=prefix,
        fallback=suffix,
    )
    system = DEFAULT_SYSTEM_PROMPT + "\n\nDataset hint: SmolTalk tool-calling trace (FR/EN)."
    text = build_chatml(system, user, assistant_json)
    meta = {"dataset": "CATIE-AQ/smoltalk2_smolagents_toolcalling_french"}
    return NormalizedExample(
        text=text,
        source="CATIE-AQ/smoltalk2_smolagents_toolcalling_french",
        meta=meta,
    )


def convert_gsm8k_toolcalls(example: Dict[str, Any]) -> Optional[NormalizedExample]:
    question = example.get("question", "").strip()
    toolcalls = example.get("toolcalls") or []
    if not question or not toolcalls:
        return None

    calls = []
    for call_str in toolcalls:
        if isinstance(call_str, list):
            call_str = call_str[0]
        match = re.match(r"<T>([^(]+)\((.*)\)=.*", call_str)
        if not match:
            continue
        name = match.group(1)
        args = match.group(2)
        # Convert arguments to JSON by wrapping into list and splitting
        arg_payload = {}
        if args:
            parts = [p.strip() for p in args.split(",")]
            for idx, part in enumerate(parts):
                key = f"arg{idx+1}"
                value = loads_json_flex(part) if re.match(r"^-?\d+(\.\d+)?$", part) else part
                arg_payload[key] = value
        normalized = normalize_tool_call(name, arg_payload)
        if normalized:
            calls.append(normalized)

    if not calls:
        return None

    assistant_json = make_plan_json(
        tool_calls=calls,
        next_step="Use final tool result to answer the math question.",
        fallback="",
    )
    system = DEFAULT_SYSTEM_PROMPT + "\n\nDataset hint: GSM8K tool-based reasoning."
    text = build_chatml(system, question, assistant_json)
    meta = {"dataset": "jvhoffbauer/gsm8k-toolcalls"}
    return NormalizedExample(text=text, source="jvhoffbauer/gsm8k-toolcalls", meta=meta)


def convert_layue_mcp(example: Dict[str, Any]) -> Optional[NormalizedExample]:
    instruction = example.get("instruction", "").strip()
    output = example.get("output", "").strip()
    if not instruction or not output:
        return None

    call = normalize_tool_call(
        name="tools.list",
        arguments={"request": "list available MCP tools"},
    )
    assistant_json = make_plan_json(
        tool_calls=[call],
        next_step="Present tool catalog to the user.",
        fallback=output,
    )
    system = DEFAULT_SYSTEM_PROMPT + "\n\nDataset hint: MCP tool listing."
    text = build_chatml(system, instruction, assistant_json)
    meta = {"dataset": "Layue13/mcp-tool-calling-dataset"}
    return NormalizedExample(
        text=text,
        source="Layue13/mcp-tool-calling-dataset",
        meta=meta,
    )


CONVERTER_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Optional[NormalizedExample]]] = {
    "interstellarninja_multiturn": convert_interstellarninja_multiturn,
    "interstellarninja_dpo": convert_interstellarninja_dpo,
    "roborovski_synthetic": convert_roborovski_synthetic,
    "roborovski_dpo": convert_roborovski_dpo,
    "bitagent_toolcalling": convert_bitagent_toolcalling,
    "llamafactory_toolcall": convert_llamafactory_toolcall,
    "tejeshbhalla_toolcall": convert_tejeshbhalla_toolcall,
    "smoltalk_toolcalling": convert_smoltalk_toolcalling,
    "gsm8k_toolcalls": convert_gsm8k_toolcalls,
    "layue_mcp": convert_layue_mcp,
}


DEFAULT_DATASET_REGISTRY: List[DatasetSpec] = [
    DatasetSpec(
        dataset_id="interstellarninja/tool-calls-multiturn",
        split="train",
        converter=convert_interstellarninja_multiturn,
        weight=2.0,
        description="Multi-turn tool planning traces.",
    ),
    DatasetSpec(
        dataset_id="interstellarninja/tool-calls-sharegpt",
        split="train",
        converter=convert_interstellarninja_multiturn,
        weight=1.0,
        description="ShareGPT-sourced planning traces.",
    ),
    DatasetSpec(
        dataset_id="interstellarninja/tool-calls-dpo",
        split="train",
        converter=convert_interstellarninja_dpo,
        weight=0.8,
        description="Preference pairs (chosen outputs).",
    ),
    DatasetSpec(
        dataset_id="roborovski/synthetic-tool-calls-v2",
        split="train",
        converter=convert_roborovski_synthetic,
        weight=0.8,
        description="Synthetic single-turn tool invocations.",
    ),
    DatasetSpec(
        dataset_id="roborovski/synthetic-tool-calls-v2-dpo-pairs",
        split="train",
        converter=convert_roborovski_dpo,
        weight=1.0,
        description="Synthetic DPO pairs (accepted).",
    ),
    DatasetSpec(
        dataset_id="BitAgent/tool_calling",
        split="train",
        converter=convert_bitagent_toolcalling,
        weight=0.1,
        description="Large-scale tool-call corpus (English).",
    ),
    DatasetSpec(
        dataset_id="llamafactory/glaive_toolcall_en",
        split="train",
        converter=convert_llamafactory_toolcall,
        weight=0.7,
        description="English tool-call traces.",
    ),
    DatasetSpec(
        dataset_id="llamafactory/glaive_toolcall_zh",
        split="train",
        converter=convert_llamafactory_toolcall,
        weight=0.4,
        description="Chinese tool-call traces.",
    ),
    DatasetSpec(
        dataset_id="tejeshbhalla/tool_calling",
        split="train",
        converter=convert_tejeshbhalla_toolcall,
        weight=0.6,
        description="E-commerce / numeric tool calls.",
    ),
    DatasetSpec(
        dataset_id="jvhoffbauer/gsm8k-toolcalls",
        split="train",
        converter=convert_gsm8k_toolcalls,
        weight=1.2,
        description="Math GSM8K with explicit tool steps.",
    ),
    DatasetSpec(
        dataset_id="Layue13/mcp-tool-calling-dataset",
        split="train",
        converter=convert_layue_mcp,
        weight=0.1,
        description="MCP metadata prompts.",
    ),
]


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def preprocess(
    registry: List[DatasetSpec],
    target_count: Optional[int],
    sample: bool,
) -> Tuple[List[NormalizedExample], Dict[str, Any]]:
    collected: List[NormalizedExample] = []
    stats: Dict[str, Any] = {"per_dataset": defaultdict(int), "attempted": defaultdict(int)}

    for spec in registry:
        print(f"\n=== Loading {spec.dataset_id} [{spec.split}] ===")
        try:
            dataset = load_dataset(spec.dataset_id, split=spec.split)
        except Exception as exc:
            print(f"[WARN] Failed to load {spec.dataset_id}: {exc}")
            continue

        # Optional weighting via sampling during conversion
        indices = list(range(len(dataset)))
        if spec.weight < 1.0 and sample:
            keep_prob = max(min(spec.weight, 1.0), 0.1)
            indices = [idx for idx in indices if random.random() <= keep_prob]

        for idx in tqdm(indices, desc=spec.dataset_id):
            raw = dataset[idx]
            stats["attempted"][spec.dataset_id] += 1
            example = spec.converter(raw)
            if not example:
                continue
            if validate_example(example):
                collected.append(example)
                stats["per_dataset"][spec.dataset_id] += 1

    # Deduplicate by ChatML text hash
    print("\n=== Deduplicating ===")
    deduped: List[NormalizedExample] = []
    seen = set()
    for example in collected:
        key = example.text
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)

    print(f"Collected: {len(collected)} -> Deduplicated: {len(deduped)} examples")

    base_examples = deduped
    base_target = None
    if target_count and sample:
        if ERROR_INJECTION_RATIO > 0:
            base_target = int(target_count / (1.0 + ERROR_INJECTION_RATIO))
        else:
            base_target = target_count
        base_target = max(base_target or target_count, 1)
        if len(base_examples) > base_target:
            print(f"Sampling base set to {base_target} examples (target {target_count})...")
            base_examples = random.sample(base_examples, base_target)

    error_examples: List[NormalizedExample] = []
    error_count = int(len(base_examples) * ERROR_INJECTION_RATIO)
    if error_count > 0:
        print(f"Injecting {error_count} synthetic error examples...")
        error_examples = inject_error_examples(base_examples, error_count)

    combined = base_examples + error_examples
    random.shuffle(combined)

    stats["base_count"] = len(base_examples)
    stats["synthetic_error_count"] = len(error_examples)
    stats["final_count"] = len(combined)
    return combined, stats


def save_dataset(
    examples: List[NormalizedExample],
    output_path: Path,
    metadata_path: Path,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps({"text": example.text}, ensure_ascii=False) + "\n")

    source_breakdown = Counter(example.source for example in examples)
    metadata = {
        "count": len(examples),
        "min_length": MIN_TEXT_LENGTH,
        "max_length": MAX_TEXT_LENGTH,
        "system_prompt_snippet": DEFAULT_SYSTEM_PROMPT[:160],
        "sources": source_breakdown,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def print_stats(stats: Dict[str, Any]) -> None:
    print("\n=== Summary ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False, default=lambda x: list(x) if isinstance(x, defaultdict) else x))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess tool/MCP datasets for expert-tools.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/train.jsonl"),
        help="Destination JSONL file (default: datasets/train.jsonl)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("datasets/train.metadata.json"),
        help="Metadata JSON output path (default: datasets/train.metadata.json)",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=10000,
        help="Target number of examples after dedupe (default: 10k). Use --no-sample to keep all.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional YAML/JSON dataset registry.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling (keep all normalized examples).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.config:
        registry = load_registry_from_file(args.config)
    else:
        registry = DEFAULT_DATASET_REGISTRY

    print(f"Using {len(registry)} dataset specs.")
    for spec in registry:
        print(f"- {spec.dataset_id} ({spec.description or 'no description'})")

    target = None if args.no_sample else args.target_count
    examples, stats = preprocess(registry, target_count=target, sample=not args.no_sample)

    total_count = len(examples)
    validation_size = max(int(total_count * VALIDATION_SPLIT), 1)
    validation_examples = examples[:validation_size]
    train_examples = examples[validation_size:]

    train_output = args.output
    validation_output = args.output.with_name("validation.jsonl")
    train_metadata = args.metadata
    validation_metadata = args.metadata.with_name("validation.metadata.json")

    save_dataset(
        train_examples,
        train_output,
        train_metadata,
        extra_metadata={"split": "train", "validation_split": VALIDATION_SPLIT},
    )
    save_dataset(
        validation_examples,
        validation_output,
        validation_metadata,
        extra_metadata={"split": "validation", "validation_split": VALIDATION_SPLIT},
    )

    counts_path = train_output.parent / "source_counts.json"
    source_counts = Counter(example.source for example in train_examples)
    counts_path.write_text(json.dumps(source_counts, indent=2, ensure_ascii=False), encoding="utf-8")

    stats["train_count"] = len(train_examples)
    stats["validation_count"] = len(validation_examples)
    print_stats(stats)

    print(f"\nSaved {len(train_examples)} train examples to {train_output}")
    print(f"Train metadata written to {train_metadata}")
    print(f"Saved {len(validation_examples)} validation examples to {validation_output}")
    print(f"Validation metadata written to {validation_metadata}")
    print(f"Source breakdown exported to {counts_path}")


if __name__ == "__main__":
    main()

