#!/usr/bin/env python3
"""
Validation scaffold for expert-tools.
Loads base model + adapter and checks tool-call structure against expected test cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def _load_test_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("test_cases", [])


def _extract_tool_name(response: str) -> str | None:
    response = response.strip()
    if not response:
        return None
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return None
    tool_call = payload.get("tool_call")
    if isinstance(tool_call, dict):
        name = tool_call.get("name")
        if isinstance(name, str):
            return name
    return None


def test_expert_tools(
    base_model_path: str,
    expert_json_adapter_path: str,
    expert_tools_adapter_path: str,
    test_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        expert_json_adapter_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    expert_model = PeftModel.from_pretrained(base_model, expert_json_adapter_path)
    expert_model.load_adapter(expert_tools_adapter_path, adapter_name="expert-tools")
    expert_model.set_adapter("expert-tools")
    expert_model.eval()

    results: Dict[str, Any] = {"passed": 0, "failed": 0, "details": []}

    default_system = "You are an assistant that routes requests to available MCP tools when appropriate."

    for case in test_cases:
        system_prompt = case.get("system", default_system)
        tool_catalog = case.get("tool_catalog", [])

        if tool_catalog:
            catalog_text = json.dumps(
                tool_catalog,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            system_prompt = f"{system_prompt}\n\nAvailable MCP tools:\n{catalog_text}"

        user_prompt = case.get("user", case.get("input", ""))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer([text], return_tensors="pt").to(expert_model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = expert_model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        expected_tool = case.get("expected_tool")
        predicted_tool = _extract_tool_name(response)
        passed = predicted_tool == expected_tool

        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1

        results["details"].append(
            {
                "name": case["name"],
                "expected_tool": expected_tool,
                "predicted_tool": predicted_tool,
                "response": response,
                "prompt_tokens": prompt_tokens,
            }
        )

    return results


def _resolve_adapter_path(base_dir: Path) -> Path:
    priority = ["final", "adapter"]
    for name in priority:
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    # Fallback: first child directory (for checkpoint folders)
    try:
        return next(p for p in sorted(base_dir.iterdir()) if p.is_dir())
    except StopIteration as exc:
        raise FileNotFoundError(f"No adapter directories found under {base_dir}") from exc


if __name__ == "__main__":
    base_path = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
    expert_root = Path(__file__).resolve().parent.parent
    expert_json_root = expert_root.parent / "expert-json"

    tools_adapter_base = expert_root / "weights" / "qwen3-06b"
    json_adapter_base = expert_json_root / "weights" / "qwen3-06b"

    try:
        json_adapter_path = _resolve_adapter_path(json_adapter_base)
        tools_adapter_path = _resolve_adapter_path(tools_adapter_base)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    cases = _load_test_cases(expert_root / "tests" / "test_cases.json")
    summary = test_expert_tools(
        str(base_path),
        str(json_adapter_path),
        str(tools_adapter_path),
        cases,
    )
    print(json.dumps(summary, indent=2))
    if summary["failed"] > 0:
        raise SystemExit(1)

