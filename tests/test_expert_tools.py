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
    adapter_path: str,
    test_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    expert_model = PeftModel.from_pretrained(base_model, adapter_path)
    expert_model.eval()

    results: Dict[str, Any] = {"passed": 0, "failed": 0, "details": []}

    for case in test_cases:
        prompt = case["input"]
        inputs = tokenizer(prompt, return_tensors="pt").to(expert_model.device)

        with torch.no_grad():
            outputs = expert_model.generate(
                **inputs,
                max_new_tokens=256,
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
            }
        )

    return results


if __name__ == "__main__":
    base_path = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
    expert_dir = Path(__file__).resolve().parent.parent
    adapter_dir = expert_dir / "weights" / "qwen3-06b"
    checkpoint = next(adapter_dir.glob("*"), None)
    if checkpoint is None:
        raise SystemExit("Adapter checkpoint not found under weights/qwen3-06b/")

    cases = _load_test_cases(expert_dir / "tests" / "test_cases.json")
    summary = test_expert_tools(str(base_path), str(checkpoint), cases)
    print(json.dumps(summary, indent=2))
    if summary["failed"] > 0:
        raise SystemExit(1)

