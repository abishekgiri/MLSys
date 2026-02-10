#!/usr/bin/env python3
"""Gemini-based agent for MLSys 2026 scheduling (Track B)."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import google.generativeai as genai
    try:
        from google.generativeai.types import GenerationConfig
    except Exception:  # pragma: no cover
        GenerationConfig = None
except Exception as exc:  # pragma: no cover
    genai = None
    GenerationConfig = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from src.official_parser import parse_official_problem
from src.official_scheduler import OfficialFusionScheduler, OfficialScheduler
from src.official_format import write_official_schedule


DEFAULT_PARAMS = {
    "scheduler": "fuse",
    "connectivity": "auto",
    "retain_budget": 0.05,
}


def _load_prompt(path: Path) -> str:
    if path.exists():
        return path.read_text().strip()
    return ""


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from text (robust to fragments + fenced blocks)."""
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("Empty model output")

    # 1) Try direct parse
    try:
        return json.loads(stripped)
    except Exception:
        pass

    # 2) Handle fenced blocks ```json { ... } ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return json.loads(m.group(1))

    # 3) Extract first {...} anywhere
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(stripped[start:end + 1])

    # 4) Fragment case: looks like key/value lines without outer braces
    if (
        '"scheduler"' in stripped
        or '"connectivity"' in stripped
        or '"retain_budget"' in stripped
    ):
        frag = stripped.strip().strip(",")
        return json.loads("{\n" + frag + "\n}")

    raise ValueError("No JSON object found in model output")


def _parse_params_from_text(text: str) -> Dict[str, Any]:
    """Best-effort parse when model does not return JSON."""
    lowered = text.lower()
    params: Dict[str, Any] = {}

    # Prefer explicit key-value matches.
    m = re.search(r"scheduler\\s*[:=]\\s*[\"']?(baseline|fuse)[\"']?", lowered)
    if m:
        params["scheduler"] = m.group(1)
    else:
        if "fuse" in lowered:
            params["scheduler"] = "fuse"
        elif "baseline" in lowered:
            params["scheduler"] = "baseline"

    m = re.search(r"connectivity\\s*[:=]\\s*[\"']?(strict|loose|auto)[\"']?", lowered)
    if m:
        params["connectivity"] = m.group(1)
    else:
        if "auto" in lowered:
            params["connectivity"] = "auto"
        elif "loose" in lowered:
            params["connectivity"] = "loose"
        elif "strict" in lowered:
            params["connectivity"] = "strict"

    # Look for a retain budget mention
    m = re.search(r"retain[_\\s-]*budget[^0-9]*([0-9]*\\.?[0-9]+)", lowered)
    if m:
        try:
            params["retain_budget"] = float(m.group(1))
        except Exception:
            pass
    else:
        # Fallback: find any float in [0, 0.2]
        for cand in re.findall(r"[0-9]*\\.?[0-9]+", lowered):
            try:
                val = float(cand)
            except Exception:
                continue
            if 0.0 <= val <= 0.2:
                params["retain_budget"] = val
                break

    if not params:
        raise ValueError("No JSON object found in model output")
    return params


def _response_text(response: Any) -> str:
    parts = []
    for cand in getattr(response, "candidates", []) or []:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(part_text)
    if parts:
        return "\n".join(parts)
    text = getattr(response, "text", None)
    if text:
        return text
    return ""


def _select_model() -> str:
    env_model = os.environ.get("GEMINI_MODEL")
    if env_model:
        return env_model
    models = list(genai.list_models())
    candidates = [
        m for m in models
        if hasattr(m, "supported_generation_methods")
        and "generateContent" in m.supported_generation_methods
    ]
    if not candidates:
        raise RuntimeError("No Gemini models support generateContent")

    name_map = {m.name: m for m in candidates}
    preferred = [
        "models/gemini-2.5-flash",
        "models/gemini-flash-latest",
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-001",
        "models/gemini-2.5-flash-lite",
        "models/gemini-flash-lite-latest",
        "models/gemini-pro-latest",
    ]
    for name in preferred:
        if name in name_map:
            return name

    # Avoid non-text variants if possible.
    def _bad(n: str) -> bool:
        lowered = n.lower()
        return any(x in lowered for x in ("image", "tts", "audio", "computer", "robot", "deep-research", "nano-banana"))

    for m in candidates:
        if not _bad(m.name):
            return m.name
    return candidates[0].name


def _gen_config():
    cfg = {
        "temperature": 0.2,
        "max_output_tokens": 1024,
    }
    if GenerationConfig is None:
        return cfg
    return GenerationConfig(**cfg)


def _call_gemini(problem_dict: Dict[str, Any]) -> Dict[str, Any]:
    if genai is None:
        raise RuntimeError(f"google-generativeai not available: {_IMPORT_ERROR}")

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    genai.configure(api_key=api_key)

    prompt_dir = Path(__file__).parent / "prompts"
    system_prompt = _load_prompt(prompt_dir / "system.txt")
    user_template = _load_prompt(prompt_dir / "user_template.txt")

    if not user_template:
        user_template = (
            "Return ONLY a JSON object with keys: scheduler, connectivity, retain_budget.\n"
            "scheduler in {baseline,fuse}; connectivity in {strict,loose,auto}; retain_budget in [0.0, 0.2].\n"
            "Problem JSON:\n{problem_json}\n"
        )

    user_prompt = user_template.replace(
        "{problem_json}", json.dumps(problem_dict, indent=2)
    )

    model_name = _select_model()
    model = genai.GenerativeModel(
        model_name,
        system_instruction=system_prompt if system_prompt else None,
    )

    gen_cfg = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "response_mime_type": "application/json",
    }

    response = model.generate_content(user_prompt, generation_config=_gen_config())
    text = _response_text(response)
    if text and text.strip().endswith(("retain_", "\"retain_", "retain")):
        raise ValueError("Gemini output truncated")
    if os.environ.get("GEMINI_DEBUG"):
        try:
            debug_path = os.environ.get("GEMINI_DEBUG_FILE", "/tmp/gemini_raw.txt")
            with open(debug_path, "w", encoding="utf-8", newline="\n") as f:
                f.write((text or "").strip() + "\n")
            preview = (text or "")[:400]
            if preview:
                print(f"[agent][debug] Gemini raw (first 400 chars):\n{preview}", file=sys.stderr)
        except Exception:
            pass

    params = None
    try:
        params = _extract_json(text)
    except Exception:
        try:
            params = _parse_params_from_text(text)
        except Exception:
            params = None

    if params is None:
        retry_prompt = (
            "Return ONLY a single JSON object that starts with '{' and ends with '}'. "
            "Keys: scheduler, connectivity, retain_budget. No prose, no markdown."
        )
        response = model.generate_content(retry_prompt, generation_config=_gen_config())
        text = _response_text(response)
        if os.environ.get("GEMINI_DEBUG"):
            try:
                debug_path = os.environ.get("GEMINI_DEBUG_FILE", "/tmp/gemini_raw_retry.txt")
                with open(debug_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write((text or "").strip() + "\n")
            except Exception:
                pass
        try:
            params = _extract_json(text)
        except Exception:
            try:
                params = _parse_params_from_text(text)
            except Exception:
                params = None

    if params is None:
        if os.environ.get("GEMINI_DEBUG"):
            print("[agent] Gemini response not parseable; using defaults", file=sys.stderr)
        return dict(DEFAULT_PARAMS)

    return params


def _normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(DEFAULT_PARAMS)
    min_budget = 0.02
    max_budget = 0.2
    if isinstance(params, dict):
        scheduler = params.get("scheduler")
        if scheduler in ("baseline", "fuse"):
            result["scheduler"] = scheduler
        connectivity = params.get("connectivity")
        if connectivity in ("strict", "loose", "auto"):
            result["connectivity"] = connectivity
        retain_budget = params.get("retain_budget")
        try:
            retain_budget = float(retain_budget)
            if retain_budget < min_budget:
                retain_budget = min_budget
            if retain_budget > max_budget:
                retain_budget = max_budget
            result["retain_budget"] = retain_budget
        except Exception:
            pass
    return result


def generate_schedule(input_path: str, output_path: str) -> None:
    problem = parse_official_problem(input_path)
    problem_dict = {
        "widths": [t.width for t in problem.tensors],
        "heights": [t.height for t in problem.tensors],
        "inputs": [op.inputs for op in problem.ops],
        "outputs": [op.outputs for op in problem.ops],
        "base_costs": [op.base_cost for op in problem.ops],
        "op_types": [op.op_type for op in problem.ops],
        "fast_memory_capacity": problem.fast_memory_capacity,
        "slow_memory_bandwidth": problem.slow_memory_bandwidth,
        "native_granularity": [problem.native_granularity.width, problem.native_granularity.height],
    }

    try:
        params = _call_gemini(problem_dict)
    except Exception as exc:
        # Fallback to safe defaults if Gemini fails.
        print(f"[agent] Gemini call failed, using defaults: {exc}", file=sys.stderr)
        params = DEFAULT_PARAMS

    params = _normalize_params(params)

    retain_budget = max(0.0, min(params["retain_budget"], 0.9))
    if params["scheduler"] == "baseline":
        scheduler = OfficialScheduler(
            problem,
            retain_budget=retain_budget,
            problem_path=input_path,
            debug_fuse=False,
            debug_fuse_file="",
            fuse_accept_slack=0.0,
            connectivity=params["connectivity"],
        )
    else:
        scheduler = OfficialFusionScheduler(
            problem,
            retain_budget=retain_budget,
            problem_path=input_path,
            debug_fuse=False,
            debug_fuse_file="",
            fuse_accept_slack=0.0,
            connectivity=params["connectivity"],
        )

    schedule = scheduler.schedule()

    write_official_schedule(
        output_path,
        [sg.ops for sg in schedule.subgraphs],
        [sg.granularity for sg in schedule.subgraphs],
        [sg.tensors_to_retain for sg in schedule.subgraphs],
        [sg.traversal_order for sg in schedule.subgraphs],
        [sg.subgraph_latency for sg in schedule.subgraphs],
        strict_output=True,
    )


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python3 agent.py <input.json> <output.json>", file=sys.stderr)
        return 2
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    generate_schedule(input_path, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
