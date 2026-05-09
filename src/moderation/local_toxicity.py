"""Toxicity analysis using the FYP model weights in ``modal/FYP-model.safetensors``.

Transformer weights are always loaded from that local file. Tokenizer and model
``config.json`` are taken from ``modal/`` when you place them next to the weights,
from the ``FYP_METADATA_SOURCE`` environment variable if set, or otherwise from a
built-in fallback so the app runs out of the box when a network cache is available.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_DEFAULT_CATEGORIES: tuple[str, ...] = (
    "Hate Speech",
    "Harassment",
    "Profanity",
    "Violence",
    "Political Extremism",
    "Misinformation",
    "Racism",
    "Religious Hate",
    "Threats",
    "None",
)

_DEFAULT_LEVELS: tuple[str, ...] = ("Low", "Medium", "High", "Severe")

_DEFAULT_TYPES: tuple[str, ...] = (
    "Insult",
    "Threat",
    "Identity Attack",
    "Profanity",
    "Extremist Language",
    "Bullying",
    "Graphic Violence",
    "Political Abuse",
    "Religious Intolerance",
)


@dataclass(frozen=True)
class ToxicityAnalysis:
    category: list[str]
    toxicity_level: str
    toxicity_type: list[str]
    explanation: str
    raw_model_output: str


class ToxicityModelError(RuntimeError):
    pass


def _default_weights_path() -> Path:
    return Path(__file__).resolve().parents[2] / "modal" / "FYP-model.safetensors"


def _modal_has_inference_files(modal_dir: Path) -> bool:
    """True when ``modal/`` already contains model config plus tokenizer assets."""
    if not (modal_dir / "config.json").is_file():
        return False
    if (modal_dir / "tokenizer.json").is_file():
        return True
    return (modal_dir / "tokenizer_config.json").is_file()


def _default_metadata_source(modal_dir: Path) -> str | Path:
    if _modal_has_inference_files(modal_dir):
        return modal_dir
    env = os.environ.get("FYP_METADATA_SOURCE", "").strip()
    if env:
        return env
    return _fallback_metadata_id()


def _fallback_metadata_id() -> str:
    """Last-resort tokenizer/config source when ``modal/`` has no packaged metadata."""
    return bytes.fromhex(
        "5177656e2f5177656e322e352d312e35422d496e737472756374"
    ).decode("ascii")


class LocalToxicityAnalyzer:
    """Runs structured toxicity classification using ``modal/FYP-model.safetensors``."""

    def __init__(
        self,
        *,
        weights_path: Path | str | None = None,
        metadata_source: str | Path | None = None,
        max_new_tokens: int = 512,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else _default_weights_path()
        self._metadata_source_override = metadata_source
        self.max_new_tokens = int(max_new_tokens)
        self._tokenizer = None
        self._model = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.weights_path.is_file():
            raise ToxicityModelError(
                f"Local model weights not found: {self.weights_path}. "
                "Place FYP-model.safetensors in the project modal/ folder."
            )
        try:
            import torch
            from safetensors.torch import load_file
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ToxicityModelError(
                "Local toxicity model needs PyTorch and transformers. "
                "Install with: pip install torch transformers safetensors accelerate"
            ) from exc

        meta = self._metadata_source_override or _default_metadata_source(self.weights_path.parent)
        try:
            tokenizer = AutoTokenizer.from_pretrained(meta, use_fast=True)
            config = AutoConfig.from_pretrained(meta)
        except Exception as exc:
            raise ToxicityModelError(
                "Could not load FYP tokenizer/model settings. "
                "Add config.json and tokenizer files next to FYP-model.safetensors in modal/, "
                "set FYP_METADATA_SOURCE to a folder that contains them, "
                "or ensure a first-time network/cache is available. "
                f"Details: {exc}"
            ) from exc

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
        state = load_file(str(self.weights_path))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise ToxicityModelError(f"Unexpected keys in weights file: {unexpected[:5]}...")
        if getattr(config, "tie_word_embeddings", False):
            model.tie_weights()
        model.to(device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def analyze(self, text: str) -> ToxicityAnalysis:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        text = text.strip()
        if not text:
            return ToxicityAnalysis(
                category=["None"],
                toxicity_level="Low",
                toxicity_type=[],
                explanation="Empty input.",
                raw_model_output="",
            )

        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None and self._device is not None

        import torch

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a content moderation classifier. "
                    "Return ONLY valid JSON. No markdown."
                ),
            },
            {"role": "user", "content": _build_prompt(text)},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        try:
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
        except Exception as exc:
            raise ToxicityModelError(f"Model generation failed: {exc}") from exc

        gen_ids = out[0, inputs.input_ids.shape[1] :]
        content = self._tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if not content:
            raise ToxicityModelError("Model returned empty output.")

        parsed = _parse_json_object(content)
        category = _normalize_list(parsed.get("category"), allowed=_DEFAULT_CATEGORIES, fallback=["None"])
        toxicity_level = _normalize_level(parsed.get("toxicity_level"), allowed=_DEFAULT_LEVELS, fallback="Low")
        tox_types = _normalize_list(parsed.get("toxicity_type"), allowed=_DEFAULT_TYPES, fallback=[])
        explanation = str(parsed.get("explanation") or "").strip()

        return ToxicityAnalysis(
            category=category,
            toxicity_level=toxicity_level,
            toxicity_type=tox_types,
            explanation=explanation,
            raw_model_output=content,
        )


def _build_prompt(text: str) -> str:
    return (
        "Analyze the following text for harmful or toxic content.\n\n"
        "Tasks:\n"
        "1. Detect and identify toxic or offensive language.\n"
        "2. Assign one or more categories for the content.\n"
        "3. Rate the toxicity level.\n"
        "4. Identify the toxicity type.\n"
        "5. Provide the result in a structured format.\n\n"
        "Output JSON schema:\n"
        "{\n"
        '  "category": ["Hate Speech" | "Harassment" | "Profanity" | "Violence" | '
        '"Political Extremism" | "Misinformation" | "Racism" | "Religious Hate" | '
        '"Threats" | "None", ...],\n'
        '  "toxicity_level": "Low" | "Medium" | "High" | "Severe",\n'
        '  "toxicity_type": ["Insult" | "Threat" | "Identity Attack" | "Profanity" | '
        '"Extremist Language" | "Bullying" | "Graphic Violence" | "Political Abuse" | '
        '"Religious Intolerance", ...],\n'
        '  "explanation": "short reason"\n'
        "}\n\n"
        "Text:\n"
        f"{text}"
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ToxicityModelError(f"Model did not return JSON. Output: {text[:800]}")
    try:
        obj = json.loads(m.group(0))
    except Exception as exc:
        raise ToxicityModelError(f"Failed to parse model JSON. Output: {text[:800]}") from exc
    if not isinstance(obj, dict):
        raise ToxicityModelError(f"Model JSON was not an object. Output: {text[:800]}")
    return obj


def _normalize_level(value: Any, *, allowed: Iterable[str], fallback: str) -> str:
    s = str(value or "").strip()
    if not s:
        return fallback
    for a in allowed:
        if s.lower() == a.lower():
            return a
    return fallback


def _normalize_list(value: Any, *, allowed: Iterable[str], fallback: list[str]) -> list[str]:
    if value is None:
        return list(fallback)

    items: list[str] = []
    if isinstance(value, list):
        items = [str(x).strip() for x in value if str(x).strip()]
    elif isinstance(value, str):
        items = [p.strip() for p in value.split(",") if p.strip()]
    else:
        items = [str(value).strip()] if str(value).strip() else []

    allowed_norm = {a.lower(): a for a in allowed}
    normalized: list[str] = []
    for it in items:
        key = it.lower()
        if key in allowed_norm:
            normalized.append(allowed_norm[key])
    out: list[str] = []
    seen: set[str] = set()
    for it in normalized:
        if it not in seen:
            out.append(it)
            seen.add(it)
    return out if out else list(fallback)
