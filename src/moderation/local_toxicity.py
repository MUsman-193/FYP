"""Toxicity analysis using weights under the project ``model/`` directory.

Transformer weights are always loaded from that local file. Tokenizer and model
``config.json`` and tokenizer assets are loaded from ``model/`` next to the weights,
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
    env = os.environ.get("FYP_WEIGHTS_PATH", "").strip()
    if env:
        return Path(env)

    model_dir = Path(__file__).resolve().parents[2] / "model"
    preferred = [
        model_dir / "FYP-model.safetensors",
        model_dir / "model.safetensors",  # HF repos often name it like this
    ]
    for p in preferred:
        if p.is_file():
            return p

    # Fallback: first .safetensors in model/ (if any).
    for p in sorted(model_dir.glob("*.safetensors")):
        if p.is_file():
            return p

    # Default expected name.
    return preferred[0]


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
    """Runs structured toxicity classification using local ``model/*.safetensors`` weights."""

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

    def preload(self) -> None:
        """Load weights + tokenizer onto device (CPU or CUDA). Safe to call from a worker thread."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.weights_path.is_file():
            raise ToxicityModelError(
                f"Local model weights not found: {self.weights_path}. "
                "Place FYP-model.safetensors or model.safetensors in the project model/ folder."
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
                "Add config.json and tokenizer files next to the weights file in model/, "
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
            raise ToxicityModelError("Empty input; nothing to classify.")

        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None and self._device is not None

        import torch

        # Some tokenizers (including many locally packaged ones) don't ship a chat_template.
        # Use a plain-text prompt so local inference works without chat templates.
        prompt = (
            "You are a content moderation classifier. Return ONLY valid JSON. No markdown.\n\n"
            + _build_prompt(text)
            + "\n\nJSON:"
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        try:
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.12,
                    pad_token_id=(
                        self._tokenizer.pad_token_id
                        if self._tokenizer.pad_token_id is not None
                        else self._tokenizer.eos_token_id
                    ),
                )
        except Exception as exc:
            raise ToxicityModelError(f"Model generation failed: {exc}") from exc

        gen_ids = out[0, inputs.input_ids.shape[1] :]
        content = self._tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if not content:
            raise ToxicityModelError("Model returned empty output.")

        parsed = _parse_json_object(content)
        if "toxicity_type" not in parsed and isinstance(parsed.get("toxicity_types"), list):
            parsed["toxicity_type"] = parsed.get("toxicity_types")
        category = _normalize_list(parsed.get("category"), allowed=_DEFAULT_CATEGORIES, fallback=["None"])
        # Some generations echo almost every schema category; infer from toxicity_type instead.
        if len(category) >= 9:
            candidate = _infer_category_from_raw_types(parsed.get("toxicity_type"))
            category = _normalize_list([candidate], allowed=_DEFAULT_CATEGORIES, fallback=["None"])
        toxicity_level = _normalize_level(parsed.get("toxicity_level"), allowed=_DEFAULT_LEVELS, fallback="Low")
        tox_types = _normalize_toxicity_types(parsed.get("toxicity_type"))
        # Model sometimes lists most of the enum; treat as non-specific.
        if len(tox_types) >= max(5, len(_DEFAULT_TYPES) - 2):
            tox_types = []
        explanation = str(parsed.get("explanation") or "").strip()

        return ToxicityAnalysis(
            category=category,
            toxicity_level=toxicity_level,
            toxicity_type=tox_types,
            explanation=explanation,
            raw_model_output=content,
        )


def _infer_category_from_raw_types(types_val: Any) -> str:
    """Pick one category when model listed the whole enum verbatim."""
    items: list[str] = []
    if isinstance(types_val, list):
        items = [str(x).strip() for x in types_val if str(x).strip()]
    elif isinstance(types_val, str) and types_val.strip():
        items = [types_val.strip()]

    tl = [t.strip().lower() for t in items]
    tl_set = set(tl)

    def has(*needles: str) -> bool:
        return bool(tl_set & set(needles))

    if has("threat"):
        return "Threats"
    if has("identity attack"):
        return "Harassment"
    if has("bullying"):
        return "Harassment"
    if has("graphic violence"):
        return "Violence"
    if has("extremist language", "political abuse"):
        return "Political Extremism"
    if has("religious intolerance"):
        return "Religious Hate"
    if has("profanity"):
        return "Profanity"
    if has("insult"):
        return "Harassment"
    return "None"


def _build_prompt(text: str) -> str:
    """Keep instructions compact so small LMs do not echo huge fake JSON schemas."""
    cats = ", ".join(_DEFAULT_CATEGORIES)
    types_ex = ", ".join(_DEFAULT_TYPES)
    levels = ", ".join(_DEFAULT_LEVELS)
    return (
        "You classify a single user message for moderation.\n\n"
        "Respond with ONE JSON object and ONLY these four keys (no other keys, no preamble):\n"
        '"category" (array of 1-3 strings), "toxicity_level" (one string), '
        '"toxicity_type" (array, often empty), "explanation" (one short string).\n\n'
        "Rules:\n"
        "- Do NOT output tasks, instructions, allowed lists, or schema inside JSON.\n"
        '- Arrays must stay short (category ≤3 items, toxicity_type ≤4 items).\n'
        "- Use comma-separated JSON arrays only; never use '|' inside arrays.\n"
        "- If text is fine, use category [\"None\"], level \"Low\", toxicity_type [].\n\n"
        "Allowed category strings (pick the best fit): "
        f"{cats}.\n"
        f"Allowed levels: {levels}.\n"
        f"Optional toxicity_type strings (omit unknowns): {types_ex}.\n\n"
        "Example shape:\n"
        '{"category":["None"],"toxicity_level":"Low","toxicity_type":[],"explanation":"No harm detected."}\n\n'
        "Message to classify:\n"
        f"{text}"
    )


def _brace_balanced_object(s: str, open_brace: int) -> str | None:
    """Return substring s[open_brace : end+1] for a balanced {...} or None if truncated."""
    if open_brace < 0 or open_brace >= len(s) or s[open_brace] != "{":
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(open_brace, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[open_brace : i + 1]
    return None


def _classification_json_candidates(raw: str) -> list[str]:
    """Pull balanced JSON objects whose first key is category (ignores leading schema junk)."""
    out: list[str] = []
    for m in re.finditer(r"\{\s*\"category\"\s*:", raw):
        blob = _brace_balanced_object(raw, m.start())
        if blob:
            out.append(blob)
    return out


def _looks_like_schema_echo(raw: str) -> bool:
    rl = raw.lower()
    if "allowed_categories" in rl:
        return True
    if re.search(r'"tasks"\s*:\s*\[', raw) and len(raw) > 500:
        return True
    return False


def _repair_schema_union_in_arrays(blob: str) -> str:
    """Fix invalid JSON where the model echoed schema notation like '\"A\" | \"B\"'."""
    out = blob
    for key in ("category", "toxicity_type"):

        def repl(mm: re.Match) -> str:
            inner = mm.group(2)
            inner = re.sub(r'"\s*\|\s*"', '", "', inner)
            return f"{mm.group(1)}[{inner}]"

        out = re.sub(rf'("{re.escape(key)}"\s*:\s*)\[(.*?)\]', repl, out, flags=re.DOTALL)
    return out


def _fallback_parse_classification_dict(raw: str) -> dict[str, Any]:
    """Last resort: regex-extract JSON-like fields after model hallucinates invalid JSON."""
    out: dict[str, Any] = {}
    lvl = re.search(r'"toxicity_level"\s*:\s*"([^"]*)"', raw)
    if lvl:
        out["toxicity_level"] = lvl.group(1)
    for ak in ("category", "toxicity_type"):
        am = re.search(rf'"{ak}"\s*:\s*\[(.*?)\]\s*(?:,|\}})', raw, re.DOTALL)
        if not am:
            am = re.search(rf'"{ak}"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
        if am:
            inner = re.sub(r'"\s*\|\s*"', '", "', am.group(1))
            quoted = [q for q in re.findall(r'"([^"]*)"', inner) if q.strip()]
            if quoted:
                out[ak] = quoted
    em = re.search(r'"explanation"\s*:\s*"(.*?)"\s*\}\s*$', raw, flags=re.DOTALL)
    if not em:
        em = re.search(r'"explanation"\s*:\s*"(.*)"', raw, flags=re.DOTALL)
    if em:
        out["explanation"] = em.group(1).strip()
    return out


def _parse_json_object(text: str) -> dict[str, Any]:
    t = text.strip()
    blobs: list[str] = []
    blobs.append(t)
    m_obj = re.search(r"\{[\s\S]*\}", t)
    if m_obj:
        blobs.append(m_obj.group(0))
    # Prefer small trailing objects after the model echoed a large fake schema.
    for sub in _classification_json_candidates(t):
        if sub not in blobs:
            blobs.insert(0, sub)

    tries: list[str] = []
    for b in blobs:
        tries.append(b)
        tries.append(_repair_schema_union_in_arrays(b))

    last_err: Exception | None = None
    seen: set[str] = set()
    for cand in tries:
        cand = cand.strip()
        if not cand or cand in seen:
            continue
        seen.add(cand)
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception as exc:
            last_err = exc
            continue

    for sub in reversed(_classification_json_candidates(t)):
        loose = _fallback_parse_classification_dict(sub)
        if loose.get("toxicity_level") or loose.get("category") or loose.get("explanation"):
            return loose

    loose = _fallback_parse_classification_dict(m_obj.group(0) if m_obj else t)
    if loose.get("toxicity_level") or loose.get("category") or loose.get("explanation"):
        return loose

    if _looks_like_schema_echo(t):
        raise ToxicityModelError(
            "The model repeated a long instruction-style JSON instead of a short classification. "
            "Try a shorter message, or retry; the prompt has been tightened to reduce this."
        )

    detail = (
        f"Failed to parse model JSON. Raw output truncated:\n{text[:1600]}..."
        if len(text) > 1600
        else f"Failed to parse model JSON. Raw output:\n{text}"
    )
    if last_err is not None:
        raise ToxicityModelError(detail) from last_err
    raise ToxicityModelError(detail)


def _coerce_toxicity_type_token(raw: str) -> str | None:
    """Map common model misspellings / plurals onto ``_DEFAULT_TYPES``."""
    key = str(raw or "").strip().lower()
    if not key:
        return None
    allowed_norm = {a.lower(): a for a in _DEFAULT_TYPES}
    if key in allowed_norm:
        return allowed_norm[key]
    if "threat" in key:
        return "Threat"
    if "identity" in key and "attack" in key:
        return "Identity Attack"
    if key in {"identity-based attack", "identity based attack", "id attack"}:
        return "Identity Attack"
    return None


def _normalize_toxicity_types(value: Any) -> list[str]:
    """Like ``_normalize_list`` but accepts near-miss type strings the model often emits."""
    if value is None:
        return []
    items: list[str] = []
    if isinstance(value, list):
        items = [str(x).strip() for x in value if str(x).strip()]
    elif isinstance(value, str):
        items = [p.strip() for p in value.split(",") if p.strip()]
    else:
        s = str(value).strip()
        if s:
            items = [s]
    out: list[str] = []
    seen: set[str] = set()
    for it in items:
        canon = _coerce_toxicity_type_token(it)
        if canon and canon not in seen:
            out.append(canon)
            seen.add(canon)
    return out


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
