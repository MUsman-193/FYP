"""Convert local model outputs into dashboard metrics.

Runs on worker threads away from Tk; never touch widgets here.
"""

from __future__ import annotations

from dataclasses import dataclass

from preprocessing import PreprocessConfig, TextPreprocessor

from .local_toxicity import LocalToxicityAnalyzer


@dataclass(frozen=True)
class ToxicityTextMetricsBundle:
    metrics: dict[str, int]
    overall: int
    lvl: str
    cats_csv: str
    types_csv: str
    explanation: str


# Categories the model often uses instead of filling toxicity_type (see local_toxicity._DEFAULT_CATEGORIES).
_IDENTITY_FROM_CATEGORY = frozenset({"racism", "religious hate", "hate speech"})
_THREAT_FROM_CATEGORY = frozenset({"threats"})
_HARASSMENT_INSULT_CAT = frozenset({"harassment"})
_POLITICAL_OR_MISINFO_CAT = frozenset({"political extremism", "misinformation"})


def metrics_from_llm_prediction(
    *,
    toxicity_level: str,
    toxicity_types: list[str],
    categories: list[str] | None = None,
) -> tuple[dict[str, int], int]:
    """Map model labels to 0–100 UI bars. ``Low`` with no types is treated as non-toxic (0), not a fixed 20%."""
    lvl = (toxicity_level or "").strip().lower()
    types = {t.strip().lower(): t for t in (toxicity_types or []) if t and t.strip()}
    cats_norm = [str(c).strip().lower() for c in (categories or []) if str(c).strip()]
    only_none_cat = (not cats_norm) or (cats_norm == ["none"])

    # Tier scores for the aggregate "Toxicity" / "Severe Toxicity" bars (type-specific bars below).
    if lvl == "low":
        base = 10 if types else 0
        severe = 0
    elif lvl == "medium":
        base, severe = 52, 28
    elif lvl == "high":
        base, severe = 78, 65
    elif lvl == "severe":
        base, severe = 96, 96
    else:
        # Unknown / misspelled level: mild hint only, not a false "40% toxic".
        base, severe = 12, 6

    # Model says harmless category and no specific types — keep UI at zero even if level string is odd.
    if only_none_cat and not types and lvl in {"", "low"}:
        base, severe = 0, 0

    cat_set = set(cats_norm)

    # Type-based (keys are lowercased canonical labels from the model output).
    identity = 85 if "identity attack" in types else 0
    if "religious intolerance" in types:
        identity = max(identity, 85)
    threat = 85 if "threat" in types else 0

    # Substring fallback: model sometimes emits non-canonical strings that still normalize to a key
    # after local_toxicity fixes; if any type key clearly indicates threat/identity, count it.
    joined_types = " ".join(types.keys())
    if threat == 0 and "threat" in joined_types:
        threat = 85
    if identity == 0 and ("identity" in joined_types and "attack" in joined_types):
        identity = 85

    # Category-based: many runs set category to "Threats" / "Racism" but leave toxicity_type empty or generic.
    if cat_set & _THREAT_FROM_CATEGORY:
        threat = max(threat, 85)
    if "violence" in cat_set:
        threat = max(threat, 70)
    if cat_set & _IDENTITY_FROM_CATEGORY:
        identity = max(identity, 85)

    profanity = 80 if "profanity" in types else 0
    insult = 70 if ("insult" in types or "bullying" in types or "political abuse" in types) else 0

    if cat_set & _HARASSMENT_INSULT_CAT:
        insult = max(insult, 58)
    if cat_set & _POLITICAL_OR_MISINFO_CAT:
        insult = max(insult, 48)

    metrics = {
        "Toxicity": int(max(0, min(100, base))),
        "Severe Toxicity": int(max(0, min(100, severe))),
        "Identity Attack": int(max(0, min(100, identity))),
        "Insult": int(max(0, min(100, insult))),
        "Profanity": int(max(0, min(100, profanity))),
        "Threat": int(max(0, min(100, threat))),
    }
    overall = int(round(sum(metrics.values()) / len(metrics)))
    return metrics, overall


def analyze_preprocessed_text_for_metrics(
    text: str,
    *,
    toxicity_analyzer: LocalToxicityAnalyzer,
) -> ToxicityTextMetricsBundle:
    analysis = toxicity_analyzer.analyze(text)
    metrics, overall = metrics_from_llm_prediction(
        toxicity_level=analysis.toxicity_level,
        toxicity_types=list(analysis.toxicity_type),
        categories=list(analysis.category),
    )
    cats = ", ".join(analysis.category) if analysis.category else ""
    types_j = ", ".join(analysis.toxicity_type) if analysis.toxicity_type else ""
    expl = (analysis.explanation or "").strip()
    return ToxicityTextMetricsBundle(
        metrics=metrics,
        overall=overall,
        lvl=str(analysis.toxicity_level),
        cats_csv=cats,
        types_csv=types_j,
        explanation=expl,
    )


def preprocess_and_analyze_single(
    text_raw: str,
    *,
    preprocessor: TextPreprocessor,
    prep_config: PreprocessConfig,
    toxicity_analyzer: LocalToxicityAnalyzer,
) -> ToxicityTextMetricsBundle:
    text = preprocessor.apply(text_raw, prep_config)
    return analyze_preprocessed_text_for_metrics(
        text,
        toxicity_analyzer=toxicity_analyzer,
    )
