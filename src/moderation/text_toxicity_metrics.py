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


def metrics_from_llm_prediction(
    *,
    toxicity_level: str,
    toxicity_types: list[str],
) -> tuple[dict[str, int], int]:
    lvl = (toxicity_level or "").strip().lower()
    base = {"low": 20, "medium": 55, "high": 80, "severe": 95}.get(lvl, 40)
    severe = {"low": 0, "medium": 35, "high": 70, "severe": 95}.get(lvl, 30)

    types = {t.strip().lower(): t for t in (toxicity_types or []) if t and t.strip()}
    identity = 85 if "identity attack" in types else 0
    threat = 85 if "threat" in types else 0
    profanity = 80 if "profanity" in types else 0
    insult = 70 if ("insult" in types or "bullying" in types or "political abuse" in types) else 0

    if base >= 55 and max(identity, threat, profanity, insult) == 0:
        insult = 55

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
