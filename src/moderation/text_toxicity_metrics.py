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


def analyze_preprocessed_text_for_metrics(
    text: str,
    *,
    toxicity_analyzer: LocalToxicityAnalyzer,
) -> ToxicityTextMetricsBundle:
    analysis = toxicity_analyzer.analyze(text)
    metrics = dict(analysis.scores)
    overall = int(analysis.overall_score)
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
