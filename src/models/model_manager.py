"""Model investigation and training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


@dataclass
class InvestigationResult:
    table: pd.DataFrame
    best_model_name: str
    best_f1_macro: float


def _safe_stratify_arg(labels: List[str]):
    """Stratified split requires every class to appear at least twice; otherwise return None."""
    from collections import Counter

    ys = [str(y) for y in labels]
    cnt = Counter(ys)
    if len(cnt) < 2:
        return None
    if min(cnt.values()) < 2:
        return None
    return ys


class ModelManager:
    """Investigates classical architectures and trains selected models."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model_factories = {
            "Logistic Regression": lambda: LogisticRegression(
                max_iter=1500, n_jobs=None, random_state=self.random_state
            ),
            "Naive Bayes": lambda: MultinomialNB(),
            "SVM (LinearSVC)": lambda: LinearSVC(random_state=self.random_state),
            "Random Forest": lambda: RandomForestClassifier(
                n_estimators=300, random_state=self.random_state, n_jobs=-1
            ),
            "KNN": lambda: KNeighborsClassifier(n_neighbors=7),
        }
        self.fitted_pipeline: Pipeline | None = None
        self.last_selected_model_name: str | None = None

    def _make_pipeline(self, model_name: str) -> Pipeline:
        if model_name not in self.model_factories:
            raise ValueError(f"Unsupported model: {model_name}")
        estimator = self.model_factories[model_name]()
        return Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
                ("model", estimator),
            ]
        )

    def investigate(
        self, texts: List[str], labels: List[str], test_size: float = 0.2
    ) -> InvestigationResult:
        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have the same length.")
        if len(texts) < 20:
            raise ValueError("At least 20 rows are recommended for model investigation.")

        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=self.random_state,
            stratify=_safe_stratify_arg(labels),
        )

        rows = []
        for model_name, factory in self.model_factories.items():
            pipeline = Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
                    ("model", clone(factory())),
                ]
            )
            pipeline.fit(x_train, y_train)
            pred = pipeline.predict(x_test)

            rows.append(
                {
                    "Model": model_name,
                    "Accuracy": round(accuracy_score(y_test, pred), 4),
                    "F1_Macro": round(f1_score(y_test, pred, average="macro"), 4),
                    "F1_Weighted": round(
                        f1_score(y_test, pred, average="weighted"), 4
                    ),
                }
            )

        table = pd.DataFrame(rows).sort_values(by="F1_Macro", ascending=False) 
        best_row = table.iloc[0]
        return InvestigationResult(
            table=table.reset_index(drop=True),
            best_model_name=str(best_row["Model"]),
            best_f1_macro=float(best_row["F1_Macro"]),
        )

    def train_selected(
        self, texts: List[str], labels: List[str], model_name: str, test_size: float = 0.2
    ) -> Tuple[Dict[str, float], str]:
        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have the same length.")

        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=self.random_state,
            stratify=_safe_stratify_arg(labels),
        )

        pipeline = self._make_pipeline(model_name)
        pipeline.fit(x_train, y_train)
        pred = pipeline.predict(x_test)

        metrics = {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
            "f1_weighted": round(f1_score(y_test, pred, average="weighted"), 4),
            "samples_train": int(len(x_train)),
            "samples_test": int(len(x_test)),
        }
        report = classification_report(y_test, pred, digits=4)

        self.fitted_pipeline = pipeline
        self.last_selected_model_name = model_name
        return metrics, report

    def predict(self, texts: List[str]) -> np.ndarray:
        if self.fitted_pipeline is None:
            raise RuntimeError("No model is trained yet.")
        return self.fitted_pipeline.predict(texts)

    @staticmethod
    def architecture_guidance() -> str:
        return (
            "Investigated families: linear models, probabilistic models, margin-based "
            "models, tree ensembles, instance-based models, and transformer-based "
            "architectures (conceptual).\n"
            "Development recommendation: start with SVM/Logistic Regression baseline "
            "for speed and interpretability; move to DistilBERT for best quality if "
            "GPU/compute is available."
        )
