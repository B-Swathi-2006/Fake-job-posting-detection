"""Load persisted models and produce real probabilities for submitted jobs."""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import sklearn

from .preprocessing import PIPELINE_VERSION, combine_job_fields

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


class PredictionError(RuntimeError):
    """A user-facing prediction problem."""


class ModelService:
    def __init__(self) -> None:
        self.ready = False
        self.model_name = "Random Forest"
        self.metrics: dict = {}
        self.model = None
        self.vectorizer = None
        self._load()

    def _load(self) -> None:
        model_path = MODEL_DIR / "random_forest.pkl"
        vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"
        metrics_path = MODEL_DIR / "metrics.pkl"
        if not model_path.exists() or not vectorizer_path.exists():
            return
        try:
            with model_path.open("rb") as handle:
                self.model = pickle.load(handle)
            with vectorizer_path.open("rb") as handle:
                self.vectorizer = pickle.load(handle)
            if metrics_path.exists():
                with metrics_path.open("rb") as handle:
                    self.metrics = pickle.load(handle)
                self.metrics = {
                    key: value
                    for key, value in self.metrics.items()
                    if key == self.model_name or key.startswith("_")
                }
            if self.metrics.get("_pipeline_version") not in (None, PIPELINE_VERSION):
                raise PredictionError("Saved model artifacts use an outdated preprocessing pipeline. Retrain the models.")
            saved_sklearn_version = self.metrics.get("_sklearn_version")
            if saved_sklearn_version and saved_sklearn_version != sklearn.__version__:
                warnings.warn(
                    f"Saved model artifacts were trained with scikit-learn {saved_sklearn_version}; "
                    f"the current environment has {sklearn.__version__}. The files may still be usable, "
                    "but retraining is recommended.",
                    UserWarning,
                    stacklevel=2,
                )
            if not hasattr(self.model, "classes_") or 1 not in self.model.classes_:
                raise PredictionError("Saved model does not contain the legitimate/fraudulent class mapping. Retrain the models.")
            if (
                hasattr(self.model, "n_features_in_")
                and hasattr(self.vectorizer, "vocabulary_")
                and self.model.n_features_in_ != len(self.vectorizer.vocabulary_)
            ):
                raise PredictionError("Saved model and vectorizer have incompatible feature dimensions. Retrain the models.")
            self.ready = True
        except PredictionError:
            raise
        except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError, KeyError, IndexError) as exc:
            raise PredictionError("Saved model files could not be loaded. Retrain the models.") from exc

    def predict(self, job: dict[str, str]) -> dict:
        if not self.ready or self.model is None or self.vectorizer is None:
            raise PredictionError(
                "No trained model is available. Place EMSCAD.csv in dataset\\ and run "
                "'python src\\train_ml_models.py' first."
            )
        text = combine_job_fields(job)
        features = self.vectorizer.transform([text])
        fraud_index = list(self.model.classes_).index(1)
        probability = float(self.model.predict_proba(features)[0][fraud_index]) * 100
        return {"model_name": self.model_name, "fraud_probability": probability}
