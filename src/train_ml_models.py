"""Train and evaluate leakage-safe EMSCAD text classifiers."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import vstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from .preprocessing import DATASET_FIELDS, PIPELINE_VERSION, dataset_text, infer_columns
except ImportError:
    from preprocessing import DATASET_FIELDS, PIPELINE_VERSION, dataset_text, infer_columns

ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42


def encode_target(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower().replace(
        {"f": "0", "false": "0", "genuine": "0", "legitimate": "0",
         "t": "1", "true": "1", "fraudulent": "1", "fake": "1"}
    )
    return pd.to_numeric(values, errors="coerce")


def oversample_training(features, target: pd.Series):
    """Duplicate minority rows only; unlike SMOTE this is safe for sparse TF-IDF."""
    counts = target.value_counts()
    majority = int(counts.max())
    indexes = []
    values = target.to_numpy()
    for label, count in counts.items():
        label_indexes = np.flatnonzero(values == label)
        indexes.extend(label_indexes.tolist())
        if count < majority:
            indexes.extend(label_indexes.tolist() * (majority // count - 1))
            indexes.extend(label_indexes[: majority % count].tolist())
    return features[indexes], target.iloc[indexes].reset_index(drop=True)


def score_model(model, features, target: pd.Series) -> dict:
    predicted = model.predict(features)
    probabilities = model.predict_proba(features)[:, list(model.classes_).index(1)]
    matrix = confusion_matrix(target, predicted, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(target, predicted)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "pr_auc": float(average_precision_score(target, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "fraudulent_detected": int(matrix[1, 1]),
        "fraudulent_missed": int(matrix[1, 0]),
    }


def main() -> None:
    dataset_path = ROOT / "dataset" / "EMSCAD.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    data = pd.read_csv(dataset_path)
    _, target_column = infer_columns(data.columns)
    data = data.drop_duplicates().dropna(subset=[target_column]).copy()
    data[target_column] = encode_target(data[target_column])
    data = data[data[target_column].isin([0, 1])].copy()
    if data[target_column].nunique() != 2:
        raise ValueError("The target column must contain both 0 (genuine) and 1 (fraudulent).")
    selected_fields = [field for field in DATASET_FIELDS if field in data.columns]
    if not selected_fields:
        raise ValueError("No supported EMSCAD job-information fields were found.")
    texts = dataset_text(data)
    targets = data[target_column].astype(int).reset_index(drop=True)
    texts = texts.reset_index(drop=True)
    x_train, x_remaining, y_train, y_remaining = train_test_split(
        texts, targets, test_size=0.30, random_state=RANDOM_STATE, stratify=targets
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_remaining, y_remaining, test_size=0.50, random_state=RANDOM_STATE, stratify=y_remaining
    )
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True
    )
    train_features = vectorizer.fit_transform(x_train)
    validation_features = vectorizer.transform(x_validation)
    test_features = vectorizer.transform(x_test)
    candidates = {
        "balanced": RandomForestClassifier(
            n_estimators=250, max_depth=None, min_samples_leaf=1,
            max_features="sqrt", random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        ),
        "balanced_regularized": RandomForestClassifier(
            n_estimators=250, max_depth=24, min_samples_leaf=2,
            max_features="sqrt", random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        ),
    }
    candidate_scores = {}
    for name, model in candidates.items():
        model.fit(train_features, y_train)
        candidate_scores[name] = score_model(model, validation_features, y_validation)
    oversampled_features, oversampled_target = oversample_training(train_features, y_train)
    oversampled_model = RandomForestClassifier(
        n_estimators=250, max_depth=24, min_samples_leaf=2,
        max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1
    )
    oversampled_model.fit(oversampled_features, oversampled_target)
    candidate_scores["oversampled"] = score_model(
        oversampled_model, validation_features, y_validation
    )
    selected_name = max(candidate_scores, key=lambda name: (
        candidate_scores[name]["pr_auc"], candidate_scores[name]["f1"]
    ))
    selected_config = {
        "balanced": candidates["balanced"],
        "balanced_regularized": candidates["balanced_regularized"],
        "oversampled": oversampled_model,
    }[selected_name]
    combined_features = vstack([train_features, validation_features])
    combined_target = pd.concat([y_train, y_validation], ignore_index=True)
    if selected_name == "oversampled":
        combined_features, combined_target = oversample_training(combined_features, combined_target)
    selected_config.fit(combined_features, combined_target)
    metrics = {
        "Random Forest": score_model(selected_config, test_features, y_test),
        "_pipeline_version": PIPELINE_VERSION,
        "_sklearn_version": sklearn.__version__,
        "_selected_fields": selected_fields,
        "_target_column": target_column,
        "_split": {"train": len(y_train), "validation": len(y_validation), "test": len(y_test)},
        "_validation_candidates": candidate_scores,
        "_selected_random_forest": selected_name,
    }
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    with (model_dir / "random_forest.pkl").open("wb") as handle:
        pickle.dump(selected_config, handle)
    with (model_dir / "tfidf_vectorizer.pkl").open("wb") as handle:
        pickle.dump(vectorizer, handle)
    with (model_dir / "metrics.pkl").open("wb") as handle:
        pickle.dump(metrics, handle)
    print(f"Trained on {len(data)} rows using fields: {', '.join(selected_fields)}")
    print(f"Target mapping: f/0=legitimate, t/1=fraudulent; selected RF: {selected_name}")
    print(metrics)


if __name__ == "__main__":
    main()
