"""Canonical job-field mapping and text preparation for training and inference."""
from __future__ import annotations

import html
import re

FORM_FIELDS = (
    "title",
    "company",
    "salary",
    "location",
    "description",
    "requirements",
    "contact",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "benefits",
)
DATASET_FIELDS = (
    "title",
    "company_profile",
    "description",
    "requirements",
    "salary_range",
    "location",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "benefits",
)
FIELD_ALIASES = {
    "company": "company_profile",
    "salary": "salary_range",
}
PIPELINE_VERSION = "emscad-text-v2"


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9@%$.,!?+\-/ ]", " ", text)).strip()


def normalize_job(job: dict[str, object]) -> dict[str, str]:
    """Return the same canonical fields for a form submission or dataset row."""
    normalized = {field: clean_text(job.get(field, "")) for field in FORM_FIELDS}
    for form_field, dataset_field in FIELD_ALIASES.items():
        normalized[dataset_field] = normalized[form_field]
    for field in DATASET_FIELDS:
        if field in job and field not in normalized:
            normalized[field] = clean_text(job.get(field, ""))
    return normalized


def combine_job_fields(job: dict[str, object]) -> str:
    normalized = normalize_job(job)
    return " ".join(clean_text(normalized.get(field, "")) for field in DATASET_FIELDS).strip()


def dataset_text(dataframe) -> object:
    """Combine only supported EMSCAD columns; absent columns are empty."""
    available = [field for field in DATASET_FIELDS if field in dataframe.columns]
    if not available:
        raise ValueError("EMSCAD.csv does not contain any supported job-information columns.")
    return dataframe.reindex(columns=available).fillna("").astype(str).agg(" ".join, axis=1).map(clean_text)


def infer_columns(columns) -> tuple[str, str]:
    normalized = {str(column).strip().lower(): column for column in columns}
    target = next((normalized[key] for key in ("fraudulent", "is_fraud", "label", "target", "class") if key in normalized), None)
    if target is None:
        raise ValueError("Could not find a fraud label column (for example 'fraudulent') in EMSCAD.csv.")
    if "title" not in normalized and "description" not in normalized:
        raise ValueError("Could not find job text columns in EMSCAD.csv.")
    return normalized.get("title", normalized.get("description")), target
