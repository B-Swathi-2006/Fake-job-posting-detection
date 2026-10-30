"""Optional BERT fine-tuning entry point.

This deliberately keeps transformer dependencies out of the baseline install.
Install the optional dependencies first, then run this script after the
dataset-adaptive preparation has been verified.
"""
from __future__ import annotations

def main() -> None:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise SystemExit(
            "BERT training is optional. Install torch, transformers, and datasets, "
            "then implement/verify the dataset-specific text/label mapping before training."
        ) from exc
    raise SystemExit(
        "BERT dependencies are installed, but EMSCAD's schema must be mapped explicitly before "
        "fine-tuning. This project does not fabricate BERT results; use the measured Trainer "
        "evaluation output before adding it to models\\metrics.pkl."
    )


if __name__ == "__main__":
    main()
