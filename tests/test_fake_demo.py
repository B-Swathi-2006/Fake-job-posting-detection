"""Reproducible synthetic-page smoke test for the Flask job-risk pipeline."""
from __future__ import annotations

import unittest
from pathlib import Path

from src.prediction import ModelService
from src.red_flags import detect_red_flags
from src.risk_analysis import adjusted_probability, risk_level
from src.scraper import extract_job_html


ROOT = Path(__file__).resolve().parents[1]
DEMO_PAGE = ROOT / "templates" / "tests_fake" / "fake_job_demo.html"


class FakeDemoPipelineTest(unittest.TestCase):
    def test_synthetic_demo_is_processed_without_forcing_model_probability(self) -> None:
        extracted = extract_job_html(DEMO_PAGE.read_text(encoding="utf-8"))
        prediction = ModelService().predict(extracted)
        flags = detect_red_flags(extracted)
        risk_score = adjusted_probability(prediction["fraud_probability"], flags)

        required_flags = {
            "Payment or fee language detected",
            "Urgent hiring language detected",
            "Unusually attractive income claims detected",
            "High income combined with no experience requirement",
            "Sensitive personal or financial information requested",
            "Suspicious recruiter contact method detected",
        }
        self.assertTrue(required_flags.issubset(set(flags)))
        self.assertGreaterEqual(prediction["fraud_probability"], 0.0)
        self.assertLessEqual(prediction["fraud_probability"], 100.0)
        self.assertGreaterEqual(risk_score, 0.0)
        self.assertLessEqual(risk_score, 100.0)

        print(f"synthetic demo flags: {flags}")
        print(f"synthetic demo model probability: {prediction['fraud_probability']:.1f}%")
        print(f"synthetic demo heuristic risk score: {risk_score:.1f}% ({risk_level(risk_score)['name']})")


if __name__ == "__main__":
    unittest.main()
