# Fake Job Posting Detection

JobShield is a Flask mini-project for identifying **potentially fraudulent** job postings. It combines dataset-trained TF-IDF classifiers with transparent red-flag checks. A prediction is a risk estimate, not proof that a posting is fraudulent.

## Features

- Manual job details and conservative public URL extraction
- TF-IDF + Random Forest training
- Persisted model, vectorizer, and held-out metrics
- Configurable 0–30% low, 30–60% medium, and 60–100% high risk thresholds
- Final risk score combining 80% model probability with 20% bounded, weighted warning-signal evidence; this is a triage score, not a calibrated probability
- Model comparison dashboard with actual evaluation values

## Windows setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The supplied EMSCAD CSV is already placed at `dataset\EMSCAD.csv`. Its verified label is `fraudulent` (`f` = legitimate and `t` = fraudulent). Training explicitly selects the EMSCAD job fields and excludes `in_balanced_dataset`, identifiers, and the target. The vectorizer is fit on the training partition only and reused for prediction. The training script compares class-weighted Random Forests with sparse-safe random oversampling; ordinary SMOTE is not applied to raw sparse TF-IDF text.

```powershell
python src\train_ml_models.py
python app.py
```

Open `http://127.0.0.1:5000`. Training must happen once after the dataset is installed; Flask loads saved artifacts and does not retrain at startup. The script prints the actual test-set metrics. No fabricated metrics or probabilities are included.

Run the synthetic demo regression test with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The test reads `templates\tests_fake\fake_job_demo.html` through the same HTML extraction path used by URL scraping. It reports the detected warning flags, the measured Random Forest probability, and the separate heuristic risk score. The page is fictional and is not treated as a confirmed real-world fraud report.

The displayed Random Forest probability is kept separate from the final risk score. The final score is calculated from configurable model and evidence weights. Evidence is normalized to 0–100 and overlapping signals in the same group count only once at their strongest weight. The combined value is a triage heuristic, not a calibrated fraud probability. Risk categories use the documented 30% and 60% thresholds. `RISK_MODEL_WEIGHT`, `RISK_EVIDENCE_WEIGHT`, `RISK_LOW_THRESHOLD`, and `RISK_MEDIUM_THRESHOLD` can override these defaults.

## Email configuration

Copy `.env.example` to `.env` and replace every empty value with real SMTP settings. Never commit `.env`. For Gmail, use `smtp.gmail.com`, port `587`, your Gmail address as `EMAIL_USERNAME`, and a Gmail App Password as `EMAIL_PASSWORD` (not your normal Gmail password). Set `AUTHORITY_EMAIL` to the authority's real receiving address. Restart Flask after changing `.env`.

For high-risk results, the result page shows the configured authority and sends a report containing the job details, final risk score, raw model estimate, and detected red flags only after the user confirms. If any SMTP values are missing, the page now lists the exact env keys that still need to be filled in.

## Project structure

`app.py` contains routes; `src\preprocessing.py` and `src\train_ml_models.py` contain the Random Forest training pipeline; `src\prediction.py` loads artifacts; `src\red_flags.py`, `src\scraper.py`, and `src\email_alert.py` handle analysis components; `templates\` and `static\` contain the interface.

## Limitations and future work

Scraping depends on a public page being accessible and readable. The baseline model can be extended with calibrated probabilities, structured features, cross-validation, and a separately trained BERT model. Any BERT result must be measured on the same held-out data before being shown.

The optional `src\train_bert.py` entry point fails clearly until the real EMSCAD schema is mapped and the optional `torch`, `transformers`, and `datasets` dependencies are installed. This prevents an untrained BERT model or invented BERT metrics from appearing in the dashboard.
