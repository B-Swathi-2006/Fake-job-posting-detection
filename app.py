"""Flask entry point for the fake-job detection demonstration application."""
from __future__ import annotations

import os

from flask import Flask, flash, redirect, render_template, request, url_for

from src.email_alert import email_is_configured, missing_email_settings, send_alert
from src.prediction import ModelService, PredictionError
from src.red_flags import detect_red_flags
from src.risk_analysis import adjusted_probability, risk_level
from src.scraper import ScrapeError, scrape_job

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "local-development-only-key")
service = ModelService()


def _add_risk_result(prediction: dict, job: dict[str, str], flags: list[str]) -> dict:
    prediction["flags"] = flags
    prediction["model_probability"] = prediction["fraud_probability"]
    prediction["risk_score"] = adjusted_probability(prediction["model_probability"], flags)
    prediction["risk"] = risk_level(prediction["risk_score"])
    prediction["label"] = prediction["risk"]["label"]
    prediction["job"] = job
    return prediction


def _job_from_form(form) -> dict[str, str]:
    fields = (
        "title", "company", "salary", "location", "description", "requirements",
        "contact", "employment_type", "required_experience", "required_education",
        "industry", "benefits", "url",
    )
    return {field: form.get(field, "").strip() for field in fields}


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/about")
def about():
    return render_template("about.html")


@app.get("/dataset")
def dataset():
    return render_template("dataset.html")


@app.get("/comparison")
def comparison():
    return render_template("comparison.html", metrics=service.metrics, model_ready=service.ready)


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "GET":
        return render_template("analyze.html")

    job = _job_from_form(request.form)
    if job["url"]:
        try:
            scraped = scrape_job(job["url"])
            scraped["url"] = job["url"]
            job.update({key: value for key, value in scraped.items() if value})
        except ScrapeError as exc:
            flash(str(exc) + " You can paste the job details manually below.", "error")
            if not any(job[key] for key in ("title", "company", "description", "requirements")):
                return render_template("analyze.html", values=job)

    if not any(job[key] for key in ("title", "company", "description", "requirements", "salary", "contact")):
        flash("Enter a job URL or at least one job detail before analyzing.", "error")
        return render_template("analyze.html", values=job)

    try:
        prediction = service.predict(job)
    except PredictionError as exc:
        flash(str(exc), "error")
        return render_template("analyze.html", values=job)

    prediction = _add_risk_result(prediction, job, detect_red_flags(job))
    return render_template(
        "result.html",
        result=prediction,
        email_configured=email_is_configured(),
        email_missing_settings=missing_email_settings(),
        authority_email=os.getenv("AUTHORITY_EMAIL", ""),
    )


@app.post("/alert")
def alert():
    job = _job_from_form(request.form)
    flags = [item for item in request.form.getlist("flags") if item]
    try:
        prediction = service.predict(job)
        prediction = _add_risk_result(prediction, job, flags)
        send_alert(job, prediction)
    except (PredictionError, ValueError, RuntimeError) as exc:
        flash(str(exc), "error")
        return redirect(url_for("analyze"))
    flash("Alert sent successfully to the configured concerned authority.", "success")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
