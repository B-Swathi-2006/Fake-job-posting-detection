"""Explicitly triggered SMTP alert delivery using environment configuration."""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv

load_dotenv()


def missing_email_settings() -> list[str]:
    required = ("EMAIL_HOST", "EMAIL_PORT", "EMAIL_USERNAME", "EMAIL_PASSWORD", "AUTHORITY_EMAIL")
    return [key for key in required if not os.getenv(key)]


def email_is_configured() -> bool:
    return not missing_email_settings()


def send_alert(job: dict[str, str], prediction: dict) -> None:
    missing = missing_email_settings()
    if missing:
        raise RuntimeError("Email is not configured. Set " + ", ".join(missing) + " in your .env file.")
    message = EmailMessage()
    message["Subject"] = (
        f"High-Risk Job Posting Alert - {prediction['risk_score']:.1f}%"
    )
    message["From"] = os.environ["EMAIL_USERNAME"]
    message["To"] = os.environ["AUTHORITY_EMAIL"]
    model_probability = prediction.get("model_probability", prediction["fraud_probability"])
    message.set_content(
        "This posting was identified as potentially fraudulent/high risk by an automated "
        "screening system and is submitted for human verification.\n\n"
        f"Job title: {job.get('title', '')}\nCompany: {job.get('company', '')}\n"
        f"URL: {job.get('url', '')}\nLocation: {job.get('location', '')}\n"
        f"Final fraud risk score: {prediction['risk_score']:.1f}%\n"
        f"Random Forest model estimate: {model_probability:.1f}%\n"
        f"Risk level: {prediction['risk']['name']}\n"
        "Detected red flags:\n- " + "\n- ".join(prediction.get("flags", []))
    )
    with smtplib.SMTP(os.environ["EMAIL_HOST"], int(os.environ["EMAIL_PORT"]), timeout=10) as smtp:
        smtp.starttls()
        smtp.login(os.environ["EMAIL_USERNAME"], os.environ["EMAIL_PASSWORD"])
        smtp.send_message(message)
