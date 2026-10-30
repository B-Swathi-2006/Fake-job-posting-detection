"""Transparent, evidence-based red-flag checks."""
from __future__ import annotations

import re


def detect_red_flags(job: dict[str, str]) -> list[str]:
    text = " ".join(str(value or "") for value in job.values()).lower()
    flags: list[str] = []
    if re.search(
        r"\b(?:registration|application|training|security|activation|account)"
        r"(?:\s+\w+){0,3}\s+(?:fee|payment|deposit|charge)\b|"
        r"\b(?:pay|send|submit)\s+(?:a\s+)?(?:fee|deposit|payment)\b|"
        r"\bupfront\s+(?:fee|payment|deposit)\b|\bbefore\s+(?:your\s+)?first\s+payout\b",
        text,
    ):
        flags.append("Payment or fee language detected")
    if re.search(
        r"\b(?:urgent(?:ly)?|immediate(?:ly)?|hiring\s+now|act\s+now|"
        r"limited\s+(?:spots|openings)|only\s+\d+\s+(?:spots|positions)|"
        r"next\s+\d+\s+minutes?)\b",
        text,
    ):
        flags.append("Urgent hiring language detected")
    if re.search(
        r"\b(?:earn|salary|income|pay|paid|make)\b.{0,60}\b"
        r"(?:huge|high|guaranteed|unlimited|easy\s+money|daily|weekly)\b|"
        r"\b(?:guaranteed|easy)\s+(?:income|money|earnings|cash)\b",
        text,
    ):
        flags.append("Unusually attractive income claims detected")
    no_experience = re.search(
        r"\b(?:no\s+experience|no\s+skills?\s+required|no\s+degree|no\s+interview)\b",
        text,
    )
    high_income = re.search(
        r"\b(?:high|guaranteed|unlimited)\s+(?:salary|income|pay)|"
        r"\b(?:earn|make)\s+\$?\d[\d,]*(?:\s+per\s+(?:day|week|month))?\b|"
        r"\b(?:easy\s+money|paid\s+every\s+(?:day|evening|week))\b",
        text,
    )
    if no_experience and high_income:
        flags.append("High income combined with no experience requirement")
    if re.search(
        r"\b(?:passport|bank\s+account|bank\s+details?|credit\s+card|otp|"
        r"aadhaar|ssn|upi|payment\s+screenshot|one[-\s]?time\s+password)\b",
        text,
    ):
        flags.append("Sensitive personal or financial information requested")
    social_job = re.search(
        r"\b(?:social\s+media|instagram|tiktok|youtube|facebook|like\s+and\s+share|"
        r"follower|influencer|content\s+promot(?:er|ion)|brand\s+ambassador)\b",
        text,
    )
    if social_job and (high_income or no_experience):
        flags.append("Social-media job makes unusually broad earnings or experience claims")
    company = str(job.get("company", "") or "").strip()
    if not company or re.search(r"\b(?:confidential|undisclosed|private)\s+(?:company|employer)\b", company, re.I):
        flags.append("Company information is missing or unverifiable (verify independently)")
    contact = str(job.get("contact", "") or "").strip()
    generic_contact = re.search(r"@(gmail|yahoo|outlook|hotmail)\.", contact, re.I)
    informal_contact = re.search(
        r"\b(?:contact|message)\s+(?:me\s+)?(?:on|via)\s+(?:telegram|whatsapp|signal)\b|"
        r"\b(?:telegram|whatsapp|signal)\s+(?:dm|message|contact)\s+only\b",
        text,
    )
    if (generic_contact and (not company or social_job)) or (informal_contact and (social_job or high_income or no_experience)):
        flags.append("Suspicious recruiter contact method detected")
    if not flags:
        flags.append("No rule-based red flags detected; review the model result carefully")
    return flags
