"""Conservative public-page extraction; never bypasses access controls."""
from __future__ import annotations

import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


class ScrapeError(RuntimeError):
    """Raised when a public job page cannot be safely extracted."""


JOB_URL_MARKERS = re.compile(
    r"(?<![a-z])(?:job|jobs|career|careers|vacanc(?:y|ies)|position|opening|"
    r"employment|workday|requisition|hiring|apply|viewjob)(?![a-z])",
    re.IGNORECASE,
)


def _is_job_posting_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    path = parsed.path.rstrip("/")
    location = f"{path}?{parsed.query}" if parsed.query else path
    if not path or path.lower() in {"/home", "/index", "/index.html"}:
        return False
    return bool(JOB_URL_MARKERS.search(location))


def scrape_job(url: str) -> dict[str, str]:
    if not _is_job_posting_url(url):
        raise ScrapeError(
            "Please provide a job posting URL containing a job, career, vacancy, "
            "position, opening, or apply link."
        )
    try:
        response = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "FakeJobDetectionDemo/1.0"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ScrapeError("The job page could not be reached.") from exc
    return extract_job_html(response.text)


def extract_job_html(page_html: str) -> dict[str, str]:
    """Extract conservative job fields from already downloaded HTML."""
    soup = BeautifulSoup(page_html, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    title = soup.find("h1") or soup.find("title")
    body = " ".join(soup.stripped_strings)
    if not body:
        raise ScrapeError("The page did not contain readable job information.")
    result = {
        "title": title.get_text(" ", strip=True) if title else "",
        "description": body[:20000],
    }
    labels = {
        "company": ("company", "employer"),
        "salary": ("salary", "pay"),
        "location": ("location", "work from"),
        "contact": ("contact", "recruitment"),
    }
    for field, terms in labels.items():
        for element in soup.find_all(["strong", "b", "dt"]):
            label = element.get_text(" ", strip=True).lower().rstrip(":")
            if label not in terms:
                continue
            value = element.find_next(["br", "dd"])
            if value is not None and value.name == "br":
                value = value.next_sibling
            if value:
                extracted = value.get_text(" ", strip=True) if hasattr(value, "get_text") else str(value).strip()
                if extracted:
                    result[field] = extracted
                    break
    return result
