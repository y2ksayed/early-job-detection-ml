"""Fetch job postings from a job board API, with a safe fallback.

This is a demo scraper. It does three things on purpose:

1. It only calls an API if JOBS_API_URL is explicitly configured.
2. It respects a simple rate limit (max requests per minute).
3. If no API is configured, it copies the bundled sample data so the
   rest of the pipeline still runs end-to-end.

Do not point this at a site that forbids scraping. Check the board's
Terms of Service and robots.txt first, and use an official API when
one exists.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import requests

from config import SETTINGS


def fetch_from_api(limit: int) -> list[dict]:
    """Fetch postings from the configured API.

    The expected response is a list of job objects, or a dict with a
    "jobs" / "results" key. We normalize both shapes here so the
    downstream code always sees a plain list.
    """
    headers = {"Accept": "application/json"}
    if SETTINGS.jobs_api_key:
        headers["Authorization"] = f"Bearer {SETTINGS.jobs_api_key}"

    params = {"limit": limit, "sort": "date_posted_desc"}
    response = requests.get(
        SETTINGS.jobs_api_url,
        headers=headers,
        params=params,
        timeout=SETTINGS.request_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("jobs", "results", "data", "postings"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError("Unexpected API response shape — expected list or dict with jobs")


def normalize_job(raw: dict) -> dict:
    """Normalize one raw posting into our standard schema.

    We keep the schema small and explicit. Extra fields from the API
    are dropped — if you need them, add them here so everyone uses
    the same names downstream.
    """
    title = str(raw.get("title", "")).strip()[: SETTINGS.max_title_length]
    description = str(raw.get("description", "")).strip()
    description = description[: SETTINGS.max_description_length]

    # Skills may come as a list or a comma-separated string.
    skills = raw.get("skills", [])
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(",") if s.strip()]

    return {
        "id": str(raw.get("id", "")),
        "title": title,
        "company": str(raw.get("company", "Unknown")),
        "location": str(raw.get("location", "Remote")),
        "category": str(raw.get("category", "Data")),
        "description": description,
        "skills": skills,
        "posted_date": str(raw.get("posted_date", "")),
        "is_remote": bool(raw.get("is_remote", False)),
        # Demo labels — replaced by real outcomes in production.
        "is_emerging": int(bool(raw.get("is_emerging", 0))),
        "is_urgent": int(bool(raw.get("is_urgent", 0))),
    }


def load_sample_data() -> list[dict]:
    """Load the bundled sample postings."""
    sample_path = Path(SETTINGS.sample_data_path)
    with open(sample_path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch job postings")
    parser.add_argument("--out", default=SETTINGS.fetched_data_path)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not SETTINGS.jobs_api_url:
        # No API configured — this is the normal path for the demo.
        print("No JOBS_API_URL set, using sample data.")
        jobs = [normalize_job(j) for j in load_sample_data()]
        jobs = jobs[: args.limit]
    else:
        # Simple rate limiting: pause between calls.
        pause = 60.0 / max(1, SETTINGS.max_requests_per_minute)
        print(f"Fetching from {SETTINGS.jobs_api_url} ...")
        time.sleep(pause)
        raw_jobs = fetch_from_api(args.limit)
        jobs = [normalize_job(j) for j in raw_jobs]

    with open(out_path, "w") as f:
        json.dump(jobs, f, indent=2)

    print(f"Wrote {len(jobs)} postings to {out_path}")

    # Keep a copy of the raw sample for debugging if we used it.
    if not SETTINGS.jobs_api_url:
        print(f"Tip: edit {SETTINGS.sample_data_path} to try different postings.")


if __name__ == "__main__":
    main()
