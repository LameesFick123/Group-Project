
from __future__ import annotations

import argparse
import html
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ADZUNA_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"

# Countries Adzuna's search endpoint supports
COUNTRIES = {
    "za": "South Africa", "us": "United States", "gb": "United Kingdom",
    "au": "Australia", "ca": "Canada", "in": "India", "de": "Germany",
    "fr": "France", "nl": "Netherlands", "sg": "Singapore", "nz": "New Zealand",
    "at": "Austria", "be": "Belgium", "br": "Brazil", "ch": "Switzerland",
    "es": "Spain", "it": "Italy", "mx": "Mexico", "pl": "Poland",
}

DEFAULT_TERMS = [
    "data scientist", "data analyst", "data engineer",
    "machine learning engineer", "AI engineer", "business analyst",
]

SNAPSHOT_COLUMNS = ["Source", "Job_ID", "Job_Title", "Description", "Country",
                    "Date", "Skills", "Search_term", "Fetched_at"]


class LiveDataError(Exception):
    """A problem the user can fix (bad keys, no network...)."""


# ----------------------------------------------------------------------------
# Skill detection
# ----------------------------------------------------------------------------
# The skill list contains everyday words ("flow", "sales", "word", "cloud"...).
# Matching those in free text would flood the results with false positives, so
# they are left out. Edit these sets if you want to change what is detected.
SKIPPED = {
    "flow", "arch", "wire", "crystal", "phoenix", "planner", "unify", "sheets", "word",
    "terminal", "cloud", "hardware", "legal", "sales", "marketing", "security", "embedded",
    "communication", "accounting", "recruiting", "robotics", "symphony", "unity", "spring",
    "express", "node", "assembly", "colocation", "sem", "wsl", "qt", "mlr", "outlook",
    "shell", "customer success", "financial analysis", "spreadsheet", "powerpoints",
    "no sql", "ibm cloud / red hat", "leaflet / folium",
}
# Real words or company names: only count when written with a capital letter
CAPITALISED_ONLY = {
    "excel", "swift", "rust", "ruby", "julia", "dart", "lua", "scala", "oracle", "slack",
    "zoom", "chef", "puppet", "yarn", "notion", "domo", "unreal", "prophet", "windows",
    "react", "spark", "go", "r", "c",
}
# Short acronyms: only count when written in capitals
UPPERCASE_ONLY = {"sas", "sap", "dax", "apl", "gis", "crm", "seo", "jax"}

_LANGS = r"(?:Python|SQL|SAS|MATLAB|Matlab|Java|Scala|Julia|Stata|SPSS|C\+\+|C#|Rust|Ruby|Kotlin|JavaScript)"
_JOIN = r"\s*(?:,|/|&|and|or)\s*"
# One-letter / very short names need their own context rules
SPECIAL = {
    "r": re.compile(
        r"(?<![A-Za-z0-9.&'/\-])R(?![A-Za-z0-9&'+#.\-])"
        r"(?:(?=\s*[,;/)\]])|(?=\.?\s*$)|(?=\s+(?:programming|language|studio|shiny|tidyverse|markdown)\b))"
        r"|" + _LANGS + _JOIN + r"R(?![A-Za-z0-9&'+#.\-])"
    ),
    "c": re.compile(
        r"(?<![A-Za-z0-9.&'/\-])C(?=\s+(?:programming|language)\b)"
        r"|" + _LANGS + _JOIN + r"C(?![A-Za-z0-9&'+#.\-/])"
    ),
    "go": re.compile(
        r"(?<![A-Za-z0-9])Go(?=\s+(?:programming|language|developer|engineer)\b)"
        r"|" + _LANGS + _JOIN + r"Go(?![A-Za-z0-9&'+#.\-])"
    ),
}


def _flexible(name: str) -> str:
    """'node.js' also matches 'NodeJS' and 'node js'; 'power bi' matches 'PowerBI'."""
    parts = [p for p in re.split(r"[\s\-_.]+", name) if p]
    return r"[\s\-_.]?".join(re.escape(p) for p in parts)


class SkillExtractor:
    """Finds known skills in free text (job title + description)."""

    def __init__(self, skills):
        self.patterns = []
        for skill in sorted(set(skills)):
            if skill in SKIPPED:
                continue
            if skill in SPECIAL:
                self.patterns.append((skill, SPECIAL[skill]))
                continue
            body = _flexible(skill)
            before, after = r"(?<![A-Za-z0-9.])", r"(?![A-Za-z0-9+#])"
            if skill in UPPERCASE_ONLY:
                rx = re.compile(before + re.escape(skill.upper()) + after)
            elif skill in CAPITALISED_ONLY:
                rx = re.compile(before + re.escape(skill.title()) + after)
            else:
                rx = re.compile(before + body + after, re.IGNORECASE)
            self.patterns.append((skill, rx))

    def find(self, text: str) -> list[str]:
        text = text or ""
        return [skill for skill, rx in self.patterns if rx.search(text)]


# ----------------------------------------------------------------------------
# Adzuna web API
# ----------------------------------------------------------------------------
_TAG = re.compile(r"<[^>]+>")


def clean_text(value) -> str:
    """Adzuna wraps search hits in <strong> tags and may include HTML entities."""
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", str(value or "")))).strip()


def parse_adzuna_item(item: dict, country: str, term: str, extractor: SkillExtractor) -> dict:
    title = clean_text(item.get("title"))
    description = clean_text(item.get("description"))
    created = pd.to_datetime(item.get("created"), errors="coerce", utc=True)
    if pd.isna(created):
        created = pd.Timestamp.now(tz="UTC")
    return {
        "Source": "adzuna",
        "Job_ID": str(item.get("id", "")),
        "Job_Title": title,
        "Description": description,
        "Country": COUNTRIES.get(country, country),
        "Date": created.strftime("%Y-%m-%d"),
        "Skills": "|".join(extractor.find(f"{title}. {description}")),
        "Search_term": term,
        "Fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


@dataclass
class FetchResult:
    postings: pd.DataFrame
    calls: int = 0
    warnings: list = field(default_factory=list)


def fetch_adzuna(app_id, app_key, country, terms, extractor, pages=2, per_page=50,
                 max_days_old=60, pause=0.4, timeout=20, session=None) -> FetchResult:
    """Download recent postings for each search term and detect skills in them."""
    if not app_id or not app_key:
        raise LiveDataError("Adzuna app_id / app_key are missing.")
    if country not in COUNTRIES:
        raise LiveDataError(f"Adzuna does not support country code '{country}'.")
    session = session or requests.Session()
    rows, calls, warnings = [], 0, []

    for term in terms:
        term = term.strip()
        if not term:
            continue
        for page in range(1, pages + 1):
            params = {
                "app_id": app_id, "app_key": app_key, "what": term,
                "results_per_page": per_page, "sort_by": "date",
                "max_days_old": max_days_old,
            }
            try:
                resp = session.get(ADZUNA_URL.format(country=country, page=page), params=params,
                                   headers={"Accept": "application/json"}, timeout=timeout)
            except requests.RequestException as exc:
                warnings.append(f"'{term}' page {page}: network problem ({type(exc).__name__}).")
                break
            calls += 1
            if resp.status_code in (401, 403):
                raise LiveDataError("Adzuna rejected the app_id / app_key. Check them in your secrets file.")
            if resp.status_code == 429:
                warnings.append("Adzuna rate limit reached, so the fetch stopped early. "
                                "What was downloaded so far is kept. Try again later.")
                return FetchResult(_to_frame(rows), calls, warnings)
            if not resp.ok:
                warnings.append(f"'{term}' page {page}: Adzuna returned HTTP {resp.status_code}.")
                break
            try:
                results = resp.json().get("results", [])
            except ValueError:
                warnings.append(f"'{term}' page {page}: unreadable response.")
                break
            rows.extend(parse_adzuna_item(r, country, term, extractor) for r in results)
            if len(results) < per_page:
                break
            time.sleep(pause)
    return FetchResult(_to_frame(rows), calls, warnings)


def _to_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)


# ----------------------------------------------------------------------------
# Snapshot file (history that builds up each time you fetch)
# ----------------------------------------------------------------------------
def read_snapshot(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return _to_frame([])
    return pd.read_csv(path, dtype=str, keep_default_na=False)[SNAPSHOT_COLUMNS]


def append_snapshot(path, new: pd.DataFrame):
    """Add new postings, skipping ones already stored. Returns (total, added)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = read_snapshot(path)
    combined = pd.concat([old, new.astype(str)], ignore_index=True)
    combined = combined[combined["Job_ID"] != ""]
    combined = combined.drop_duplicates(subset=["Source", "Job_ID"], keep="first")
    combined.to_csv(path, index=False)
    return len(combined), len(combined) - len(old)


# ----------------------------------------------------------------------------
# Converting live postings into the same tables the stored files give
# ----------------------------------------------------------------------------
def derive_tables(snapshot: pd.DataFrame, framework_map=None, labels=None) -> dict:
    """Returns dict with postings, skill_count, skill_job and matches tables."""
    framework_map, labels = framework_map or {}, labels or {}
    df = snapshot.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Month"] = df["Date"].dt.strftime("%Y-%m")
    df["Title_key"] = df["Job_Title"].astype(str).str.strip().str.casefold()
    df["Skills"] = df["Skills"].fillna("")

    postings = df[["Job_ID", "Job_Title", "Skills", "Country", "Date", "Month", "Title_key"]].reset_index(drop=True)
    exploded = postings.assign(Skill=postings["Skills"].str.split("|")).explode("Skill")
    exploded = exploded[exploded["Skill"].notna() & (exploded["Skill"] != "")].copy()
    exploded["Label"] = exploded["Skill"].map(lambda s: labels.get(s, s))

    skill_count = (exploded.groupby(["Month", "Label"]).size().reset_index(name="Posting_count")
                   .rename(columns={"Label": "Skill"}))
    skill_job = (exploded.groupby(["Month", "Label", "Job_Title", "Title_key"]).size()
                 .reset_index(name="Posting_count").rename(columns={"Label": "Skill"}))
    matches = exploded.groupby("Skill").size().reset_index(name="mentions").rename(columns={"Skill": "skill_clean"})
    matches["framework"] = matches["skill_clean"].map(lambda s: framework_map.get(s, "NONE"))
    matches["method"] = "live"
    return {"postings": postings, "skill_count": skill_count, "skill_job": skill_job,
            "matches": matches[["skill_clean", "framework", "method", "mentions"]]}


# ----------------------------------------------------------------------------
# Command line: python live_data.py --country za --pages 3
# ----------------------------------------------------------------------------
def _main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Fetch live job postings into the dashboard's snapshot file.")
    parser.add_argument("--country", default="za", choices=sorted(COUNTRIES))
    parser.add_argument("--terms", default=",".join(DEFAULT_TERMS), help="comma-separated search terms")
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--skills-file", default=None, help="path to all_skill_matches.csv")
    parser.add_argument("--out", default=str(here / "live_cache" / "live_postings.csv"))
    args = parser.parse_args()

    skills_file = args.skills_file or next(
        (str(p) for p in (here / "all_skill_matches.csv", here / "data" / "all_skill_matches.csv") if p.exists()), None)
    if not skills_file:
        raise SystemExit("Could not find all_skill_matches.csv. Pass --skills-file.")
    skills = pd.read_csv(skills_file)["skill_clean"].dropna().unique().tolist()

    result = fetch_adzuna(os.environ.get("ADZUNA_APP_ID", ""), os.environ.get("ADZUNA_APP_KEY", ""),
                          args.country, args.terms.split(","), SkillExtractor(skills),
                          pages=args.pages, max_days_old=args.days)
    total, added = append_snapshot(args.out, result.postings)
    print(f"{result.calls} API calls, {added} new postings added, {total} stored in {args.out}")
    for w in result.warnings:
        print("warning:", w)


if __name__ == "__main__":
    _main()
