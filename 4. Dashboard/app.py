import csv
import hmac
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import live_data

st.set_page_config(page_title="AI Job Market Dashboard", page_icon="📊", layout="wide")

BASE = Path(__file__).parent
FILES = {
    "postings": "Fixed.csv",
    "skill_count": "Skill_Count.txt",
    "skill_job": "Skill_Job.txt",
    "matches": "all_skill_matches.csv",
    "titles": "job_title_results_filled.csv",
}
# The team's cleaned dataset (read by the Prophet, EDA and modelling notebooks). Any of these names works.
EDA_FILES = ("Final_cleanedDataset.csv", "Combined_Datasets_EDA 1.xls", "Combined_Datasets_EDA.xls")
MODEL_FILES = ("Combined_Datasets_EDA 1.xls", "Combined_Datasets_EDA.xls", "Final_cleanedDataset.csv")
LIVE_FILE = BASE / "live_cache" / "live_postings.csv"
STORED, LIVE = "Stored dataset", "Live data"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def find_file(name):
    """Look for a data file next to app.py, then in a 'data' folder."""
    for folder in (BASE, BASE / "data"):
        path = folder / name
        if path.exists():
            return str(path)
    return None


def require(*keys):
    """Return the paths of the files a page needs, or show an error and stop."""
    paths, missing = [], []
    for key in keys:
        path = find_file(FILES[key])
        if path is None:
            missing.append(FILES[key])
        else:
            paths.append(path)
    if missing:
        st.error(
            "Missing file(s): " + ", ".join(missing)
            + ". Put them in the same folder as app.py (or in a folder called 'data' next to it)."
        )
        st.stop()
    return paths


def title_key(s):
    return s.astype(str).str.strip().str.casefold()


def show(fig, height=None):
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch")


def month_range(months, key):
    """Sidebar slider that returns (first_month, last_month) as 'YYYY-MM' text."""
    if len(months) < 2:
        st.sidebar.caption(f"Only one month of data so far: {months[0]}")
        return months[0], months[0]
    return st.sidebar.select_slider(
        "Months", options=months, value=(months[0], months[-1]), key=key
    )


def tfidf_cosine(matrix, row):
    """Cosine similarity of one row against all rows after TF-IDF weighting.
    Same maths as sklearn's TfidfTransformer + cosine_similarity."""
    x = matrix.values.astype(float)
    n = x.shape[0]
    doc_freq = (x > 0).sum(axis=0)
    idf = np.log((1 + n) / (1 + doc_freq)) + 1
    t = x * idf
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    t = t / np.where(norms == 0, 1, norms)
    return t @ t[row]


def display_names(sj):
    """Map lower-case job title -> the most common way it is written."""
    best = (sj.groupby(["Title_key", "Job_Title"])["Posting_count"].sum()
              .reset_index().sort_values("Posting_count", ascending=False)
              .drop_duplicates("Title_key"))
    return dict(zip(best["Title_key"], best["Job_Title"]))


# ----------------------------------------------------------------------------
# Data loaders (cached so files are only read once)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading postings...")
def load_postings(path):
    """Fixed.csv. Some job titles contain ';', which shifts the later columns.
    Reading the last four fields from the right repairs those rows."""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader, None)
        for parts in reader:
            while parts and parts[-1].strip() == "":
                parts.pop()
            if len(parts) < 5:
                continue
            title = ", ".join(p.strip() for p in parts[:-4])
            rows.append((title, parts[-4].strip(), parts[-3].strip(),
                         parts[-2].strip(), parts[-1].strip()))
    df = pd.DataFrame(rows, columns=["Job_Title", "Skills", "Work_mode", "Country", "Date"])
    df["Date"] = pd.to_datetime(df["Date"], format="%Y/%m/%d", errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    df["Month"] = df["Date"].dt.strftime("%Y-%m")
    df["Title_key"] = title_key(df["Job_Title"])
    return df


@st.cache_data(show_spinner="Loading skill counts...")
def load_skill_count(path):
    df = pd.read_csv(path, index_col=0)
    df["Skill"] = df["Skill"].str.strip()
    df["Month"] = df["Month_year"].astype(str).str.strip()
    return df[["Month", "Skill", "Posting_count"]]


@st.cache_data(show_spinner="Loading job-skill data...")
def load_skill_job(path):
    df = pd.read_csv(path, index_col=0)
    df["Skill"] = df["Skill"].str.strip()
    df["Month"] = df["Month_year"].astype(str).str.strip()
    df["Title_key"] = title_key(df["Job_Title"])
    return df[["Month", "Skill", "Job_Title", "Title_key", "Posting_count"]]


@st.cache_data
def load_matches(path):
    df = pd.read_csv(path)
    df["framework"] = df["framework"].fillna("NONE").astype(str).str.strip()
    # a skill listed twice (e.g. svn, two O*NET codes) must only be counted once
    return df.groupby("skill_clean", as_index=False).agg(
        framework=("framework", "first"),
        method=("method", "first"),
        mentions=("mentions", "max"),
    )


@st.cache_data
def load_titles(path):
    df = pd.read_csv(path)
    df["framework"] = df["framework"].fillna("NONE").astype(str).str.strip()
    return df.groupby("title_clean", as_index=False).agg(
        framework=("framework", "first"), postings=("postings", "sum"))


def postings_per_title(fx, m0, m1):
    sub = fx[(fx["Month"] >= m0) & (fx["Month"] <= m1)]
    return sub.groupby("Title_key").size()


# ----------------------------------------------------------------------------
# Stored vs live data
# ----------------------------------------------------------------------------
def pick_source(page_key):
    """Sidebar switch between the stored files and live postings. The choice
    is remembered when you move between pages."""
    options = [STORED, LIVE]
    current = st.session_state.get("source", STORED)
    choice = st.sidebar.radio("Data source", options, index=options.index(current),
                              key=f"source_{page_key}")
    st.session_state["source"] = choice
    if choice == LIVE:
        st.sidebar.caption("Live postings: Jobs by Adzuna. Fetch them on the Live data page.")
    return choice


@st.cache_data(show_spinner="Preparing live data...")
def build_live_tables(snapshot_path, snapshot_mtime, matches_path, count_path):
    snapshot = live_data.read_snapshot(snapshot_path)
    matches = load_matches(matches_path)
    framework_map = dict(zip(matches["skill_clean"], matches["framework"]))
    labels = {}
    if count_path:
        # reuse the nicer capitalisation from Skill_Count.txt (e.g. 'SQL', 'Scikit-learn')
        norm = lambda s: s.strip().casefold().replace("-", " ").replace(".", " ").replace("_", " ")
        stored = {norm(s): s for s in load_skill_count(count_path)["Skill"].unique()}
        labels = {k: stored[norm(k)] for k in matches["skill_clean"] if norm(k) in stored}
    return live_data.derive_tables(snapshot, framework_map, labels)


def live_tables():
    snap = Path(LIVE_FILE)
    if not snap.exists() or snap.stat().st_size == 0:
        st.info("No live data stored yet. Open the **Live data** page and fetch postings first.")
        st.stop()
    (matches_path,) = require("matches")
    tables = build_live_tables(str(snap), snap.stat().st_mtime, matches_path,
                               find_file(FILES["skill_count"]) or "")
    if tables["postings"].empty:
        st.warning("The stored live postings have no usable dates.")
        st.stop()
    return tables


def get_data(source, *needs):
    """Return the requested tables (matches, skill_count, skill_job, postings)
    from either the stored files or the live snapshot."""
    if source == LIVE:
        tables = live_tables()
        return [tables[k] for k in needs]
    loaders = {"matches": load_matches, "skill_count": load_skill_count,
               "skill_job": load_skill_job, "postings": load_postings}
    paths = require(*needs)
    return [loaders[k](p) for k, p in zip(needs, paths)]


def live_note():
    st.caption("Live data comes from Adzuna (Jobs by Adzuna). Skills are detected in the job title "
               "and the short description excerpt the API returns, so live counts are lower than "
               "the stored dataset, which used full descriptions, and words in a job's own title count "
               "(a Machine Learning Engineer posting counts as mentioning machine learning). "
               "Compare patterns, not exact numbers.")


# ----------------------------------------------------------------------------
# Cleaned dataset, country cleaning and the notebook charts (explorer page)
# ----------------------------------------------------------------------------
NOT_STATED = "Not stated"
US_PLACES = {"Missouri", "Kansas", "Shawnee County, KS"}
COUNTRY_ALIASES = {
    'US': 'United States', 'USA': 'United States', 'UK': 'United Kingdom', 'MX': 'Mexico',
    'ZA': 'South Africa', 'FI': 'Finland', 'NO': 'Norway', 'HK': 'Hong Kong',
    'PR': 'Puerto Rico', 'VI': 'United States Virgin Islands',
    'U.S. Virgin Islands': 'United States Virgin Islands', 'Czechia': 'Czech Republic',
    'Macedonia (FYROM)': 'North Macedonia', "Côte d'Ivoire": 'Ivory Coast',
}
COUNTRY_ISO3 = {
    'United States': 'USA', 'Canada': 'CAN', 'Germany': 'DEU', 'United Kingdom': 'GBR',
    'Australia': 'AUS', 'India': 'IND', 'Singapore': 'SGP', 'Mexico': 'MEX',
    'South Africa': 'ZAF', 'Finland': 'FIN', 'Norway': 'NOR', 'Hong Kong': 'HKG',
    'Puerto Rico': 'PRI', 'United States Virgin Islands': 'VIR', 'Romania': 'ROU',
    'Belgium': 'BEL', 'Turkey': 'TUR', 'Kenya': 'KEN', 'Hungary': 'HUN', 'Vietnam': 'VNM',
    'China': 'CHN', 'Japan': 'JPN', 'Ukraine': 'UKR', 'Indonesia': 'IDN', 'Poland': 'POL',
    'Portugal': 'PRT', 'Israel': 'ISR', 'Netherlands': 'NLD', 'Egypt': 'EGY', 'France': 'FRA',
    'Nigeria': 'NGA', 'Costa Rica': 'CRI', 'Philippines': 'PHL', 'Argentina': 'ARG',
    'Bulgaria': 'BGR', 'Italy': 'ITA', 'Brazil': 'BRA', 'Czech Republic': 'CZE',
    'Taiwan': 'TWN', 'United Arab Emirates': 'ARE', 'Spain': 'ESP', 'Colombia': 'COL',
    'Thailand': 'THA', 'Bangladesh': 'BGD', 'Ireland': 'IRL', 'Pakistan': 'PAK',
    'Malaysia': 'MYS', 'Sweden': 'SWE', 'Saudi Arabia': 'SAU', 'Morocco': 'MAR',
    'Lithuania': 'LTU', 'Serbia': 'SRB', 'Denmark': 'DNK', 'South Korea': 'KOR',
    'Switzerland': 'CHE', 'Austria': 'AUT', 'Estonia': 'EST', 'New Zealand': 'NZL',
    'Greece': 'GRC', 'Peru': 'PER', 'Slovenia': 'SVN', 'Uruguay': 'URY', 'Latvia': 'LVA',
    'Chile': 'CHL', 'Croatia': 'HRV', 'Ghana': 'GHA', 'Slovakia': 'SVK', 'Sudan': 'SDN',
    'Luxembourg': 'LUX', 'Uzbekistan': 'UZB', 'Guatemala': 'GTM', 'Jordan': 'JOR',
    'Cyprus': 'CYP', 'Qatar': 'QAT', 'Bahrain': 'BHR', 'Uganda': 'UGA', 'Kyrgyzstan': 'KGZ',
    'Malta': 'MLT', 'North Macedonia': 'MKD', 'Lebanon': 'LBN', 'Jamaica': 'JAM',
    'Russia': 'RUS', 'Belarus': 'BLR', 'Sri Lanka': 'LKA', 'Honduras': 'HND', 'Tunisia': 'TUN',
    'Dominican Republic': 'DOM', 'El Salvador': 'SLV', 'Kuwait': 'KWT', 'Nicaragua': 'NIC',
    'Ecuador': 'ECU', 'Moldova': 'MDA', 'Mauritius': 'MUS', 'Panama': 'PAN',
    'Kazakhstan': 'KAZ', 'Armenia': 'ARM', 'Iraq': 'IRQ', 'Namibia': 'NAM', 'Guam': 'GUM',
    'Nepal': 'NPL', 'Zimbabwe': 'ZWE', 'Trinidad and Tobago': 'TTO', 'Montenegro': 'MNE',
    'Suriname': 'SUR', 'Oman': 'OMN', 'Algeria': 'DZA', 'Albania': 'ALB', 'Mongolia': 'MNG',
    'Cambodia': 'KHM', 'Ivory Coast': 'CIV', 'Iran': 'IRN',
    'Iran, Islamic Republic of...': 'IRN', 'Ethiopia': 'ETH', 'Hong Kong (S.A.R.)': 'HKG',
    'Viet Nam': 'VNM', 'Russian Federation': 'RUS',
}


def clean_country(value):
    """Same rules as the EDA notebook: merge USA, US cities/states and spelling variants."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value in ("Anywhere", "nan", ""):
        return None
    if "," in value and not value.startswith("United States"):
        suffix = value.split(",")[-1].strip()
        if len(suffix) == 2 and suffix.isupper():
            return "United States"
    if value.startswith("United States") or value in US_PLACES:
        return "United States"
    return COUNTRY_ALIASES.get(value, value)


def tidy_country(series):
    return series.map(clean_country).fillna(NOT_STATED)


def find_first(names):
    for name in names:
        path = find_file(name)
        if path:
            return path
    return None


@st.cache_data(show_spinner="Loading postings...")
def load_cleaned(path):
    """The team's cleaned dataset (Final_cleanedDataset.csv / Combined_Datasets_EDA)."""
    df = pd.read_csv(path, dtype=str)
    if df.shape[1] == 1:
        df = pd.read_csv(path, dtype=str, sep=";")
    needed = ["Job_Title", "Skills", "Work_mode", "Country", "Posting_date"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError("missing columns: " + ", ".join(missing))
    out = df[needed].rename(columns={"Posting_date": "Date"})
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"]).reset_index(drop=True)
    out["Job_Title"] = out["Job_Title"].fillna("").astype(str)
    out["Work_mode"] = out["Work_mode"].fillna(NOT_STATED)
    out["Country"] = tidy_country(out["Country"])
    out["Month"] = out["Date"].dt.strftime("%Y-%m")
    out["Title_key"] = title_key(out["Job_Title"])
    return out


@st.cache_data(show_spinner="Loading postings...")
def load_fixed_for_explorer(path):
    df = load_postings(path).copy()
    df["Country"] = tidy_country(df["Country"])
    return df


def main_postings(required=True):
    """Postings used by the front page and explorer: the cleaned file the notebooks
    read if it is in the folder, otherwise Fixed.csv. Returns (data, file name)."""
    path = find_first(EDA_FILES)
    if path:
        try:
            return load_cleaned(path), Path(path).name
        except ValueError as exc:
            st.warning(f"{Path(path).name} could not be used ({exc}). Using {FILES['postings']} instead.")
    fixed = find_file(FILES["postings"])
    if fixed is None:
        if required:
            require("postings")  # shows the error and stops the page
        return None, None
    return load_fixed_for_explorer(fixed), FILES["postings"]


def monthly_counts(dates):
    """Postings per month (month-start dates), as in the Prophet notebook."""
    counts = pd.Series(1, index=pd.DatetimeIndex(dates.dropna().values)).resample("MS").size()
    return pd.DataFrame({"ds": counts.index, "y": counts.values})


@st.cache_data(show_spinner="Fitting the forecast model...")
def prophet_forecast(ds, y, horizon, mode="multiplicative", width=0.95):
    """Prophet fitted as in the notebook: history plus `horizon` future months."""
    import logging
    for name in ("cmdstanpy", "prophet"):
        logging.getLogger(name).setLevel(logging.WARNING)
    from prophet import Prophet

    ts = pd.DataFrame({"ds": pd.to_datetime(list(ds)), "y": list(y)})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                    seasonality_mode=mode, interval_width=width)
    model.fit(ts)
    np.random.seed(42)  # Prophet simulates its interval; a fixed seed keeps the numbers stable
    out = model.predict(model.make_future_dataframe(periods=horizon, freq="MS"))
    return out[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def forecast_chart(ts, fc, horizon=24):
    """The notebook's 'Historical Actuals vs 2-Year Prophet Projection' chart."""
    last, end = ts["ds"].max(), fc["ds"].max()
    top = float(max(fc["yhat_upper"].max(), ts["y"].max()) * 1.05)
    bottom = float(max(0, min(fc["yhat_lower"].min(), ts["y"].min()) * 0.9))
    blue = "#0072B2"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_lower"], fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(0,114,178,0.2)", name="Uncertainty interval", legendrank=3))
    fig.add_trace(go.Scatter(x=[last, end], y=[top, top], fill="tozeroy", line=dict(width=0),
                             fillcolor="rgba(255,0,0,0.3)", name=f"{horizon // 12}-Year Forecast Window ({horizon} Months)",
                             hoverinfo="skip", legendrank=4))
    fig.add_trace(go.Scatter(x=[last, last], y=[bottom, top], mode="lines", line=dict(color="gray", dash="dash", width=2),
                             name=f"Forecast Start ({last:%Y-%m})", hoverinfo="skip", legendrank=5))
    fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines", line=dict(color=blue, width=2.2),
                             name="Forecast", legendrank=2))
    fig.add_trace(go.Scatter(x=ts["ds"], y=ts["y"], mode="markers", marker=dict(color="black", size=6),
                             name="Observed data points", legendrank=1))
    fig.update_layout(title=dict(text="<b>Aggregate Job Postings: Historical Actuals vs 2-Year Prophet Projection</b>"),
                      xaxis_title="Date", yaxis_title="Total Monthly Job Postings",
                      yaxis=dict(range=[bottom, top]),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    return fig


@st.cache_data(show_spinner="Training the models...")
def skill_direction_results(path):
    """From forest_linear.ipynb: can a model predict whether a skill's mention rate rises next month?
    Trained on the older months and tested on the final 12."""
    import re
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    jobs = pd.read_csv(path, dtype=str)
    if jobs.shape[1] == 1:
        jobs = pd.read_csv(path, dtype=str, sep=";")
    if "Posting_date" not in jobs.columns or "Skills" not in jobs.columns:
        raise ValueError("the file needs Posting_date and Skills columns")
    jobs["date"] = pd.to_datetime(jobs["Posting_date"], errors="coerce")
    jobs = jobs[(jobs["date"] >= "2021-01-01") & (jobs["date"] < "2026-07-01")].copy()
    jobs["month"] = jobs["date"].dt.to_period("M").dt.to_timestamp()

    def tidy_skill(s):
        s = re.sub(r"[-_]", " ", str(s).lower())
        return re.sub(r"\s+", " ", s).strip()

    def split_skills(text):
        text = str(text)
        found = re.findall(r"'([^']*)'", text) if "'" in text else text.strip("[]").split(",")
        return list(dict.fromkeys(tidy_skill(x) for x in found if x.strip() != ""))

    split_map = {s: split_skills(s) for s in jobs["Skills"].dropna().unique()}
    jobs["skill_list"] = jobs["Skills"].map(split_map)
    jobs = jobs.dropna(subset=["skill_list"])
    adverts_per_month = jobs.groupby("month").size()
    long = jobs[["month", "skill_list"]].explode("skill_list").dropna()
    counts = long.groupby(["month", "skill_list"]).size().unstack(fill_value=0)
    share = counts.div(adverts_per_month, axis=0) * 100
    top_skills = long["skill_list"].value_counts().head(10).index.tolist()

    rows = []
    for skill in top_skills:
        s = share[skill]
        df = pd.DataFrame({"share": s})
        df["change1"] = s.diff(1)
        df["change2"] = s.diff(2)
        df["roll3"] = s.rolling(3).mean()
        df["gap_to_roll3"] = s - df["roll3"]
        df["month_no"] = s.index.month
        df["target"] = (s.shift(-1) > s).astype(int)
        df = df.iloc[:-1].dropna()
        df["skill"] = skill
        rows.append(df)
    data = pd.concat(rows)
    features = ["share", "change1", "change2", "roll3", "gap_to_roll3", "month_no"]
    months = sorted(data.index.unique())
    if len(months) < 24:
        raise ValueError("not enough months to train and test")
    cut = months[-12]
    train, test = data[data.index < cut], data[data.index >= cut]

    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    logit.fit(train[features], train["target"])
    forest = RandomForestClassifier(n_estimators=300, random_state=42)
    forest.fit(train[features], train["target"])
    majority = int(train["target"].mean() >= 0.5)

    acc = np.array([
        round(accuracy_score(test["target"], [majority] * len(test)) * 100, 1),
        round(accuracy_score(test["target"], logit.predict(test[features])) * 100, 1),
        round(accuracy_score(test["target"], forest.predict(test[features])) * 100, 1),
    ])
    n = len(test)
    ci = 1.96 * np.sqrt((acc / 100) * (1 - acc / 100) / n) * 100
    return {"acc": acc.tolist(), "ci": ci.tolist(), "test_rows": n, "train_rows": len(train),
            "skills": len(top_skills), "months": 12, "top_skills": top_skills}


ACCURACY_LABELS = ["Always guess<br>the most common", "Logistic regression<br>(baseline)", "Random forest<br>(main model)"]


def accuracy_text(res):
    names = ["Always guessing the most common", "Logistic regression", "Random forest"]
    acc, ci = res["acc"], res["ci"]
    # the headline judges the two real models; the first bar is only a baseline
    clear = [(a, name) for name, a, c in zip(names[1:], acc[1:], ci[1:]) if a - c > 50]
    title = (f"On the final {res['months']} months, {max(clear)[1]} beats a coin flip" if clear
             else f"On the final {res['months']} months, no model beats a coin flip")
    line2 = f"Accuracy on {res['months']} unseen months | {res['skills']} skills, {res['test_rows']} predictions"
    includes_50 = all(a - c <= 50 <= a + c for a, c in zip(acc, ci))
    line3 = ("Every range includes 50%, but this is one test period." if includes_50
             else "This is one test period, so treat the result with caution.")
    return title, line2, line3


def accuracy_chart(res):
    acc, ci = np.array(res["acc"]), np.array(res["ci"])
    fig = go.Figure(go.Bar(
        x=ACCURACY_LABELS, y=acc, width=0.55, marker_color=["#B8BEC9", "#4C78A8", "#E8833A"],
        text=[f"{v:.1f}%" for v in acc], textposition="inside", insidetextanchor="start",
        textfont=dict(size=26, color="white"),
        error_y=dict(type="data", array=ci, color="#444444", thickness=1.6, width=8),
        hovertemplate="%{y:.1f}%<extra></extra>"))
    fig.add_hline(y=50, line_dash="dash", line_color="#6B7280", annotation_text="coin flip",
                  annotation_position="top right")
    fig.update_yaxes(range=[0, 75], tickvals=list(range(0, 71, 10)),
                     ticktext=[f"{t}%" for t in range(0, 71, 10)])
    fig.update_layout(showlegend=False)
    return fig


def country_map(counts):
    """The notebook's 'Job Postings by Country' map. Returns (figure, postings left off the map)."""
    frame = pd.DataFrame({"Country": counts.index, "Postings": counts.values})
    frame["iso"] = frame["Country"].map(COUNTRY_ISO3)
    left_off = int(frame.loc[frame["iso"].isna(), "Postings"].sum())
    placed = frame.dropna(subset=["iso"])
    fig = px.choropleth(placed, locations="iso", locationmode="ISO-3", color="Postings",
                        hover_name="Country", color_continuous_scale="Plasma",
                        title="<b>Job Postings by Country</b>", template="plotly_white")
    fig.update_geos(showocean=True, oceancolor="#fafaf9", showland=True, landcolor="#e7e5e4",
                    showcoastlines=True, coastlinecolor="#d6d3d1", projection_type="natural earth")
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0),
                      coloraxis_colorbar=dict(title="Postings", thickness=14, len=0.6))
    return fig, left_off


# ----------------------------------------------------------------------------
# Front page - Forecasting the future
# ----------------------------------------------------------------------------
# Edit the wording here
HOME_TAGLINE = "Which skills will employers ask for next?"
HOME_INTRO = (
    "AI is changing the skills employers look for. This dashboard studies real job postings "
    "to show which skills are in demand, how that demand is changing over time, and where it "
    "may be heading, to help students, job seekers and employers plan their next move."
)
HOME_ABOUT = ("It aims to close the knowledge gap that has been created due to the impact "
              "AI has had on the job market.")
HOME_CARDS = [
    ("Posting explorer", "Forecast, work mode, country and skill-model charts, with filters for month, country and job title."),
    ("Job overviews", "The most and least posted job titles."),
    ("Skills overview", "The most and least requested skills."),
    ("Skill trends", "How demand for chosen skills moves over time, with a forecast of where each is heading."),
    ("Analysis of Skills per Job", "What a job title asks for, and how two roles compare."),
    ("Similarity Calculation", "Roles with a similar skill mix, and how their top skills compare."),
    ("Live data", "Fetch current postings from the Adzuna API."),
]


def page_home():
    st.title("Forecasting the future")
    st.subheader(HOME_TAGLINE)
    st.write(HOME_INTRO)

    numbers = []
    postings, _ = main_postings(required=False)
    matches_path = find_file(FILES["matches"])
    if postings is not None:
        numbers.append(("Job postings analysed", f"{len(postings):,}"))
    if matches_path:
        numbers.append(("Skills tracked", f"{len(load_matches(matches_path)):,}"))
    if postings is not None:
        numbers.append(("Distinct job titles", f"{postings['Title_key'].nunique():,}"))
    if numbers:
        for column, (label, value) in zip(st.columns(len(numbers)), numbers):
            column.metric(label, value)

    st.divider()
    st.subheader("About")
    st.write(HOME_ABOUT)

    st.divider()
    st.subheader("Explore the dashboard")
    lookup = {page.title: page for page in pages}
    cards = [(t, b) for t, b in HOME_CARDS if t in lookup]
    columns = st.columns(3)
    for i, (title, blurb) in enumerate(cards):
        with columns[i % 3]:
            with st.container(border=True):
                st.page_link(lookup[title])
                st.caption(blurb)


# ----------------------------------------------------------------------------
# Page 1 - Posting explorer
# ----------------------------------------------------------------------------
def page_explorer():
    st.title("Posting explorer")
    fx, source_label = main_postings()
    months = sorted(fx["Month"].unique())

    st.sidebar.header("Filters")
    m0, m1 = month_range(months, "explorer_months")
    modes = sorted(fx["Work_mode"].unique())
    chosen_modes = st.sidebar.multiselect("Work mode", modes, default=modes, key="explorer_modes")
    countries = st.sidebar.multiselect(
        "Country (empty = all)", sorted(fx["Country"].unique()), key="explorer_countries")
    query = st.sidebar.text_input("Job title contains", key="explorer_query")

    df = fx[(fx["Month"] >= m0) & (fx["Month"] <= m1) & fx["Work_mode"].isin(chosen_modes)]
    if countries:
        df = df[df["Country"].isin(countries)]
    if query:
        df = df[df["Job_Title"].str.contains(query, case=False, regex=False)]
    if df.empty:
        st.warning("No postings match these filters.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Postings", f"{len(df):,}")
    c2.metric("Distinct job titles", f"{df['Title_key'].nunique():,}")
    c3.metric("Countries", f"{df.loc[df['Country'] != NOT_STATED, 'Country'].nunique():,}")
    c4.metric("Remote share", f"{(df['Work_mode'] == 'Remote').mean() * 100:.1f}%")

    # Prophet forecast of all postings (from the time series notebook)
    series = monthly_counts(fx["Date"])
    try:
        if len(series) < 24:
            raise ValueError("fewer than 24 months")
        forecast = prophet_forecast(tuple(series["ds"].dt.strftime("%Y-%m-%d")), tuple(series["y"]), 24)
    except ImportError:
        st.info("Install Prophet to see the forecast (in the terminal: pip install prophet). "
                "Showing postings per month instead.")
        show(px.bar(series, x="ds", y="y", labels={"ds": "Month", "y": "Postings"},
                    title="Postings per month"))
    except ValueError:
        show(px.bar(series, x="ds", y="y", labels={"ds": "Month", "y": "Postings"},
                    title="Postings per month"))
        st.caption("A forecast needs at least 24 months of data.")
    else:
        show(forecast_chart(series, forecast), height=480)

    # Work mode share over time, as lines
    by_mode = df.groupby(["Month", "Work_mode"]).size().unstack(fill_value=0)
    mode_share = by_mode.div(by_mode.sum(axis=1), axis=0) * 100
    long = mode_share.reset_index().melt(id_vars="Month", var_name="Work mode", value_name="Share of postings (%)")
    fig = px.line(long, x="Month", y="Share of postings (%)", color="Work mode",
                  title="Work mode share over time (%)")
    fig.update_yaxes(ticksuffix="%", rangemode="tozero")
    show(fig)

    left, right = st.columns(2)
    with left:
        counts = df.loc[df["Country"] != NOT_STATED, "Country"].value_counts()
        top_c = counts.head(15).reset_index()
        top_c.columns = ["Country", "Postings"]
        fig = px.bar(top_c, x="Postings", y="Country", orientation="h", title="Top 15 countries")
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        show(fig)
        if counts.empty:
            st.info("No countries to show on the map.")
        else:
            world, left_off = country_map(counts)
            show(world, height=360)
            if left_off:
                st.caption(f"{left_off:,} postings from countries that could not be placed on the map are left out.")

    with right:
        model_path = find_first(MODEL_FILES)
        if model_path is None:
            pass
        else:
            try:
                res = skill_direction_results(model_path)
            except ImportError:
                st.info("Install scikit-learn to see the accuracy chart (in the terminal: pip install scikit-learn).")
            except ValueError as exc:
                st.warning(f"The accuracy chart could not be built from {Path(model_path).name}: {exc}.")
            else:
                title, line2, line3 = accuracy_text(res)
                st.markdown(f"#### {title}")
                st.caption(f"{line2}  \n{line3}")
                show(accuracy_chart(res), height=460)
                st.caption("Each model tries to predict whether a skill's mention rate will rise next month. "
                           "Whiskers: 95% range if predictions were independent (the true range is wider). "
                           f"Data: {Path(model_path).name}. The sidebar filters do not apply.")

    st.subheader("Data")
    view = df[["Job_Title", "Skills", "Work_mode", "Country", "Date"]].copy()
    view["Date"] = view["Date"].dt.strftime("%Y-%m-%d")
    st.caption(f"Showing the first {min(len(view), 5000):,} of {len(view):,} rows.")
    st.dataframe(view.head(5000), width="stretch", hide_index=True)
    st.download_button("Download filtered data (CSV)", view.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_postings.csv", mime="text/csv")


# ----------------------------------------------------------------------------
# Page 2 - Skills overview
# ----------------------------------------------------------------------------
def page_skills(show_frameworks=True):
    st.title("Skills overview")
    st.sidebar.header("Options")
    source = pick_source("skills")
    (sk,) = get_data(source, "matches")
    if source == LIVE:
        live_note()
    if sk.empty:
        st.warning("No skills were detected in the stored live postings yet.")
        st.stop()

    n = st.sidebar.slider("Skills in bar charts", 5, 50, 20, key="skills_n")

    total = sk["mentions"].sum()
    metric_cols = st.columns(3 if show_frameworks else 2)
    metric_cols[0].metric("Skills", f"{len(sk):,}")
    metric_cols[1].metric("Total mentions", f"{total:,}")
    if show_frameworks:
        unmapped = sk.loc[sk["framework"] == "NONE", "mentions"].sum()
        metric_cols[2].metric("Mentions with no framework match", f"{unmapped / total * 100:.1f}%")
    fw_color = "framework" if show_frameworks else None

    left, right = st.columns(2)
    top = sk.nlargest(n, "mentions")
    fig = px.bar(top, x="mentions", y="skill_clean", orientation="h", color=fw_color,
                 title=f"Top {min(n, len(sk))} most mentioned skills")
    fig.update_layout(yaxis=dict(categoryorder="total ascending", title=""))
    with left:
        show(fig, height=max(400, 22 * min(n, len(sk))))

    least = sk.nsmallest(n, "mentions")
    fig = px.bar(least, x="mentions", y="skill_clean", orientation="h", color=fw_color,
                 title=f"{min(n, len(sk))} least mentioned skills")
    fig.update_layout(yaxis=dict(categoryorder="total descending", title=""))
    with right:
        show(fig, height=max(400, 22 * min(n, len(sk))))

    if show_frameworks:
        st.subheader("Framework coverage")
        fw = sk.groupby("framework", as_index=False)["mentions"].sum()
        fw["percentage"] = (fw["mentions"] / fw["mentions"].sum() * 100).round(2)
        show(px.treemap(fw, path=["framework"], values="mentions", color="mentions",
                        color_continuous_scale="Viridis", hover_data=["mentions", "percentage"],
                        title="Mentions by framework (with % share)"))
        show(px.treemap(sk, path=["framework", "skill_clean"], values="mentions", color="mentions",
                        color_continuous_scale="Viridis", title="Skills nested within frameworks"),
             height=650)


# ----------------------------------------------------------------------------
# Page 3 - Skill trends
# ----------------------------------------------------------------------------
def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def page_trends():
    st.title("Skill trends")
    st.sidebar.header("Options")
    source = pick_source("trends")
    sc, fx = get_data(source, "skill_count", "postings")
    if source == LIVE:
        live_note()
    months = sorted(sc["Month"].unique())
    if not months:
        st.warning("No skills were detected in the stored live postings yet.")
        st.stop()

    m0, m1 = month_range(months, f"trends_months_{source}")
    in_range = [m for m in months if m0 <= m <= m1]
    sub = sc[(sc["Month"] >= m0) & (sc["Month"] <= m1)]
    totals = sub.groupby("Skill")["Posting_count"].sum().sort_values(ascending=False)

    chosen = st.sidebar.multiselect("Skills", totals.index.tolist(),
                                    default=totals.index[:5].tolist(), key=f"trends_skills_{source}")
    measure = st.sidebar.radio("Measure", ["Postings", "Share of postings (%)"], key="trends_measure")
    horizon = st.sidebar.slider("Months to forecast", 6, 24, 12, key="trends_horizon")
    show_band = st.sidebar.checkbox("Show uncertainty interval", key="trends_band")

    monthly_total = fx.groupby("Month").size()
    if not chosen:
        st.info("Pick at least one skill in the sidebar.")
        st.stop()

    piv = (sub[sub["Skill"].isin(chosen)]
           .pivot_table(index="Month", columns="Skill", values="Posting_count", aggfunc="sum")
           .reindex(in_range).fillna(0))
    if measure.startswith("Share"):
        piv = piv.div(monthly_total.reindex(piv.index), axis=0) * 100
        st.caption("Share = postings mentioning the skill / all postings that month. "
                   "Use this when posting volume changes a lot between months.")

    can_forecast = len(in_range) >= 24
    if not can_forecast:
        st.info("Forecasting needs at least 24 months of history, so only the actual values are shown. "
                "Select a longer range, or switch to the stored dataset.")

    colors = px.colors.qualitative.Plotly
    fig = go.Figure()
    last_actual = None
    try:
        for i, skill in enumerate(piv.columns):
            color = colors[i % len(colors)]
            values = piv[skill].dropna()
            if values.empty:
                continue
            x_hist = pd.to_datetime(values.index.astype(str) + "-01")
            last_actual = x_hist.max() if last_actual is None else max(last_actual, x_hist.max())
            fig.add_trace(go.Scatter(x=x_hist, y=values.values, mode="lines", name=skill,
                                     legendgroup=skill, line=dict(color=color, width=2.5)))
            if not can_forecast:
                continue
            fc = prophet_forecast(tuple(x_hist.strftime("%Y-%m-%d")), tuple(values.values), horizon)
            future = fc.iloc[-horizon:]
            line_x = [x_hist[-1]] + list(future["ds"])
            line_y = [values.iloc[-1]] + list(future["yhat"].clip(lower=0))
            fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", legendgroup=skill, showlegend=False,
                                     name=f"{skill} (forecast)", line=dict(color=color, width=2.5, dash="dash")))
            if show_band:
                upper = list(future["yhat_upper"].clip(lower=0))
                lower = list(future["yhat_lower"].clip(lower=0))
                fig.add_trace(go.Scatter(x=list(future["ds"]) + list(future["ds"])[::-1], y=upper + lower[::-1],
                                         fill="toself", fillcolor=hex_to_rgba(color, 0.15), line=dict(width=0),
                                         hoverinfo="skip", showlegend=False, legendgroup=skill))
    except ImportError:
        st.error("The Prophet package is not installed. In the terminal run:  pip install prophet")
        st.stop()

    if can_forecast and last_actual is not None:
        fig.add_shape(type="line", x0=last_actual, x1=last_actual, y0=0, y1=1, yref="paper",
                      line=dict(color="gray", dash="dash", width=1.5))
    title = f"{measure}: actual (solid) and forecast (dashed)" if can_forecast else measure
    fig.update_layout(title=title, yaxis_title=measure, xaxis_title="Month",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    show(fig, height=520)


# ----------------------------------------------------------------------------
# Page 4 - Analysis of Skills per Job
# ----------------------------------------------------------------------------
def trim_incomplete_tail(trend):
    """Drop trailing months with almost no postings (under 10% of the average of the 12 months
    before them). Such months are usually incomplete data, and would drag a forecast down."""
    counts = trend["Postings"].reset_index(drop=True)
    end = len(counts)
    while end > 12:
        recent = counts.iloc[end - 13:end - 1].mean()
        if counts.iloc[end - 1] < 0.1 * recent:
            end -= 1
        else:
            break
    return trend.iloc[:end]


def page_job_skills():
    st.title("Analysis of Skills per Job")
    st.sidebar.header("Options")
    source = pick_source("jobs")
    sj, fx = get_data(source, "skill_job", "postings")
    if source == LIVE:
        live_note()
    names = display_names(sj)
    months = sorted(sj["Month"].unique())
    if not months:
        st.warning("No skills were detected in the stored live postings yet.")
        st.stop()

    m0, m1 = month_range(months, f"jobs_months_{source}")
    min_posts = st.sidebar.slider("Only list jobs with at least this many postings",
                                  1, 500, 50 if source == STORED else 3, key=f"jobs_min_{source}")
    posts = postings_per_title(fx, m0, m1)
    pool = posts[(posts >= min_posts) & posts.index.isin(list(names))].sort_values(ascending=False)
    if pool.empty:
        st.warning("No job titles meet that minimum. Lower it in the sidebar.")
        st.stop()
    keys = list(pool.index)

    def label(k):
        return "(none)" if k is None else f"{names.get(k, k)} ({pool[k]:,} postings)"

    key_a = st.sidebar.selectbox("Job title", keys, format_func=label,
                                 index=keys.index("data scientist") if "data scientist" in keys else 0,
                                 key=f"jobs_a_{source}")
    key_b = st.sidebar.selectbox("Compare with (optional)", [None] + keys, format_func=label,
                                 key=f"jobs_b_{source}")
    if key_b == key_a:
        key_b = None
    top_n = st.sidebar.slider("Skills to show", 5, 30, 15, key="jobs_n")
    horizon = st.sidebar.slider("Months to forecast", 6, 24, 12, key="jobs_horizon")
    show_band = st.sidebar.checkbox("Show uncertainty interval", key="jobs_band")
    trim_tail = st.sidebar.checkbox(
        "Ignore trailing months with almost no postings", value=True, key="jobs_trim",
        help="Months at the end with almost no postings are probably incomplete data. "
             "Leaving them out stops them dragging the forecast down.")

    sub = sj[(sj["Month"] >= m0) & (sj["Month"] <= m1)]

    def profile(key):
        counts = sub[sub["Title_key"] == key].groupby("Skill")["Posting_count"].sum()
        return (counts / pool[key] * 100).clip(upper=100)

    prof_a = profile(key_a)
    name_a = names.get(key_a, key_a)
    c1, c2 = st.columns(2)
    c1.metric(f"Postings: {name_a}", f"{pool[key_a]:,}")
    c2.metric("Distinct skills mentioned", f"{len(prof_a):,}")

    if prof_a.empty:
        st.warning("No skill data for this job in the selected months.")
        st.stop()

    if key_b:
        name_b = names.get(key_b, key_b)
        both = pd.DataFrame({name_a: prof_a, name_b: profile(key_b)}).fillna(0)
        both = both.loc[both.mean(axis=1).nlargest(top_n).index]
        long = both.reset_index(names="Skill").melt(id_vars="Skill", var_name="Job", value_name="Share")
        fig = px.bar(long, x="Share", y="Skill", color="Job", orientation="h", barmode="group",
                     title="Share of postings that mention each skill (%)")
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        show(fig, height=max(450, 32 * top_n))
    else:
        top = prof_a.nlargest(top_n).reset_index()
        top.columns = ["Skill", "Share"]
        fig = px.bar(top, x="Share", y="Skill", orientation="h",
                     title=f"Most requested skills for {name_a} (% of postings)")
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        show(fig, height=max(450, 26 * len(top)))

    in_range = [m for m in months if m0 <= m <= m1]
    trend = (fx[fx["Title_key"] == key_a].groupby("Month").size()
             .reindex(in_range, fill_value=0).reset_index())
    trend.columns = ["Month", "Postings"]
    trend["Date"] = pd.to_datetime(trend["Month"] + "-01")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=trend["Date"], y=trend["Postings"], name="Actual", marker_color="#7FC4FD"))
    fit = trim_incomplete_tail(trend) if trim_tail else trend
    if len(fit) >= 24:
        try:
            fc = prophet_forecast(tuple(fit["Date"].dt.strftime("%Y-%m-%d")), tuple(fit["Postings"]), horizon)
        except ImportError:
            st.error("The Prophet package is not installed. In the terminal run:  pip install prophet")
        else:
            future = fc.iloc[-horizon:]
            last_fit = fit["Date"].iloc[-1]
            if show_band:
                upper = list(future["yhat_upper"].clip(lower=0))
                lower = list(future["yhat_lower"].clip(lower=0))
                fig.add_trace(go.Scatter(x=list(future["ds"]) + list(future["ds"])[::-1], y=upper + lower[::-1],
                                         fill="toself", fillcolor="rgba(255,127,14,0.18)", line=dict(width=0),
                                         hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=[last_fit] + list(future["ds"]),
                                     y=[fit["Postings"].iloc[-1]] + list(future["yhat"].clip(lower=0)),
                                     mode="lines+markers", name="Forecast",
                                     line=dict(color="#FF7F0E", width=2.5, dash="dash")))
            fig.add_shape(type="line", x0=last_fit, x1=last_fit, y0=0, y1=1, yref="paper",
                          line=dict(color="gray", dash="dash", width=1.5))
    else:
        st.caption("A forecast needs at least 24 months of history for this job. "
                   "Select a longer range in the sidebar.")
    fig.update_layout(title=f"Postings per month: {name_a}", xaxis_title="Month", yaxis_title="Postings",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    show(fig)

    with st.expander("Full skill table"):
        table = prof_a.sort_values(ascending=False).round(1).reset_index()
        table.columns = ["Skill", "% of postings"]
        st.dataframe(table, width="stretch", hide_index=True)


# ----------------------------------------------------------------------------
# Page 5 - Similarity Calculation (cosine similarity on Skill_Job.txt)
# ----------------------------------------------------------------------------
def page_similar(show_gap_table=True):
    st.title("Similarity Calculation")
    p_sj, p_fx = require("skill_job", "postings")
    sj = load_skill_job(p_sj)
    fx = load_postings(p_fx)
    names = display_names(sj)
    months = sorted(sj["Month"].unique())

    st.sidebar.header("Options")
    m0, m1 = month_range(months, "sim_months")
    min_posts = st.sidebar.slider("Only compare jobs with at least this many postings",
                                  5, 200, 20, step=5, key="sim_min")
    top_n = st.sidebar.slider("Similar jobs to show", 5, 30, 15, key="sim_n")

    posts = postings_per_title(fx, m0, m1)
    pool = posts[posts >= min_posts]
    sub = sj[(sj["Month"] >= m0) & (sj["Month"] <= m1) & sj["Title_key"].isin(pool.index)]
    matrix = sub.pivot_table(index="Title_key", columns="Skill", values="Posting_count",
                             aggfunc="sum", fill_value=0)
    if len(matrix) < 3:
        st.warning("Too few job titles to compare. Lower the minimum postings in the sidebar.")
        st.stop()

    keys = [k for k in pool.sort_values(ascending=False).index if k in matrix.index]

    def label(k):
        return f"{names.get(k, k)} ({pool[k]:,} postings)"

    key = st.selectbox("Job title", keys, format_func=label,
                       index=keys.index("data scientist") if "data scientist" in keys else 0,
                       key="sim_job")

    scores = pd.Series(tfidf_cosine(matrix, matrix.index.get_loc(key)), index=matrix.index)
    scores = scores.drop(key).sort_values(ascending=False).head(top_n)
    result = pd.DataFrame({
        "Job title": [names.get(k, k) for k in scores.index],
        "Similarity": [f"{v * 100:.1f}%" for v in scores.values],
        "Postings": [int(pool[k]) for k in scores.index],
    })
    chart_data = result.assign(**{"Similarity (%)": (scores.values * 100).round(1)})

    st.subheader(f"Most similar to {names.get(key, key)}")
    left, right = st.columns([2, 3])
    with left:
        st.dataframe(result, width="stretch", hide_index=True)
    with right:
        fig = px.bar(chart_data, x="Similarity (%)", y="Job title", orientation="h")
        fig.update_layout(yaxis=dict(categoryorder="total ascending", title=""),
                          xaxis=dict(range=[0, 100], ticksuffix="%"))
        show(fig, height=max(350, 26 * len(result)))

    st.subheader("Skill gap")
    match_key = st.selectbox("Compare skills with", list(scores.index),
                             format_func=lambda k: names.get(k, k), key="sim_match")

    def profile(k):
        return (matrix.loc[k] / pool[k] * 100).clip(upper=100)

    pa, pb = profile(key), profile(match_key)
    name_a, name_b = names.get(key, key), names.get(match_key, match_key)
    both = pd.DataFrame({name_a: pa, name_b: pb}).fillna(0)
    shown = both.loc[both.mean(axis=1).nlargest(10).index]
    long = shown.reset_index(names="Skill").melt(id_vars="Skill", var_name="Job", value_name="Share")
    fig = px.bar(long, x="Share", y="Skill", color="Job", orientation="h", barmode="group",
                 title="Top skills across both jobs (% of postings)")
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    show(fig, height=480)

    if show_gap_table:
        gap = (pb - pa).sort_values(ascending=False)
        gap = gap[gap > 0].head(10)
        st.markdown(f"**Skills {name_b} asks for more often than {name_a}**")
        if gap.empty:
            st.write("None: this job asks for every skill at least as often.")
        else:
            table = pd.DataFrame({
                "Skill": gap.index,
                f"% of {name_b}": pb[gap.index].round(1).values,
                f"% of {name_a}": pa[gap.index].round(1).values,
            })
            st.dataframe(table, width="stretch", hide_index=True)


# ----------------------------------------------------------------------------
# Page 6 - Job overviews
# ----------------------------------------------------------------------------
def page_titles(show_frameworks=True):
    st.title("Job overviews")
    (path,) = require("titles")
    jt = load_titles(path)

    st.sidebar.header("Options")
    n = st.sidebar.slider("Titles in bar charts", 5, 50, 20, key="titles_n")
    if show_frameworks:
        min_nested = st.sidebar.slider("Nested treemap: minimum postings", 1, 100, 10, key="titles_nested")

    total = jt["postings"].sum()
    metric_cols = st.columns(3 if show_frameworks else 2)
    metric_cols[0].metric("Job titles", f"{len(jt):,}")
    metric_cols[1].metric("Total postings", f"{total:,}")
    if show_frameworks:
        unmapped = jt.loc[jt["framework"] == "NONE", "postings"].sum()
        metric_cols[2].metric("Postings with no framework match", f"{unmapped / total * 100:.1f}%")
    fw_color = "framework" if show_frameworks else None

    left, right = st.columns(2)
    top = jt.nlargest(n, "postings")
    fig = px.bar(top, x="postings", y="title_clean", orientation="h", color=fw_color,
                 title=f"Top {n} most posted jobs")
    fig.update_layout(yaxis=dict(categoryorder="total ascending", title=""))
    with left:
        show(fig, height=max(400, 22 * n))
    least = jt.nsmallest(n, "postings")
    fig = px.bar(least, x="postings", y="title_clean", orientation="h", color=fw_color,
                 title=f"{n} least posted jobs")
    fig.update_layout(yaxis=dict(categoryorder="total descending", title=""))
    with right:
        show(fig, height=max(400, 22 * n))

    if show_frameworks:
        st.subheader("Framework coverage")
        fw = jt.groupby("framework", as_index=False)["postings"].sum()
        fw["percentage"] = (fw["postings"] / fw["postings"].sum() * 100).round(2)
        show(px.treemap(fw, path=["framework"], values="postings", color="postings",
                        color_continuous_scale="Viridis", hover_data=["postings", "percentage"],
                        title="Postings by framework (with % share)"))

        nested = jt[jt["postings"] >= min_nested]
        show(px.treemap(nested, path=["framework", "title_clean"], values="postings", color="postings",
                        color_continuous_scale="Viridis",
                        title=f"Jobs nested within frameworks (titles with {min_nested}+ postings)"),
             height=650)


# ----------------------------------------------------------------------------
# Live data (fetch real postings from the Adzuna API)
# ----------------------------------------------------------------------------
def adzuna_credentials():
    """Keys come from .streamlit/secrets.toml (or the host's secrets), then environment variables."""
    try:
        section = st.secrets["adzuna"]
        return section["app_id"], section["app_key"]
    except Exception:
        return os.environ.get("ADZUNA_APP_ID", ""), os.environ.get("ADZUNA_APP_KEY", "")


@st.cache_resource
def get_extractor(matches_path):
    skills = load_matches(matches_path)["skill_clean"].tolist()
    return live_data.SkillExtractor(skills)


def page_live():
    st.title("Live data")
    st.markdown("Fetch current job postings from the [Adzuna](https://www.adzuna.com) API. "
                "Skills are detected in each posting and added to a stored history, which the "
                "**Skills overview**, **Skill trends** and **Analysis of Skills per Job** pages can use "
                "when you switch their *Data source* to *Live data*.")
    st.caption("Jobs by Adzuna. Why not LinkedIn? LinkedIn has no open job-search API and its terms "
               "restrict automated access, so this dashboard does not scrape it.")
    (matches_path,) = require("matches")
    app_id, app_key = adzuna_credentials()

    if not (app_id and app_key):
        st.warning("Adzuna keys not found. They are free:")
        st.markdown(
            "1. Register at [developer.adzuna.com](https://developer.adzuna.com) and copy your **app id** and **app key**.\n"
            "2. In your project folder create a folder called `.streamlit` and inside it a file called `secrets.toml`.\n"
            "3. Put this in it, with your own values, then restart the app:")
        st.code('[adzuna]\napp_id = "YOUR_APP_ID"\napp_key = "YOUR_APP_KEY"', language="toml")
        st.caption("Never share this file or upload it to GitHub. When you deploy, paste the same text "
                   "into your host's secrets settings instead.")

    st.subheader("Fetch postings")
    c1, c2 = st.columns([1, 2])
    with c1:
        country = st.selectbox("Country", list(live_data.COUNTRIES), index=0,
                               format_func=lambda c: f"{live_data.COUNTRIES[c]} ({c})", key="live_country")
        pages = st.slider("Pages per search term (50 postings each)", 1, 5, 2, key="live_pages")
        days = st.slider("Only postings from the last N days", 7, 90, 60, key="live_days")
    with c2:
        terms_text = st.text_area("Search terms (one per line)", "\n".join(live_data.DEFAULT_TERMS),
                                  height=190, key="live_terms")
    terms = [t.strip() for t in terms_text.splitlines() if t.strip()][:10]
    calls = len(terms) * pages
    st.caption(f"This will make up to {calls} API calls. Adzuna's free plan has usage limits, so "
               "fetch when you need fresh data rather than on every visit.")

    if st.button("Fetch latest postings", type="primary", disabled=not (app_id and app_key and terms)):
        try:
            with st.spinner("Contacting Adzuna..."):
                result = live_data.fetch_adzuna(app_id, app_key, country, terms, get_extractor(matches_path),
                                                pages=pages, max_days_old=days)
            total, added = live_data.append_snapshot(LIVE_FILE, result.postings)
        except live_data.LiveDataError as exc:
            st.error(str(exc))
        else:
            st.success(f"{result.calls} API calls made. {added:,} new postings added, "
                       f"{total:,} stored in total.")
            for warning in result.warnings:
                st.warning(warning)

    st.subheader("What is stored")
    snap = live_data.read_snapshot(LIVE_FILE)
    if snap.empty:
        st.info("Nothing stored yet. Fetch postings above.")
        return
    dates = pd.to_datetime(snap["Date"], errors="coerce")
    with_skills = (snap["Skills"] != "").mean() * 100
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Postings stored", f"{len(snap):,}")
    m2.metric("Earliest posting", f"{dates.min():%Y-%m-%d}")
    m3.metric("Latest posting", f"{dates.max():%Y-%m-%d}")
    m4.metric("With at least one skill detected", f"{with_skills:.0f}%")
    if with_skills < 50:
        st.info("Many postings have no detected skill because the API only returns a short excerpt of "
                "each description. Skill counts from live data are therefore lower than the stored dataset.")

    preview = snap[["Job_Title", "Country", "Date", "Skills", "Search_term"]].copy()
    preview["Skills"] = preview["Skills"].str.replace("|", ", ", regex=False)
    st.dataframe(preview.sort_values("Date", ascending=False).head(500), width="stretch", hide_index=True)
    st.download_button("Download live postings (CSV)", snap.to_csv(index=False).encode("utf-8"),
                       file_name="live_postings.csv", mime="text/csv")

    with st.expander("Running this on a server"):
        st.markdown(
            "Live history only grows when you fetch, so for a deployed dashboard schedule a fetch "
            "(for example daily) with the same code the button uses:\n\n"
            "`python live_data.py --country za --pages 3 --days 60`\n\n"
            "It reads `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` from environment variables. Many free hosts "
            "reset their disk when the app restarts, so keep `live_cache/live_postings.csv` somewhere "
            "persistent (a database or storage bucket) if you need the history to last.")


# ----------------------------------------------------------------------------
# User and administrator interfaces
# ----------------------------------------------------------------------------
def page_titles_user():
    page_titles(show_frameworks=False)


def page_titles_admin():
    page_titles(show_frameworks=True)


def page_skills_user():
    page_skills(show_frameworks=False)


def page_skills_admin():
    page_skills(show_frameworks=True)


def page_similar_user():
    page_similar(show_gap_table=False)


def page_similar_admin():
    page_similar(show_gap_table=True)


def admin_password():
    """The admin password comes from .streamlit/secrets.toml (or the host's secrets), then the
    ADMIN_PASSWORD environment variable. There is no built-in default password."""
    try:
        return str(st.secrets["admin"]["password"])
    except Exception:
        return os.environ.get("ADMIN_PASSWORD", "")


def page_admin_login():
    st.title("Admin login")
    expected = admin_password()
    if not expected:
        st.warning("Admin access is not set up yet. In your project folder, open (or create) "
                   "`.streamlit/secrets.toml` and add this with your own password, then restart the app:")
        st.code('[admin]\npassword = "choose-a-strong-password"', language="toml")
        st.caption("Never upload this file to GitHub. When you deploy, paste the same text into your "
                   "host's secrets settings instead.")
        return
    with st.form("admin_login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")
    if submitted:
        if hmac.compare_digest(password.encode("utf-8"), expected.encode("utf-8")):
            st.session_state["is_admin"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")


def log_out():
    st.session_state["is_admin"] = False


# ----------------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------------
if st.session_state.get("is_admin", False):
    # Administrator interface
    pages = [
        st.Page(page_explorer, title="Posting explorer", icon="🔎", url_path="explorer", default=True),
        st.Page(page_titles_admin, title="Job overviews", icon="🏷️", url_path="titles"),
        st.Page(page_skills_admin, title="Skills overview", icon="🧩", url_path="skills"),
        st.Page(page_trends, title="Skill trends", icon="📈", url_path="trends"),
        st.Page(page_job_skills, title="Analysis of Skills per Job", icon="💼", url_path="skills-per-job"),
        st.Page(page_similar_admin, title="Similarity Calculation", icon="🔗", url_path="similarity"),
        st.Page(page_live, title="Live data", icon="🌐", url_path="live"),
    ]
    st.sidebar.caption("Signed in as administrator")
    st.sidebar.button("Log out", on_click=log_out)
else:
    # User interface
    pages = [
        st.Page(page_home, title="Forecasting the future", icon="🏠", url_path="home", default=True),
        st.Page(page_explorer, title="Posting explorer", icon="🔎", url_path="explorer"),
        st.Page(page_titles_user, title="Job overviews", icon="🏷️", url_path="titles"),
        st.Page(page_skills_user, title="Skills overview", icon="🧩", url_path="skills"),
        st.Page(page_job_skills, title="Analysis of Skills per Job", icon="💼", url_path="skills-per-job"),
        st.Page(page_similar_user, title="Similarity Calculation", icon="🔗", url_path="similarity"),
        st.Page(page_admin_login, title="Admin login", icon="🔒", url_path="admin"),
    ]
st.navigation(pages).run()