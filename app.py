import os
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from scipy.stats import norm

st.set_page_config(page_title="Prédicteur de Panier", page_icon="🏀", layout="centered")

# -----------------------------
# Helpers
# -----------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "basket-predictor/1.0"})

def _get_json(url, params=None):
    r = SESSION.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# =============================
# 1) NBA – gratuit (balldontlie)
# =============================
BDL_BASE = "https://www.balldontlie.io/api/v1"

def nba_list_teams():
    teams = []
    page = 1
    while True:
        data = _get_json(f"{BDL_BASE}/teams", params={"per_page": 100, "page": page})
        teams.extend(data["data"])
        if page >= data["meta"]["total_pages"]:
            break
        page += 1
    df = pd.DataFrame(teams)
    # Garde un id + nom court joli
    df["label"] = df["full_name"] + " (" + df["abbreviation"] + ")"
    return df[["id", "label", "full_name", "abbreviation"]].sort_values("label").reset_index(drop=True)

def nba_team_games(team_id: int, season: int, n: int = 10):
    """Retourne les n derniers matchs d'une équipe (terminés)."""
    games = []
    page = 1
    while len(games) < n and page <= 20:
        data = _get_json(
            f"{BDL_BASE}/games",
            params={
                "team_ids[]": team_id,
                "seasons[]": season,
                "per_page": 100,
                "page": page,
                "postseason": "false",
            },
        )
        games.extend([g for g in data["data"] if g["status"] == "Final"])
        if page >= data["meta"]["total_pages"]:
            break
        page += 1
    games = games[:n]
    if not games:
        return pd.DataFrame()
    rows = []
    for g in games:
        is_home = g["home_team"]["id"] == team_id
        pts_for = g["home_team_score"] if is_home else g["visitor_team_score"]
        pts_opp = g["visitor_team_score"] if is_home else g["home_team_score"]
        opp_id = g["visitor_team"]["id"] if is_home else g["home_team"]["id"]
        rows.append(
            {
                "game_id": g["id"],
                "date": pd.to_datetime(g["date"]),
                "team_id": team_id,
                "opp_id": opp_id,
                "pts_for": pts_for,
                "pts_against": pts_opp,
                "home": is_home,
            }
        )
    df = pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)
    return df

def form_metrics_from_games(g: pd.DataFrame):
    """Renvoie des métriques simples de forme récente."""
    if g.empty:
        return {"pf": np.nan, "pa": np.nan, "pace": np.nan, "std": np.nan}

    pf = g["pts_for"].mean()
    pa = g["pts_against"].mean()
    tot = (g["pts_for"] + g["pts_against"]).mean()
    std = (g["pts_for"] - pf).std(ddof=1) if len(g) >= 2 else 12.0  # dispersion de l’attaque
    # approx 'pace' via total points moyen
    return {"pf": pf, "pa": pa, "pace": tot, "std": std if not np.isnan(std) else 12.0}

def build_prediction(mA, mB):
    """
    Petit modèle additif :
      Attendu A = moyenne( attaque A vs défense B , rythme moyen )
      Attendu B = moyenne( attaque B vs défense A , rythme moyen )
    Proba de victoire via Normal approx sur (A-B) avec variance combinée.
    """
    sA = 0.5 * (mA["pf"] + (mB["pace"] + mA["pace"]) / 2 - mB["pa"]/2)
    sB = 0.5 * (mB["pf"] + (mA["pace"] + mB["pace"]) / 2 - mA["pa"]/2)

    # bornes raisonnables
    sA = float(np.clip(sA, 80, 140))
    sB = float(np.clip(sB, 80, 140))

    # écart-type combiné (bruit attaque des deux équipes)
    sigma = math.sqrt(max(mA["std"], 8.0) ** 2 + max(mB["std"], 8.0) ** 2)
    diff = sA - sB
    # Probabilité A gagne ~ P(diff > 0)
    pA = float(norm.cdf(diff / sigma))
    return round(sA, 1), round(sB, 1), pA

# =====================================
# 2) Multi-ligues (optionnel, via RapidAPI)
# =====================================
RAPID_KEY = os.environ.get("RAPIDAPI_KEY", "").strip()
HAS_MULTI = len(RAPID_KEY) > 0

def multi_leagues():
    """Quelques ligues communes (API-BASKETBALL)."""
    return {
        "EuroLeague": {"league": 120, "season": "2023-2024"},
        "France Pro A": {"league": 2, "season": "2023-2024"},
        "NBA (alt)": {"league": 12, "season": "2023-2024"},
    }

def api_basketball(path, params):
    url = f"https://api-basketball.p.rapidapi.com/{path}"
    headers = {"X-RapidAPI-Key": RAPID_KEY, "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def multi_list_teams(league_id, season):
    data = api_basketball("teams", {"league": league_id, "season": season})
    rows = []
    for t in data.get("response", []):
        rows.append({"id": t["id"], "label": t["name"]})
    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)

def multi_team_games(team_id, season, n=10):
    data = api_basketball("games", {"team": team_id, "season": season})
    games = []
    for g in data.get("response", []):
        if g["status"]["short"] not in ("FT", "AOT"):  # terminé
            continue
        is_home = g["teams"]["home"]["id"] == team_id
        pts_for = g["scores"]["home"]["total"] if is_home else g["scores"]["away"]["total"]
        pts_opp = g["scores"]["away"]["total"] if is_home else g["scores"]["home"]["total"]
        opp_id = g["teams"]["away"]["id"] if is_home else g["teams"]["home"]["id"]
        games.append(
            {
                "date": pd.to_datetime(g["date"]),
                "team_id": team_id,
                "opp_id": opp_id,
                "pts_for": pts_for,
                "pts_against": pts_opp,
                "home": is_home,
            }
        )
    df = pd.DataFrame(games).sort_values("date", ascending=False).head(n).reset_index(drop=True)
    return df

# -----------------------------
# UI
# -----------------------------
st.title("Prédicteur de Panier 🏀")

mode = st.radio(
    "Source de données",
    ["NBA (gratuit – balldontlie)"] + (["Multi-ligues (API-Basketball)"] if HAS_MULTI else []),
    horizontal=False,
)

# -----------------------------
# NBA
# -----------------------------
if mode.startswith("NBA"):
    st.subheader("NBA (gratuit)")
    season = st.number_input("Saison (ex: 2024 pour 2023-24)", min_value=1979, max_value=2025, value=2024, step=1)

    teams_df = nba_list_teams()
    tA = st.selectbox("Équipe A", options=teams_df["label"].tolist(), index=0 if len(teams_df)>0 else None)
    tB = st.selectbox("Équipe B", options=teams_df["label"].tolist(), index=1 if len(teams_df)>1 else None)
    idA = int(teams_df.iloc[teams_df.index[teams_df["label"]==tA][0]]["id"])
    idB = int(teams_df.iloc[teams_df.index[teams_df["label"]==tB][0]]["id"])

    n_last = st.slider("Nombre de derniers matchs pris en compte", 5, 20, 10, 1)
    line = st.slider("Ligne Over/Under (points totaux)", 150, 260, 180, 1)

    if st.button("🔮 Prédire"):
        gA = nba_team_games(idA, season, n=n_last)
        gB = nba_team_games(idB, season, n=n_last)

        if gA.empty or gB.empty:
            st.error("Pas assez de matchs terminés trouvés pour l’une des équipes.")
            st.stop()

        mA = form_metrics_from_games(gA)
        mB = form_metrics_from_games(gB)
        sA, sB, pA = build_prediction(mA, mB)

        st.success(f"Score attendu : **{tA} {sA} – {sB} {tB}** | P({tA}) = **{pA*100:.1f}%**")

        total_points = round(sA + sB, 1)
        st.info(f"Total points attendu : **{total_points}**")

        verdict = "Plus" if total_points > line else "Moins"
        st.write(f"👉 Pronostic : **{verdict} de {line} points**")

        with st.expander("Détails des matchs (Équipe A)"):
            st.dataframe(gA)
        with st.expander("Détails des matchs (Équipe B)"):
            st.dataframe(gB)

# -----------------------------
# Multi-ligues (optionnel)
# -----------------------------
else:
    if not HAS_MULTI:
        st.warning("Ajoute une clé RapidAPI (API-BASKETBALL) dans **Settings → Secrets** pour activer ce mode.")
        st.stop()

    st.subheader("Multi-ligues (via API-Basketball)")
    leagues = multi_leagues()

    ligue = st.selectbox("Ligue", list(leagues.keys()))
    league_id = leagues[ligue]["league"]
    season = leagues[ligue]["season"]

    tdf = multi_list_teams(league_id, season)
    tA = st.selectbox("Équipe A", tdf["label"].tolist())
    tB = st.selectbox("Équipe B", tdf["label"].tolist(), index=1 if len(tdf)>1 else 0)
    idA = int(tdf.iloc[tdf.index[tdf["label"]==tA][0]]["id"])
    idB = int(tdf.iloc[tdf.index[tdf["label"]==tB][0]]["id"])

    n_last = st.slider("Nombre de derniers matchs pris en compte", 5, 20, 10, 1)
    line = st.slider("Ligne Over/Under (points totaux)", 120, 220, 160, 1)

    if st.button("🔮 Prédire"):
        gA = multi_team_games(idA, season, n=n_last)
        gB = multi_team_games(idB, season, n=n_last)

        if gA.empty or gB.empty:
            st.error("Pas assez de matchs terminés trouvés pour l’une des équipes.")
            st.stop()

        mA = form_metrics_from_games(gA)
        mB = form_metrics_from_games(gB)
        sA, sB, pA = build_prediction(mA, mB)

        st.success(f"Score attendu : **{tA} {sA} – {sB} {tB}** | P({tA}) = **{pA*100:.1f}%**")

        total_points = round(sA + sB, 1)
        st.info(f"Total points attendu : **{total_points}**")

        verdict = "Plus" if total_points > line else "Moins"
        st.write(f"👉 Pronostic : **{verdict} de {line} points**")

        with st.expander("Détails des matchs (Équipe A)"):
            st.dataframe(gA)
        with st.expander("Détails des matchs (Équipe B)"):
            st.dataframe(gB)

st.caption("Modèle simple basé sur la forme récente (points pour/contre + rythme). À affiner avec d’autres features pour un niveau pro.")
