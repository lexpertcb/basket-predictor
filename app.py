import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt

SPORT_MODE = "basketball"  # verrouillage permanent du mode basket

# =========================
# ⚙️ CONFIGURATION GLOBALE
# =========================
st.set_page_config(page_title="🏀 BasketAI Ultra+ v12.5 — Prédictions IA Mondiales 🌍", layout="wide")

API_KEY = "bb3ba63a8bfab1020390fe28bd180522"  # ta clé API Sports
API_URL = "https://v1.basketball.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# =========================
# 🧠 FONCTIONS API
# =========================
@st.cache_data
def get_leagues():
    """Récupère toutes les ligues disponibles depuis l’API"""
    r = requests.get(f"{API_URL}/leagues", headers=HEADERS)
    data = r.json()
    return [(l["id"], f'{l["name"]} ({l["country"]})') for l in data["response"] if "country" in l]

@st.cache_data
def get_teams(league_id):
    """Récupère les équipes d’une ligue"""
    r = requests.get(f"{API_URL}/teams?league={league_id}&season=2024-2025", headers=HEADERS)
    data = r.json()
    teams = {t["id"]: t["name"] for t in data["response"]}
    logos = {t["id"]: t["logo"] for t in data["response"]}
    return teams, logos

@st.cache_data
def get_team_stats(team_id, league_id):
    """Récupère les statistiques d’une équipe"""
    r = requests.get(f"{API_URL}/teams/statistics?team={team_id}&league={league_id}&season=2024-2025", headers=HEADERS)
    return r.json().get("response", {})

@st.cache_data
def get_team_form(team_id):
    """Retourne la forme récente (5 derniers matchs)"""
    r = requests.get(f"{API_URL}/games?team={team_id}&season=2024-2025&last=5", headers=HEADERS)
    data = r.json().get("response", [])
    if not data:
        return 0, 0
    pts_for = np.mean([
        g["teams"]["home"]["points"] if g["teams"]["home"]["id"] == team_id else g["teams"]["visitors"]["points"]
        for g in data
    ])
    pts_against = np.mean([
        g["teams"]["visitors"]["points"] if g["teams"]["home"]["id"] == team_id else g["teams"]["home"]["points"]
        for g in data
    ])
    return pts_for, pts_against

def compute_strength(stats, form_for, form_against):
    """Combine stats générales et forme récente"""
    base_for = stats.get("points", {}).get("for", {}).get("average", {}).get("all", 85)
    base_against = stats.get("points", {}).get("against", {}).get("average", {}).get("all", 80)
    pts_for = (0.7 * base_for) + (0.3 * form_for)
    pts_against = (0.7 * base_against) + (0.3 * form_against)
    return pts_for, pts_against

def simulate_match(home_strength, away_strength, sims=200000):
    """Simulation Monte Carlo"""
    home_off, home_def = home_strength
    away_off, away_def = away_strength
    mu_home = (home_off + (100 - away_def)) / 2
    mu_away = (away_off + (100 - home_def)) / 2
    s_home = np.random.normal(mu_home, 10, sims)
    s_away = np.random.normal(mu_away, 10, sims)
    total = s_home + s_away
    return s_home, s_away, total

# =========================
# 🎨 INTERFACE
# =========================
st.title("🏀 BasketAI Ultra+ v12.5 — Prédictions IA Mondiales 🌍")
st.caption("Analyse complète avec forme récente, moyenne des points et simulation Monte Carlo (200 000 itérations).")

leagues = get_leagues()
selected_league_id, selected_league_name = st.selectbox("🌍 Choisis une ligue :", leagues)

teams, logos = get_teams(selected_league_id)
col1, col2 = st.columns(2)
with col1:
    team_home = st.selectbox("🏠 Équipe domicile :", list(teams.values()))
with col2:
    team_away = st.selectbox("🚀 Équipe extérieure :", list(teams.values()))

home_id = [k for k, v in teams.items() if v == team_home][0]
away_id = [k for k, v in teams.items() if v == team_away][0]

col1.image(logos[home_id], width=120)
col2.image(logos[away_id], width=120)

if st.button("🎯 Lancer la prédiction IA complète"):
    st.info("Analyse des statistiques et forme récente des équipes en cours...")

    stats_home = get_team_stats(home_id, selected_league_id)
    stats_away = get_team_stats(away_id, selected_league_id)

    form_home_for, form_home_against = get_team_form(home_id)
    form_away_for, form_away_against = get_team_form(away_id)

    home_strength = compute_strength(stats_home, form_home_for, form_home_against)
    away_strength = compute_strength(stats_away, form_away_for, form_away_against)

    s_home, s_away, total = simulate_match(home_strength, away_strength)

    mean_home = np.mean(s_home)
    mean_away = np.mean(s_away)
    mean_total = np.mean(total)
    std_total = np.std(total)

    fiability = 100 - (std_total / mean_total * 100)
    fiability = np.clip(fiability, 55, 99)

    st.subheader("📊 Résultats IA")
    st.write(f"🏠 {team_home} — Moyenne : {mean_home:.1f} pts (forme récente : {form_home_for:.1f})")
    st.write(f"🚀 {team_away} — Moyenne : {mean_away:.1f} pts (forme récente : {form_away_for:.1f})")
    st.write(f"🎯 Total simulé : {mean_total:.1f} pts ± {std_total:.1f}")
    st.write(f"💪 Fiabilité IA : {fiability:.1f}%")

    prob_over = np.mean(total > mean_total) * 100
    prob_under = 100 - prob_over

    if prob_over > prob_under:
        st.success(f"✅ Tendance : OVER {mean_total:.1f} points ({prob_over:.1f}%)")
    else:
        st.warning(f"📉 Tendance : UNDER {mean_total:.1f} points ({prob_under:.1f}%)")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(total, bins=50, color="orange", alpha=0.7)
    ax.axvline(mean_total, color="blue", linestyle="--", label=f"Moyenne IA {mean_total:.1f}")
    ax.set_xlabel("Total Points")
    ax.legend()
    st.pyplot(fig)
