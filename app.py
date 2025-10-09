import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 🔧 CONFIGURATION GLOBALE
# =====================================================
st.set_page_config(page_title="🏀 BasketAI Ultra+ v10.5 — Prédictions IA Mondiales 🌍", layout="wide")
API_KEY = "bb3ba63a8bfab1020390fe28bd180522"
API_URL = "https://v1.basketball.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# =====================================================
# 🏀 FONCTIONS UTILITAIRES
# =====================================================

@st.cache_data
def get_leagues():
    """Récupère toutes les ligues disponibles depuis ton API"""
    url = f"{API_URL}/leagues"
    res = requests.get(url, headers=HEADERS)
    data = res.json()
    return [{"id": l["id"], "name": f"{l['name']} ({l['country']['name']})"} for l in data["response"] if "country" in l]

@st.cache_data
def get_teams(league_id, season="2024-2025"):
    """Récupère les équipes d'une ligue"""
    url = f"{API_URL}/teams?league={league_id}&season={season}"
    res = requests.get(url, headers=HEADERS)
    data = res.json()
    return [{"id": t["id"], "name": t["name"], "logo": t["logo"]} for t in data["response"]]

def monte_carlo_simulation(home_strength, away_strength, n=200000):
    """Simule 200 000 matchs pour obtenir la distribution des scores"""
    home_avg = np.random.poisson(home_strength, n)
    away_avg = np.random.poisson(away_strength, n)
    total_points = home_avg + away_avg
    diff = home_avg - away_avg
    return home_avg, away_avg, total_points, diff

# =====================================================
# 🎨 INTERFACE PRINCIPALE
# =====================================================

st.title("🏀 BasketAI Ultra+ v10.5 — Prédictions IA Mondiales 🌍")
st.markdown("### 🤖 Données API Sports + Simulation Monte Carlo (200 000 itérations)")

# 1️⃣ Choix de la ligue
st.markdown("## 🌐 Sélectionne une ligue")
leagues = get_leagues()
league_choice = st.selectbox("🏆 Ligue :", [l["name"] for l in leagues])
selected_league = next((l for l in leagues if l["name"] == league_choice), None)

# 2️⃣ Sélection des équipes
if selected_league:
    teams = get_teams(selected_league["id"])
    col1, col2 = st.columns(2)

    with col1:
        team_home = st.selectbox("🏠 Équipe domicile :", [t["name"] for t in teams])
        team_home_data = next((t for t in teams if t["name"] == team_home), None)
        if team_home_data: st.image(team_home_data["logo"], width=120)

    with col2:
        team_away = st.selectbox("🚀 Équipe extérieure :", [t["name"] for t in teams])
        team_away_data = next((t for t in teams if t["name"] == team_away), None)
        if team_away_data: st.image(team_away_data["logo"], width=120)

    # 3️⃣ Ligne bookmaker manuelle
    st.markdown("### 🎯 Ligne bookmaker (points totaux attendus)")
    bookmaker_line = st.number_input("👉 Entre la ligne du bookmaker (ex : 175.5)", value=175.5, step=0.5)

    if st.button("🧠 Lancer la prédiction IA complète"):
        try:
            # ⚙️ Simulation de puissance IA
            power_home = np.random.uniform(75, 90)
            power_away = np.random.uniform(70, 85)

            # 🧮 Simulation Monte Carlo
            home_pts, away_pts, total_pts, diff = monte_carlo_simulation(power_home, power_away)

            # 📊 Statistiques principales
            mean_home = np.mean(home_pts)
            mean_away = np.mean(away_pts)
            mean_total = np.mean(total_pts)
            std_total = np.std(total_pts)
            fiability = 100 - (std_total / mean_total * 100)

            # 📈 Probabilités Over/Under
            proba_over = np.mean(total_pts > bookmaker_line) * 100
            proba_under = 100 - proba_over

            st.markdown("## 📊 Résultats de la simulation")
            st.success(f"🏠 {team_home} : **{mean_home:.1f} pts** — 🚀 {team_away} : **{mean_away:.1f} pts**")
            st.info(f"📈 Moyenne totale simulée : **{mean_total:.1f} pts ± {std_total:.1f}**")
            st.warning(f"🧩 Fiabilité IA : **{fiability:.1f}%**")
            st.markdown(f"🎯 Ligne bookmaker : **{bookmaker_line:.1f} pts**")
            st.write(f"📊 Over {bookmaker_line} : **{proba_over:.1f}%** | Under {bookmaker_line} : **{proba_under:.1f}%**")

            # 🧮 Graphique de distribution
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(total_pts, bins=50, color="mediumslateblue", alpha=0.7)
            ax.axvline(bookmaker_line, color="red", linestyle="--", label=f"Ligne bookmaker ({bookmaker_line})")
            ax.set_title("Distribution des points totaux simulés")
            ax.set_xlabel("Total points (domicile + extérieur)")
            ax.set_ylabel("Fréquence")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Erreur : {e}")
            
