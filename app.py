
import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt

# =========================================
# ⚙️ CONFIG
# =========================================
SPORT_MODE = "basketball"  # verrouillage mode basket
st.set_page_config(page_title="🏀 BasketAI Ultra+ v13 — IA Basket Mondiale", layout="wide")

API_KEY = "bb3ba63a8bfab1020390fe28bd180522"  # <- ta clé API Sports
API_URL = "https://v1.basketball.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

# =========================================
# 🔧 FONCTIONS API
# =========================================
@st.cache_data(ttl=3600)
def api_get(path):
    try:
        r = requests.get(f"{API_URL}{path}", headers=HEADERS, timeout=25)
        r.raise_for_status()
        js = r.json()
        # format attendu: {"response":[...]}
        return js.get("response", [])
    except Exception as e:
        return {"_error": str(e)}

@st.cache_data(ttl=3600)
def get_leagues():
    """Liste (id, libellé) des ligues"""
    data = api_get("/leagues")
    if isinstance(data, dict) and "_error" in data:
        return [], data["_error"]
    leagues = []
    for l in data:
        name = l.get("name", "Unknown")
        country = l.get("country") or {}
        country_name = country.get("name", "")
        lid = l.get("id")
        if lid and name:
            label = f"{name}" + (f" ({country_name})" if country_name else "")
            leagues.append((lid, label))
    leagues.sort(key=lambda x: x[1].lower())
    return leagues, None

@st.cache_data(ttl=1800)
def get_teams(league_id, season="2024-2025"):
    """Liste des équipes d'une ligue"""
    data = api_get(f"/teams?league={league_id}&season={season}")
    if isinstance(data, dict) and "_error" in data:
        return [], data["_error"]
    teams = []
    for t in data:
        tid = t.get("id")
        name = t.get("name")
        if tid and name:
            teams.append({"id": tid, "name": name})
    teams.sort(key=lambda x: x["name"].lower())
    return teams, None

def monte_carlo_simulation(home_strength, away_strength, n=200_000):
    """Distribution des points totaux (Poisson)"""
    home_scores = np.random.poisson(home_strength, n)
    away_scores = np.random.poisson(away_strength, n)
    return home_scores + away_scores

# =========================================
# 🎮 UI
# =========================================
st.title("🏀 BasketAI Ultra+ v13")
st.info("Mode verrouillé : **Basket** mondial uniquement.")

# Ligues
leagues, err_leagues = get_leagues()
if err_leagues:
    st.error(f"Erreur API ligues : {err_leagues}")
elif not leagues:
    st.warning("Aucune ligue trouvée (clé API invalide, quota épuisé ou API indisponible).")
else:
    league_name = st.selectbox(
        "🌐 Sélectionne une ligue :",
        options=[name for _, name in leagues],
        index=0
    )
    league_id = next((lid for lid, name in leagues if name == league_name), None)

    # Équipes
    teams, err_teams = ([], None)
    if league_id:
        teams, err_teams = get_teams(league_id)
    if err_teams:
        st.error(f"Erreur API équipes : {err_teams}")

    if teams:
        team_names = [t["name"] for t in teams]
        col1, col2 = st.columns(2)
        with col1:
            team_home_name = st.selectbox("🏠 Équipe domicile :", team_names, index=None, placeholder="Choisis l'équipe domicile")
        with col2:
            team_away_name = st.selectbox("🚀 Équipe extérieure :", team_names, index=None, placeholder="Choisis l'équipe extérieure")

        # Lancer la prédiction
        if st.button("🔮 Lancer la prédiction IA"):
            if not team_home_name or not team_away_name:
                st.warning("⚠️ Sélectionne **deux équipes** avant de lancer la prédiction.")
                st.stop()

            team_home = next((t for t in teams if t["name"] == team_home_name), None)
            team_away = next((t for t in teams if t["name"] == team_away_name), None)
            if not team_home or not team_away:
                st.error("❌ Les équipes sélectionnées n’ont pas été trouvées.")
                st.stop()
            if team_home["id"] == team_away["id"]:
                st.error("❌ Choisis **deux équipes différentes**.")
                st.stop()

            st.info(f"📊 Simulation en cours : **{team_home_name}** vs **{team_away_name}**…")

            # paramètres par défaut "basket pro"
            base_home, base_away = 85, 82
            total_points = monte_carlo_simulation(base_home, base_away, n=200_000)
            mean_points = float(np.mean(total_points))
            std_points = float(np.std(total_points))

            st.success(f"✅ Moyenne totale simulée : **{mean_points:.1f} pts ± {std_points:.1f}** (200 000 matchs)")
            st.caption("Modèle Poisson basique. Ajuste ensuite selon les tendances réelles de la ligue.")

            # Histogramme
            fig = plt.figure(figsize=(8, 3.8))
            plt.hist(total_points, bins=50, edgecolor="black")
            plt.axvline(mean_points, linestyle="dashed", linewidth=2)
            plt.title("Distribution simulée des points totaux 🏀")
            plt.xlabel("Points totaux")
            plt.ylabel("Fréquence")
            st.pyplot(fig)
    else:
        st.warning("Sélectionne une ligue pour afficher les équipes disponibles.")
