# /root/basketai/basket-predictor/app.py
import os
import math
import time
import json
import requests
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="🏀 BasketAI Ultra+ v15 — Over/Under IA", layout="wide")
API_URL = "https://v1.basketball.api-sports.io"
API_KEY = os.getenv("APISPORTS_KEY", "bb3ba63a8bfab1020390fe28bd180522").strip()
HEADERS = {"x-apisports-key": API_KEY}
DEFAULT_SEASON = "2024-2025"

# -----------------------------
# UTILS
# -----------------------------
def api_get(path: str, params: dict = None, timeout=20):
    """GET helper avec gestion d'erreurs."""
    url = f"{API_URL}{path}"
    try:
        r = requests.get(url, headers=HEADERS, params=params or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"get": path, "error": str(e), "response": None}

@st.cache_data(ttl=1800)
def check_api_status():
    data = api_get("/status")
    try:
        ok = data.get("response", {}).get("account", {}).get("active", False)
        plan = data.get("response", {}).get("subscription", {}).get("plan", "?")
        end  = data.get("response", {}).get("subscription", {}).get("end", "?")
        used = data.get("response", {}).get("requests", {}).get("current", 0)
        limit= data.get("response", {}).get("requests", {}).get("limit_day", 0)
        return ok, {"plan": plan, "end": end, "used": used, "limit": limit}
    except Exception:
        return False, {}

@st.cache_data(ttl=1800)
def get_leagues(season: str):
    # essaye par saison d’abord
    data = api_get("/leagues", params={"season": season})
    resp = data.get("response", [])
    if not resp:  # fallback: toutes
        data = api_get("/leagues")
        resp = data.get("response", [])
    leagues = []
    for L in resp:
        lid = L.get("id")
        name = L.get("name")
        country = L.get("country")
        if lid and name:
            label = f"{name}" + (f" ({country})" if country else "")
            leagues.append({"id": lid, "label": label})
    # tri alpha
    leagues.sort(key=lambda x: x["label"].lower())
    return leagues

@st.cache_data(ttl=900)
def get_teams(league_id: int, season: str):
    data = api_get("/teams", params={"league": league_id, "season": season})
    teams = []
    for t in data.get("response", []):
        tid = t.get("id")
        name = t.get("name")
        if tid and name:
            teams.append({"id": tid, "name": name})
    teams.sort(key=lambda x: x["name"].lower())
    return teams

@st.cache_data(ttl=300)
def get_ou_line(league_id: int, season: str):
    """
    Récupère une ligne Over/Under (total points) depuis /odds
    Retourne un float (ex: 165.5) ou None si non trouvé.
    """
    # Beaucoup de books renvoient des bets structurés différemment.
    # On scanne les libellés les plus courants.
    candidates = {"Over/Under", "Total Points", "Totals", "Match Points"}
    data = api_get("/odds", params={"league": league_id, "season": season})
    try:
        for item in data.get("response", []):
            for book in item.get("bookmakers", []):
                for bet in book.get("bets", []):
                    label = str(bet.get("name") or bet.get("label") or "").strip()
                    if any(k.lower() in label.lower() for k in candidates):
                        # On cherche une valeur de type "165.5"
                        # parfois "Over 165.5" / "Under 165.5"
                        lines = []
                        for v in bet.get("values", []):
                            val = v.get("value") or v.get("handicap") or ""
                            # tente d’extraire un nombre
                            try:
                                # extrait le dernier nombre présent
                                import re
                                nums = re.findall(r"[-+]?\d+(\.\d+)?", str(val))
                                if nums:
                                    lines.append(float(nums[-1]))
                            except:
                                pass
                        if lines:
                            # médiane des lignes trouvées
                            return float(np.median(lines))
        return None
    except Exception:
        return None

# -----------------------------
# IA — modèles simples (ensemble)
# -----------------------------
def poisson_sim(mu_home: float, mu_away: float, n=120000):
    home = np.random.poisson(mu_home, n)
    away = np.random.poisson(mu_away, n)
    return home + away

def m_lin(home, away):         # linéaire simple
    return 0.52 if home + away >= 168 else 0.48

def m_poly(home, away):        # poly léger
    x = home + away
    z = -0.00015*(x-170)**2 + 0.55
    return float(np.clip(z, 0.05, 0.95))

def m_logit(home, away):       # logistique
    x = home + away - 165
    return 1/(1 + np.exp(-x/6.5))

def m_elo_mix(home, away):     # proxy ELO → centrage 166
    base = (home*1.02 + away*0.98)
    return float(np.clip((base-166)/12 + 0.5, 0.05, 0.95))

def m_noise(home, away):       # bruit contrôlé
    return float(np.clip(0.5 + (home-away)/120.0, 0.05, 0.95))

def m_clip(home, away):        # clip moyenne
    return float(np.clip((home+away-160)/16, 0.05, 0.95))

def ensemble_over_prob(total_mu, sim_over):
    # 7 modèles + Poisson
    home_mu = total_mu/2.0
    away_mu = total_mu/2.0
    models = [
        m_lin(home_mu, away_mu),
        m_poly(home_mu, away_mu),
        m_logit(home_mu, away_mu),
        m_elo_mix(home_mu, away_mu),
        m_noise(home_mu, away_mu),
        m_clip(home_mu, away_mu),
        0.5 + (total_mu-165)/30.0  # heuristique
    ]
    models = [float(np.clip(m, 0.01, 0.99)) for m in models]
    # ajoute la proba Poisson simulée (sim_over)
    models.append(float(np.clip(sim_over, 0.01, 0.99)))
    return float(np.mean(models))

# -----------------------------
# UI
# -----------------------------
st.title("🏀 BasketAI Ultra+ v15 — Plus/Moins IA (ensemble)")

ok, info = check_api_status()
if ok:
    st.success(f"✅ Connecté à API-Sports · Plan **{info['plan']}** · "
               f"Requêtes aujourd’hui: **{info['used']} / {info['limit']}**")
    st.session_state["offline"] = False
else:
    st.warning("🟡 Mode hors-ligne : l’interface fonctionnera avec des valeurs par défaut (ligues & stats API désactivées).")
    st.session_state["offline"] = True

col_top1, col_top2 = st.columns([1,1])
with col_top1:
    if st.button("🔍 Tester la clé API"):
        ok2, info2 = check_api_status()
        if ok2:
            st.success(f"Clé OK — Plan **{info2['plan']}** (jusqu’au {info2['end']}).")
            st.session_state["offline"] = False
        else:
            st.error("Clé invalide / quota / réseau — Mode hors-ligne.")
            st.session_state["offline"] = True

season = st.selectbox("📅 Saison", [DEFAULT_SEASON, "2023-2024", "2022-2023"], index=0)

if not st.session_state.get("offline"):
    leagues = get_leagues(season)
    league_label = st.selectbox("🌍 Ligue", [l["label"] for l in leagues])
    league_id = next(l["id"] for l in leagues if l["label"] == league_label)

    teams = get_teams(league_id, season)
    names = [t["name"] for t in teams]
    team_home_name = st.selectbox("🏠 Équipe domicile", names, key="home_sel")
    team_away_name = st.selectbox("🚀 Équipe extérieure", names, key="away_sel")

    # Récupère ligne O/U réelle
    ou_col1, ou_col2 = st.columns([1,1])
    with ou_col1:
        if st.button("📡 Récupérer la ligne Over/Under (bookmakers)"):
            line = get_ou_line(league_id, season)
            if line:
                st.session_state["ou_line"] = float(line)
                st.success(f"🧾 Ligne détectée : **{line:.2f}**")
            else:
                st.warning("Aucune ligne trouvée pour cette ligue/saison (books indisponibles).")
    default_line = st.session_state.get("ou_line", 165.5)
else:
    league_label = st.selectbox("🌍 Ligue (mode hors-ligne)", ["NBA", "Euroligue", "Pro A (France)", "Liga ACB (Espagne)"])
    team_home_name = st.selectbox("🏠 Équipe domicile (hors ligne)", ["Team A", "Team B", "Team C"])
    team_away_name = st.selectbox("🚀 Équipe extérieure (hors ligne)", ["Team X", "Team Y", "Team Z"])
    default_line = st.session_state.get("ou_line", 165.5)

# Ligne Over/Under (modifiable)
ou_line = st.number_input("✍️ Ligne Over/Under (si vide, utilisez le bouton de récupération ci-dessus)", value=float(default_line), step=0.5)

st.divider()
if st.button("🏀 Lancer la prédiction"):
    if team_home_name == team_away_name:
        st.error("Choisis deux équipes différentes.")
        st.stop()

    # Hypothèse de points moyens si pas de stats détaillées (peut être amélioré via endpoint fixtures/stats)
    mu_home = 84.5
    mu_away = 82.5
    total_mu = mu_home + mu_away

    # Simulation Poisson
    sim = poisson_sim(mu_home, mu_away, n=180000)
    mean_pts = float(np.mean(sim))
    std_pts  = float(np.std(sim))
    over_prob_sim = float(np.mean(sim > ou_line))
    under_prob_sim= 1.0 - over_prob_sim

    # Ensemble
    over_prob = ensemble_over_prob(total_mu, over_prob_sim)
    under_prob = 1.0 - over_prob

    # Résultats
    st.subheader("Résultats")
    st.write(f"**Moyenne simulée** : {mean_pts:.1f} pts ± {std_pts:.1f}")
    st.write(f"**Ligne** {ou_line:.2f} — **P(Over)** = {over_prob*100:.1f}% · **P(Under)** = {under_prob*100:.1f}%")

    # Histogramme
    fig = plt.figure(figsize=(8,3.8))
    plt.hist(sim, bins=60)
    plt.axvline(mean_pts, linestyle="--", linewidth=2)
    plt.axvline(ou_line, linestyle="--", linewidth=2)
    plt.xlabel("Total points")
    plt.ylabel("Fréquence")
    plt.title("Distribution simulée du total de points")
    st.pyplot(fig)

st.caption("Créé avec 💜 — BasketAI Ultra+ v15")
