# app.py — Football : Ensemble 7 algorithmes + Stacking (sans API)
import math, functools, time
import numpy as np
import pandas as pd
import requests as rq
import streamlit as st
from scipy.stats import poisson

# ML
from sklearn.linear_model import PoissonRegressor, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="Prédicteur Foot — 7 algos + Stacking", page_icon="⚽", layout="wide")
st.title("⚽ Prédicteur de Football — Ensemble de 7 algorithmes + Stacking")
st.caption("Données : football-data.co.uk (CSV publics). Aucune clé API requise.")

# ------------------ Ligues (codes football-data) ------------------
LEAGUES = {
    "Premier League (Angleterre)": "E0",
    "La Liga (Espagne)": "SP1",
    "Serie A (Italie)": "I1",
    "Bundesliga (Allemagne)": "D1",
    "Ligue 1 (France)": "F1",
}

def season_code(start_year: int) -> str:
    end_year = (start_year + 1) % 100
    return f"{str(start_year % 100).zfill(2)}{str(end_year).zfill(2)}"

@st.cache_data(ttl=3600)
def load_csv(league_code: str, start_year: int) -> pd.DataFrame:
    sc = season_code(start_year)
    url = f"https://www.football-data.co.uk/mmz4281/{sc}/{league_code}.csv"
    r = rq.get(url, timeout=25, headers={"User-Agent":"foot-ensemble/1.0"})
    r.raise_for_status()
    df = pd.read_csv(pd.compat.StringIO(r.text))
    # Harmoniser
    df = df.rename(columns={"HomeTeam":"Home","AwayTeam":"Away","FTHG":"HG","FTHA":"AG","Date":"Date"})
    keep = [c for c in ["Date","Home","Away","HG","AG"] if c in df.columns]
    df = df[keep].dropna()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df["HG"] = pd.to_numeric(df["HG"], errors="coerce")
    df["AG"] = pd.to_numeric(df["AG"], errors="coerce")
    df = df.dropna(subset=["Date","HG","AG"]).sort_values("Date").reset_index(drop=True)
    return df

def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    last = {}
    rh, ra = [], []
    for _, r in df.iterrows():
        d, h, a = r["Date"], r["Home"], r["Away"]
        rh.append((d - last.get(h, d)).days if h in last else 7)
        ra.append((d - last.get(a, d)).days if a in last else 7)
        last[h] = d; last[a] = d
    out = df.copy()
    out["rest_home"], out["rest_away"] = rh, ra
    return out

def ema_team_rates(df: pd.DataFrame, alpha=0.25):
    """
    Calcule des taux attaque/défense lissés (EMA) séparés domicile/extérieur.
    Retourne dicts + moyennes ligue (dom/ext).
    """
    teams = sorted(set(df["Home"]).union(set(df["Away"])))
    att_home, def_home, att_away, def_away = {}, {}, {}, {}
    for t in teams:
        H = df[df["Home"]==t][["HG","AG"]].copy()
        if not H.empty:
            H["ema_for"] = H["HG"].ewm(alpha=alpha, adjust=False).mean()
            H["ema_against"] = H["AG"].ewm(alpha=alpha, adjust=False).mean()
            att_home[t] = float(H["ema_for"].iloc[-1]); def_home[t] = float(H["ema_against"].iloc[-1])
        else:
            att_home[t] = np.nan; def_home[t] = np.nan
        A = df[df["Away"]==t][["HG","AG"]].copy()
        if not A.empty:
            A["ema_for"] = A["AG"].ewm(alpha=alpha, adjust=False).mean()
            A["ema_against"] = A["HG"].ewm(alpha=alpha, adjust=False).mean()
            att_away[t] = float(A["ema_for"].iloc[-1]); def_away[t] = float(A["ema_against"].iloc[-1])
        else:
            att_away[t] = np.nan; def_away[t] = np.nan
    avgH, avgA = df["HG"].mean(), df["AG"].mean()
    for t in teams:
        if np.isnan(att_home[t]): att_home[t] = avgH
        if np.isnan(def_home[t]): def_home[t] = avgA
        if np.isnan(att_away[t]): att_away[t] = avgA
        if np.isnan(def_away[t]): def_away[t] = avgH
    return att_home, def_home, att_away, def_away, avgH, avgA

def compute_elo(df: pd.DataFrame, K=20, home_adv_elo=60):
    teams = sorted(set(df["Home"]).union(set(df["Away"])))
    elo = {t:1500.0 for t in teams}
    rec = []
    for _, r in df.iterrows():
        h,a,hg,ag = r["Home"], r["Away"], r["HG"], r["AG"]
        Eh = 1/(1+10**((elo[a]-elo[h]-home_adv_elo)/400))
        Ra_h = 1.0 if hg>ag else (0.5 if hg==ag else 0.0)
        Ra_a = 1.0 - Ra_h
        elo[h] += K*(Ra_h-Eh)
        elo[a] += K*(Ra_a-(1-Eh))
        rec.append({"Date":r["Date"],"Home":h,"Away":a,"elo_home":elo[h],"elo_away":elo[a]})
    return df.merge(pd.DataFrame(rec), on=["Date","Home","Away"])

def build_feature_table(df: pd.DataFrame, alpha=0.25):
    df = add_rest_days(df)
    df = compute_elo(df, K=20, home_adv_elo=60)
    attH, defH, attA, defA, avgH, avgA = ema_team_rates(df, alpha=alpha)
    rows = []
    for _, r in df.iterrows():
        h, a = r["Home"], r["Away"]
        row = {
            "elo_diff": r["elo_home"] - r["elo_away"],
            "att_diff": attH[h] - defA[a],
            "def_diff": attA[a] - defH[h],
            "form_diff": (attH[h]-defH[h]) - (attA[a]-defA[a]),
            "rest_diff": r["rest_home"] - r["rest_away"],
            "home_adv": 1,
            "HG": r["HG"],
            "AG": r["AG"],
            "Home": h, "Away": a
        }
        rows.append(row)
    Xy = pd.DataFrame(rows)
    X = Xy[["elo_diff","att_diff","def_diff","form_diff","rest_diff","home_adv"]].copy()
    yH = Xy["HG"].to_numpy()
    yA = Xy["AG"].to_numpy()
    meta = {"attH":attH,"defH":defH,"attA":attA,"defA":defA,"avgH":avgH,"avgA":avgA}
    return Xy, X, yH, yA, meta

# -------------- Ensemble 7 modèles + stacking --------------
def get_base_models():
    # pipelines avec standardisation selon besoin
    return [
        ("poisson", PoissonRegressor(alpha=0.5, max_iter=500)),
        ("ridge",   make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("enet",    make_pipeline(StandardScaler(), ElasticNet(alpha=0.02, l1_ratio=0.3))),
        ("knn",     make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15, weights="distance"))),
        ("rf",      RandomForestRegressor(n_estimators=300, random_state=42)),
        ("et",      ExtraTreesRegressor(n_estimators=500, random_state=42)),
        ("gbr",     GradientBoostingRegressor(random_state=42)),
    ]

def fit_stack(X: pd.DataFrame, y: np.ndarray):
    base = get_base_models()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    OOF = np.zeros((len(X), len(base)))
    fitted = []
    for j,(name,model) in enumerate(base):
        for tr, va in kf.split(X):
            m = model
            m.fit(X.iloc[tr], y[tr])
            OOF[va, j] = m.predict(X.iloc[va])
        model.fit(X, y)    # fit final sur tout
        fitted.append((name, model))
    # méta-modèle simple : ridge
    meta = Ridge(alpha=1.0)
    meta.fit(OOF, y)
    return fitted, meta

def predict_stack(fitted, meta, Xnew: pd.DataFrame) -> np.ndarray:
    P = np.column_stack([m.predict(Xnew) for _,m in fitted])
    yhat = meta.predict(P)
    return np.clip(yhat, 0.01, 10.0)

# -------------- Probabilités issues des lambdas (Poisson) --------------
def probs_from_lambdas(mu_h, mu_a, line=2.5, max_goals=10):
    pH = [poisson.pmf(i, mu_h) for i in range(max_goals+1)]
    pA = [poisson.pmf(i, mu_a) for i in range(max_goals+1)]
    M = np.outer(pH, pA)
    win_home  = np.tril(M, -1).sum()
    draw      = np.trace(M)
    win_away  = np.triu(M, 1).sum()
    over_line = sum(M[i,j] for i in range(max_goals+1) for j in range(max_goals+1) if i+j > line)
    i_star, j_star = np.unravel_index(np.argmax(M), M.shape)
    return win_home, draw, win_away, over_line, M, (i_star, j_star)

# ========================== UI ==========================
left, right = st.columns(2)
league_label = left.selectbox("Ligue", list(LEAGUES.keys()))
season = right.number_input("Saison (année de début, ex: 2023 pour 2023-24)", 2012, 2025, 2023, 1)

with st.spinner("Chargement des données…"):
    df = load_csv(LEAGUES[league_label], int(season))
if df.empty:
    st.error("Données indisponibles pour cette ligue/saison.")
    st.stop()

Xy, X, yH, yA, meta = build_feature_table(df, alpha=0.25)

teams = sorted(set(Xy["Home"]).union(set(Xy["Away"])))
colA, colB = st.columns(2)
home_team = colA.selectbox("Équipe à domicile", teams, index=0)
away_team = colB.selectbox("Équipe à l'extérieur", [t for t in teams if t != home_team], index=1 if len(teams)>1 else 0)

st.sidebar.header("Paramètres")
alpha = st.sidebar.slider("EMA (forme récente)", 0.05, 0.50, 0.25, 0.01)
line = st.sidebar.slider("Ligne Over/Under (total buts)", 1.5, 4.5, 2.5, 0.5)
retrain = st.sidebar.checkbox("Recalculer les features avec ce niveau d'EMA", value=False)

# Rebuild features si l’utilisateur change alpha
if retrain:
    Xy, X, yH, yA, meta = build_feature_table(df, alpha=float(alpha))

with st.spinner("Apprentissage des 7 modèles + stacking…"):
    fitted_H, meta_H = fit_stack(X, yH)
    fitted_A, meta_A = fit_stack(X, yA)

# Construire la ligne de features pour le match demandé
def build_one_row(Xy, meta, home_team, away_team):
    # prend la dernière valeur connue des différences pour ces équipes ; fallback moyenne
    lastH = Xy[(Xy["Home"]==home_team) & (Xy["Away"]==away_team)]
    if lastH.empty:
        # si pas d'affrontement, on prend les dernières lignes de chaque équipe
        last_home = Xy[Xy["Home"]==home_team].tail(1)
        last_away = Xy[Xy["Away"]==away_team].tail(1)
        elo_diff = float((last_home["elo_diff"].mean() - (-last_away["elo_diff"].mean())) if not last_home.empty and not last_away.empty else Xy["elo_diff"].mean())
        att_diff = float((last_home["att_diff"].mean() - Xy["att_diff"].mean()) if not last_home.empty else Xy["att_diff"].mean())
        def_diff = float((last_away["def_diff"].mean() - Xy["def_diff"].mean()) if not last_away.empty else Xy["def_diff"].mean())
        form_diff= float((last_home["form_diff"].mean() - last_away["form_diff"].mean()) if not last_home.empty and not last_away.empty else Xy["form_diff"].mean())
        rest_diff= float((last_home["rest_diff"].mean() - last_away["rest_diff"].mean()) if not last_home.empty and not last_away.empty else Xy["rest_diff"].mean())
    else:
        row = lastH.iloc[-1]
        elo_diff, att_diff, def_diff, form_diff, rest_diff = map(float, [
            row["elo_diff"], row["att_diff"], row["def_diff"], row["form_diff"], row["rest_diff"]
        ])
    return pd.DataFrame([{
        "elo_diff": elo_diff, "att_diff": att_diff, "def_diff": def_diff,
        "form_diff": form_diff, "rest_diff": rest_diff, "home_adv": 1
    }])

x_new = build_one_row(Xy, meta, home_team, away_team)

if st.button("🔮 Prédire"):
    mu_home = float(predict_stack(fitted_H, meta_H, x_new)[0])
    mu_away = float(predict_stack(fitted_A, meta_A, x_new)[0])

    win_home, draw, win_away, over_line, M, (i_star, j_star) = probs_from_lambdas(mu_home, mu_away, line=float(line))

    st.subheader("Résultats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Buts attendus (dom.)", f"{mu_home:.2f}")
    c2.metric("Buts attendus (ext.)", f"{mu_away:.2f}")
    c3.metric("Total attendu", f"{(mu_home+mu_away):.2f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("1 (dom.)", f"{win_home*100:.1f}%")
    c5.metric("N",        f"{draw*100:.1f}%")
    c6.metric("2 (ext.)", f"{win_away*100:.1f}%")

    st.info(f"Over {line:.1f} : **{over_line*100:.1f}%**  |  Under {line:.1f} : **{(1-over_line)*100:.1f}%**")

    st.write(f"**Score le plus probable** : {home_team} **{i_star} - {j_star}** {away_team}  "
             f"(p={M[i_star, j_star]*100:.1f}%)")

    with st.expander("10 derniers matchs (saison)"):
        lastH = df[(df["Home"]==home_team) | (df["Away"]==home_team)].tail(10)[["Date","Home","HG","AG","Away"]]
        lastA = df[(df["Home"]==away_team) | (df["Away"]==away_team)].tail(10)[["Date","Home","HG","AG","Away"]]
        st.write(f"**{home_team}**"); st.dataframe(lastH, use_container_width=True)
        st.write(f"**{away_team}**"); st.dataframe(lastA, use_container_width=True)

st.caption("Ensemble : Poisson, Ridge, ElasticNet, KNN, RandomForest, ExtraTrees, GradientBoosting + méta Ridge. "
           "Features : Elo, EMA attaque/défense, forme, jours de repos, domicile.")
