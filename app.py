# app.py — Foot sans API : Score exact, Over/Under, BTTS
# 20 algos + Bayésien + Elo + Dixon–Coles + Calibration (Isotonic)
import os, math, difflib
import numpy as np
import pandas as pd
import streamlit as st
from math import exp, factorial

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor,
    HuberRegressor, RANSACRegressor, TheilSenRegressor, LassoLars,
    PoissonRegressor, TweedieRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
)

st.set_page_config(page_title="⚽ Score exact / O-U / BTTS — PRO", page_icon="⚽", layout="wide")
st.title("⚽ Football — Score exact • Over/Under • BTTS (20 algos + Bayésien + Elo + Dixon–Coles + Calibration)")
st.caption("Données publiques via data_etl.py — aucun abonnement API. Tape uniquement les 2 équipes.")

# ----------------------- Chargement données -----------------------
@st.cache_data(ttl=3600)
def load_data():
    path = "data/all_leagues.csv"
    if not os.path.exists(path):
        st.warning("Données introuvables. Lance d'abord data_etl.py pour télécharger les CSV publics.")
        return pd.DataFrame(columns=["date","league","home_team","away_team","goals_home","goals_away"])
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date","home_team","away_team","goals_home","goals_away"]).reset_index(drop=True)
    return df.sort_values("date")

df = load_data()
if df.empty:
    st.stop()

# ----------------------- Paramètres UI -----------------------
colL, colK, colMax = st.columns([2,1,1])
league = colL.selectbox("Ligue (optionnel)", ["(Toutes)"] + sorted(df["league"].unique().tolist()), index=0)
K = colK.slider("Matchs récents utilisés (K)", 5, 15, 10)
max_g = int(colMax.number_input("Max buts par équipe (matrice score)", 4, 10, 5, 1))

dfl = df if league == "(Toutes)" else df[df["league"] == league].copy()
if dfl.empty:
    st.error("Pas de données pour cette ligue.")
    st.stop()

teams_all = sorted(set(dfl["home_team"]).union(set(dfl["away_team"])))
t_home_raw = st.text_input("Équipe à domicile (approximatif ok)", "")
t_away_raw = st.text_input("Équipe à l'extérieur (approximatif ok)", "")

def fuzzy_best(name_raw):
    if not name_raw:
        return None, []
    choices = difflib.get_close_matches(name_raw, teams_all, n=5, cutoff=0.5)
    best = choices[0] if choices else None
    return best, choices

best_home, sugg_home = fuzzy_best(t_home_raw)
best_away, sugg_away = fuzzy_best(t_away_raw)

if t_home_raw:
    st.write("➡️ Interprété (domicile) :", f"**{best_home}**" if best_home else "Aucune correspondance")
    if len(sugg_home) > 1: st.caption("Suggestions : " + ", ".join(sugg_home))
if t_away_raw:
    st.write("➡️ Interprété (extérieur) :", f"**{best_away}**" if best_away else "Aucune correspondance")
    if len(sugg_away) > 1: st.caption("Suggestions : " + ", ".join(sugg_away))

line = st.number_input("Ligne Over/Under (total buts)", 0.5, 6.5, 2.5, 0.5)

# ----------------------- Fonctions utilitaires -----------------------
def team_history(d, team, K=10):
    rows = []
    for _, r in d.iterrows():
        if r["home_team"] == team:
            rows.append({"date": r["date"], "pf": r["goals_home"], "pa": r["goals_away"], "home": 1})
        elif r["away_team"] == team:
            rows.append({"date": r["date"], "pf": r["goals_away"], "pa": r["goals_home"], "home": 0})
    t = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if t.empty: return t
    t["pf_k"] = t["pf"].rolling(K, min_periods=1).mean().shift(1)
    t["pa_k"] = t["pa"].rolling(K, min_periods=1).mean().shift(1)
    t["tot_k"] = (t["pf"] + t["pa"]).rolling(K, min_periods=1).mean().shift(1)
    t["home_rate_k"] = t["home"].rolling(K, min_periods=1).mean().shift(1)
    return t

def poisson(lam, k):  # P(X=k)
    return (lam**k * exp(-lam)) / factorial(k)

# ----------------------- Elo (features) -----------------------
@st.cache_data(ttl=3600)
def build_elo_table(d, base=1500, Kfac=20, H=60):
    teams = sorted(set(d["home_team"]).union(set(d["away_team"])))
    R = {t: base for t in teams}
    recs = []
    for _, r in d.sort_values("date").iterrows():
        h, a = r["home_team"], r["away_team"]
        Rh = R[h] + H; Ra = R[a]
        Eh = 1 / (1 + 10 ** ((Ra - Rh)/400))
        Ea = 1 - Eh
        gh, ga = r["goals_home"], r["goals_away"]
        if gh > ga: Sh, Sa = 1.0, 0.0
        elif gh < ga: Sh, Sa = 0.0, 1.0
        else: Sh, Sa = 0.5, 0.5
        R[h] = R[h] + Kfac * (Sh - Eh)
        R[a] = R[a] + Kfac * (Sa - Ea)
        recs.append({"date": r["date"], "home": h, "away": a, "elo_h": R[h], "elo_a": R[a]})
    elo_df = pd.DataFrame(recs)
    return R, elo_df

def last_elo_for(d, team, R_final):
    return float(R_final.get(team, 1500.0))

# ----------------------- Dataset de features -----------------------
def build_dataset(d, K=10, R_final=None):
    games = []
    teams_all_local = sorted(set(d["home_team"]).union(set(d["away_team"])))
    hist = {t: team_history(d, t, K=K) for t in teams_all_local}
    for _, r in d.iterrows():
        h, a, dt = r["home_team"], r["away_team"], r["date"]
        th, ta = hist[h], hist[a]
        if th.empty or ta.empty: continue
        thp = th[th["date"] < dt].tail(1)
        tap = ta[ta["date"] < dt].tail(1)
        if thp.empty or tap.empty: continue
        row = {
            "date": dt, "league": r["league"], "home": h, "away": a,
            "y_home": r["goals_home"], "y_away": r["goals_away"],
            "H_pf_k": float(thp["pf_k"]), "H_pa_k": float(thp["pa_k"]),
            "A_pf_k": float(tap["pf_k"]), "A_pa_k": float(tap["pa_k"]),
            "H_tot_k": float(thp["tot_k"]), "A_tot_k": float(tap["tot_k"]),
            "H_home_rate": float(thp["home_rate_k"]), "A_home_rate": float(tap["home_rate_k"]),
            "home_adv": 1.0
        }
        if R_final is not None:
            row["elo_home"] = last_elo_for(d, h, R_final)
            row["elo_away"] = last_elo_for(d, a, R_final)
            row["elo_diff"] = row["elo_home"] - row["elo_away"]
        games.append(row)
    return pd.DataFrame(games).dropna()

# ----------------------- 20 modèles -----------------------
X_BASE = ["H_pf_k","H_pa_k","A_pf_k","A_pa_k","H_tot_k","A_tot_k","H_home_rate","A_home_rate","home_adv"]
X_ELO  = ["elo_home","elo_away","elo_diff"]
MODELS = [
    ("LinReg",  make_pipeline(StandardScaler(), LinearRegression())),
    ("Ridge",   make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
    ("Lasso",   make_pipeline(StandardScaler(), Lasso(alpha=0.01))),
    ("Elastic", make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.3))),
    ("LassoLars", make_pipeline(StandardScaler(), LassoLars(alpha=0.005))),
    ("SGD",     make_pipeline(StandardScaler(), SGDRegressor(max_iter=2000, tol=1e-3))),
    ("Huber",   make_pipeline(StandardScaler(), HuberRegressor())),
    ("TheilSen", TheilSenRegressor()),
    ("RANSAC",  RANSACRegressor(base_estimator=LinearRegression())),
    ("KNN",     make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15, weights="distance"))),
    ("SVR",     make_pipeline(StandardScaler(), SVR(kernel="rbf", C=2.0, epsilon=0.1))),
    ("DT",      DecisionTreeRegressor(random_state=42)),
    ("RF",      RandomForestRegressor(n_estimators=400, random_state=42)),
    ("ET",      ExtraTreesRegressor(n_estimators=600, random_state=42)),
    ("GB",      GradientBoostingRegressor(random_state=42)),
    ("HGB",     HistGradientBoostingRegressor(random_state=42)),
    ("Ada",     AdaBoostRegressor(random_state=42)),
    ("Bagging", BaggingRegressor(random_state=42)),
    ("Poisson", make_pipeline(StandardScaler(with_mean=False), PoissonRegressor(alpha=1e-8, max_iter=2000))),
    ("Tweedie", TweedieRegressor(power=1.5, alpha=0.001)),
]

def fit_predict_targets(X, y, x_new):
    preds, used = [], []
    for name, m in MODELS:
        try:
            m.fit(X, y)
            preds.append(float(m.predict(x_new)[0])); used.append(name)
        except Exception:
            continue
    mu = float(np.mean(preds)); sd = float(np.std(preds)) if len(preds)>1 else 0.0
    return mu, sd, preds, used

# ----------------------- Bayésien Gamma–Poisson -----------------------
def league_gamma_prior(d):
    goals = pd.concat([d["goals_home"], d["goals_away"]]).astype(float)
    m = goals.mean() if not goals.empty else 1.3
    v = goals.var(ddof=1) if len(goals)>1 else 1.0
    v = max(v, 0.5)
    alpha0 = (m**2)/v
    beta0  = m/v  # rate
    return float(alpha0), float(beta0)

def bayes_lambda(sum_goals_lastK, K, alpha0, beta0):
    alpha_p = alpha0 + max(0.0, float(sum_goals_lastK))
    beta_p  = beta0 + max(1.0, float(K))
    return float(alpha_p / beta_p)

# ----------------------- Dixon–Coles -----------------------
def dixon_coles_correction(i, j, rho):
    if i == 0 and j == 0:
        return 1 - rho
    if i == 0 and j == 1:
        return 1 + rho
    if i == 1 and j == 0:
        return 1 + rho
    if i == 1 and j == 1:
        return 1 - rho
    return 1.0

def dc_matrix(lh, la, max_g, rho):
    mat = np.zeros((max_g+1, max_g+1))
    for i in range(max_g+1):
        for j in range(max_g+1):
            base = poisson(lh, i)*poisson(la, j)
            mat[i,j] = base * dixon_coles_correction(i, j, rho)
    s = mat.sum()
    return mat / (s if s>0 else 1.0)

@st.cache_data(ttl=3600)
def estimate_rho(d, K=10, grid=np.linspace(-0.2, 0.2, 21)):
    ds = build_dataset(d, K=K, R_final=None)
    if ds.empty:
        return 0.0
    best_rho, best_ll = 0.0, -1e18
    for rho in grid:
        ll = 0.0
        for _, r in ds.iterrows():
            lh = max(r["H_pf_k"], 1e-4); la = max(r["A_pf_k"], 1e-4)
            i = int(r["y_home"]); j = int(r["y_away"])
            p = poisson(lh, i)*poisson(la, j) * dixon_coles_correction(i, j, rho)
            if p > 0: ll += np.log(p)
        if ll > best_ll:
            best_ll, best_rho = ll, rho
    return float(best_rho)

# ----------------------- Calibration (Isotonic) -----------------------
@st.cache_data(ttl=3600)
def fit_isotonic_calibrators(d, K=10, ou_line=2.5):
    ds = build_dataset(d, K=K, R_final=None)
    if ds.empty:
        return None, None
    lh = np.clip(ds["H_pf_k"].values, 1e-4, 10)
    la = np.clip(ds["A_pf_k"].values, 1e-4, 10)
    max_g = 8
    p_over_raw, p_btts_raw, y_over, y_btts = [], [], [], []
    for k in range(len(ds)):
        mat = np.zeros((max_g+1, max_g+1))
        for i in range(max_g+1):
            for j in range(max_g+1):
                mat[i,j] = poisson(lh[k], i)*poisson(la[k], j)
        mat /= mat.sum()
        p_over_raw.append(float(np.sum([mat[i,j] for i in range(max_g+1) for j in range(max_g+1) if (i+j) > ou_line])))
        p_btts_raw.append(float(np.sum([mat[i,j] for i in range(1, max_g+1) for j in range(1, max_g+1)])))
        tot = int(ds.iloc[k]["y_home"] + ds.iloc[k]["y_away"])
        y_over.append(int(tot > ou_line))
        yb = int((ds.iloc[k]["y_home"] > 0) and (ds.iloc[k]["y_away"] > 0))
        y_btts.append(yb)
    iso_over = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(np.array(p_over_raw), np.array(y_over))
    iso_btts = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(np.array(p_btts_raw), np.array(y_btts))
    return iso_over, iso_btts

# ----------------------- Construction ligne actuelle -----------------------
def current_row(d, home, away, K, R_final):
    th = team_history(d, home, K=K); ta = team_history(d, away, K=K)
    if th.empty or ta.empty: return None, None, None
    thp = th.tail(1).iloc[0]; tap = ta.tail(1).iloc[0]
    row = pd.DataFrame([{
        "H_pf_k": thp["pf_k"], "H_pa_k": thp["pa_k"], "A_pf_k": tap["pf_k"], "A_pa_k": tap["pa_k"],
        "H_tot_k": thp["tot_k"], "A_tot_k": tap["tot_k"],
        "H_home_rate": thp["home_rate_k"], "A_home_rate": tap["home_rate_k"],
        "home_adv": 1.0,
        "elo_home": last_elo_for(d, home, R_final),
        "elo_away": last_elo_for(d, away, R_final),
        "elo_diff": last_elo_for(d, home, R_final) - last_elo_for(d, away, R_final),
    }])
    sum_home_gf = float(th.tail(K)["pf"].sum()) if len(th)>=1 else 0.0
    sum_away_gf = float(ta.tail(K)["pf"].sum()) if len(ta)>=1 else 0.0
    return row, sum_home_gf, sum_away_gf

# ----------------------- Pipeline principal -----------------------
ready = (best_home is not None) and (best_away is not None) and (best_home != best_away)

if st.button("🔮 Prédire") and ready:
    home = best_home
    away = best_away

    R_final, _elo_df = build_elo_table(dfl)
    dataset = build_dataset(dfl, K=K, R_final=R_final)
    if dataset.empty or len(dataset) < 100:
        st.warning("Peu de données featurisées. Laisse tourner l’ETL sur plusieurs ligues/jours pour plus d'historique.")
    X_cols = X_BASE + X_ELO

    row_now, home_sumK, away_sumK = current_row(dfl, home, away, K, R_final)
    if row_now is None:
        st.error("Données insuffisantes pour construire les features (K trop grand ?).")
        st.stop()

    x_new = row_now[X_cols].fillna(row_now.median(numeric_only=True))
    X = dataset[X_cols].copy()
    y_home = dataset["y_home"].values
    y_away = dataset["y_away"].values

    lam_h_ens, sd_h, preds_h, used_h = fit_predict_targets(X, y_home, x_new)
    lam_a_ens, sd_a, preds_a, used_a = fit_predict_targets(X, y_away, x_new)

    a0, b0 = league_gamma_prior(dfl)
    lam_h_bayes = bayes_lambda(home_sumK, K, a0, b0)
    lam_a_bayes = bayes_lambda(away_sumK, K, a0, b0)

    w = 0.45 if K < 10 else 0.55
    lam_h = max(1e-4, w*lam_h_bayes + (1-w)*lam_h_ens)
    lam_a = max(1e-4, w*lam_a_bayes + (1-w)*lam_a_ens)

    rho = estimate_rho(dfl, K=K)
    mat = dc_matrix(lam_h, lam_a, max_g=max_g, rho=rho)

    scores = [{"Score": f"{i}-{j}", "Prob%": 100*mat[i,j]} for i in range(max_g+1) for j in range(max_g+1)]
    score_df = pd.DataFrame(scores).sort_values("Prob%", ascending=False)
    top5 = score_df.head(5)

    total_mean = sum((i+j)*mat[i,j] for i in range(max_g+1) for j in range(max_g+1))
    var_tot = sum(((i+j)-total_mean)**2 * mat[i,j] for i in range(max_g+1) for j in range(max_g+1))
    std_tot = float(np.sqrt(var_tot))

    over_prob_raw = float(sum(mat[i,j] for i in range(max_g+1) for j in range(max_g+1) if (i+j) > line))
    under_prob_raw = 1 - over_prob_raw
    p_btts_yes_raw = float(sum(mat[i,j] for i in range(1, max_g+1) for j in range(1, max_g+1)))
    p_btts_no_raw  = 1 - p_btts_yes_raw

    iso_over, iso_btts = fit_isotonic_calibrators(dfl, K=K, ou_line=line)
    if (iso_over is not None) and (iso_btts is not None):
        over_prob = float(iso_over.predict([over_prob_raw])[0])
        p_btts_yes = float(iso_btts.predict([p_btts_yes_raw])[0])
    else:
        over_prob = over_prob_raw
        p_btts_yes = p_btts_yes_raw
    under_prob = 1 - over_prob
    p_btts_no  = 1 - p_btts_yes

    st.success(f"{home} vs {away} — λ_home ≈ {lam_h:.2f} • λ_away ≈ {lam_a:.2f} | ρ (Dixon–Coles) ≈ {rho:.3f}")
    st.info(f"Total attendu ≈ **{total_mean:.2f}** (± {std_tot:.2f})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Over {line:.1f}", f"{over_prob*100:.1f}%")
    c2.metric(f"Under {line:.1f}", f"{under_prob*100:.1f}%")
    c3.metric("BTTS = OUI", f"{p_btts_yes*100:.1f}%")
    c4.metric("BTTS = NON", f"{(1-p_btts_yes)*100:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 5 scores probables (Dixon–Coles)")
        st.dataframe(top5, use_container_width=True)
    with col2:
        st.subheader("Détail modèles (λ)")
        st.write(pd.DataFrame({"Modèle": used_h, "λ_home": preds_h[:len(used_h)]}))
        st.write(pd.DataFrame({"Modèle": used_a, "λ_away": preds_a[:len(used_a)]}))
else:
    st.info("Entre uniquement les 2 équipes (domicile / extérieur), puis clique **🔮 Prédire**.")
