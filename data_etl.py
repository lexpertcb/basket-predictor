import os, io, requests, pandas as pd

LEAGUES = {
    "EPL":    ("Premier League", "https://www.football-data.co.uk/mmz4281/2425/E0.csv"),
    "LALIGA": ("LaLiga",         "https://www.football-data.co.uk/mmz4281/2425/SP1.csv"),
    "SERIEA": ("Serie A",        "https://www.football-data.co.uk/mmz4281/2425/I1.csv"),
    "BUND":   ("Bundesliga",     "https://www.football-data.co.uk/mmz4281/2425/D1.csv"),
    "L1":     ("Ligue 1",        "https://www.football-data.co.uk/mmz4281/2425/F1.csv"),
}

os.makedirs("data", exist_ok=True)
frames = []
for code, (name, url) in LEAGUES.items():
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content), encoding="latin1")
        keep = ["Date","HomeTeam","AwayTeam","FTHG","FTAG"]
        df = df[[c for c in keep if c in df.columns]].copy()
        df["league"] = name
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date","HomeTeam","AwayTeam","FTHG","FTAG"])
        df.rename(columns={
            "Date":"date","HomeTeam":"home_team","AwayTeam":"away_team",
            "FTHG":"goals_home","FTAG":"goals_away"
        }, inplace=True)
        df.to_csv(f"data/{code}.csv", index=False)
        frames.append(df)
        print(f"[OK] {name}: {len(df)} lignes")
    except Exception as e:
        print(f"[WARN] {name}: {e}")

if frames:
    big = pd.concat(frames).sort_values("date").reset_index(drop=True)
    big.to_csv("data/all_leagues.csv", index=False)
    print(f"[MERGE] total {len(big)} -> data/all_leagues.csv")
else:
    print("Aucune donnée téléchargée.")
