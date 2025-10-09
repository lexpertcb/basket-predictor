
import os
import requests
import pandas as pd

# ============================================================
# 🔧 CONFIGURATION GLOBALE
# ============================================================

API_KEY = "bb3ba63a8bfab1020390fe28bd180522"
API_URL = "https://v1.basketball.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 📦 FONCTIONS UTILITAIRES
# ============================================================

def get_leagues():
    """Récupère toutes les ligues disponibles via l’API"""
    url = f"{API_URL}/leagues"
    res = requests.get(url, headers=HEADERS)
    data = res.json()
    leagues = [
        {"id": l["id"], "name": l["name"], "country": l["country"]["name"]}
        for l in data.get("response", []) if "country" in l
    ]
    return leagues


def get_teams(league_id, season="2024-2025"):
    """Récupère toutes les équipes d’une ligue"""
    url = f"{API_URL}/teams?league={league_id}&season={season}"
    res = requests.get(url, headers=HEADERS)
    data = res.json()
    teams = []
    for t in data.get("response", []):
        teams.append({
            "id": t["id"],
            "name": t["name"],
            "logo": t.get("logo"),
            "country": t.get("country", {}).get("name", "")
        })
    return teams


def get_team_stats(team_id, season="2024-2025"):
    """Récupère les statistiques d’une équipe"""
    url = f"{API_URL}/teams/statistics?team={team_id}&season={season}"
    res = requests.get(url, headers=HEADERS)
    data = res.json()
    if not data.get("response"):
        return None
    r = data["response"]
    return {
        "games_played": r.get("games", {}).get("all", {}).get("played", 0),
        "points_for": r.get("points", {}).get("for", {}).get("average", {}).get("all", 0),
        "points_against": r.get("points", {}).get("against", {}).get("average", {}).get("all", 0),
        "rebounds": r.get("rebounds", {}).get("total", {}).get("average", {}).get("all", 0),
        "assists": r.get("assists", {}).get("average", {}).get("all", 0),
        "steals": r.get("steals", {}).get("average", {}).get("all", 0),
        "turnovers": r.get("turnovers", {}).get("average", {}).get("all", 0),
        "team_id": team_id
    }


# ============================================================
# 🚀 ETL PRINCIPAL
# ============================================================

def extract_basket_data():
    all_data = []
    leagues = get_leagues()
    print(f"📊 {len(leagues)} ligues trouvées.\n")

    for lg in leagues:
        print(f"➡️ {lg['name']} ({lg['country']}) ...")
        teams = get_teams(lg["id"])
        for t in teams:
            stats = get_team_stats(t["id"])
            if stats:
                stats["team_name"] = t["name"]
                stats["league_name"] = lg["name"]
                stats["country"] = lg["country"]
                all_data.append(stats)

    df = pd.DataFrame(all_data)
    output_file = os.path.join(OUTPUT_DIR, "basket_leagues.csv")
    df.to_csv(output_file, index=False)
    print(f"\n✅ Données enregistrées dans : {output_file}")
    print(f"📈 Total équipes enregistrées : {len(df)}")


# ============================================================
# 🏁 EXÉCUTION
# ============================================================

if __name__ == "__main__":
    extract_basket_data()
