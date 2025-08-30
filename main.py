from fastapi import FastAPI, Query
import os, requests
from typing import List, Dict, Optional

app = FastAPI(title="Bet Helper")

# 👇 klucz czytamy z Railway → Settings → Shared Variables (ODDS_API_KEY)
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
BASE = "https://api.the-odds-api.com/v4"

def implied(price: Optional[float]) -> Optional[float]:
    if not price or price <= 1.0:
        return None
    return 1.0 / price

def normalize_3way(h, a, d):
    probs = [p for p in (h, a, d) if p]
    s = sum(probs) or 1.0
    scale = (lambda x: (x / s) if x else None)
    return scale(h), scale(a), scale(d)

@app.get("/")
def home():
    return {"message": "Bet Helper działa!"}

@app.get("/picks")
def picks(
    sport: str = Query("soccer_epl", description="np. soccer_epl, soccer_poland_ekstraklasa, basketball_nba"),
    min_ev: float = Query(0.03, description="Minimalne EV, np. 0.03 = 3%"),
    limit: int = Query(20, ge=1, le=100),
    region: str = Query("eu", description="eu | uk | us | au")
):
    """
    Pobiera kursy H2H, wybiera najlepszy kurs HOME/AWAY/DRAW, normalizuje prawdopodobieństwa,
    liczy EV = fair_prob * price - 1 i zwraca tylko value-bety (EV ≥ min_ev).
    """
    if not ODDS_API_KEY:
        return {"error": "Brak ODDS_API_KEY w środowisku Railway (Shared Variables)."}

    url = f"{BASE}/sports/{sport}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": region,
        "markets": "h2h",
        "oddsFormat": "decimal"
    }

    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        return {"error": f"API error: {e}"}

    results: List[Dict] = []

    for ev in events:
        home = ev.get("home_team")
        away = ev.get("away_team")
        commence = ev.get("commence_time")
        league = ev.get("sport_title", "")

        best_home = best_away = best_draw = None
        best_home_book = best_away_book = best_draw_book = None

        for b in ev.get("bookmakers", []):
            bname = b.get("title")
            for m in b.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                for o in m.get("outcomes", []):
                    name, price = o.get("name"), o.get("price")
                    if name == home and (not best_home or price > best_home):
                        best_home, best_home_book = price, bname
                    elif name == away and (not best_away or price > best_away):
                        best_away, best_away_book = price, bname
                    elif name == "Draw" and (not best_draw or price > best_draw):
                        best_draw, best_draw_book = price, bname

        # policz EV
        h, a, d = implied(best_home), implied(best_away), implied(best_draw)
        h, a, d = normalize_3way(h, a, d)

        candidates: List[Dict] = []
        if h and best_home:
            candidates.append({
                "match": f"{home} vs {away}",
                "selection": "HOME",
                "price": round(best_home, 3),
                "bookmaker": best_home_book,
                "fair_prob": round(h, 3),
                "ev": round(h * best_home - 1, 3),
                "commence": commence,
                "league": league
            })
        if a and best_away:
            candidates.append({
                "match": f"{home} vs {away}",
                "selection": "AWAY",
                "price": round(best_away, 3),
                "bookmaker": best_away_book,
                "fair_prob": round(a, 3),
                "ev": round(a * best_away - 1, 3),
                "commence": commence,
                "league": league
            })
        if d and best_draw:
            candidates.append({
                "match": f"{home} vs {away}",
                "selection": "DRAW",
                "price": round(best_draw, 3),
                "bookmaker": best_draw_book,
                "fair_prob": round(d, 3),
                "ev": round(d * best_draw - 1, 3),
                "commence": commence,
                "league": league
            })

        if candidates:
            best = max(candidates, key=lambda x: x["ev"])
            if best["ev"] >= min_ev:
                results.append(best)

    results.sort(key=lambda x: (-x["ev"], x["commence"] or ""))
    return results[:limit]
@app.get("/sports")
def list_sports(region: str = "eu"):
    url = f"{BASE}/sports"
    params = {"apiKey": ODDS_API_KEY, "all": "true"}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        # zwróć tylko najpotrzebniejsze pola
        return [
            {"key": s.get("key"), "title": s.get("title"), "active": s.get("active")}
            for s in data
        ]
    except Exception as e:
        return {"error": f"API error: {e}"}
