from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import os
import time
import requests

# -----------------------------
# Konfiguracja
# -----------------------------
app = FastAPI(title="Bet Helper", version="1.3.0")

# CORS – żeby frontend mógł pytać API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # w produkcji wpisz swoją domenę
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
BASE = "https://api.the-odds-api.com/v4"

# Cache w pamięci
CACHE_TTL_SECONDS = 60
_cache: Dict[str, Tuple[float, dict]] = {}
_last_limits: Dict[str, str] = {}

# -----------------------------
# Narzędzia
# -----------------------------
def _cache_get(key: str) -> Optional[dict]:
    now = time.time()
    item = _cache.get(key)
    if not item:
        return None
    ts, value = item
    if now - ts > CACHE_TTL_SECONDS:
        _cache.pop(key, None)
        return None
    return value

def _cache_set(key: str, value: dict) -> None:
    _cache[key] = (time.time(), value)

def implied(price: Optional[float]) -> Optional[float]:
    if not price or price <= 1.0:
        return None
    return 1.0 / price

def normalize_3way(h, a, d):
    probs = [p for p in (h, a, d) if p]
    s = sum(probs) or 1.0
    scale = (lambda x: (x / s) if x else None)
    return scale(h), scale(a), scale(d)

def kelly_fraction(prob: float, price: float, cap: float = 0.25, fraction: float = 0.5) -> float:
    """
    Kelly na kurs dziesiętny (decimal).
    prob = fair probability, price = kurs.
    fraction = 0.5 -> half-Kelly.
    cap = max % banku na jeden bet (np. 0.25 = 25%).
    """
    if not prob or not price or price <= 1.0:
        return 0.0
    b = price - 1.0
    q = 1.0 - prob
    raw = (b * prob - q) / b
    k = max(0.0, raw) * fraction
    return min(k, cap)

def fetch_json(url: str, params: dict, cache_key: Optional[str] = None) -> dict:
    if cache_key:
        cached = _cache_get(cache_key)
        if cached is not None:
            return {"ok": True, "data": cached, "cached": True}
    try:
        r = requests.get(url, params=params, timeout=20)
        for k, v in r.headers.items():
            lk = k.lower()
            if "limit" in lk or lk.startswith("x-requests"):
                _last_limits[lk] = v
        r.raise_for_status()
        data = r.json()
        if cache_key:
            _cache_set(cache_key, data)
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# -----------------------------
# Modele odpowiedzi
# -----------------------------
class Pick(BaseModel):
    match: str
    selection: str
    price: float
    bookmaker: Optional[str] = None
    fair_prob: Optional[float] = None
    ev: Optional[float] = None
    commence: Optional[str] = None
    league: Optional[str] = None
    type: str
    kelly: Optional[float] = None
    stake: Optional[float] = None

# -----------------------------
# Endpointy
# -----------------------------
@app.get("/")
def home():
    return {"message": "Bet Helper działa!"}

@app.get("/status")
def status():
    return {
        "ok": True,
        "have_api_key": bool(ODDS_API_KEY),
        "rate_limit_headers": _last_limits,
        "cache_size": len(_cache),
    }

@app.get("/sports")
def list_sports(all: bool = True):
    if not ODDS_API_KEY:
        return {"error": "Brak ODDS_API_KEY"}
    url = f"{BASE}/sports"
    params = {"apiKey": ODDS_API_KEY}
    if all:
        params["all"] = "true"
    res = fetch_json(url, params, cache_key=f"sports:{all}")
    if not res["ok"]:
        return {"error": f"API error: {res['error']}"}
    return [
        {"key": s.get("key"), "title": s.get("title"), "active": s.get("active")}
        for s in res["data"]
    ]

@app.get("/picks", response_model=List[Pick])
def picks(
    sport: str = Query("soccer_epl"),
    region: str = Query("eu,uk"),
    min_ev: float = Query(0.03, ge=0.0),
    limit: int = Query(20, ge=1, le=100),
    bankroll: Optional[float] = Query(None, description="Twój bank, np. 1000"),
    kelly_cap: float = Query(0.25, ge=0.0, le=1.0),
    kelly_fraction_param: float = Query(0.5, ge=0.0, le=1.0),
):
    if not ODDS_API_KEY:
        return [{"match": "ERROR: Brak ODDS_API_KEY", "selection": "", "price": 0, "type": "error"}]

    url = f"{BASE}/sports/{sport}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": region, "markets": "h2h", "oddsFormat": "decimal"}
    res = fetch_json(url, params, cache_key=f"odds:{sport}:{region}")
    if not res["ok"]:
        return [{"match": f"ERROR: {res['error']}", "selection": "", "price": 0, "type": "error"}]

    events = res["data"]
    values: List[Pick] = []

    for ev in events:
        home, away = ev.get("home_team"), ev.get("away_team")
        commence, league = ev.get("commence_time"), ev.get("sport_title", "")
        best_home = best_away = best_draw = None
        book_home = book_away = book_draw = None

        for b in ev.get("bookmakers", []):
            for m in b.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                for o in m.get("outcomes", []):
                    name, price = o.get("name"), o.get("price")
                    if not price:
                        continue
                    if name == home and (not best_home or price > best_home):
                        best_home, book_home = price, b.get("title")
                    elif name == away and (not best_away or price > best_away):
                        best_away, book_away = price, b.get("title")
                    elif name == "Draw" and (not best_draw or price > best_draw):
                        best_draw, book_draw = price, b.get("title")

        h, a, d = implied(best_home), implied(best_away), implied(best_draw)
        if any([h, a, d]):
            h, a, d = normalize_3way(h, a, d)

            def make(sel, price, prob, book):
                if not (price and prob):
                    return None
                evv = prob * price - 1
                if evv < min_ev:
                    return None
                kfrac = kelly_fraction(prob, price, cap=kelly_cap, fraction=kelly_fraction_param)
                stake = bankroll * kfrac if bankroll else None
                return Pick(
                    match=f"{home} vs {away}",
                    selection=sel,
                    price=round(price, 3),
                    bookmaker=book,
                    fair_prob=round(prob, 3),
                    ev=round(evv, 3),
                    commence=commence,
                    league=league,
                    type="value",
                    kelly=round(kfrac, 4),
                    stake=round(stake, 2) if stake else None,
                )

            for pick in [
                make("HOME", best_home, h, book_home),
                make("AWAY", best_away, a, book_away),
                make("DRAW", best_draw, d, book_draw),
            ]:
                if pick:
                    values.append(pick)

    values.sort(key=lambda x: (-(x.ev or 0), x.commence or ""))
    return values[:limit]

@app.get("/debug")
def debug(
    sport: str = Query("soccer_poland_ekstraklasa"),
    region: str = Query("eu,uk"),
):
    if not ODDS_API_KEY:
        return {"error": "Brak ODDS_API_KEY"}
    url = f"{BASE}/sports/{sport}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": region, "markets": "h2h", "oddsFormat": "decimal"}
    res = fetch_json(url, params, cache_key=f"debug:{sport}:{region}")
    if not res["ok"]:
        return {"error": f"API error: {res['error']}"}
    events = res["data"]
    return {
        "sport": sport,
        "region": region,
        "events_from_api": len(events),
        "examples": [
            {"match": f"{e.get('home_team')} vs {e.get('away_team')}",
             "bookmakers": [b.get("title") for b in e.get("bookmakers", [])]}
            for e in events[:3]
        ],
        "rate_limit_headers": _last_limits,
    }
