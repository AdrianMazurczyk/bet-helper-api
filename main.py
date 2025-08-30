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
app = FastAPI(title="Bet Helper", version="1.2.0")

# CORS: pozwól na wywołania z przeglądarki (np. własny frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # w produkcji wpisz swoją domenę
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
BASE = "https://api.the-odds-api.com/v4"

# Cache w pamięci (bardzo proste, ale skuteczne)
CACHE_TTL_SECONDS = 60  # odśwież co minutę
_cache: Dict[str, Tuple[float, dict]] = {}

# Trzymamy ostatnie nagłówki limitów (opcjonalne)
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


def fetch_json(url: str, params: dict, cache_key: Optional[str] = None) -> dict:
    """
    GET z prostym cache + odczytem nagłówków limitów.
    Zwraca dict: {"ok": bool, "data": ..., "error": "..."}
    """
    if cache_key:
        cached = _cache_get(cache_key)
        if cached is not None:
            return {"ok": True, "data": cached, "cached": True}

    try:
        r = requests.get(url, params=params, timeout=20)
        # zapisz info o limitach (o ile są)
        for k, v in r.headers.items():
            lk = k.lower()
            if lk.startswith("x-requests") or lk.startswith("x-remaining") or "limit" in lk:
                _last_limits[lk] = v

        r.raise_for_status()
        data = r.json()
        if cache_key:
            _cache_set(cache_key, data)
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# -----------------------------
# Modele odpowiedzi (opcjonalne, Swagger ładniej wygląda)
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
    type: str  # "value" albo "fallback"


# -----------------------------
# Endpointy
# -----------------------------
@app.get("/")
def home():
    return {"message": "Bet Helper działa!"}


@app.get("/status")
def status():
    """
    Prosty healthcheck + ostatnie nagłówki limitów (jeśli były zapytania).
    """
    return {
        "ok": True,
        "have_api_key": bool(ODDS_API_KEY),
        "rate_limit_headers": _last_limits,
        "cache_size": len(_cache),
    }


@app.get("/sports")
def list_sports(all: bool = True):
    """
    Lista sportów z The Odds API. all=true zwraca także nieaktywne.
    """
    if not ODDS_API_KEY:
        return {"error": "Brak ODDS_API_KEY w środowisku Railway (Shared Variables)."}

    url = f"{BASE}/sports"
    params = {"apiKey": ODDS_API_KEY}
    if all:
        params["all"] = "true"

    res = fetch_json(url, params, cache_key=f"sports:{all}")
    if not res["ok"]:
        return {"error": f"API error: {res['error']}"}

    data = res["data"]
    return [
        {"key": s.get("key"), "title": s.get("title"), "active": s.get("active")}
        for s in data
    ]


@app.get("/picks", response_model=List[Pick])
def picks(
    sport: str = Query("soccer_epl", description="np. soccer_epl, soccer_poland_ekstraklasa, basketball_nba"),
    region: str = Query("eu,uk", description="np. eu, uk, us, au lub kombinacje: eu,uk"),
    min_ev: float = Query(0.03, ge=0.0, description="Minimalne EV, np. 0.03 = 3%"),
    limit: int = Query(20, ge=1, le=100),
    market: str = Query("h2h", description="Domyślnie h2h"),
    bookmakers: Optional[str] = Query(None, description="Lista buków oddzielona przecinkami, np. Unibet,Betfair")
):
    """
    Pobiera kursy (domyślnie H2H), wybiera najlepszy kurs HOME/AWAY/DRAW,
    normalizuje prawdopodobieństwa, liczy EV i zwraca:
    - value bety (EV ≥ min_ev), a jeśli ich brak:
    - fallback: top kursy (żeby front nie był pusty)
    """
    if not ODDS_API_KEY:
        return [{"match": "", "selection": "", "price": 0, "bookmaker": None, "fair_prob": None,
                 "ev": None, "commence": None, "league": None, "type": "error"}]

    url = f"{BASE}/sports/{sport}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": region,
        "markets": market,
        "oddsFormat": "decimal"
    }

    books_filter = None
    if bookmakers:
        books_filter = set([b.strip().lower() for b in bookmakers.split(",") if b.strip()])

    ck = f"odds:{sport}:{region}:{market}"
    res = fetch_json(url, params, cache_key=ck)
    if not res["ok"]:
        return [{"match": f"ERROR: {res['error']}", "selection": "", "price": 0, "bookmaker": None,
                 "fair_prob": None, "ev": None, "commence": None, "league": None, "type": "error"}]

    events = res["data"]
    values: List[Pick] = []
    fallback: List[Pick] = []

    for ev in events:
        home = ev.get("home_team")
        away = ev.get("away_team")
        commence = ev.get("commence_time")
        league = ev.get("sport_title", "")

        best_home = best_away = best_draw = None
        best_home_book = best_away_book = best_draw_book = None

        for b in ev.get("bookmakers", []):
            bname = b.get("title") or ""
            if books_filter and bname.lower() not in books_filter:
                continue

            for m in b.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                for o in m.get("outcomes", []):
                    name, price = o.get("name"), o.get("price")
                    if not price:
                        continue
                    if name == home and (not best_home or price > best_home):
                        best_home, best_home_book = price, bname
                    elif name == away and (not best_away or price > best_away):
                        best_away, best_away_book = price, bname
                    elif name == "Draw" and (not best_draw or price > best_draw):
                        best_draw, best_draw_book = price, bname

        # fallback – pokaż co najmniej top kursy
        if best_home:
            fallback.append(Pick(
                match=f"{home} vs {away}", selection="HOME", price=round(best_home, 3),
                bookmaker=best_home_book, fair_prob=None, ev=None, commence=commence,
                league=league, type="fallback"
            ))
        if best_away:
            fallback.append(Pick(
                match=f"{home} vs {away}", selection="AWAY", price=round(best_away, 3),
                bookmaker=best_away_book, fair_prob=None, ev=None, commence=commence,
                league=league, type="fallback"
            ))
        if best_draw:
            fallback.append(Pick(
                match=f"{home} vs {away}", selection="DRAW", price=round(best_draw, 3),
                bookmaker=best_draw_book, fair_prob=None, ev=None, commence=commence,
                league=league, type="fallback"
            ))

        # EV – jeśli mamy komplet danych
        h, a, d = implied(best_home), implied(best_away), implied(best_draw)
        if any([h, a, d]):
            h, a, d = normalize_3way(h, a, d)

            def cand(sel: str, price: Optional[float], prob: Optional[float], book: Optional[str]) -> Optional[Pick]:
                if not (price and prob):
                    return None
                evv = prob * price - 1
                return Pick(
                    match=f"{home} vs {away}", selection=sel, price=round(price, 3),
                    bookmaker=book, fair_prob=round(prob, 3), ev=round(evv, 3),
                    commence=commence, league=league, type="value"
                )

            cands = [
                cand("HOME", best_home, h, best_home_book),
                cand("AWAY", best_away, a, best_away_book),
                cand("DRAW", best_draw, d, best_draw_book),
            ]
            cands = [c for c in cands if c]
            if cands:
                best = max(cands, key=lambda x: x.ev or -999)
                if (best.ev or -1) >= min_ev:
                    values.append(best)

    # posortuj
    values.sort(key=lambda x: (-(x.ev or 0), x.commence or ""))
    fallback.sort(key=lambda x: (x.match, x.selection))

    # jeśli są value-bety – zwróć je; inaczej fallback
    out = values[:limit] if values else fallback[:limit]
    return out


@app.get("/debug")
def debug(
    sport: str = Query("soccer_poland_ekstraklasa"),
    region: str = Query("eu,uk"),
    market: str = Query("h2h")
):
    """
    Szybkie podsumowanie: ile eventów, ile z rynkiem H2H, przykłady.
    """
    if not ODDS_API_KEY:
        return {"error": "Brak ODDS_API_KEY"}

    url = f"{BASE}/sports/{sport}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": region,
        "markets": market,
        "oddsFormat": "decimal"
    }

    res = fetch_json(url, params, cache_key=f"debug:{sport}:{region}:{market}")
    if not res["ok"]:
        return {"error": f"API error: {res['error']}"}

    events = res["data"]
    total_events = len(events)
    with_h2h = 0
    examples = []

    for ev in events:
        has = False
        example_book = ""
        example_outcomes = []
        for b in ev.get("bookmakers", []):
            for m in b.get("markets", []):
                if m.get("key") == "h2h" and m.get("outcomes"):
                    has = True
                    example_book = b.get("title", "")
                    example_outcomes = [o.get("name") for o in m.get("outcomes", [])]
                    break
            if has:
                break
        if has:
            with_h2h += 1
            if len(examples) < 3:
                examples.append({
                    "match": f"{ev.get('home_team')} vs {ev.get('away_team')}",
                    "bookmaker": example_book,
                    "outcomes": example_outcomes
                })

    return {
        "sport": sport,
        "region": region,
        "events_from_api": total_events,
        "events_with_h2h": with_h2h,
        "examples": examples,
        "rate_limit_headers": _last_limits,
    }

