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

# CORS (dla frontendu)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # w produkcji wpisz swoją domenę
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
BASE = "https://api.the-odds-api.com/v4"

# Prosty cache w pamięci
CACHE_TTL_SECONDS = 60
_cache: Dict[str, Tuple[float, dict]] = {}

# Ostatnie nagłówki dot. limitów zapytań
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
    """Konwersja kursu dziesiętnego na „fair” prawdopodobieństwo."""
    if not price or price <= 1.0:
        return None
    return 1.0 / price


def normalize_3way(h, a, d):
    """Normalizacja prawdopodobieństw (HOME/AWAY/DRAW) do 1."""
    probs = [p for p in (h, a, d) if p]
    s = sum(probs) or 1.0
    scale = (lambda x: (x / s) if x else None)
    return scale(h), scale(a), scale(d)


def kelly_fraction(prob: float, price: float, cap: float = 0.25, fraction: float = 0.5) -> float:
    """
    Kelly dla kursu dziesiętnego.
    fraction=0.5 -> tzw. „half Kelly” (bezpieczniej),
    cap=0.25 -> maksymalnie 25% banku na jeden zakład.
    """
    if not prob or not price or price <= 1.0:
        return 0.0
    b = price - 1.0
    q = 1.0 - prob
    raw = (b * prob - q) / b  # klasyczny Kelly
    k = max(0.0, raw) * fraction
    return min(k, cap)


def fetch_json(url: str, params: dict, cache_key: Optional[str] = None) -> dict:
    """GET z cache + zbieraniem nagłówków limitów. Zwraca {"ok":bool,"data":...|None,"error":str|None}"""
    if cache_key:
        cached = _cache_get(cache_key)
        if cached is not None:
            return {"ok": True, "data": cached, "cached": True}

    try:
        r = requests.get(url, params=params, timeout=20)
        # zbierz nagłówki dot. limitów
        for k, v in r.headers.items():
            lk = k.lower()
            if lk.startswith("x-requests") or "remaining" in lk or "limit" in lk:
                _last_limits[lk] = v
        r.raise_for_status()
        data = r.json()
        if cache_key:
            _cache_set(cache_key, data)
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# -----------------------------
# Modele (ładniej w Swaggerze)
# -----------------------------
class Pick(BaseModel):
    match: str
    selection: str
    price: float
    bookmaker: Optional[str] = None
    fair_prob: Optional[float] = None
    ev: Optional[float] = None
    kelly: Optional[float] = None
    stake: Optional[float] = None
    commence: Optional[str] = None
    league: Optional[str] = None
    type: str  # "value" | "low_ev" | "no_prob" | "error"


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
        return {"error": "Brak ODDS_API_KEY w środowisku Railway (Shared Variables)."}

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
    sport: str = Query("soccer_epl", description="np. soccer_epl, soccer_poland_ekstraklasa, basketball_nba"),
    region: str = Query("eu,uk", description="np. eu, uk, us, au (lub mieszane: eu,uk)"),
    min_ev: float = Query(0.03, ge=0.0, description="Minimalne EV, np. 0.03 = 3%"),
    limit: int = Query(20, ge=1, le=100),
    market: str = Query("h2h", description="Domyślnie h2h"),
    bookmakers: Optional[str] = Query(None, description="Filtr: np. Unibet,Betfair"),
    bankroll: float = Query(0.0, ge=0.0, description="Bankroll do stawki Kelly; 0 = nie licz stawki"),
    kelly_fraction_param: float = Query(0.5, ge=0.1, le=1.0, description="Część Kelly (np. 0.5 = half Kelly)"),
    kelly_cap: float = Query(0.25, ge=0.05, le=1.0, description="Maksymalna część banku na zakład")
):
    """
    Zwraca:
      - type="value" (EV ≥ min_ev) – value bety,
      - type="low_ev" (EV < min_ev) – mamy fair_prob, ale próg nie spełniony,
      - type="no_prob" – kursy są, ale nie dało się policzyć fair_prob (np. brak kompletu H/A/D).
    Dla "value" i "low_ev" pokazujemy też Kelly/stake (jeśli bankroll>0).
    """
    if not ODDS_API_KEY:
        return [Pick(match="ERROR", selection="", price=0, type="error")]

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
        return [Pick(match=f"ERROR: {res['error']}", selection="", price=0, type="error")]

    events = res["data"]
    values: List[Pick] = []
    lows: List[Pick] = []
    no_prob: List[Pick] = []

    for ev in events:
        home = ev.get("home_team")
        away = ev.get("away_team")
        commence = ev.get("commence_time")
        league = ev.get("sport_title", "")

        best_home = best_away = best_draw = None
        best_home_book = best_away_book = best_draw_book = None

        for b in ev.get("bookmakers", []):
            bname = (b.get("title") or "")
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

        # policz fair_prob (jeśli możliwe)
        h, a, d = implied(best_home), implied(best_away), implied(best_draw)
        can_prob = any([h, a, d])
        if can_prob:
            h, a, d = normalize_3way(h, a, d)

        # pomocnicza funkcja do budowy pozycji
        def mk_pick(sel: str, price: Optional[float], prob: Optional[float], book: Optional[str]) -> Optional[Pick]:
            if not price:
                return None
            kelly = None
            stake = None
            evv = None
            if prob:
                evv = prob * price - 1
                kelly = kelly_fraction(prob, price, cap=kelly_cap, fraction=kelly_fraction_param)
                if bankroll > 0:
                    stake = round(kelly * bankroll, 2)

            p = Pick(
                match=f"{home} vs {away}",
                selection=sel,
                price=round(price, 3),
                bookmaker=book,
                fair_prob=round(prob, 3) if prob is not None else None,
                ev=round(evv, 3) if evv is not None else None,
                kelly=round(kelly, 4) if kelly is not None else None,
                stake=stake,
                commence=commence,
                league=league,
                type="value"  # tymczasowo; zaraz przypiszemy właściwy typ
            )
            if prob is None:
                p.type = "no_prob"
            elif p.ev is not None and p.ev >= min_ev:
                p.type = "value"
            else:
                p.type = "low_ev"
            return p

        picks_for_event: List[Pick] = []
        picks_for_event += [mk_pick("HOME", best_home, h if can_prob else None, best_home_book)]
        picks_for_event += [mk_pick("AWAY", best_away, a if can_prob else None, best_away_book)]
        picks_for_event += [mk_pick("DRAW", best_draw, d if can_prob else None, best_draw_book)]
        picks_for_event = [p for p in picks_for_event if p]

        # rozdziel do kubełków
        for p in picks_for_event:
            if p.type == "value":
                values.append(p)
            elif p.type == "low_ev":
                lows.append(p)
            else:
                no_prob.append(p)

    # sortowanie i ograniczenie
    values.sort(key=lambda x: (-(x.ev or 0), x.commence or "", x.match))
    lows.sort(key=lambda x: (-(x.ev or -1), x.commence or "", x.match))
    no_prob.sort(key=lambda x: (x.match, x.selection))

    out: List[Pick] = []
    if values:
        out = values[:limit]
    elif lows:
        out = lows[:limit]
    else:
        out = no_prob[:limit]

    return out


@app.get("/debug")
def debug(
    sport: str = Query("soccer_poland_ekstraklasa"),
    region: str = Query("eu,uk"),
    market: str = Query("h2h")
):
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
    example_books = []

    for ev in events:
        has = False
        local_books = set()
        for b in ev.get("bookmakers", []):
            btitle = b.get("title", "")
            for m in b.get("markets", []):
                if m.get("key") == "h2h" and m.get("outcomes"):
                    has = True
                    local_books.add(btitle)
            # kontynuujemy, żeby zebrać pełną listę buków w przykładzie
        if has:
            with_h2h += 1
            if len(examples) < 3:
                examples.append({
                    "match": f"{ev.get('home_team')} vs {ev.get('away_team')}",
                    "bookmakers": sorted(list(local_books))
                })
        example_books.extend(list(local_books))

    return {
        "sport": sport,
        "region": region,
        "events_from_api": total_events,
        "events_with_h2h": with_h2h,
        "examples": examples,
        "rate_limit_headers": _last_limits,
    }
