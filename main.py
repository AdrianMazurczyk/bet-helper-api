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

# CORS – w produkcji dodaj swoją domenę zamiast "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
BASE = "https://api.the-odds-api.com/v4"

# Prosty cache w pamięci, żeby nie klepać API co sekundę
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


def fetch_json(url: str, params: dict, cache_key: Optional[str] = None) -> dict:
    """
    GET z prostym cache + zebraniem nagłówków limitów.
    Zwraca dict: {"ok": bool, "data": ..., "error": "..." }
    """
    if cache_key:
        cached = _cache_get(cache_key)
        if cached is not None:
            return {"ok": True, "data": cached, "cached": True}

    try:
        r = requests.get(url, params=params, timeout=20)
        # podejrzyj limity
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


def implied(price: Optional[float]) -> Optional[float]:
    if not price or price <= 1.0:
        return None
    return 1.0 / price


def normalize_3way(h, a, d):
    probs = [p for p in (h, a, d) if p]
    s = sum(probs) or 1.0
    scale = (lambda x: (x / s) if x else None)
    return scale(h), scale(a), scale(d)


def kelly_fraction(prob: Optional[float], price: Optional[float],
                   cap: float = 0.25, fraction: float = 0.5) -> float:
    """
    Kelly dla kursu dziesiętnego.
    fraction=0.5 to „half Kelly”, cap – maksymalny % bankrolu na zakład.
    """
    if not prob or not price or price <= 1.0:
        return 0.0
    b = price - 1.0
    q = 1.0 - prob
    raw = (b * prob - q) / b  # klasyczny Kelly
    k = max(0.0, raw) * fraction
    return min(k, cap)


# -----------------------------
# Modele (dla ładnego Swaggera)
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
    type: str  # "value" | "low_ev" | "fallback"
    kelly: float
    stake: float


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
    region: str = Query("eu,uk", description="np. eu, uk, us, au lub kombinacje: eu,uk"),
    min_ev: float = Query(0.03, ge=0.0, description="Minimalne EV, np. 0.03 = 3%"),
    limit: int = Query(20, ge=1, le=100),
    market: str = Query("h2h", description="Domyślnie h2h"),
    bookmakers: Optional[str] = Query(None, description="Lista buków, np. Unibet,Betfair"),
    bankroll: float = Query(0.0, ge=0.0, description="Bankrol do wyliczenia stawki"),
    include_low_ev: bool = Query(True, description="Pokazuj też low_ev, gdy brak value"),
    include_fallback: bool = Query(True, description="Pokazuj fallback, gdy brak sensownych kursów"),
):
    """
    1) liczymy EV i Kelly dla najlepszych kursów HOME/AWAY/DRAW,
    2) value -> EV ≥ min_ev,
    3) low_ev -> EV < min_ev (stawka=0, ale z kelly),
    4) fallback -> gdy nie ma kompletu danych do EV.
    """
    if not ODDS_API_KEY:
        return []

    url = f"{BASE}/sports/{sport}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": region, "markets": market, "oddsFormat": "decimal"}

    books_filter = None
    if bookmakers:
        books_filter = set([b.strip().lower() for b in bookmakers.split(",") if b.strip()])

    res = fetch_json(url, params, cache_key=f"odds:{sport}:{region}:{market}:{bookmakers or 'all'}")
    if not res["ok"]:
        return []

    events = res["data"]
    values: List[Pick] = []
    low_ev: List[Pick] = []
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

        # policz EV/kelly dla kandydatów
        h, a, d = implied(best_home), implied(best_away), implied(best_draw)
        has_any_price = any([best_home, best_away, best_draw])

        if has_any_price:
            # normalizacja tylko jeśli mamy jakieś prawdopodobieństwa
            h, a, d = normalize_3way(h, a, d)

            def mk(sel: str, price: Optional[float], prob: Optional[float],
                   book: Optional[str]) -> Optional[Pick]:
                if not price:
                    return None
                ev_val = None
                kelly = 0.0
                if prob:
                    ev_val = prob * price - 1
                    kelly = kelly_fraction(prob, price)
                stake = round((bankroll * kelly), 2) if ev_val is not None and ev_val >= min_ev else 0.0
                ptype = "value" if (ev_val is not None and ev_val >= min_ev) else "low_ev"
                return Pick(
                    match=f"{home} vs {away}",
                    selection=sel,
                    price=round(price, 3),
                    bookmaker=book,
                    fair_prob=round(prob, 3) if prob else None,
                    ev=round(ev_val, 3) if ev_val is not None else None,
                    commence=commence,
                    league=league,
                    type=ptype,
                    kelly=round(kelly, 4),
                    stake=stake
                )

            cands = [
                mk("HOME", best_home, h, best_home_book),
                mk("AWAY", best_away, a, best_away_book),
                mk("DRAW", best_draw, d, best_draw_book),
            ]
            cands = [c for c in cands if c]

            # podziel na value / low_ev
            for c in cands:
                if c.type == "value":
                    values.append(c)
                elif include_low_ev:
                    low_ev.append(c)
        else:
            # brak cen – fallback
            if include_fallback:
                for sel, price, book in (
                    ("HOME", best_home, best_home_book),
                    ("AWAY", best_away, best_away_book),
                    ("DRAW", best_draw, best_draw_book),
                ):
                    if price:
                        fallback.append(Pick(
                            match=f"{home} vs {away}",
                            selection=sel,
                            price=round(price, 3),
                            bookmaker=book,
                            fair_prob=None,
                            ev=None,
                            commence=commence,
                            league=league,
                            type="fallback",
                            kelly=0.0,
                            stake=0.0
                        ))

    # sortowanie
    values.sort(key=lambda x: (-(x.ev or 0), x.commence or ""))
    low_ev.sort(key=lambda x: (-(x.ev or -1e9), x.commence or ""))
    fallback.sort(key=lambda x: (x.match, x.selection))

    # logika zwrotu
    if values:
        return values[:limit]
    if include_low_ev and low_ev:
        return low_ev[:limit]
    if include_fallback and fallback:
        return fallback[:limit]
    return []


@app.get("/debug")
def debug(
    sport: str = Query("soccer_poland_ekstraklasa"),
    region: str = Query("eu,uk"),
    market: str = Query("h2h")
):
    if not ODDS_API_KEY:
        return {"error": "Brak ODDS_API_KEY"}

    url = f"{BASE}/sports/{sport}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": region, "markets": market, "oddsFormat": "decimal"}

    res = fetch_json(url, params, cache_key=f"debug:{sport}:{region}:{market}")
    if not res["ok"]:
        return {"error": f"API error: {res['error']}"}

    events = res["data"]
    examples = []
    for ev in events[:3]:
        books = set()
        for b in ev.get("bookmakers", []):
            books.add(b.get("title", ""))
        examples.append({
            "match": f"{ev.get('home_team')} vs {ev.get('away_team')}",
            "bookmakers": sorted([x for x in books if x]),
        })

    return {
        "sport": sport,
        "region": region,
        "events_from_api": len(events),
        "examples": examples,
        "rate_limit_headers": _last_limits,
    }
