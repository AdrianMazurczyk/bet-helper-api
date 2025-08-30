# main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
import os
import time
import csv
import io
import requests

# ---------------------------------
# Konfiguracja i inicjalizacja
# ---------------------------------
app = FastAPI(title="Bet Helper", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # w produkcji wpisz swoją domenę
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
BASE = "https://api.the-odds-api.com/v4"

CACHE_TTL_SECONDS = 60
_cache: Dict[str, Tuple[float, dict]] = {}
_last_limits: Dict[str, str] = {}

# ---------------------------------
# Narzędzia
# ---------------------------------
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
    """GET z prostym cache + zapis nagłówków limitów."""
    if cache_key:
        cached = _cache_get(cache_key)
        if cached is not None:
            return {"ok": True, "data": cached, "cached": True}
    try:
        r = requests.get(url, params=params, timeout=20)
        # nagłówki limitów (jeśli są)
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

def implied(price: Optional[float]) -> Optional[float]:
    if not price or price <= 1.0:
        return None
    return 1.0 / price

def normalize_3way(h, a, d):
    """Skalowanie, by (h + a + d) = 1 wśród nie-None."""
    probs = [p for p in (h, a, d) if p]
    s = sum(probs) or 1.0
    scale = (lambda x: (x / s) if x else None)
    return scale(h), scale(a), scale(d)

def kelly_fraction(prob: float, price: float, commission: float = 0.0,
                   cap: float = 0.25, fraction: float = 0.5) -> float:
    """
    Kelly dla kursu dziesiętnego z uwzględnieniem prowizji (np. giełdy).
    net_price = price * (1 - commission)
    Kelly raw: (b*p - q)/b, gdzie b = net_price - 1, q = 1-p
    Zwraca część banku (0..cap) po zastosowaniu 'fraction' (np. half-Kelly).
    """
    if not prob or not price or price <= 1.0:
        return 0.0
    net_price = price * (1.0 - commission)
    if net_price <= 1.0:
        return 0.0
    b = net_price - 1.0
    q = 1.0 - prob
    raw = (b * prob - q) / b
    k = max(0.0, raw) * max(0.0, fraction)
    return min(k, max(0.0, cap))

def parse_commence(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    # API zwraca ISO8601 z Z; datetime.fromisoformat nie lubi 'Z' w Py3.10-
    # więc zamienimy na +00:00
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        return None

# ---------------------------------
# Modele odpowiedzi
# ---------------------------------
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
    type: str  # "value" | "low_ev"

# ---------------------------------
# Endpointy
# ---------------------------------
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

    data = res["data"]
    return [
        {"key": s.get("key"), "title": s.get("title"), "active": s.get("active")}
        for s in data
    ]

@app.get("/picks", response_model=List[Pick])
def picks(
    sport: str = Query("soccer_epl"),
    region: str = Query("eu,uk"),
    min_ev: float = Query(0.03, ge=0.0),
    limit: int = Query(20, ge=1, le=200),
    market: str = Query("h2h"),
    bankroll: float = Query(1000.0, ge=0.0),
    bookmakers: Optional[str] = Query(None, description="np. Unibet,Betfair"),
    # Filtry dodatkowe
    kelly_fraction_q: float = Query(0.5, ge=0.0, le=1.0, alias="kelly_fraction"),
    kelly_cap: float = Query(0.25, ge=0.0, le=1.0),
    commission: float = Query(0.0, ge=0.0, le=0.2),
    since_hours: Optional[int] = Query(None, ge=0),
    until_hours: Optional[int] = Query(None, ge=0),
    min_price: Optional[float] = Query(None, ge=1.0),
    max_price: Optional[float] = Query(None, ge=1.0),
    show: Optional[str] = Query(None, regex="^(value|low_ev|all)$"),
    format: Optional[str] = Query(None, regex="^(csv)$"),
):
    """
    Zwraca listę typów z EV, Kelly i stake.
    - 'show': value | low_ev | all (domyślnie: value jeśli są, inaczej low_ev)
    - 'format=csv' zwraca CSV
    """
    if not ODDS_API_KEY:
        return []

    url = f"{BASE}/sports/{sport}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": region,
        "markets": market,
        "oddsFormat": "decimal"
    }
    ck = f"odds:{sport}:{region}:{market}"
    res = fetch_json(url, params, cache_key=ck)
    if not res["ok"]:
        return []

    books_filter = None
    if bookmakers:
        books_filter = set([b.strip().lower() for b in bookmakers.split(",") if b.strip()])

    now = datetime.now(timezone.utc)
    win_start = now + timedelta(hours=since_hours or 0) if since_hours is not None else None
    win_end   = now + timedelta(hours=until_hours) if until_hours is not None else None

    value_bets: List[Pick] = []
    low_ev_bets: List[Pick] = []

    for ev in res["data"]:
        home = ev.get("home_team")
        away = ev.get("away_team")
        commence = ev.get("commence_time")
        league = ev.get("sport_title", "")

        # filtr czasu
        cdt = parse_commence(commence)
        if win_start and (cdt is None or cdt < win_start):
            continue
        if win_end and (cdt is None or cdt > win_end):
            continue

        best_home = best_away = best_draw = None
        best_home_book = best_away_book = best_draw_book = None

        for b in ev.get("bookmakers", []):
            bname = (b.get("title") or "").strip()
            if books_filter and bname.lower() not in books_filter:
                continue
            for m in b.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                for o in m.get("outcomes", []):
                    name, price = o.get("name"), o.get("price")
                    if not price:
                        continue
                    if min_price and price < min_price:
                        continue
                    if max_price and price > max_price:
                        continue
                    if name == home and (not best_home or price > best_home):
                        best_home, best_home_book = price, bname
                    elif name == away and (not best_away or price > best_away):
                        best_away, best_away_book = price, bname
                    elif name == "Draw" and (not best_draw or price > best_draw):
                        best_draw, best_draw_book = price, bname

        # policz fair probs
        h, a, d = implied(best_home), implied(best_away), implied(best_draw)
        h, a, d = normalize_3way(h, a, d)

        def build(sel: str, price: Optional[float], prob: Optional[float], book: Optional[str]) -> Optional[Pick]:
            if not price:
                return None
            # EV liczymy na cenie netto po prowizji
            ev = None
            kly = 0.0
            stk = 0.0
            if prob:
                net_price = price * (1 - commission)
                ev = prob * net_price - 1.0
                kly = kelly_fraction(prob, price, commission=commission,
                                     cap=kelly_cap, fraction=kelly_fraction_q)
                stk = round(bankroll * kly, 2)
            typ = "value" if (ev is not None and ev >= min_ev) else "low_ev"
            return Pick(
                match=f"{home} vs {away}",
                selection=sel,
                price=round(price, 3),
                bookmaker=book,
                fair_prob=round(prob, 3) if prob is not None else None,
                ev=round(ev, 3) if ev is not None else None,
                kelly=round(kly, 4) if kly is not None else None,
                stake=stk,
                commence=commence,
                league=league,
                type=typ
            )

        cands: List[Pick] = []
        if best_home: cands.append(build("HOME", best_home, h, best_home_book))
        if best_away: cands.append(build("AWAY", best_away, a, best_away_book))
        if best_draw: cands.append(build("DRAW", best_draw, d, best_draw_book))

        for c in filter(None, cands):
            (value_bets if c.type == "value" else low_ev_bets).append(c)

    # sortowanie: najpierw po największej stawce, potem EV, potem najbliższy start
    def sort_key(p: Pick):
        ev_s = p.ev if p.ev is not None else -999
        cdt = parse_commence(p.commence) or datetime.max.replace(tzinfo=timezone.utc)
        # stawka None -> 0
        st = p.stake or 0.0
        return (-st, -ev_s, cdt)

    value_bets.sort(key=sort_key)
    low_ev_bets.sort(key=sort_key)

    # logika 'show'
    if show == "value":
        out = value_bets
    elif show == "low_ev":
        out = low_ev_bets
    elif show == "all":
        out = value_bets + low_ev_bets
    else:
        # domyślnie: jeśli są value-bets -> je pokazuj; w przeciwnym razie low_ev
        out = value_bets if value_bets else low_ev_bets

    out = out[:limit]

    # CSV?
    if format == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "match", "selection", "price", "bookmaker",
            "fair_prob", "ev", "kelly", "stake", "commence", "league", "type"
        ])
        for p in out:
            writer.writerow([
                p.match, p.selection, p.price, p.bookmaker or "",
                ("" if p.fair_prob is None else p.fair_prob),
                ("" if p.ev is None else p.ev),
                ("" if p.kelly is None else p.kelly),
                ("" if p.stake is None else p.stake),
                p.commence or "", p.league or "", p.type
            ])
        return PlainTextResponse(buf.getvalue(), media_type="text/csv; charset=utf-8")

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

    for ev in events:
        has = False
        example_books = []
        for b in ev.get("bookmakers", []):
            example_books.append(b.get("title", ""))
            for m in b.get("markets", []):
                if m.get("key") == "h2h" and m.get("outcomes"):
                    has = True
        if has:
            with_h2h += 1
            if len(examples) < 3:
                examples.append({
                    "match": f"{ev.get('home_team')} vs {ev.get('away_team')}",
                    "bookmakers": example_books[:20],
                })

    return {
        "sport": sport,
        "region": region,
        "events_from_api": total_events,
        "events_with_h2h": with_h2h,
        "examples": examples,
        "rate_limit_headers": _last_limits,
    }
