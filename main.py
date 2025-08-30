from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import os
import time
import csv
import io
import requests

app = FastAPI(title="Bet Helper", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # w produkcji ustaw swoją domenę
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
BASE = "https://api.the-odds-api.com/v4"

# Cache prosty w pamięci
CACHE_TTL_SECONDS = 60
_cache: Dict[str, Tuple[float, dict]] = {}
_last_limits: Dict[str, str] = {}

# ---------- narzędzia ----------
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
    """Normalizacja tylko gdy mamy >=2 prawdopodobieństwa."""
    probs = [p for p in (h, a, d) if p is not None]
    if len(probs) < 2:
        return None, None, None
    s = sum(probs) or 1.0
    scale = (lambda x: (x / s) if x is not None else None)
    return scale(h), scale(a), scale(d)

def kelly_fraction(prob: float, price_net: float, cap: float = 0.25, fraction: float = 0.5) -> float:
    """
    Kelly dla kursu NETTO (po prowizji). Zwraca ułamek banku [0..cap].
    """
    if not prob or not price_net or price_net <= 1.0:
        return 0.0
    b = price_net - 1.0
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
            if lk.startswith("x-requests") or lk.startswith("x-remaining") or "limit" in lk:
                _last_limits[lk] = v
        r.raise_for_status()
        data = r.json()
        if cache_key:
            _cache_set(cache_key, data)
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------- modele ----------
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
    type: str  # value | low_ev

# ---------- endpoints ----------
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
        return {"error": "Brak ODDS_API_KEY w Shared Variables."}
    url = f"{BASE}/sports"
    params = {"apiKey": ODDS_API_KEY}
    if all:
        params["all"] = "true"
    res = fetch_json(url, params, cache_key=f"sports:{all}")
    if not res["ok"]:
        return {"error": f"API error: {res['error']}"}
    data = res["data"]
    return [{"key": s.get("key"), "title": s.get("title"), "active": s.get("active")} for s in data]

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
    total_events = len(events)
    with_h2h = 0
    examples = []
    for ev in events:
        found = []
        for b in ev.get("bookmakers", []):
            for m in b.get("markets", []):
                if m.get("key") == "h2h" and m.get("outcomes"):
                    found.extend([o.get("name") for o in m.get("outcomes", []) if o.get("price")])
        if found:
            with_h2h += 1
            if len(examples) < 3:
                examples.append({
                    "match": f"{ev.get('home_team')} vs {ev.get('away_team')}",
                    "bookmakers": list({bk.get('title') for bk in ev.get("bookmakers", []) if bk.get('title')})
                })
    return {
        "sport": sport,
        "region": region,
        "events_from_api": total_events,
        "events_with_h2h": with_h2h,
        "examples": examples,
        "rate_limit_headers": _last_limits,
    }

@app.get("/picks", response_model=List[Pick])
def picks(
    sport: str = Query("soccer_epl"),
    region: str = Query("eu,uk"),
    market: str = Query("h2h"),
    limit: int = Query(20, ge=1, le=200),

    # value/kelly
    min_ev: float = Query(0.02, ge=0.0),
    min_kelly: float = Query(0.0, ge=0.0),
    bankroll: float = Query(1000.0, ge=0.0),
    kelly_fraction_param: float = Query(0.5, ge=0.0, le=1.0, alias="kelly_fraction"),
    kelly_cap: float = Query(0.2, ge=0.0, le=1.0),
    commission: float = Query(0.0, ge=0.0, le=0.1, description="Prowizja giełdy, np. 0.02 = 2%"),

    # filtry dodatkowe
    bookmakers: Optional[str] = Query(None, description="CSV buków, np. Unibet,Betfair"),
    min_price: Optional[float] = Query(None, ge=1.0),
    max_price: Optional[float] = Query(None, ge=1.0),
    min_outcomes: int = Query(2, ge=1, le=3, description="Minimalna liczba wyników z kursem (2 albo 3)"),

    # prezentacja
    show: str = Query("value", regex="^(value|low_ev|all)$"),
    format: str = Query("json", regex="^(json|csv)$"),
):
    """
    Zwraca value-bety + Kelly. Zabezpieczenia:
    - liczymy tylko jeśli w meczu są >= min_outcomes kursów (HOME/AWAY/DRAW) po filtrach.
    - Kelly liczone od kursu NETTO (po prowizji), tylko gdy EV>0.
    """
    if not ODDS_API_KEY:
        return []

    url = f"{BASE}/sports/{sport}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": region, "markets": market, "oddsFormat": "decimal"}

    books_filter = None
    if bookmakers:
        books_filter = set([b.strip().lower() for b in bookmakers.split(",") if b.strip()])

    res = fetch_json(url, params, cache_key=f"odds:{sport}:{region}:{market}")
    if not res["ok"]:
        return []

    events = res["data"]
    out_all: List[Pick] = []
    out_value: List[Pick] = []
    out_low: List[Pick] = []

    for ev in events:
        home = ev.get("home_team")
        away = ev.get("away_team")
        commence = ev.get("commence_time")
        league = ev.get("sport_title", "")

        # Zbieramy najlepsze kursy
        best = {"HOME": (None, None), "AWAY": (None, None), "DRAW": (None, None)}  # price, book
        for b in ev.get("bookmakers", []):
            bname = (b.get("title") or "").strip()
            if books_filter and bname.lower() not in books_filter:
                continue
            for m in b.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                for o in m.get("outcomes", []):
                    name = o.get("name")
                    price = o.get("price")
                    if not price:
                        continue
                    if min_price and price < min_price:
                        continue
                    if max_price and price > max_price:
                        continue
                    sel = None
                    if name == home: sel = "HOME"
                    elif name == away: sel = "AWAY"
                    elif name == "Draw": sel = "DRAW"
                    if not sel:
                        continue
                    cur_price, _ = best[sel]
                    if (cur_price is None) or (price > cur_price):
                        best[sel] = (price, bname)

        # Sprawdź ile realnych wyników mamy po filtrach
        present = [(sel, p, bk) for sel, (p, bk) in best.items() if p is not None]
        if len(present) < min_outcomes:
            # zbyt mało danych => pomiń mecz (blokuje fair_prob=1.0 i chore Kelly)
            continue

        # policz fair_prob (normalizacja) na podstawie dostępnych cen
        h_i = implied(best["HOME"][0]) if best["HOME"][0] else None
        a_i = implied(best["AWAY"][0]) if best["AWAY"][0] else None
        d_i = implied(best["DRAW"][0]) if best["DRAW"][0] else None
        h, a, d = normalize_3way(h_i, a_i, d_i)
        if h is None and a is None and d is None:
            continue  # brak ≥2 wyników

        def make_pick(sel: str, prob: Optional[float], price: Optional[float], book: Optional[str]) -> Optional[Pick]:
            if price is None:
                return None
            price_net = price * (1.0 - commission)  # prowizja
            ev = (prob * price_net - 1.0) if (prob is not None) else None
            kelly = kelly_fraction(prob or 0.0, price_net, cap=kelly_cap, fraction=kelly_fraction_param) if (ev is not None and ev > 0) else 0.0
            stake = round(kelly * bankroll, 2) if kelly > 0 else 0.0
            typ = "value" if (ev is not None and ev >= min_ev and kelly >= min_kelly) else "low_ev"
            return Pick(
                match=f"{home} vs {away}",
                selection=sel,
                price=round(price, 3),
                bookmaker=book,
                fair_prob=round(prob, 3) if prob is not None else None,
                ev=round(ev, 3) if ev is not None else None,
                kelly=round(kelly, 4) if kelly else 0.0,
                stake=stake,
                commence=commence,
                league=league,
                type=typ
            )

        cand = []
        cand.append(make_pick("HOME", h, best["HOME"][0], best["HOME"][1]))
        cand.append(make_pick("AWAY", a, best["AWAY"][0], best["AWAY"][1]))
        if best["DRAW"][0] is not None:
            cand.append(make_pick("DRAW", d, best["DRAW"][0], best["DRAW"][1]))
        cand = [c for c in cand if c]

        for c in cand:
            out_all.append(c)
            if c.type == "value":
                out_value.append(c)
            else:
                out_low.append(c)

    # sorty
    out_value.sort(key=lambda x: (-(x.ev or 0), -(x.kelly or 0), x.commence or ""))
    out_low.sort(key=lambda x: (x.match, x.selection))
    out_all.sort(key=lambda x: (x.match, x.selection))

    chosen: List[Pick]
    if show == "value":
        chosen = out_value[:limit]
    elif show == "low_ev":
        chosen = out_low[:limit]
    else:
        chosen = out_all[:limit]

    # CSV?
    if format == "csv":
        headers = ["match", "selection", "price", "bookmaker", "fair_prob", "ev", "kelly", "stake", "commence", "league", "type"]
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(headers)
        for p in chosen:
            writer.writerow([
                p.match, p.selection, p.price, p.bookmaker or "",
                p.fair_prob if p.fair_prob is not None else "",
                p.ev if p.ev is not None else "",
                p.kelly if p.kelly is not None else "",
                p.stake if p.stake is not None else "",
                p.commence or "", p.league or "", p.type
            ])
        return PlainTextResponse(buf.getvalue(), media_type="text/csv")

    return chosen
