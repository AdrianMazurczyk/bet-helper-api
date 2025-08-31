# main.py
from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import os
import time
import requests
from datetime import datetime, timedelta, timezone
import csv
import io

# -----------------------------------------------------------------------------
# Konfiguracja
# -----------------------------------------------------------------------------
app = FastAPI(title="Bet Helper", version="1.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # w produkcji wpisz domenę
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

# Popularne aliasy sportów (wygoda w UI)
SPORT_ALIASES = {
    # Piłka nożna
    "epl": "soccer_epl",
    "pl": "soccer_poland_ekstraklasa",
    "ekstraklasa": "soccer_poland_ekstraklasa",
    "laliga": "soccer_spain_la_liga",
    "seriea": "soccer_italy_serie_a",
    "bundes": "soccer_germany_bundesliga",
    "ucl": "soccer_uefa_champions_league",
    # Koszykówka
    "nba": "basketball_nba",
    # Hokej / baseball / american football
    "nhl": "ice_hockey_nhl",
    "mlb": "baseball_mlb",
    "nfl": "americanfootball_nfl",
    # Tenis
    "tennis": "tennis_atp",
    "atp": "tennis_atp",
    "wta": "tennis_wta",
    # Rugby
    "rugby": "rugby_union",
    "rugbyu": "rugby_union",
}

# -----------------------------------------------------------------------------
# Narzędzia
# -----------------------------------------------------------------------------
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
    Kelly dla kursu dziesiętnego.
    prob: fair probability po normalizacji
    price: kurs (decimal)
    fraction: 0.5 = half Kelly (bezpieczniej)
    cap: maks. część banku na zakład (np. 25%)
    """
    if not prob or not price or price <= 1.0:
        return 0.0
    b = price - 1.0
    q = 1.0 - prob
    raw = (b * prob - q) / b
    k = max(0.0, raw) * fraction
    return min(k, cap)

def fetch_json(url: str, params: dict, cache_key: Optional[str] = None) -> dict:
    """
    GET z prostym cache + nagłówkami limitów.
    Zwraca dict: {"ok": bool, "data": ..., "error": "..." }
    """
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

def parse_iso(dt: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(dt.replace("Z", "+00:00"))
    except Exception:
        return None

def smart_sport_key(s: str) -> str:
    if not s:
        return s
    return SPORT_ALIASES.get(s.lower().strip(), s)

# -----------------------------------------------------------------------------
# Modele (dla ładnego Swaggera)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Endpointy bazowe
# -----------------------------------------------------------------------------
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
        return {"error": "Brak ODDS_API_KEY w środowisku."}
    url = f"{BASE}/sports"
    params = {"apiKey": ODDS_API_KEY}
    if all:
        params["all"] = "true"
    res = fetch_json(url, params, cache_key=f"sports:{all}")
    if not res["ok"]:
        return {"error": f"API error: {res['error']}"}
    data = res["data"]
    return [{"key": s.get("key"), "title": s.get("title"), "active": s.get("active")} for s in data]

# -----------------------------------------------------------------------------
# /picks – główne API z EV/Kelly/stake + CSV
# -----------------------------------------------------------------------------
@app.get("/picks", response_model=List[Pick])
def picks(
    sport: str = Query("soccer_epl", description="np. soccer_epl, epl, soccer_poland_ekstraklasa, nba, atp"),
    region: str = Query("eu,uk", description="np. eu, uk, us, au, kombinacje: eu,uk"),
    min_ev: float = Query(0.03, ge=0.0, description="Minimalne EV, np. 0.03 = 3%"),
    limit: int = Query(40, ge=1, le=400),
    market: str = Query("h2h"),
    bookmakers: Optional[str] = Query(None, description="CSV buków, np. Unibet,Betfair"),
    min_price: Optional[float] = Query(None, ge=1.0),
    max_price: Optional[float] = Query(None, ge=1.0),
    bankroll: float = Query(1000.0, ge=0.0),
    kelly_fraction_q: float = Query(0.5, ge=0.0, le=1.0, description="np. 0.5 = half Kelly"),
    kelly_cap: float = Query(0.25, ge=0.0, le=1.0),
    commission: float = Query(0.0, ge=0.0, le=0.1, description="Prowizja (0–0.1)"),
    show: str = Query("value", regex="^(value|all)$"),
    format: str = Query("json", regex="^(json|csv)$"),
    stake_all: bool = Query(False, description="Jeśli True – pokazuj Kelly/stake także dla low_ev"),
    since_hours: int = Query(0, ge=0, description="Od teraz + Xh (filtr czasu)"),
    until_hours: int = Query(120, ge=1, description="Do teraz + Yh (filtr czasu)"),
):
    if not ODDS_API_KEY:
        return [{"match": "", "selection": "", "price": 0, "bookmaker": None,
                 "fair_prob": None, "ev": None, "kelly": None, "stake": None,
                 "commence": None, "league": None, "type": "error"}]

    sport = smart_sport_key(sport)
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

    # okno czasowe
    now = datetime.now(timezone.utc)
    start_dt = now + timedelta(hours=since_hours)
    end_dt = now + timedelta(hours=until_hours)

    ck = f"odds:{sport}:{region}:{market}"
    res = fetch_json(url, params, cache_key=ck)
    if not res["ok"]:
        return [{"match": f"ERROR: {res['error']}", "selection": "", "price": 0, "bookmaker": None,
                 "fair_prob": None, "ev": None, "kelly": None, "stake": None,
                 "commence": None, "league": None, "type": "error"}]

    events = res["data"]
    out_rows: List[Dict] = []

    for ev in events:
        commence = ev.get("commence_time")
        commence_dt = parse_iso(commence)
        if commence_dt and not (start_dt <= commence_dt <= end_dt):
            continue

        home = ev.get("home_team")
        away = ev.get("away_team")
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
                    if price:
                        if min_price is not None and price < min_price:
                            continue
                        if max_price is not None and price > max_price:
                            continue
                    if name == home and (not best_home or price > best_home):
                        best_home, best_home_book = price, bname
                    elif name == away and (not best_away or price > best_away):
                        best_away, best_away_book = price, bname
                    elif name == "Draw" and (not best_draw or price > best_draw):
                        best_draw, best_draw_book = price, bname

        # fair probs
        h, a, d = implied(best_home), implied(best_away), implied(best_draw)
        if any([h, a, d]):
            h, a, d = normalize_3way(h, a, d)

        def build(sel: str, price: Optional[float], prob: Optional[float], book: Optional[str]) -> Optional[Dict]:
            if not price:
                return None
            eff_price = price * (1.0 - commission)
            ev_val = (prob * eff_price - 1.0) if prob else None
            kel = kelly_fraction(prob, eff_price, cap=kelly_cap, fraction=kelly_fraction_q) if prob else 0.0
            is_value = (ev_val is not None) and (ev_val >= min_ev)
            row_type = "value" if is_value else "low_ev"
            stake_amt = round(kel * bankroll, 2) if (kel and bankroll) else 0.0
            if not is_value and not stake_all:
                stake_amt = 0.0
            return {
                "match": f"{home} vs {away}",
                "selection": sel,
                "price": round(price, 3),
                "bookmaker": book,
                "fair_prob": round(prob, 3) if prob is not None else None,
                "ev": round(ev_val, 3) if ev_val is not None else None,
                "kelly": round(kel, 4) if kel is not None else 0.0,
                "stake": stake_amt,
                "commence": commence,
                "league": league,
                "type": row_type,
            }

        cands = [
            build("HOME", best_home, h, best_home_book),
            build("AWAY", best_away, a, best_away_book),
            build("DRAW", best_draw, d, best_draw_book),
        ]
        cands = [c for c in cands if c]

        if show == "value":
            cands = [c for c in cands if c["type"] == "value"]

        out_rows.extend(cands)

    # sort: value>low_ev, EV desc, czas rosnąco
    def sort_key(x: Dict):
        t = 1 if x["type"] == "value" else 0
        evv = x["ev"] if x["ev"] is not None else -999
        c = x["commence"] or ""
        return (-t, -evv, c)

    out_rows.sort(key=sort_key)
    out_rows = out_rows[:limit]

    if format == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["match", "selection", "price", "bookmaker", "fair_prob", "ev", "kelly", "stake", "commence", "league", "type"])
        for r in out_rows:
            writer.writerow([
                r.get("match"), r.get("selection"), r.get("price"), r.get("bookmaker"),
                r.get("fair_prob"), r.get("ev"), r.get("kelly"), r.get("stake"),
                r.get("commence"), r.get("league"), r.get("type"),
            ])
        csv_bytes = buf.getvalue().encode("utf-8")
        return Response(content=csv_bytes, media_type="text/csv")

    return out_rows

# -----------------------------------------------------------------------------
# /debug – szybkie info
# -----------------------------------------------------------------------------
@app.get("/debug")
def debug(
    sport: str = Query("soccer_epl"),
    region: str = Query("eu,uk"),
    market: str = Query("h2h")
):
    if not ODDS_API_KEY:
        return {"error": "Brak ODDS_API_KEY"}
    sport = smart_sport_key(sport)
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
        found = False
        books = []
        for b in ev.get("bookmakers", []):
            books.append(b.get("title"))
            for m in b.get("markets", []):
                if m.get("key") == "h2h" and m.get("outcomes"):
                    found = True
        if found:
            with_h2h += 1
            if len(examples) < 3:
                examples.append({
                    "match": f"{ev.get('home_team')} vs {ev.get('away_team')}",
                    "bookmakers": books[:30],
                })
    return {
        "sport": sport,
        "region": region,
        "events_from_api": total_events,
        "events_with_h2h": with_h2h,
        "examples": examples,
        "rate_limit_headers": _last_limits,
    }

# -----------------------------------------------------------------------------
# /match_summary – jeden mecz: HOME/DRAW/AWAY, has_draw dla 2-way
# -----------------------------------------------------------------------------
@app.get("/match_summary")
def match_summary(
    match: str = Query(..., description="format: 'HOME TEAM vs AWAY TEAM'"),
    sport: str = Query("soccer_epl"),
    region: str = Query("eu,uk"),
    commission: float = Query(0.0, ge=0.0, le=0.1),
):
    if not ODDS_API_KEY:
        return {"error": "Brak ODDS_API_KEY"}

    sport = smart_sport_key(sport)
    url = f"{BASE}/sports/{sport}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": region,
        "markets": "h2h",
        "oddsFormat": "decimal"
    }

    res = fetch_json(url, params, cache_key=f"summary:{sport}:{region}:h2h")
    if not res["ok"]:
        return {"error": f"API error: {res['error']}"}

    # parsowanie "Home vs Away"
    if " vs " not in match:
        return {"error": "Parametr 'match' musi być w formacie 'HOME vs AWAY'."}
    expect_home, expect_away = [x.strip() for x in match.split(" vs ", 1)]

    found_ev = None
    for ev in res["data"]:
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        # dopasuj w obie strony (różne API potrafią różnie pisać)
        if {home.lower(), away.lower()} == {expect_home.lower(), expect_away.lower()}:
            found_ev = ev
            break

    if not found_ev:
        return {"error": f"Nie znaleziono meczu '{match}' w {sport} ({region})."}

    home = found_ev.get("home_team")
    away = found_ev.get("away_team")
    commence = found_ev.get("commence_time")
    league = found_ev.get("sport_title", "")

    best_home = best_away = best_draw = None
    for b in found_ev.get("bookmakers", []):
        for m in b.get("markets", []):
            if m.get("key") != "h2h":
                continue
            for o in m.get("outcomes", []):
                name, price = o.get("name"), o.get("price")
                if not price:
                    continue
                if name == home and (not best_home or price > best_home):
                    best_home = price
                elif name == away and (not best_away or price > best_away):
                    best_away = price
                elif name == "Draw" and (not best_draw or price > best_draw):
                    best_draw = price

    # implied & normalize
    h, a, d = implied(best_home), implied(best_away), implied(best_draw)
    if any([h, a, d]):
        h, a, d = normalize_3way(h, a, d)

    # has_draw: jeśli nie ma sensownego kursu na remis – to 2-way
    has_draw = best_draw is not None and d is not None

    # rekomendacja: wybierz najwyższe prawdopodobieństwo
    probs = {
        "HOME": h or 0.0,
        "AWAY": a or 0.0,
    }
    if has_draw:
        probs["DRAW"] = d or 0.0

    best_pick = max(probs.items(), key=lambda x: x[1])
    pick_name, pick_prob = best_pick[0], best_pick[1]

    return {
        "match": f"{home} vs {away}",
        "league": league,
        "start": commence,
        "has_draw": has_draw,
        "probabilities": {
            "HOME": round((h or 0.0) * 100, 1),
            "DRAW": round((d or 0.0) * 100, 1) if has_draw else None,
            "AWAY": round((a or 0.0) * 100, 1),
        },
        "raw_prices": {
            "HOME": best_home,
            "DRAW": best_draw if has_draw else None,
            "AWAY": best_away,
        },
        "recommendation": {
            "pick": pick_name,
            "confidence_percent": round(pick_prob * 100, 1),
        }
    }

# -----------------------------------------------------------------------------
# /ui – dashboard
# -----------------------------------------------------------------------------
@app.get("/ui")
def ui():
    html = """
<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Bet Helper – Dashboard</title>
<style>
:root{--bg:#0f1220;--card:#171a2b;--text:#e9ecf1;--muted:#9aa3b2;--green:#27c28a;--gray:#2a2f45;--red:#f87171;--amber:#f59e0b;--btn:#4251f5;}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,Ubuntu}
.wrap{max-width:1100px;margin:32px auto;padding:0 16px}
h1{margin:0 0 16px 0;font-size:22px}
.card{background:var(--card);border-radius:14px;padding:16px;margin-bottom:16px;box-shadow:0 6px 20px rgba(0,0,0,.25)}
.row{display:flex;gap:8px;flex-wrap:wrap}
input,select{background:#0e1120;border:1px solid #2b324d;color:var(--text);border-radius:10px;padding:10px 12px;outline:none}
button{background:var(--btn);border:none;color:#fff;border-radius:10px;padding:10px 14px;cursor:pointer}
button.ghost{background:#252a43}
.badge{padding:4px 8px;border-radius:999px;font-size:12px}
.badge.value{background:rgba(39,194,138,.18);color:#27c28a}
.badge.low{background:var(--gray);color:#c9d2e3}
table{width:100%;border-collapse:collapse}
th,td{padding:10px;border-bottom:1px solid #22263b;text-align:left;vertical-align:top}
th{color:#c5cbe0;font-weight:600}
small{color:var(--muted)}
.k{white-space:nowrap}
.right{text-align:right}
.footer{display:flex;gap:12px;align-items:center;justify-content:space-between}
a.link{color:#8ab4ff;text-decoration:none}
</style>
</head>
<body>
<div class="wrap">
  <h1>Bet Helper – Dashboard</h1>

  <div class="card">
    <div class="row">
      <label>Sport<br><input id="sport" value="epl" placeholder="np. epl, nba, atp"/></label>
      <label>Region<br><input id="region" value="eu,uk"/></label>
      <label>Show<br>
        <select id="show">
          <option value="value" selected>value</option>
          <option value="all">all</option>
        </select>
      </label>
      <label>Min EV<br><input id="min_ev" type="number" step="0.001" value="0.02"/></label>
      <label>Bankroll<br><input id="bankroll" type="number" step="1" value="1000"/></label>
      <label>Bookmakers (CSV)<br><input id="books" placeholder="np. Unibet,Betfair"/></label>
      <label>Min price<br><input id="min_price" type="number" step="0.01" placeholder=""/></label>
      <label>Max price<br><input id="max_price" type="number" step="0.01" placeholder=""/></label>
      <label>Since h<br><input id="since" type="number" value="0"/></label>
      <label>Until h<br><input id="until" type="number" value="120"/></label>
      <label>Commission<br><input id="comm" type="number" step="0.005" value="0.00"/></label>
      <label>Stake all<br>
        <select id="stake_all"><option value="false">false</option><option value="true">true</option></select>
      </label>
    </div>
    <div style="margin-top:10px" class="row">
      <button onclick="loadPicks()">Refresh</button>
      <button class="ghost" onclick="exportCSV()">Export CSV</button>
      <small id="meta"></small>
    </div>
  </div>

  <div class="card">
    <div class="footer"><div><b>Typy</b></div><div>Σ stake: <b id="sumStake">0</b></div></div>
    <div style="overflow:auto">
      <table id="tbl">
        <thead>
          <tr>
            <th>Mecz</th><th>Pick</th><th class="right">Kurs</th><th>Buk</th>
            <th class="right">Fair</th><th class="right">EV</th><th class="right">Kelly</th><th class="right">Stake</th>
            <th>Start</th><th>Typ</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <div class="footer">
      <b>Dual view (HOME vs AWAY, DRAW zawsze żółty)</b>
      <small id="dual_meta"></small>
    </div>
    <div id="dual"></div>
  </div>
</div>

<script>
const col = {HOME:'#22c55e', DRAW:'#f59e0b', AWAY:'#ef4444'};
function pct(x){ return x!=null ? (x*100).toFixed(1)+'%' : ''; }

async function loadPicks(){
  const q = new URLSearchParams();
  q.set('sport', document.getElementById('sport').value);
  q.set('region', document.getElementById('region').value);
  q.set('show', document.getElementById('show').value);
  q.set('min_ev', document.getElementById('min_ev').value || '0');
  q.set('bankroll', document.getElementById('bankroll').value || '1000');
  const books = document.getElementById('books').value.trim();
  if(books) q.set('bookmakers', books);
  const minp = document.getElementById('min_price').value; if(minp) q.set('min_price', minp);
  const maxp = document.getElementById('max_price').value; if(maxp) q.set('max_price', maxp);
  q.set('since_hours', document.getElementById('since').value || '0');
  q.set('until_hours', document.getElementById('until').value || '120');
  q.set('commission', document.getElementById('comm').value || '0');
  q.set('stake_all', document.getElementById('stake_all').value);
  q.set('limit','200');

  document.getElementById('meta').innerText = 'Loading…';
  try{
    const res = await fetch('/picks?' + q.toString());
    const data = await res.json();
    const tb = document.querySelector('#tbl tbody');
    tb.innerHTML = '';
    let sumStake = 0;
    data.forEach(r=>{
      const tr = document.createElement('tr');
      const badge = r.type === 'value' ? '<span class="badge value">value</span>' : '<span class="badge low">low_ev</span>';
      const ev = r.ev!=null ? r.ev.toFixed(3) : '';
      const fair = r.fair_prob!=null ? r.fair_prob.toFixed(3) : '';
      const kelly = r.kelly!=null ? r.kelly.toFixed(4) : '';
      const stake = (r.stake||0).toFixed(2);
      sumStake += parseFloat(stake)||0;

      tr.innerHTML = `
        <td><div><a class="link" href="/ui_match?match=${encodeURIComponent(r.match)}&sport=${encodeURIComponent(document.getElementById('sport').value)}&region=${encodeURIComponent(document.getElementById('region').value)}" target="_blank">${r.match}</a></div><small>${r.league||''}</small></td>
        <td>${r.selection}</td>
        <td class="right k">${r.price?.toFixed ? r.price.toFixed(2) : r.price}</td>
        <td>${r.bookmaker||''}</td>
        <td class="right">${fair}</td>
        <td class="right">${ev}</td>
        <td class="right">${kelly}</td>
        <td class="right">${stake}</td>
        <td>${r.commence ? new Date(r.commence).toLocaleString() : ''}</td>
        <td>${badge}</td>
      `;
      tb.appendChild(tr);
    });
    document.getElementById('sumStake').innerText = sumStake.toFixed(2);
    document.getElementById('meta').innerText = `Łącznie: ${data.length} pozycji`;

    // Dual view (agregacja po meczu)
    const byMatch = {};
    data.forEach(r=>{
      if(!byMatch[r.match]) byMatch[r.match] = {league:r.league, start:r.commence, HOME:null, DRAW:null, AWAY:null};
      byMatch[r.match][r.selection] = r.fair_prob;
    });
    const rows = Object.entries(byMatch).map(([m,v])=>({match:m, ...v}));
    rows.sort((a,b)=> (a.start||'').localeCompare(b.start||''));

    const wrap = document.getElementById('dual');
    wrap.innerHTML = '';
    rows.forEach(r=>{
      const home = r.HOME!=null ? (r.HOME*100).toFixed(1)+'%' : '';
      const draw = r.DRAW!=null ? (r.DRAW*100).toFixed(1)+'%' : '';
      const away = r.AWAY!=null ? (r.AWAY*100).toFixed(1)+'%' : '';
      const div = document.createElement('div');
      div.style.margin = '8px 0';
      div.innerHTML = `
        <div><b>${r.match}</b> <small>${r.league||''} • ${r.start? new Date(r.start).toLocaleString():''}</small></div>
        <div style="display:flex;gap:6px;margin-top:6px">
          <span style="background:${col.HOME};color:#07110b;padding:4px 8px;border-radius:8px">HOME ${home}</span>
          ${r.DRAW!=null ? `<span style="background:${col.DRAW};color:#201300;padding:4px 8px;border-radius:8px">DRAW ${draw}</span>` : ''}
          <span style="background:${col.AWAY};color:#200707;padding:4px 8px;border-radius:8px">AWAY ${away}</span>
        </div>
      `;
      wrap.appendChild(div);
    });
    document.getElementById('dual_meta').innerText = `Mecze: ${rows.length}`;
  }catch(e){
    document.getElementById('meta').innerText = 'Błąd: ' + e;
  }
}

function exportCSV(){
  const q = new URLSearchParams();
  q.set('sport', document.getElementById('sport').value);
  q.set('region', document.getElementById('region').value);
  q.set('show', document.getElementById('show').value);
  q.set('min_ev', document.getElementById('min_ev').value || '0');
  const books = document.getElementById('books').value.trim();
  if(books) q.set('bookmakers', books);
  const minp = document.getElementById('min_price').value; if(minp) q.set('min_price', minp);
  const maxp = document.getElementById('max_price').value; if(maxp) q.set('max_price', maxp);
  q.set('since_hours', document.getElementById('since').value || '0');
  q.set('until_hours', document.getElementById('until').value || '120');
  q.set('commission', document.getElementById('comm').value || '0');
  q.set('stake_all', document.getElementById('stake_all').value);
  q.set('format','csv');
  q.set('limit','200');
  window.location = '/picks?' + q.toString();
}

loadPicks();
</script>
</body>
</html>
    """.strip()
    return Response(content=html, media_type="text/html")

# -----------------------------------------------------------------------------
# /ui_match – widok pojedynczego meczu (zielony/czerwony/żółty)
# -----------------------------------------------------------------------------
@app.get("/ui_match")
def ui_match(
    match: str = Query(..., description="HOME vs AWAY"),
    sport: str = Query("epl"),
    region: str = Query("eu,uk")
):
    html = f"""
<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Bet Helper – Match View</title>
<style>
:root{{--bg:#0f1220;--card:#171a2b;--text:#e9ecf1;--muted:#9aa3b2;--green:#22c55e;--red:#ef4444;--amber:#f59e0b;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Ubuntu}}
.wrap{{max-width:900px;margin:28px auto;padding:0 16px}}
.card{{background:var(--card);border-radius:14px;padding:18px;box-shadow:0 10px 30px rgba(0,0,0,.32)}}
h1{{font-size:22px;margin:0 0 12px}}
small{{color:var(--muted)}}
.bar{{height:42px;border-radius:10px;margin:6px 0;display:flex;align-items:center;justify-content:space-between;padding:0 12px;font-weight:600}}
.bar.home{{background:rgba(34,197,94,.18);color:#8af0ba;border:1px solid rgba(34,197,94,.35)}}
.bar.draw{{background:rgba(245,158,11,.18);color:#ffd18c;border:1px solid rgba(245,158,11,.35)}}
.bar.away{{background:rgba(239,68,68,.18);color:#ff9d9d;border:1px solid rgba(239,68,68,.35)}}
.badge{{padding:4px 8px;border-radius:999px;font-size:12px;background:#2a314c;color:#d4d9ea}}
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>Match view</h1>
    <small id="meta">Ładowanie…</small>
    <div id="out" style="margin-top:14px"></div>
  </div>
</div>
<script>
function mkBar(label, p, cls){ 
  const val = (p!=null)? p.toFixed(1)+'%' : '';
  return `<div class="bar ${cls}"><span>${label}</span><span>${val}</span></div>`;
}
(async ()=>{
  const q = new URLSearchParams({match: {match!r}, sport: {sport!r}, region: {region!r}});
  const res = await fetch('/match_summary?' + q.toString());
  const data = await res.json();
  const m = document.getElementById('meta');
  if(data.error){{ m.innerText = data.error; return; }}
  m.innerText = `${{data.league||''}} • ${{data.start? new Date(data.start).toLocaleString():''}}`;

  const p = data.probabilities || {{}};
  const hasDraw = !!data.has_draw;
  const blocks = [
    mkBar('HOME', p.HOME, 'home'),
    hasDraw ? mkBar('DRAW', p.DRAW, 'draw') : '',
    mkBar('AWAY', p.AWAY, 'away'),
  ].join('');
  document.getElementById('out').innerHTML = `
    <h2>${{data.match}}</h2>
    <div style="height:8px"></div>
    ${blocks}
    <div style="margin-top:10px"><b>Rekomendacja:</b>
      <span class="badge">${{data.recommendation.pick}}</span>
      <small>(pewność ${{data.recommendation.confidence_percent.toFixed(1)}}%)</small>
    </div>
  `;
}})();
</script>
</body>
</html>
    """.strip()
    return Response(content=html, media_type="text/html")
