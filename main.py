from fastapi import FastAPI, Query, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import os
import time
import requests
from datetime import datetime, timedelta, timezone
import csv
import io

# =========================================================
# Konfiguracja
# =========================================================
app = FastAPI(title="Bet Helper", version="1.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # w produkcji ogranicz do swojej domeny
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
BASE = "https://api.the-odds-api.com/v4"

# prosty cache
CACHE_TTL_SECONDS = 60
_cache: Dict[str, Tuple[float, dict]] = {}
_last_limits: Dict[str, str] = {}

# aliasy sportów (w UI możesz pisać "epl")
SPORT_ALIASES = {
    "epl": "soccer_epl",
    "pl": "soccer_poland_ekstraklasa",
    "ekstraklasa": "soccer_poland_ekstraklasa",
    "nba": "basketball_nba",
}

def normalize_sport_key(sport: str) -> str:
    if not sport:
        return "soccer_epl"
    s = sport.strip().lower()
    return SPORT_ALIASES.get(s, sport)

# =========================================================
# Narzędzia
# =========================================================
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
    price: kurs
    fraction: 0.5 = half Kelly (bezpieczniej)
    cap: maks. część banku na jeden zakład
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
        # The Odds API zwraca UTC ISO
        return datetime.fromisoformat(dt.replace("Z", "+00:00"))
    except Exception:
        return None

# =========================================================
# Modele (dla ładnego Swaggera)
# =========================================================
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
    type: str

# =========================================================
# Root / Status
# =========================================================
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

# =========================================================
# Sports
# =========================================================
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

# =========================================================
# Picks
# =========================================================
@app.get("/picks", response_model=List[Pick])
def picks(
    sport: str = Query("soccer_epl", description="np. soccer_epl, epl (alias wspierany)"),
    region: str = Query("eu,uk", description="np. eu, uk, us, au lub kombinacje: eu,uk"),
    min_ev: float = Query(0.0, ge=0.0, description="Minimalne EV, np. 0.03 = 3%"),
    limit: int = Query(200, ge=1, le=300),
    market: str = Query("h2h", description="Domyślnie h2h"),
    bookmakers: Optional[str] = Query(None, description="Lista buków przecinkiem, np. Unibet,Betfair"),
    min_price: Optional[float] = Query(None, ge=1.0),
    max_price: Optional[float] = Query(None, ge=1.0),
    bankroll: float = Query(1000.0, ge=0.0),
    kelly_fraction_q: float = Query(0.5, ge=0.0, le=1.0, description="np. 0.5 = half Kelly"),
    kelly_cap: float = Query(0.25, ge=0.0, le=1.0),
    commission: float = Query(0.0, ge=0.0, le=0.1, description="Prowizja bukm. (0–0.1)"),
    show: str = Query("all", regex="^(value|all)$"),
    format: str = Query("json", regex="^(json|csv)$"),
    stake_all: bool = Query(True, description="Jeśli True, pokazuj Kelly/stake również dla low_ev"),
    since_hours: int = Query(0, ge=0, description="Od teraz + Xh (filtr czasu)"),
    until_hours: int = Query(120, ge=1, description="Do teraz + Yh (filtr czasu)"),
):
    """
    Pobiera kursy H2H, wybiera najlepszy kurs HOME/AWAY/DRAW, normalizuje fair prob,
    liczy EV (po prowizji), Kelly i proponowaną stawkę.
    - show=value → tylko value-bety (EV ≥ min_ev)
    - show=all   → value + low_ev
    - format=csv → CSV zamiast JSON
    - stake_all  → pokaż kelly/stake również dla low_ev
    """
    if not ODDS_API_KEY:
        return [{"match": "", "selection": "", "price": 0, "bookmaker": None, "fair_prob": None,
                 "ev": None, "kelly": None, "stake": None, "commence": None, "league": None, "type": "error"}]

    sport = normalize_sport_key(sport)

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
                 "fair_prob": None, "ev": None, "kelly": None, "stake": None, "commence": None,
                 "league": None, "type": "error"}]

    events = res["data"]
    out_rows: List[Dict] = []

    for ev in events:
        commence = ev.get("commence_time")
        commence_dt = parse_iso(commence)
        if commence_dt:
            if not (start_dt <= commence_dt <= end_dt):
                continue

        home = ev.get("home_team")
        away = ev.get("away_team")
        league = ev.get("sport_title", "")

        best_home = best_away = best_draw = None
        best_home_book = best_away_book = best_draw_book = None

        # wybór najlepszego kursu per wynik
        for b in ev.get("bookmakers", []):
            bname = b.get("title") or ""
            if books_filter and bname.lower() not in books_filter:
                continue
            for m in b.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                for o in m.get("outcomes", []):
                    name, price = o.get("name"), o.get("price")

                    # filtr cen
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

        # policz fair i EV
        h, a, d = implied(best_home), implied(best_away), implied(best_draw)
        if any([h, a, d]):
            h, a, d = normalize_3way(h, a, d)

        def build(sel: str, price: Optional[float], prob: Optional[float], book: Optional[str]) -> Optional[Dict]:
            if not price:
                return None
            # prowizja (np. 0.02 = 2%) obniża efektywny kurs
            eff_price = price * (1.0 - commission)
            ev_val = (prob * eff_price - 1.0) if prob else None
            kel = kelly_fraction(prob, eff_price, cap=kelly_cap, fraction=kelly_fraction_q) if prob else 0.0

            is_value = (ev_val is not None) and (ev_val >= min_ev)
            row_type = "value" if is_value else "low_ev"

            # stake – zawsze licz, ale pokaż przy low_ev tylko gdy stake_all=True
            stake_amt = round(kel * bankroll, 2) if kel and bankroll else 0.0
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

    # sortowanie: najpierw po typie (value > low_ev), potem EV desc, potem czas
    def sort_key(x: Dict):
        t = 1 if x["type"] == "value" else 0
        ev_ = x["ev"] if x["ev"] is not None else -999
        c = x["commence"] or ""
        return (-t, -ev_, c)

    out_rows.sort(key=sort_key)
    out_rows = out_rows[:limit]

    # CSV?
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

# =========================================================
# Debug
# =========================================================
@app.get("/debug")
def debug(
    sport: str = Query("soccer_epl"),
    region: str = Query("eu,uk"),
    market: str = Query("h2h")
):
    if not ODDS_API_KEY:
        return {"error": "Brak ODDS_API_KEY"}

    sport = normalize_sport_key(sport)

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
                    "bookmakers": books[:25],
                })

    return {
        "sport": sport,
        "region": region,
        "events_from_api": total_events,
        "events_with_h2h": with_h2h,
        "examples": examples,
        "rate_limit_headers": _last_limits,
    }

# =========================================================
# Dashboard /ui
# =========================================================
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
:root{--bg:#0f1220;--card:#171a2b;--text:#e9ecf1;--muted:#9aa3b2;--green:#27c28a;--gray:#2a2f45;--red:#f87171;}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, Ubuntu}
.wrap{max-width:1100px;margin:32px auto;padding:0 16px}
h1{margin:0 0 16px 0;font-size:22px}
.card{background:var(--card);border-radius:14px;padding:16px;margin-bottom:16px;box-shadow:0 6px 20px rgba(0,0,0,.25)}
.row{display:flex;gap:8px;flex-wrap:wrap}
input,select{background:#0e1120;border:1px solid #2b324d;color:var(--text);border-radius:10px;padding:10px 12px;outline:none}
button{background:#4251f5;border:none;color:#fff;border-radius:10px;padding:10px 14px;cursor:pointer}
button.ghost{background:#252a43}
.badge{padding:4px 8px;border-radius:999px;font-size:12px}
.badge.value{background:rgba(39,194,138,.16);color:#27c28a}
.badge.low{background:var(--gray);color:#c9d2e3}
table{width:100%;border-collapse:collapse}
th,td{padding:10px;border-bottom:1px solid #22263b;text-align:left;vertical-align:top}
th{color:#c5cbe0;font-weight:600}
small{color:var(--muted)}
.k{white-space:nowrap}
.right{text-align:right}
.footer{display:flex;gap:12px;align-items:center;justify-content:space-between}
a.btn{color:#fff;text-decoration:none;background:#2b355a;padding:6px 10px;border-radius:8px}
</style>
</head>
<body>
<div class="wrap">
  <h1>Bet Helper – Dashboard</h1>

  <div class="card">
    <div class="row">
      <label>Sport<br><input id="sport" value="epl"/></label>
      <label>Region<br><input id="region" value="eu,uk"/></label>
      <label>Show<br>
        <select id="show">
          <option value="all" selected>all</option>
          <option value="value">value</option>
        </select>
      </label>
      <label>Min EV<br><input id="min_ev" type="number" step="0.001" value="0"/></label>
      <label>Bankroll<br><input id="bankroll" type="number" step="1" value="1000"/></label>
      <label>Bookmakers (CSV)<br><input id="books" placeholder="np. Unibet,Betfair"/></label>
      <label>Min price<br><input id="min_price" type="number" step="0.01" placeholder=""/></label>
      <label>Max price<br><input id="max_price" type="number" step="0.01" placeholder=""/></label>
      <label>Since h<br><input id="since" type="number" value="0"/></label>
      <label>Until h<br><input id="until" type="number" value="120"/></label>
      <label>Commission<br><input id="comm" type="number" step="0.005" value="0.00"/></label>
      <label>Stake all<br>
        <select id="stake_all"><option value="true">true</option><option value="false">false</option></select>
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
            <th>Start</th><th>Typ</th><th>Compare</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>

<script>
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

      // link Compare -> /ui_match z uzupełnionymi parametrami
      const parts = (r.match||'').split(' vs ');
      const home = parts[0]||''; const away = parts[1]||'';
      const u = new URLSearchParams();
      u.set('sport', document.getElementById('sport').value);
      u.set('region', document.getElementById('region').value);
      u.set('home', home); u.set('away', away);
      const books = document.getElementById('books').value.trim();
      if(books) u.set('bookmakers', books);
      const comm = document.getElementById('comm').value || '0';
      u.set('commission', comm);
      const compareLink = '/ui_match#'+u.toString();

      tr.innerHTML = `
        <td><div>${r.match}</div><small>${r.league||''}</small></td>
        <td>${r.selection}</td>
        <td class="right k">${r.price?.toFixed ? r.price.toFixed(2) : r.price}</td>
        <td>${r.bookmaker||''}</td>
        <td class="right">${fair}</td>
        <td class="right">${ev}</td>
        <td class="right">${kelly}</td>
        <td class="right">${stake}</td>
        <td>${r.commence ? new Date(r.commence).toLocaleString() : ''}</td>
        <td>${badge}</td>
        <td><a class="btn" href="${compareLink}" target="_blank">🔍 Compare</a></td>
      `;
      tb.appendChild(tr);
    });
    document.getElementById('sumStake').innerText = sumStake.toFixed(2);
    document.getElementById('meta').innerText = `Łącznie: ${data.length} pozycji`;
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

# =========================================================
# MATCH SUMMARY – proste zestawienie jednego meczu (JSON)
# =========================================================
def _find_event_by_teams(events, home_q: str, away_q: str):
    """Znajdź event po 'zawiera' (case-insensitive) nazwach drużyn."""
    hq = (home_q or "").casefold()
    aq = (away_q or "").casefold()
    best = None
    for ev in events:
        home = (ev.get("home_team") or "").casefold()
        away = (ev.get("away_team") or "").casefold()
        if hq and aq and (hq in home and aq in away):
            return ev
        # fallback: pasuj wg dowolnej kolejności
        if hq and aq and (hq in away and aq in home):
            return ev
        # zapamiętaj częściowe trafienie (na wypadek literówek)
        if (hq and hq in home) or (aq and aq in away) or (hq and hq in away) or (aq and aq in home):
            best = best or ev
    return best

@app.get("/match_summary")
def match_summary(
    sport: str = Query("soccer_epl", description="np. soccer_epl / epl alias"),
    region: str = Query("eu,uk"),
    home: str = Query(..., description="Nazwa gospodarzy (fragment)"),
    away: str = Query(..., description="Nazwa gości (fragment)"),
    bookmakers: Optional[str] = Query(None, description="CSV, np. Unibet,Betfair"),
    commission: float = Query(0.0, ge=0.0, le=0.1),
):
    """
    Zwraca prosty podgląd 1 meczu: procenty HOME / DRAW / AWAY + rekomendacja.
    """
    if not ODDS_API_KEY:
        raise HTTPException(status_code=500, detail="Brak ODDS_API_KEY")

    sport = normalize_sport_key(sport)

    # pobierz kursy
    url = f"{BASE}/sports/{sport}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": region, "markets": "h2h", "oddsFormat": "decimal"}
    res = fetch_json(url, params, cache_key=f"summary:{sport}:{region}:h2h")
    if not res["ok"]:
        raise HTTPException(status_code=502, detail=f"API error: {res['error']}")

    events = res["data"]
    ev = _find_event_by_teams(events, home, away)
    if not ev:
        raise HTTPException(status_code=404, detail="Nie znaleziono meczu. Sprawdź nazwy.")

    H = ev.get("home_team"); A = ev.get("away_team")
    league = ev.get("sport_title", "")
    start = ev.get("commence_time")

    # filtry buków
    books_filter = None
    if bookmakers:
        books_filter = set([b.strip().lower() for b in bookmakers.split(",") if b.strip()])

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
                if name == H and (not best_home or price > best_home):
                    best_home, best_home_book = price, bname
                elif name == A and (not best_away or price > best_away):
                    best_away, best_away_book = price, bname
                elif name == "Draw" and (not best_draw or price > best_draw):
                    best_draw, best_draw_book = price, bname

    # implied & normalizacja
    h, a, d = implied(best_home), implied(best_away), implied(best_draw)
    if any([h, a, d]):
        h, a, d = normalize_3way(h, a, d)

    # kurs efektywny po prowizji (tylko do informacji)
    def eff(p): return round(p * (1.0 - commission), 3) if p else None

    probs = {
        "HOME": round((h or 0.0) * 100, 1),
        "DRAW": round((d or 0.0) * 100, 1),
        "AWAY": round((a or 0.0) * 100, 1),
    }
    prices = {
        "HOME": best_home, "DRAW": best_draw, "AWAY": best_away,
        "HOME_eff": eff(best_home), "DRAW_eff": eff(best_draw), "AWAY_eff": eff(best_away),
    }
    books = {"HOME": best_home_book, "DRAW": best_draw_book, "AWAY": best_away_book}

    # rekomendacja: porównujemy tylko HOME vs AWAY (remis zawsze żółty)
    ha = [("HOME", h or 0.0), ("AWAY", a or 0.0)]
    winner = max(ha, key=lambda x: x[1])[0]
    loser  = "HOME" if winner == "AWAY" else "AWAY"

    return {
        "match": f"{H} vs {A}",
        "league": league,
        "start": start,
        "probs_percent": probs,            # np. {"HOME": 52.1, "DRAW": 24.7, "AWAY": 23.2}
        "prices": prices,                  # kursy + kursy po prowizji
        "bookmakers": books,               # skąd kurs
        "colors": {                        # jakie kolory w UI
            "HOME": "green" if winner == "HOME" else ("red" if loser == "HOME" else "yellow"),
            "DRAW": "yellow",
            "AWAY": "green" if winner == "AWAY" else ("red" if loser == "AWAY" else "yellow"),
        },
        "recommendation": {                # jasno: na co grać
            "pick": winner,
            "confidence_percent": round(max(h or 0.0, a or 0.0)*100, 1)
        }
    }

# =========================================================
# UI dla jednego meczu
# =========================================================
@app.get("/ui_match")
def ui_match():
    html = """
<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Match Summary</title>
<style>
:root{--bg:#0f1220;--card:#171a2b;--text:#e9ecf1;--muted:#9aa3b2;--green:#27c28a;--red:#f87171;--yellow:#f5c84a;}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto}
.wrap{max-width:820px;margin:28px auto;padding:0 16px}
.card{background:var(--card);border-radius:14px;padding:16px;box-shadow:0 6px 20px rgba(0,0,0,.25)}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
input{background:#0e1120;border:1px solid #2b324d;color:var(--text);border-radius:10px;padding:10px 12px;outline:none}
button{background:#4251f5;border:none;color:#fff;border-radius:10px;padding:10px 14px;cursor:pointer}
h2{margin:0 0 8px 0}
.badge{padding:4px 8px;border-radius:999px;font-size:12px}
.green{background:rgba(39,194,138,.16);color:#27c28a}
.red{background:rgba(248,113,113,.16);color:#f87171}
.yellow{background:rgba(245,200,74,.18);color:#f5c84a}
.bar{height:12px;border-radius:8px;background:#202542}
.bar>span{display:block;height:12px;border-radius:8px}
.label{width:72px;display:inline-block}
small{color:var(--muted)}
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <div class="row">
      <input id="sport" value="soccer_epl" style="width:160px" placeholder="sport"/>
      <input id="region" value="eu,uk" style="width:120px" placeholder="region"/>
      <input id="home" placeholder="Home (fragment)"/>
      <input id="away" placeholder="Away (fragment)"/>
      <input id="books" placeholder="np. Unibet,Betfair"/>
      <input id="comm" type="number" step="0.005" value="0" style="width:90px" placeholder="commission"/>
      <button onclick="go()">Pokaż</button>
    </div>
    <div id="out"><small>Wpisz nazwy drużyn i kliknij Pokaż. Możesz też przejść tu z linku „🔍 Compare” w /ui.</small></div>
  </div>
</div>

<script>
function color(c){return c==='green'?'green':(c==='red'?'red':'yellow');}

// jeśli przyszliśmy z /ui przez hash, wczytaj parametry
(function preset(){
  if(location.hash && location.hash.length>1){
    const p = new URLSearchParams(location.hash.slice(1));
    if(p.get('sport')) document.getElementById('sport').value = p.get('sport');
    if(p.get('region')) document.getElementById('region').value = p.get('region');
    if(p.get('home')) document.getElementById('home').value = p.get('home');
    if(p.get('away')) document.getElementById('away').value = p.get('away');
    if(p.get('bookmakers')) document.getElementById('books').value = p.get('bookmakers');
    if(p.get('commission')) document.getElementById('comm').value = p.get('commission');
    setTimeout(go, 50);
  }
})();

function go(){
  const q=new URLSearchParams();
  q.set('sport',document.getElementById('sport').value||'soccer_epl');
  q.set('region',document.getElementById('region').value||'eu,uk');
  q.set('home',document.getElementById('home').value||'');
  q.set('away',document.getElementById('away').value||'');
  const b=document.getElementById('books').value.trim(); if(b) q.set('bookmakers',b);
  const c=document.getElementById('comm').value||'0'; q.set('commission',c);
  fetch('/match_summary?'+q.toString()).then(r=>{
    if(!r.ok) throw new Error('HTTP '+r.status);
    return r.json();
  }).then(data=>{
    const p=data.probs_percent;
    const col=data.colors;
    const prices=data.prices;
    const books=data.bookmakers;

    const mkBar=(lab,val,c)=>{
      const w=Math.max(1, Math.min(100, val));
      const cls=color(c);
      return `
        <div style="margin:10px 0">
          <div class="label">${lab}</div>
          <div class="bar"><span style="width:${w}%; background: var(--${cls});"></span></div>
          <div style="display:flex; gap:12px; align-items:center; margin-top:4px">
            <span class="badge ${cls}">${val.toFixed(1)}%</span>
            <small>Kurs: ${prices[lab]?.toFixed ? prices[lab].toFixed(2) : (prices[lab]||'-')} (${books[lab]||'-'})</small>
          </div>
        </div>`;
    };

    document.getElementById('out').innerHTML = `
      <h2>${data.match}</h2>
      <small>${data.league||''} • ${data.start ? new Date(data.start).toLocaleString() : ''}</small>
      <div style="margin-top:12px"></div>
      ${mkBar('HOME', p.HOME, col.HOME)}
      ${mkBar('DRAW', p.DRAW, col.DRAW)}
      ${mkBar('AWAY', p.AWAY, col.AWAY)}
      <div style="margin-top:8px"><b>Rekomendacja:</b> <span class="badge green">${data.recommendation.pick}</span>
      <small>(pewność ${data.recommendation.confidence_percent.toFixed(1)}%)</small></div>
    `;
  }).catch(e=>{
    document.getElementById('out').innerHTML = '<small style="color:#f87171">Błąd: '+e+'</small>';
  });
}
</script>
</body>
</html>
    """.strip()
    return Response(content=html, media_type="text/html")
