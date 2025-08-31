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

# -----------------------------
# Konfiguracja
# -----------------------------
app = FastAPI(title="Bet Helper", version="1.5.0")

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

# Proste aliasy sportów
SPORT_ALIASES = {
    "epl": "soccer_epl",
    "ekstraklasa": "soccer_poland_ekstraklasa",
    "pl": "soccer_poland_ekstraklasa",
    "laliga": "soccer_spain_la_liga",
    "la_liga": "soccer_spain_la_liga",
    "serie_a": "soccer_italy_serie_a",
    "bundesliga": "soccer_germany_bundesliga",
    "ligue1": "soccer_france_ligue_one",
    "ucl": "soccer_uefa_champs_league",
    "nba": "basketball_nba",
}
def resolve_sport(s: str) -> str:
    if not s:
        return s
    return SPORT_ALIASES.get(s.strip().lower(), s)

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

# -----------------------------
# Modele (Swagger)
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
    type: str
    rec_color: Optional[str] = None  # "green" | "red" | "yellow"

# -----------------------------
# Wspólna logika zbierania typów
# -----------------------------
def collect_picks_rows(
    sport: str,
    region: str,
    market: str,
    bookmakers: Optional[str],
    min_price: Optional[float],
    max_price: Optional[float],
    commission: float,
    since_hours: int,
    until_hours: int,
    min_ev: float,
    show: str,
    bankroll: float,
    kelly_fraction_q: float,
    kelly_cap: float,
    stake_all: bool,
) -> List[Dict]:

    if not ODDS_API_KEY:
        return []

    url = f"{BASE}/sports/{sport}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": region,
        "markets": market,
        "oddsFormat": "decimal",
    }

    books_filter = None
    if bookmakers:
        books_filter = set([b.strip().lower() for b in bookmakers.split(",") if b.strip()])

    now = datetime.now(timezone.utc)
    start_dt = now + timedelta(hours=since_hours)
    end_dt = now + timedelta(hours=until_hours)

    res = fetch_json(url, params, cache_key=f"odds:{sport}:{region}:{market}")
    if not res["ok"]:
        return []

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
            stake_amt = round(kel * bankroll, 2) if kel and bankroll else 0.0
            if not is_value and not stake_all:
                stake_amt = 0.0
            return {
                "match": f"{home} vs {away}",
                "event_id": f"{home}||{away}",
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
                "rec_color": None,  # uzupełnimy niżej
            }

        cands = [
            build("HOME", best_home, h, best_home_book),
            build("AWAY", best_away, a, best_away_book),
            build("DRAW", best_draw, d, best_draw_book),
        ]
        cands = [c for c in cands if c]

        # --- rekomendacja kolorami:
        # 1) remis zawsze żółty
        # 2) pozostale – najwyższe fair_prob w meczu = zielony, reszta czerwony
        best_prob = -1.0
        best_idx = -1
        for i, c in enumerate(cands):
            if c["selection"] == "DRAW":
                c["rec_color"] = "yellow"
            p = c.get("fair_prob") or 0.0
            if p > best_prob and c["selection"] != "DRAW":
                best_prob = p
                best_idx = i
        if best_idx >= 0:
            cands[best_idx]["rec_color"] = "green"
        for i, c in enumerate(cands):
            if c["rec_color"] is None:
                c["rec_color"] = "red"

        if show == "value":
            cands = [c for c in cands if c["type"] == "value"]

        out_rows.extend(cands)

    return out_rows

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
        return {"error": "Brak ODDS_API_KEY w środowisku Railway."}
    url = f"{BASE}/sports"
    params = {"apiKey": ODDS_API_KEY}
    if all:
        params["all"] = "true"
    res = fetch_json(url, params, cache_key=f"sports:{all}")
    if not res["ok"]:
        return {"error": f"API error: {res['error']}"}
    data = res["data"]
    return [{"key": s.get("key"), "title": s.get("title"), "active": s.get("active")} for s in data]

@app.get("/picks", response_model=List[Pick])
def picks(
    sport: str = Query("epl", description="np. soccer_epl lub alias: epl"),
    region: str = Query("eu,uk"),
    min_ev: float = Query(0.0, ge=0.0),
    limit: int = Query(60, ge=1, le=200),
    market: str = Query("h2h"),
    bookmakers: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None, ge=1.0),
    max_price: Optional[float] = Query(None, ge=1.0),
    bankroll: float = Query(1000.0, ge=0.0),
    kelly_fraction_q: float = Query(0.5, ge=0.0, le=1.0),
    kelly_cap: float = Query(0.25, ge=0.0, le=1.0),
    commission: float = Query(0.0, ge=0.0, le=0.1),
    show: str = Query("all", regex="^(value|all)$"),
    format: str = Query("json", regex="^(json|csv)$"),
    stake_all: bool = Query(True),
    since_hours: int = Query(0, ge=0),
    until_hours: int = Query(120, ge=1),
):
    sport = resolve_sport(sport)

    rows = collect_picks_rows(
        sport=sport, region=region, market=market,
        bookmakers=bookmakers, min_price=min_price, max_price=max_price,
        commission=commission, since_hours=since_hours, until_hours=until_hours,
        min_ev=min_ev, show=show, bankroll=bankroll,
        kelly_fraction_q=kelly_fraction_q, kelly_cap=kelly_cap, stake_all=stake_all
    )

    def sort_key(x: Dict):
        # najpierw kolor: green > yellow > red, potem EV, potem czas
        color_rank = {"green":2, "yellow":1, "red":0}.get(x.get("rec_color"), 0)
        evv = x["ev"] if x["ev"] is not None else -999
        c = x["commence"] or ""
        return (-color_rank, -evv, c)

    rows.sort(key=sort_key)
    rows = rows[:limit]

    if format == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            ["match","selection","price","bookmaker","fair_prob","ev","kelly","stake","commence","league","type","rec_color"]
        )
        for r in rows:
            writer.writerow([
                r.get("match"), r.get("selection"), r.get("price"), r.get("bookmaker"),
                r.get("fair_prob"), r.get("ev"), r.get("kelly"), r.get("stake"),
                r.get("commence"), r.get("league"), r.get("type"), r.get("rec_color"),
            ])
        return Response(content=buf.getvalue().encode("utf-8"), media_type="text/csv")

    return rows

@app.get("/accas")
def accas(
    sport: str = Query("epl"),
    region: str = Query("eu,uk"),
    min_ev: float = Query(0.0, ge=0.0),
    show: str = Query("all", regex="^(value|all)$"),
    market: str = Query("h2h"),
    bookmakers: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None, ge=1.0),
    max_price: Optional[float] = Query(None, ge=1.0),
    commission: float = Query(0.0, ge=0.0, le=0.1),
    since_hours: int = Query(0, ge=0),
    until_hours: int = Query(120, ge=1),
    bankroll: float = Query(0.0, ge=0.0),
    kelly_fraction_q: float = Query(0.5, ge=0.0, le=1.0),
    kelly_cap: float = Query(0.25, ge=0.0, le=1.0),
    stake_all: bool = Query(True),
    min_legs: int = Query(2, ge=1),
    max_legs: int = Query(10, ge=1, le=15),
    limit_candidates: int = Query(60, ge=5, le=150),
    stake: float = Query(2.0, ge=0.0),
):
    sport = resolve_sport(sport)

    rows = collect_picks_rows(
        sport=sport, region=region, market=market,
        bookmakers=bookmakers, min_price=min_price, max_price=max_price,
        commission=commission, since_hours=since_hours, until_hours=until_hours,
        min_ev=min_ev, show=show, bankroll=bankroll,
        kelly_fraction_q=kelly_fraction_q, kelly_cap=kelly_cap, stake_all=stake_all
    )

    # 1 najlepszy pick z meczu (po fair_prob), DRAW traktujemy jak inne
    best_per_match: Dict[str, Dict] = {}
    for r in rows:
        prob = r.get("fair_prob") or 0.0
        eid = r.get("event_id")
        if not eid or prob <= 0:
            continue
        prev = best_per_match.get(eid)
        if (prev is None) or (prob > (prev.get("fair_prob") or 0.0)):
            best_per_match[eid] = r

    cands = list(best_per_match.values())
    cands.sort(key=lambda x: x.get("fair_prob") or 0.0, reverse=True)
    cands = cands[:limit_candidates]

    out = []
    for L in range(min_legs, max_legs + 1):
        if len(cands) < L:
            continue
        picks_L = cands[:L]
        prob_total = 1.0
        odds_total = 1.0
        for p in picks_L:
            prob_total *= float(p.get("fair_prob") or 0.0)
            odds_total *= float(p.get("price") or 1.0)
        ev_rel = prob_total * odds_total - 1.0
        payout = round(odds_total * stake, 2)
        ev_cash = round(stake * ev_rel, 2)
        out.append({
            "legs": L,
            "prob": round(prob_total, 6),
            "prob_pct": round(prob_total * 100.0, 2),
            "odds": round(odds_total, 3),
            "stake": round(stake, 2),
            "payout": payout,
            "ev_rel": round(ev_rel, 4),
            "ev_cash": ev_cash,
            "picks": [{
                "match": p["match"],
                "selection": p["selection"],
                "price": p["price"],
                "bookmaker": p["bookmaker"],
                "fair_prob": p["fair_prob"],
                "ev": p["ev"],
                "commence": p["commence"],
                "rec_color": p.get("rec_color"),
            } for p in picks_L],
        })
    return out

@app.get("/debug")
def debug(
    sport: str = Query("ekstraklasa"),
    region: str = Query("eu,uk"),
    market: str = Query("h2h")
):
    if not ODDS_API_KEY:
        return {"error": "Brak ODDS_API_KEY"}
    sport = resolve_sport(sport)
    url = f"{BASE}/sports/{sport}/odds"
    params = {"apiKey": ODDS_API_KEY,"regions": region,"markets": market,"oddsFormat": "decimal"}
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
    return {"sport": sport,"region": region,"events_from_api": total_events,
            "events_with_h2h": with_h2h,"examples": examples,"rate_limit_headers": _last_limits}

# -----------------------------
# Dashboard /ui z kolorami
# -----------------------------
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
:root{--bg:#0f1220;--card:#171a2b;--text:#e9ecf1;--muted:#9aa3b2;--green:#27c28a;--gray:#2a2f45;--red:#f87171;--blue:#4251f5;--yellow:#facc15}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, Ubuntu}
.wrap{max-width:1100px;margin:32px auto;padding:0 16px}
h1{margin:0 0 16px 0;font-size:22px}
.card{background:var(--card);border-radius:14px;padding:16px;margin-bottom:16px;box-shadow:0 6px 20px rgba(0,0,0,.25)}
.row{display:flex;gap:8px;flex-wrap:wrap}
input,select{background:#0e1120;border:1px solid #2b324d;color:var(--text);border-radius:10px;padding:10px 12px;outline:none}
button{background:var(--blue);border:none;color:#fff;border-radius:10px;padding:10px 14px;cursor:pointer}
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
.tag{display:inline-block;padding:.2rem .5rem;border-radius:999px;font-weight:700}
.tag.green{background:rgba(39,194,138,.18);color:#19d198;border:1px solid rgba(39,194,138,.35)}
.tag.red{background:rgba(248,113,113,.18);color:#ff9da1;border:1px solid rgba(248,113,113,.35)}
.tag.yellow{background:rgba(250,204,21,.22);color:#ffe58a;border:1px solid rgba(250,204,21,.45)}
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
          <option value="value">value</option>
          <option value="all" selected>all</option>
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
        <select id="stake_all"><option value="false">false</option><option value="true" selected>true</option></select>
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
    <div class="row">
      <label>Acca min legs<br><input id="acca_min" type="number" value="2"/></label>
      <label>Acca max legs<br><input id="acca_max" type="number" value="10"/></label>
      <label>Stake per acca<br><input id="acca_stake" type="number" step="0.5" value="2"/></label>
      <label>Limit candidates<br><input id="acca_lim" type="number" value="60"/></label>
    </div>
    <div style="margin-top:10px" class="row">
      <button onclick="buildAccas()">Build accas</button>
      <small id="accameta"></small>
    </div>

    <div style="overflow:auto;margin-top:8px">
      <table id="accatbl">
        <thead>
          <tr>
            <th>Legs</th><th class="right">Prob</th><th class="right">Kurs</th>
            <th class="right">Stake</th><th class="right">Payout</th><th class="right">EV (cash)</th><th>Details</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

</div>

<script>
function tagFor(selection, color){
  const cls = color==='green'?'green':(color==='yellow'?'yellow':'red');
  return `<span class="tag ${cls}">${selection}</span>`;
}

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
        <td><div>${r.match}</div><small>${r.league||''}</small></td>
        <td>${tagFor(r.selection, r.rec_color)}</td>
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

async function buildAccas(){
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
  q.set('min_legs', document.getElementById('acca_min').value || '2');
  q.set('max_legs', document.getElementById('acca_max').value || '10');
  q.set('stake', document.getElementById('acca_stake').value || '2');
  q.set('limit_candidates', document.getElementById('acca_lim').value || '60');

  document.getElementById('accameta').innerText = 'Building…';
  try{
    const res = await fetch('/accas?' + q.toString());
    const data = await res.json();
    const tb = document.querySelector('#accatbl tbody');
    tb.innerHTML = '';
    data.forEach(a=>{
      const picksHtml = a.picks.map(p => `${p.match} — ${tagFor(p.selection, p.rec_color)} @ ${p.price} <small>(${(p.fair_prob*100).toFixed(1)}%)</small>`).join('<br>');
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${a.legs}</td>
        <td class="right">${a.prob_pct.toFixed(2)}%</td>
        <td class="right">${a.odds.toFixed ? a.odds.toFixed(2) : a.odds}</td>
        <td class="right">${a.stake.toFixed ? a.stake.toFixed(2) : a.stake}</td>
        <td class="right">${a.payout.toFixed ? a.payout.toFixed(2) : a.payout}</td>
        <td class="right">${a.ev_cash.toFixed ? a.ev_cash.toFixed(2) : a.ev_cash}</td>
        <td>${picksHtml}</td>
      `;
      tb.appendChild(tr);
    });
    document.getElementById('accameta').innerText = `Zbudowano: ${data.length} kuponów`;
  }catch(e){
    document.getElementById('accameta').innerText = 'Błąd: ' + e;
  }
}

loadPicks();
</script>
</body>
</html>
    """.strip()
    return Response(content=html, media_type="text/html")
