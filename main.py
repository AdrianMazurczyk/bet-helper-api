from fastapi import FastAPI
import requests
import os

app = FastAPI()

API_KEY = "74079997f2e6463585d5060a2dec6aca"
BASE_URL = "https://api.the-odds-api.com/v4/sports"

@app.get("/")
def home():
    return {"message": "Bet Helper działa z The Odds API!"}


@app.get("/picks")
def get_picks():
    # przykładowo: piłka nożna - liga angielska
    sport = "soccer_epl"
    url = f"{BASE_URL}/{sport}/odds/"
    params = {
        "apiKey": API_KEY,
        "regions": "eu",   # region bukmacherów
        "markets": "h2h",  # zakłady 1x2
        "oddsFormat": "decimal"
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        picks = []
        for match in data[:5]:  # weźmy pierwsze 5 meczów
            teams = match.get("bookmakers", [])[0].get("markets", [])[0].get("outcomes", [])
            for outcome in teams:
                picks.append({
                    "match": f"{match['home_team']} vs {match['away_team']}",
                    "selection": outcome["name"],
                    "price": outcome["price"]
                })

        return picks

    except Exception as e:
        return {"error": str(e)}
