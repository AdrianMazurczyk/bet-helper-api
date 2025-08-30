from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Bet Helper działa!"}
from typing import List, Dict

# proste, przykładowe typy (na start)
SAMPLE_PICKS: List[Dict] = [
    {"match": "Team A vs Team B", "selection": "HOME", "price": 2.10, "ev": 0.07},
    {"match": "Team C vs Team D", "selection": "AWAY", "price": 1.95, "ev": 0.05},
]

@app.get("/picks")
def picks():
    return SAMPLE_PICKS
