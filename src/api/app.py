from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from src.engine.scorer import DraftScorer
from src.models.schemas import DraftState, Recommendation

app = FastAPI(title="Brawl Draft Assistant")
STATIC_DIR = Path("static")

scorer = DraftScorer()


@app.get("/")
def serve_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/meta")
def get_meta() -> dict[str, list[str]]:
    brawlers = {b.name for b in scorer.my_brawlers}
    brawlers.update(scorer.tier_list.tiers.keys())
    for map_data in scorer.map_stats.values():
        brawlers.update(map_data.keys())

    return {
        "maps": sorted(scorer.map_stats.keys()),
        "brawlers": sorted(brawlers),
    }


@app.post("/api/recommend", response_model=list[Recommendation])
def recommend_picks(state: DraftState) -> list[Recommendation]:
    return scorer.recommend(state, top_n=8)


@app.post("/api/reload")
def reload_data() -> dict[str, str]:
    global scorer
    scorer = DraftScorer()
    return {"status": "reloaded"}