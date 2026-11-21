from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from src.api.supercell import SupercellClient
from src.engine.scorer import DraftScorer
from src.models.schemas import BanRecommendation, DraftState, Recommendation

app = FastAPI(title="Brawl Draft Assistant")
STATIC_DIR = Path("static")

scorer = DraftScorer()


@app.get("/")
def serve_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/meta")
def get_meta() -> dict[str, Any]:
    brawlers = {b.name for b in scorer.my_brawlers}
    brawlers.update(scorer.tier_list.tiers.keys())
    for map_data in scorer.map_stats.values():
        brawlers.update(map_data.keys())

    return {
        "maps": sorted(scorer.map_stats.keys()),
        "brawlers": sorted(brawlers),
        "brawler_ids": {b.name: b.id for b in scorer.my_brawlers},
        "ml_active": scorer.predictor is not None,
        "ml_auc": round(scorer.predictor.val_auc, 4) if scorer.predictor else None,
        "ready_count": len(scorer.ready_brawlers),
    }


@app.post("/api/recommend", response_model=list[Recommendation])
def recommend_picks(state: DraftState) -> list[Recommendation]:
    return scorer.recommend(state, top_n=8)


@app.post("/api/bans", response_model=list[BanRecommendation])
def recommend_bans(state: DraftState) -> list[BanRecommendation]:
    return scorer.recommend_bans(state, top_n=5)


@app.post("/api/reload")
def reload_data() -> dict[str, str]:
    global scorer
    scorer = DraftScorer()
    return {"status": "reloaded"}


@app.post("/api/sync-account")
def sync_account() -> dict[str, Any]:
    global scorer
    try:
        client = SupercellClient()
        brawlers = client.fetch_my_brawlers(save_cache=True)
        scorer = DraftScorer()
        return {
            "status": "ok",
            "total": len(brawlers),
            "ready": len(scorer.ready_brawlers),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc