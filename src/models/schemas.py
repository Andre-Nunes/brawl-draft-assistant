from pydantic import BaseModel, Field
from typing import Literal

class PlayerBrawler(BaseModel):
    id: int
    name: str
    power: int
    trophies: int
    has_star_power: bool = False
    has_gadget: bool = False

    @property
    def is_ranked_ready(self) -> bool:
        """Considera pronto para Ranked se estiver a nível 11."""
        return self.power >= 11

class TierList(BaseModel):
    patch_date: str
    source: str
    weights: dict[str, float] = Field(
        default={"S": 1.0, "A": 0.8, "B": 0.6, "C": 0.4, "D": 0.2, "F": 0.0}
    )
    tiers: dict[str, str]  # Ex: {"MELODIE": "S", "SHELLY": "C"}

class DraftState(BaseModel):
    map_name: str
    bans: list[str] = Field(default_factory=list)
    my_team: list[str] = Field(default_factory=list)
    enemy_team: list[str] = Field(default_factory=list)
    profile: Literal["solo_queue", "pro_meta"] = "solo_queue"
    only_level_11: bool = True

class Recommendation(BaseModel):
    brawler: str
    score: float
    tier: str
    risk: Literal["Baixo", "Médio", "Alto"] = "Baixo"
    reasons: list[str]

class BanRecommendation(BaseModel):
    brawler: str
    score: float
    tier: str
    owned_lvl11: bool
    reasons: list[str]