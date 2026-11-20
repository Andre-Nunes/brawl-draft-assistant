from pydantic import BaseModel, Field

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