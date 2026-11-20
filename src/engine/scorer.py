import json
from pathlib import Path

from src.models.schemas import DraftState, PlayerBrawler, Recommendation, TierList

WEIGHT_PROFILES = {
    "solo_queue": {"map": 0.40, "counter": 0.30, "synergy": 0.15, "tier": 0.15},
    "pro_meta":   {"map": 0.25, "counter": 0.30, "synergy": 0.15, "tier": 0.30},
}


class DraftScorer:
    def __init__(self, data_dir: Path = Path("data")) -> None:
        self.my_brawlers = self._load_my_brawlers(data_dir / "raw/my_brawlers.json")
        self.tier_list = TierList.model_validate_json(
            (data_dir / "tier_list.json").read_text(encoding="utf-8")
        )
        self.map_stats: dict[str, dict[str, float]] = json.loads(
            (data_dir / "map_stats.json").read_text(encoding="utf-8")
        )
        matchups = json.loads((data_dir / "matchups.json").read_text(encoding="utf-8"))
        self.counters: dict[str, list[str]] = matchups.get("counters", {})
        self.synergies: dict[str, list[str]] = matchups.get("synergies", {})

    @staticmethod
    def _load_my_brawlers(path: Path) -> list[PlayerBrawler]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [PlayerBrawler(**item) for item in raw]

    def _calc_counter_score(self, brawler: str, enemies: list[str]) -> tuple[float, list[str]]:
        if not enemies:
            return 0.5, []  # Neutro se ainda não há inimigos (ex: first pick)

        hits = [e for e in enemies if e in self.counters.get(brawler, [])]
        penalties = [e for e in enemies if brawler in self.counters.get(e, [])]

        # Normaliza entre 0.0 e 1.0 a partir de uma base neutra (0.5)
        raw_score = 0.5 + (0.25 * len(hits)) - (0.30 * len(penalties))
        score = max(0.0, min(1.0, raw_score))

        reasons = []
        if hits:
            reasons.append(f"Counter a {', '.join(hits)}")
        if penalties:
            reasons.append(f"Vulnerável a {', '.join(penalties)}")
        return score, reasons

    def _calc_synergy_score(self, brawler: str, allies: list[str]) -> tuple[float, list[str]]:
        if not allies:
            return 0.5, []

        hits = [a for a in allies if a in self.synergies.get(brawler, [])]
        score = min(1.0, 0.5 + (0.25 * len(hits)))
        reasons = [f"Sinergia com {', '.join(hits)}"] if hits else []
        return score, reasons

    def recommend(self, state: DraftState, top_n: int = 5) -> list[Recommendation]:
        map_key = state.map_name.upper()
        map_data = self.map_stats.get(map_key, {})
        weights = WEIGHT_PROFILES[state.profile]

        unavailable = {
            b.upper() for b in (state.bans + state.my_team + state.enemy_team)
        }
        enemies = [e.upper() for e in state.enemy_team]
        allies = [a.upper() for a in state.my_team]

        results: list[Recommendation] = []

        for b in self.my_brawlers:
            if b.name in unavailable:
                continue
            if state.only_level_11 and not b.is_ranked_ready:
                continue

            # 1. Map Winrate (fallback neutro 0.50 se não houver dados no mapa)
            map_score = map_data.get(b.name, 0.50)

            # 2. Tier List Score
            tier_label = self.tier_list.tiers.get(b.name, "B")
            tier_score = self.tier_list.weights.get(tier_label, 0.50)

            # 3. Counters & Sinergias
            counter_score, counter_reasons = self._calc_counter_score(b.name, enemies)
            synergy_score, synergy_reasons = self._calc_synergy_score(b.name, allies)

            final_score = (
                weights["map"] * map_score
                + weights["counter"] * counter_score
                + weights["synergy"] * synergy_score
                + weights["tier"] * tier_score
            )

            reasons = []
            if map_score >= 0.58:
                reasons.append(f"Winrate alta no mapa ({map_score:.0%})")
            if tier_label in ("S", "A"):
                reasons.append(f"Pro Tier {tier_label}")
            reasons.extend(counter_reasons)
            reasons.extend(synergy_reasons)

            results.append(
                Recommendation(
                    brawler=b.name,
                    score=round(final_score * 100, 1),
                    tier=tier_label,
                    reasons=reasons or ["Pick equilibrado"],
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n]


if __name__ == "__main__":
    scorer = DraftScorer()
    sample_draft = DraftState(
        map_name="CENTER STAGE",
        bans=["MELODIE", "MOE"],
        my_team=["SPIKE"],
        enemy_team=["BULL", "EL PRIMO"],
        profile="solo_queue",
    )

    print(f"=== Recomendações para {sample_draft.map_name} ({sample_draft.profile}) ===")
    for i, rec in enumerate(scorer.recommend(sample_draft), start=1):
        print(f"{i}. {rec.brawler:<10} | Score: {rec.score:>5}% | Tier: {rec.tier} | {' · '.join(rec.reasons)}")