import json
from pathlib import Path

from src.ml.predictor import DraftPredictor
from src.models.schemas import (
    BanRecommendation,
    DraftState,
    PlayerBrawler,
    Recommendation,
    TierList,
)

WEIGHT_PROFILES = {
    "solo_queue": {"map": 0.40, "counter": 0.30, "synergy": 0.15, "tier": 0.15},
    "pro_meta":   {"map": 0.25, "counter": 0.30, "synergy": 0.15, "tier": 0.30},
}

ML_WEIGHT_PROFILES = {
    "solo_queue": {"ml": 0.65, "map": 0.15, "tier": 0.20},
    "pro_meta":   {"ml": 0.50, "map": 0.15, "tier": 0.35},
}


class DraftScorer:
    def __init__(self, data_dir: Path = Path("data")) -> None:
        model_path = data_dir / "draft_model.pt"
        self.predictor = DraftPredictor(model_path) if model_path.exists() else None
        self.my_brawlers = self._load_my_brawlers(data_dir / "raw/my_brawlers.json")
        self.ready_brawlers: set[str] = {
            b.name for b in self.my_brawlers if b.is_ranked_ready
        }
        self.tier_list = TierList.model_validate_json(
            (data_dir / "tier_list.json").read_text(encoding="utf-8")
        )
        self.map_stats: dict[str, dict[str, float]] = json.loads(
            (data_dir / "map_stats.json").read_text(encoding="utf-8")
        )
        matchups = json.loads((data_dir / "matchups.json").read_text(encoding="utf-8"))
        self.counters: dict[str, list[str]] = matchups.get("counters", {})
        self.countered_by: dict[str, list[str]] = {}
        for attacker, targets in self.counters.items():
            for target in targets:
                self.countered_by.setdefault(target, []).append(attacker)
        self.synergies: dict[str, list[str]] = matchups.get("synergies", {})

    @staticmethod
    def _load_my_brawlers(path: Path) -> list[PlayerBrawler]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [PlayerBrawler(**item) for item in raw]

    def _calc_counter_score(self, brawler: str, enemies: list[str]) -> tuple[float, list[str]]:
        if not enemies:
            return 0.5, []

        hits = [e for e in enemies if e in self.counters.get(brawler, [])]
        penalties = [e for e in enemies if brawler in self.counters.get(e, [])]

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

    def _calc_early_pick_risk(
        self,
        brawler: str,
        unavailable: set[str],
        map_data: dict[str, float],
        enemy_count: int,
    ) -> tuple[float, str, list[str]]:
        remaining_enemy_picks = max(0, 3 - enemy_count)
        if remaining_enemy_picks == 0:
            return 0.0, "Baixo", []

        exposure = remaining_enemy_picks / 3.0
        potential_counters = self.countered_by.get(brawler, [])

        viable_open = [
            c for c in potential_counters
            if c not in unavailable
            and (
                map_data.get(c, 0.0) >= 0.48
                or self.tier_list.tiers.get(c) in ("S", "A")
            )
        ]
        viable_open.sort(key=lambda c: map_data.get(c, 0.50), reverse=True)

        count = len(viable_open)
        if count == 0 and remaining_enemy_picks >= 2:
            return -0.03, "Baixo", ["First pick seguro"]

        penalty = min(0.16, count * 0.03) * exposure

        if penalty >= 0.08:
            risk = "Alto"
            top_threats = ", ".join(viable_open[:3])
            reasons = [f"Risco alto de counter aberto ({top_threats})"]
        elif penalty >= 0.04:
            risk = "Médio"
            top_threats = ", ".join(viable_open[:2])
            reasons = [f"Vulnerável a {top_threats}"]
        else:
            risk = "Baixo"
            reasons = []

        return penalty, risk, reasons

    def recommend_bans(self, state: DraftState, top_n: int = 5) -> list[BanRecommendation]:
        map_key = state.map_name.upper()
        map_data = self.map_stats.get(map_key, {})
        unavailable = {
            b.upper() for b in (state.bans + state.my_team + state.enemy_team)
        }

        candidates = set(map_data.keys()) | {
            b for b, t in self.tier_list.tiers.items() if t in ("S", "A")
        }
        map_meta_pool = {b for b, wr in map_data.items() if wr >= 0.50}

        ban_weights = (
            {"map": 0.50, "tier": 0.25, "threat": 0.25}
            if state.profile == "solo_queue"
            else {"map": 0.35, "tier": 0.40, "threat": 0.25}
        )

        results: list[BanRecommendation] = []
        for brawler in candidates:
            if brawler in unavailable:
                continue

            map_score = map_data.get(brawler, 0.50)
            tier_label = self.tier_list.tiers.get(brawler, "B")
            tier_score = self.tier_list.weights.get(tier_label, 0.50)

            countered_on_map = [
                t for t in self.counters.get(brawler, []) if t in map_meta_pool
            ]
            threat_score = min(1.0, 0.45 + 0.08 * len(countered_on_map))

            owned_lvl11 = brawler in self.ready_brawlers
            unowned_bonus = 0.04 if not owned_lvl11 else 0.0

            final_score = (
                ban_weights["map"] * map_score
                + ban_weights["tier"] * tier_score
                + ban_weights["threat"] * threat_score
                + unowned_bonus
            )
            final_score = max(0.0, min(1.0, final_score))

            reasons = []
            if map_score >= 0.55:
                reasons.append(f"Dominante no mapa ({map_score:.0%})")
            if tier_label in ("S", "A"):
                reasons.append(f"Pro Tier {tier_label}")
            if len(countered_on_map) >= 3:
                reasons.append(f"Counter a {len(countered_on_map)} picks fortes aqui")
            if not owned_lvl11:
                reasons.append("Não tens a Nível 11")

            results.append(
                BanRecommendation(
                    brawler=brawler,
                    score=round(final_score * 100, 1),
                    tier=tier_label,
                    owned_lvl11=owned_lvl11,
                    reasons=reasons or ["Ameaça geral no mapa"],
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n]

    def recommend(self, state: DraftState, top_n: int = 5) -> list[Recommendation]:
        map_key = state.map_name.upper()
        map_data = self.map_stats.get(map_key, {})
        weights = WEIGHT_PROFILES[state.profile]

        unavailable = {
            b.upper() for b in (state.bans + state.my_team + state.enemy_team)
        }
        enemies = [e.upper() for e in state.enemy_team]
        allies = [a.upper() for a in state.my_team]

        eligible = [
            b for b in self.my_brawlers
            if b.name not in unavailable and (not state.only_level_11 or b.is_ranked_ready)
        ]

        ml_probs = (
            self.predictor.predict_candidates(
                map_key, allies, enemies, [b.name for b in eligible]
            )
            if self.predictor
            else {}
        )

        results: list[Recommendation] = []
        for b in eligible:
            map_score = map_data.get(b.name, 0.50)

            tier_label = self.tier_list.tiers.get(b.name, "B")
            tier_score = self.tier_list.weights.get(tier_label, 0.50)

            counter_score, counter_reasons = self._calc_counter_score(b.name, enemies)
            synergy_score, synergy_reasons = self._calc_synergy_score(b.name, allies)

            risk_penalty, risk_label, risk_reasons = self._calc_early_pick_risk(
                b.name, unavailable, map_data, len(enemies)
            )

            if b.name in ml_probs:
                ml_w = ML_WEIGHT_PROFILES[state.profile]
                ml_score = ml_probs[b.name]
                final_score = (
                    ml_w["ml"] * ml_score
                    + ml_w["map"] * map_score
                    + ml_w["tier"] * tier_score
                    - risk_penalty
                )
            else:
                final_score = (
                    weights["map"] * map_score
                    + weights["counter"] * counter_score
                    + weights["synergy"] * synergy_score
                    + weights["tier"] * tier_score
                    - risk_penalty
                )
            final_score = max(0.0, min(1.0, final_score))

            reasons = []
            if b.name in ml_probs:
                reasons.append(f"ML Win Prob: {ml_probs[b.name]:.0%}")
            if map_score >= 0.58:
                reasons.append(f"Winrate alta no mapa ({map_score:.0%})")
            if tier_label in ("S", "A"):
                reasons.append(f"Pro Tier {tier_label}")
            reasons.extend(counter_reasons)
            reasons.extend(synergy_reasons)
            reasons.extend(risk_reasons)

            results.append(
                Recommendation(
                    brawler=b.name,
                    score=round(final_score * 100, 1),
                    tier=tier_label,
                    risk=risk_label,
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
        print(f"{i}. {rec.brawler:<10} | Score: {rec.score:>5}% | Tier: {rec.tier} | Risco: {rec.risk:<5} | {' · '.join(rec.reasons)}")