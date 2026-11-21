from pathlib import Path

import torch

from src.ml.model import DraftNet


class DraftPredictor:
    def __init__(self, model_path: Path = Path("data/draft_model.pt")) -> None:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.brawler_to_idx: dict[str, int] = checkpoint["brawler_to_idx"]
        self.map_to_idx: dict[str, int] = checkpoint["map_to_idx"]
        self.val_auc: float = checkpoint.get("val_auc", 0.0)

        self.model = DraftNet(**checkpoint["config"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def _encode_team(self, brawlers: list[str]) -> list[int]:
        ids = [self.brawler_to_idx.get(b.upper(), 0) for b in brawlers[:3]]
        return ids + [0] * (3 - len(ids))

    def predict_candidates(
        self,
        map_name: str,
        my_team: list[str],
        enemy_team: list[str],
        candidates: list[str],
    ) -> dict[str, float]:
        if not candidates:
            return {}

        map_id = self.map_to_idx.get(map_name.upper(), 0)
        t2_encoded = self._encode_team(enemy_team)

        t1_batch = [self._encode_team(my_team[:2] + [c]) for c in candidates]
        t2_batch = [t2_encoded] * len(candidates)
        m_batch = [map_id] * len(candidates)

        with torch.no_grad():
            logits = self.model(
                torch.tensor(t1_batch, dtype=torch.long),
                torch.tensor(t2_batch, dtype=torch.long),
                torch.tensor(m_batch, dtype=torch.long),
            )
            probs = torch.sigmoid(logits).tolist()

        return dict(zip(candidates, probs))