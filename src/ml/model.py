import torch
import torch.nn as nn


class DraftNet(nn.Module):
    def __init__(
        self,
        num_brawlers: int,
        num_maps: int,
        brawler_dim: int = 32,
        map_dim: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.brawler_emb = nn.Embedding(num_brawlers, brawler_dim, padding_idx=0)
        self.map_emb = nn.Embedding(num_maps, map_dim, padding_idx=0)

        in_dim = (brawler_dim * 4) + map_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _pool_team(self, team_ids: torch.Tensor) -> torch.Tensor:
        emb = self.brawler_emb(team_ids)                 # (B, 3, D)
        mask = (team_ids != 0).unsqueeze(-1).float()     # (B, 3, 1)
        counts = mask.sum(dim=1).clamp(min=1.0)          # (B, 1)
        return (emb * mask).sum(dim=1) / counts          # (B, D)

    def forward(
        self,
        t1_ids: torch.Tensor,
        t2_ids: torch.Tensor,
        map_ids: torch.Tensor,
    ) -> torch.Tensor:
        v1 = self._pool_team(t1_ids)
        v2 = self._pool_team(t2_ids)
        vm = self.map_emb(map_ids)

        features = torch.cat([v1, v2, v1 - v2, v1 * v2, vm], dim=-1)
        return self.head(features).squeeze(-1)