from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.api.sync_hf import load_dataframe
from src.ml.model import DraftNet

MODEL_PATH = Path("data/draft_model.pt")


def prepare_tensors(df: pd.DataFrame, min_skill_ns: float = 0.0):
    if "skill_ns_ok" in df.columns:
        df = df[df["skill_ns_ok"] == 1]
    if "skill_ns" in df.columns:
        df = df[df["skill_ns"] >= min_skill_ns]

    t1_cols = ["t1_b0_name", "t1_b1_name", "t1_b2_name"]
    t2_cols = ["t2_b0_name", "t2_b1_name", "t2_b2_name"]
    df = df[["map", "record", *t1_cols, *t2_cols]].dropna().copy()

    # Extrai vencedor a partir da coluna record (ex: "T2-T2" ou "T1-T2-T1")
    t1_wins = df["record"].str.upper().str.count("T1")
    t2_wins = df["record"].str.upper().str.count("T2")
    valid = t1_wins != t2_wins
    df = df[valid].reset_index(drop=True)
    y = (t1_wins[valid] > t2_wins[valid]).astype(np.float32).to_numpy()

    # Normaliza nomes e cria vocabulários (0 reservado para PAD/UNK)
    maps = df["map"].str.upper().to_numpy()
    brawler_vals = df[t1_cols + t2_cols].apply(lambda s: s.str.upper())

    unique_brawlers = sorted(pd.unique(brawler_vals.to_numpy().ravel()))
    unique_maps = sorted(pd.unique(maps))

    brawler_to_idx = {b: i + 1 for i, b in enumerate(unique_brawlers)}
    map_to_idx = {m: i + 1 for i, m in enumerate(unique_maps)}

    t1 = np.column_stack([brawler_vals[c].map(brawler_to_idx).to_numpy() for c in t1_cols])
    t2 = np.column_stack([brawler_vals[c].map(brawler_to_idx).to_numpy() for c in t2_cols])
    m_ids = np.array([map_to_idx[m] for m in maps], dtype=np.int64)

    return t1.astype(np.int64), t2.astype(np.int64), m_ids, y, brawler_to_idx, map_to_idx


def apply_draft_augmentation(
    t1: torch.Tensor, t2: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 1. Troca aleatória de equipas (50%) para garantir simetria T1 vs T2
    swap = torch.rand(len(y), device=y.device) < 0.5
    t1_aug = torch.where(swap.unsqueeze(1), t2, t1)
    t2_aug = torch.where(swap.unsqueeze(1), t1, t2)
    y_aug = torch.where(swap, 1.0 - y, y)

    # 2. Masking parcial em 25% do batch (simula drafts a meio)
    partial = torch.rand(len(y), device=y.device) < 0.25
    mask1 = (torch.rand_like(t1_aug.float()) < 0.35) & partial.unsqueeze(1)
    mask2 = (torch.rand_like(t2_aug.float()) < 0.35) & partial.unsqueeze(1)
    t1_aug = t1_aug.masked_fill(mask1, 0)
    t2_aug = t2_aug.masked_fill(mask2, 0)

    return t1_aug, t2_aug, y_aug


def train(epochs: int = 15, batch_size: int = 1024, lr: float = 1e-3) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"A treinar em: {device}")

    df = load_dataframe(n_files=1)
    t1, t2, m_ids, y, brawler_to_idx, map_to_idx = prepare_tensors(df)
    print(f"Dataset pronto: {len(y):,} partidas | {len(brawler_to_idx)} brawlers | {len(map_to_idx)} mapas")

    idx_train, idx_val = train_test_split(np.arange(len(y)), test_size=0.15, random_state=42, stratify=y)

    train_ds = TensorDataset(
        torch.tensor(t1[idx_train]), torch.tensor(t2[idx_train]),
        torch.tensor(m_ids[idx_train]), torch.tensor(y[idx_train]),
    )
    val_ds = TensorDataset(
        torch.tensor(t1[idx_val]), torch.tensor(t2[idx_val]),
        torch.tensor(m_ids[idx_val]), torch.tensor(y[idx_val]),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2)

    config = {
        "num_brawlers": len(brawler_to_idx) + 1,
        "num_maps": len(map_to_idx) + 1,
        "brawler_dim": 32,
        "map_dim": 16,
        "hidden_dim": 128,
        "dropout": 0.2,
    }
    model = DraftNet(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bt1, bt2, bm, by in train_loader:
            bt1, bt2, bm, by = bt1.to(device), bt2.to(device), bm.to(device), by.to(device)
            bt1, bt2, by = apply_draft_augmentation(bt1, bt2, by)

            optimizer.zero_grad()
            logits = model(bt1, bt2, bm)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(by)

        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for bt1, bt2, bm, by in val_loader:
                logits = model(bt1.to(device), bt2.to(device), bm.to(device))
                val_probs.append(torch.sigmoid(logits).cpu().numpy())
                val_targets.append(by.numpy())

        probs = np.concatenate(val_probs)
        targets = np.concatenate(val_targets)
        auc = roc_auc_score(targets, probs)
        acc = ((probs >= 0.5) == targets).mean()
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {train_loss / len(train_ds):.4f} | Val Acc: {acc:.2%} | Val AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": config,
                    "brawler_to_idx": brawler_to_idx,
                    "map_to_idx": map_to_idx,
                    "val_auc": float(best_auc),
                },
                MODEL_PATH,
            )

    print(f"\nMelhor modelo guardado em {MODEL_PATH} (Val AUC: {best_auc:.4f})")


if __name__ == "__main__":
    train()