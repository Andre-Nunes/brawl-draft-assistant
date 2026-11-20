import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "EliF77/brawlstars-ranked"
DATA_DIR = Path("data")


def download_latest_parquets(n_files: int = 2) -> list[Path]:
    api = HfApi()
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    parquets = sorted(f for f in files if f.endswith(".parquet"))
    if not parquets:
        raise FileNotFoundError("Nenhum ficheiro .parquet encontrado no dataset.")

    selected = parquets[-n_files:]
    print(f"A obter os {len(selected)} ficheiros mais recentes: {selected}")
    return [
        Path(hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset"))
        for f in selected
    ]


def load_dataframe(n_files: int = 2) -> pd.DataFrame:
    paths = download_latest_parquets(n_files=n_files)
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def build_meta_from_hf(
    df: pd.DataFrame,
    min_skill_ns: float = 0.0,
    min_map_games: int = 50,
    min_pair_games: int = 50,
) -> None:
    # Filtra partidas com Elo válido e acima da média da season
    if "skill_ns_ok" in df.columns:
        df = df[df["skill_ns_ok"] == 1]
    if "skill_ns" in df.columns:
        df = df[df["skill_ns"] >= min_skill_ns]
    print(f"Partidas após filtro (skill_ns_ok=1, skill_ns>={min_skill_ns}): {len(df):,}")

    t1_cols = ["t1_b0_name", "t1_b1_name", "t1_b2_name"]
    t2_cols = ["t2_b0_name", "t2_b1_name", "t2_b2_name"]
    cols = ["map", "record", *t1_cols, *t2_cols]

    map_counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    vs_counts = defaultdict(lambda: [0, 0])
    syn_counts = defaultdict(lambda: [0, 0])

    for row in df[cols].dropna().itertuples(index=False, name=None):
        map_name = str(row[0]).upper()
        rounds = str(row[1]).upper().split("-")
        t1_wins, t2_wins = rounds.count("T1"), rounds.count("T2")

        if t1_wins == t2_wins:
            continue  # Ignora séries empatadas ou incompletas

        t1 = [str(b).upper() for b in row[2:5]]
        t2 = [str(b).upper() for b in row[5:8]]
        winners, losers = (t1, t2) if t1_wins > t2_wins else (t2, t1)

        for b in winners:
            map_counts[map_name][b][0] += 1
            map_counts[map_name][b][1] += 1
        for b in losers:
            map_counts[map_name][b][1] += 1

        for w in winners:
            for l in losers:
                if w != l:
                    vs_counts[(w, l)][0] += 1
                    vs_counts[(w, l)][1] += 1
                    vs_counts[(l, w)][1] += 1

        for a, b in combinations(sorted(winners), 2):
            if a != b:
                syn_counts[(a, b)][0] += 1
                syn_counts[(a, b)][1] += 1
                syn_counts[(b, a)][0] += 1
                syn_counts[(b, a)][1] += 1
        for a, b in combinations(sorted(losers), 2):
            if a != b:
                syn_counts[(a, b)][1] += 1
                syn_counts[(b, a)][1] += 1

    # Suavização bayesiana calibrada para amostras grandes
    map_stats = {}
    for map_name, brawlers in sorted(map_counts.items()):
        stats = {
            b: round((w + 25.0) / (t + 50.0), 3)
            for b, (w, t) in brawlers.items()
            if t >= min_map_games
        }
        if stats:
            map_stats[map_name] = dict(
                sorted(stats.items(), key=lambda x: x[1], reverse=True)
            )

    counters = defaultdict(list)
    for (b, opp), (w, t) in vs_counts.items():
        if t >= min_pair_games and (w / t) >= 0.55:
            counters[b].append(opp)

    synergies = defaultdict(list)
    for (a, b), (w, t) in syn_counts.items():
        if t >= min_pair_games and (w / t) >= 0.54:
            synergies[a].append(b)

    (DATA_DIR / "map_stats.json").write_text(json.dumps(map_stats, indent=2), encoding="utf-8")
    (DATA_DIR / "matchups.json").write_text(
        json.dumps({"counters": dict(counters), "synergies": dict(synergies)}, indent=2),
        encoding="utf-8",
    )
    print(f"Atualizado: {len(map_stats)} mapas | {len(counters)} brawlers com counters | {len(synergies)} com sinergias.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true", help="Mostra colunas e primeira linha do dataset")
    parser.add_argument("--files", type=int, default=2, help="Número de ficheiros Parquet mais recentes")
    args = parser.parse_args()

    dataframe = load_dataframe(n_files=1 if args.inspect else args.files)
    if args.inspect:
        print("Shape:", dataframe.shape)
        print("Colunas:", list(dataframe.columns))
        print("\nExemplo de linha:\n", dataframe.iloc[0].to_dict())
    else:
        build_meta_from_hf(dataframe)