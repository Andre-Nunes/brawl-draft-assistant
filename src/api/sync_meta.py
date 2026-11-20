import json
import os
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.brawlstars.com/v1"
DATA_DIR = Path("data")
MATCHES_CACHE = DATA_DIR / "raw" / "ranked_matches.json"


class MetaSyncer:
    def __init__(self) -> None:
        api_key = os.getenv("BRAWL_API_KEY")
        if not api_key:
            raise ValueError("Define BRAWL_API_KEY no ficheiro .env")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _get(self, endpoint: str) -> dict | None:
        resp = self.session.get(f"{BASE_URL}{endpoint}", timeout=10)
        if resp.status_code != 200:
            return None
        return resp.json()

    def fetch_top_player_tags(self, limit: int = 100) -> list[str]:
        data = self._get("/rankings/global/players")
        if not data:
            return []
        return [p["tag"] for p in data.get("items", [])[:limit]]

    def collect_ranked_matches(self, player_limit: int = 100) -> dict[str, dict]:
        # Carrega jogos anteriores para ir acumulando histórico do patch atual
        matches: dict[str, dict] = {}
        if MATCHES_CACHE.exists():
            matches = json.loads(MATCHES_CACHE.read_text(encoding="utf-8"))

        tags = self.fetch_top_player_tags(limit=player_limit)
        print(f"A recolher battle logs de {len(tags)} jogadores do Top Global...")

        new_count = 0
        for idx, tag in enumerate(tags, start=1):
            encoded = quote(tag, safe="")
            log = self._get(f"/players/{encoded}/battlelog")
            if not log:
                continue

            for item in log.get("items", []):
                battle = item.get("battle", {})
                event = item.get("event", {})
                b_type = battle.get("type")
                result = battle.get("result")
                teams = battle.get("teams", [])
                map_name = event.get("map")

                if (
                    b_type not in {"soloRanked", "teamRanked"}
                    or result not in {"victory", "defeat"}
                    or len(teams) != 2
                    or not map_name
                ):
                    continue

                # Identifica qual das equipas pertence ao jogador consultado
                t0_tags = {p["tag"].upper() for p in teams[0]}
                t1_tags = {p["tag"].upper() for p in teams[1]}
                tag_upper = tag.upper()

                if tag_upper in t0_tags:
                    my_team, enemy_team = teams[0], teams[1]
                elif tag_upper in t1_tags:
                    my_team, enemy_team = teams[1], teams[0]
                else:
                    continue

                winners = my_team if result == "victory" else enemy_team
                losers = enemy_team if result == "victory" else my_team

                all_tags = sorted(t0_tags | t1_tags)
                match_id = f"{item.get('battleTime')}|{map_name.upper()}|{','.join(all_tags)}"

                if match_id not in matches:
                    matches[match_id] = {
                        "map": map_name.upper(),
                        "mode": event.get("mode", "unknown"),
                        "winners": [p["brawler"]["name"].upper() for p in winners],
                        "losers": [p["brawler"]["name"].upper() for p in losers],
                    }
                    new_count += 1

            if idx % 25 == 0:
                print(f"  Progresso: {idx}/{len(tags)} jogadores | +{new_count} novos jogos Ranked")
            time.sleep(0.05)  # Respeita rate limits da API

        MATCHES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        MATCHES_CACHE.write_text(json.dumps(matches, indent=2), encoding="utf-8")
        print(f"Total acumulado: {len(matches)} partidas Ranked (+{new_count} novas).")
        return matches

    @staticmethod
    def build_meta_files(
        matches: dict[str, dict],
        min_map_games: int = 3,
        min_pair_games: int = 3,
    ) -> None:
        map_counts: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(lambda: [0, 0])
        )
        vs_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        syn_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])

        for m in matches.values():
            map_name = m["map"]
            winners = m["winners"]
            losers = m["losers"]

            # 1. Contagem por mapa
            for b in winners:
                map_counts[map_name][b][0] += 1
                map_counts[map_name][b][1] += 1
            for b in losers:
                map_counts[map_name][b][1] += 1

            # 2. Counters (vencedor vs derrotado)
            for w in winners:
                for l in losers:
                    if w != l:
                        vs_counts[(w, l)][0] += 1
                        vs_counts[(w, l)][1] += 1
                        vs_counts[(l, w)][1] += 1

            # 3. Sinergias (pares na mesma equipa)
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

        # Suavização bayesiana: (wins + 2.5) / (total + 5) evita extremos com amostras pequenas
        map_stats: dict[str, dict[str, float]] = {}
        for map_name, brawlers in sorted(map_counts.items()):
            stats = {
                b: round((w + 2.5) / (t + 5.0), 3)
                for b, (w, t) in brawlers.items()
                if t >= min_map_games
            }
            if stats:
                map_stats[map_name] = dict(
                    sorted(stats.items(), key=lambda x: x[1], reverse=True)
                )

        counters: dict[str, list[str]] = defaultdict(list)
        for (b, opp), (w, t) in vs_counts.items():
            if t >= min_pair_games and (w / t) >= 0.58:
                counters[b].append(opp)

        synergies: dict[str, list[str]] = defaultdict(list)
        for (a, b), (w, t) in syn_counts.items():
            if t >= min_pair_games and (w / t) >= 0.56:
                synergies[a].append(b)

        (DATA_DIR / "map_stats.json").write_text(
            json.dumps(map_stats, indent=2), encoding="utf-8"
        )
        (DATA_DIR / "matchups.json").write_text(
            json.dumps(
                {"counters": dict(counters), "synergies": dict(synergies)}, indent=2
            ),
            encoding="utf-8",
        )
        print(
            f"Gerado: {len(map_stats)} mapas em map_stats.json | "
            f"{len(counters)} brawlers com counters | {len(synergies)} com sinergias."
        )


if __name__ == "__main__":
    syncer = MetaSyncer()
    all_matches = syncer.collect_ranked_matches(player_limit=100)
    syncer.build_meta_files(all_matches)