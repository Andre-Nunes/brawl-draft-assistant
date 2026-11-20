import json
import os
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv

from src.models.schemas import PlayerBrawler

load_dotenv()

BASE_URL = "https://api.brawlstars.com/v1"
CACHE_PATH = Path("data/raw/my_brawlers.json")


class SupercellClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("BRAWL_API_KEY")
        self.player_tag = os.getenv("PLAYER_TAG")
        if not self.api_key or not self.player_tag:
            raise ValueError("Define BRAWL_API_KEY e PLAYER_TAG no ficheiro .env")

        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def fetch_my_brawlers(self, save_cache: bool = True) -> list[PlayerBrawler]:
        encoded_tag = quote(self.player_tag, safe="")
        url = f"{BASE_URL}/players/{encoded_tag}"

        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        raw_brawlers = response.json().get("brawlers", [])

        brawlers = [
            PlayerBrawler(
                id=b["id"],
                name=b["name"].upper(),
                power=b["power"],
                trophies=b["trophies"],
                has_star_power=len(b.get("starPowers", [])) > 0,
                has_gadget=len(b.get("gadgets", [])) > 0,
            )
            for b in raw_brawlers
        ]

        if save_cache:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_text(
                json.dumps([b.model_dump() for b in brawlers], indent=2),
                encoding="utf-8",
            )

        return brawlers


if __name__ == "__main__":
    client = SupercellClient()
    my_brawlers = client.fetch_my_brawlers()
    ready = [b.name for b in my_brawlers if b.is_ranked_ready]
    print(f"Total: {len(my_brawlers)} brawlers | Nível 11 (Ranked ready): {len(ready)}")
    print("Prontos:", ", ".join(ready[:10]), "...")