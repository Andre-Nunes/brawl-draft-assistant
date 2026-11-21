# Brawl Draft Assistant

Assistente local de **draft para Brawl Stars Ranked** alimentado por um modelo de **Machine Learning (PyTorch)**, dados de mais de 500 mil partidas reais (Hugging Face), Tier List de pro players e integração direta com a **API oficial da Supercell** para filtrar apenas os brawlers que tens a Nível 11.

---

## Funcionalidades Principais

- **Filtro pela tua conta (API Supercell):** Importa automaticamente os teus brawlers e foca as recomendações nos que estão prontos para Ranked (Nível 11).
- **Motor Híbrido (ML + Heurísticas):** Combina a probabilidade de vitória prevista por uma rede neuronal (*DeepSets*) com winrate no mapa, Tier List de pro players e perfis selecionáveis (`Solo Queue` vs. `Pro Meta`).
- **Risco Dinâmico de First Pick:** Penaliza brawlers vulneráveis a *hard counters* ainda livres e viáveis no mapa, escalando a penalização consoante o número de picks restantes do inimigo.
- **Sugestão Personalizada de Bans:** Prioriza brawlers dominantes no mapa, ameaças de counter e dá bónus a picks fortes que não tens a Nível 11.
- **Interface Web Rápida (FastAPI + Tailwind):** Otimizada para usar durante o tempo limite do draft com pesquisa por teclado (`Enter`, `Alt+1/2/3`), ícones via CDN, seleção em 1 clique e botão de sincronização de conta.

---

## Arquitetura & Pipeline de Dados

```text
API Supercell (Perfil) ──► data/raw/my_brawlers.json ──┐
Hugging Face (Parquet) ──► map_stats + matchups.json ──┼──► DraftScorer (Híbrido) ──► FastAPI ──► Web UI
Pro Tier List (Manual) ──► data/tier_list.json       ──┤         ▲
Hugging Face (Parquet) ──► PyTorch DraftNet (.pt)    ──┘         │
                                                         Inferência < 2ms