# Brawl Draft Assistant

Assistente local de **draft para Brawl Stars Ranked** alimentado por um modelo de **Machine Learning (PyTorch)**, dados de mais de 500 mil partidas reais (Hugging Face), Tier List de pro players e integração direta com a **API oficial da Supercell** para filtrar apenas os brawlers que tens prontos a Nível 11.

---

## Funcionalidades Principais

* **Filtro pela tua conta (API Supercell):** Importa automaticamente os teus brawlers e foca as recomendações nos que estão prontos para Ranked (Nível 11).
* **Motor Híbrido (ML + Heurísticas):** Combina a probabilidade de vitória prevista por uma rede neuronal (*DeepSets*) com winrate no mapa, Tier List de pro players e perfis selecionáveis (`Solo Queue` vs. `Pro Meta`).
* **Risco Dinâmico de First Pick:** Penaliza brawlers vulneráveis a *hard counters* ainda livres e viáveis no mapa, escalando a penalização consoante o número de picks restantes da equipa inimiga.
* **Sugestão Personalizada de Bans:** Prioriza brawlers dominantes no mapa, ameaças de counter e atribui bónus a picks fortes que não tens a Nível 11.
* **Interface Web Rápida (FastAPI + Tailwind):** Otimizada para usar durante o tempo limite do draft com pesquisa por teclado (`Enter`, `Alt+1/2/3`), ícones via CDN, seleção de aliados/inimigos em 1 clique e botão de sincronização de conta.

---

## Arquitetura & Pipeline de Dados

```text
API Supercell (Perfil) ──► data/raw/my_brawlers.json ──┐
Hugging Face (Parquet) ──► map_stats + matchups.json ──┼──► DraftScorer (Híbrido) ──► FastAPI ──► Web UI
Pro Tier List (Manual) ──► data/tier_list.json       ──┤         ▲
Hugging Face (Parquet) ──► PyTorch DraftNet (.pt)    ──┘         │
                                                         Inferência < 2ms
```

### Modelo de Machine Learning (`src/ml/`)

* **Arquitetura:** `DraftNet` — rede *DeepSets* invariante à ordem dos picks com *Embeddings* de Brawlers (32d) e Mapas (16d), combinando os vetores das equipas por diferença (`v1 - v2`) e interação elemento-a-elemento (`v1 * v2`).
* **Data Augmentation:** Simetria de equipas (`T1 ↔ T2`) e *masking* parcial de slots (25% do batch) para suportar previsões em drafts incompletos (1v0, 2v1, 2v2).
* **Dataset:** `EliF77/brawlstars-ranked` (Season 56, ~509k partidas filtradas por `skill_ns_ok == 1` e `skill_ns >= 0.0`).
* **Métrica em Validação:** **0.6501 ROC-AUC**.

---

## Estrutura do Projeto

```text
brawl-draft-assistant/
├── data/
│   ├── raw/                 # Cache local da conta e partidas (ignorado pelo Git)
│   ├── draft_model.pt       # Checkpoint treinado do DraftNet (PyTorch)
│   ├── map_stats.json       # Winrates por mapa com suavização bayesiana
│   ├── matchups.json        # Matriz de counters e sinergias
│   └── tier_list.json       # Tier list de pro players (S a F)
├── src/
│   ├── api/
│   │   ├── app.py           # Servidor FastAPI e endpoints REST
│   │   ├── supercell.py     # Cliente oficial da API Brawl Stars
│   │   ├── sync_hf.py       # Ingestão de dados Ranked do Hugging Face
│   │   └── sync_meta.py     # Recolha alternativa via battle logs do Top Global
│   ├── engine/
│   │   └── scorer.py        # Motor híbrido, risco de first pick e bans
│   ├── ml/
│   │   ├── model.py         # Arquitetura DraftNet (PyTorch)
│   │   ├── train.py         # Script de treino e validação
│   │   └── predictor.py     # Inferência em batch na CPU
│   └── models/
│       └── schemas.py       # Modelos Pydantic
├── static/
│   └── index.html           # Interface web interativa
├── .env.example
└── requirements.txt
```

---

## Como Executar

### 1. Clonar e configurar ambiente

```bash
git clone https://github.com/Andre-Nunes/brawl-draft-assistant.git
cd brawl-draft-assistant

python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Linux / macOS:        source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente

Cria uma API Key em [developer.brawlstars.com](https://developer.brawlstars.com/) associada ao teu IP público e cria um ficheiro `.env` na raiz baseado no `.env.example`:

```env
BRAWL_API_KEY="tua_api_key_aqui"
PLAYER_TAG="#O_TEU_TAG"
```

### 3. Sincronizar conta e arrancar a aplicação

```bash
# Importar os teus brawlers a Nível 11
python -m src.api.supercell

# Iniciar servidor local
uvicorn src.api.app:app --reload
```

Abre **http://127.0.0.1:8000** no browser (a documentação interativa da API fica disponível em `/docs`).

---

## Atalhos da Interface

| Atalho | Ação |
| :--- | :--- |
| `Alt + 1` | Selecionar destino **Bans** |
| `Alt + 2` | Selecionar destino **Minha Equipa** |
| `Alt + 3` | Selecionar destino **Equipa Inimiga** |
| `Enter` | Adicionar o primeiro brawler da pesquisa |
| `Esc` | Limpar caixa de pesquisa |

---

## Atualizar Meta e Retreinar o Modelo (Nova Season / Patch)

Sempre que sair um novo *balance change* ou rotação de mapas Ranked:

```bash
# 1. Atualizar estatísticas de mapas e matchups com o Parquet mais recente
python -m src.api.sync_hf --files 1

# 2. Retreinar o modelo PyTorch
python -m src.ml.train
```

Atualiza também `data/tier_list.json` caso o meta competitivo mude significativamente.

---

## Stack Tecnológica

* **Backend & API:** Python, FastAPI, Uvicorn, Pydantic
* **Machine Learning & Dados:** PyTorch, Pandas, PyArrow, Scikit-learn, Hugging Face Hub
* **Frontend:** HTML5, Vanilla JS, Tailwind CSS, Brawlify CDN