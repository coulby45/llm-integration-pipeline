## LLM Integration Study (FR)

Pipeline de recherche appliquee pour comparer plusieurs strategies d'integration de connaissances recentes dans un LLM (baseline, RAG, fine-tuning, FT+RAG, RAFT, reranking, function calling) sur un corpus 100% francophone.

## Objectif

Mesurer, dans un protocole reproductible, le compromis entre:
- qualite des reponses,
- fidelite au contexte,
- hallucination,
- latence et cout.

Le projet est organise autour de notebooks executes en sequence et d'une aggregation finale dans `09_evaluation.ipynb`.

## Methodes comparees

- Baseline (API, sans contexte)
- RAG dense (FAISS + API)
- Fine-tuning LoRA (local)
- FT+RAG (local + retrieval)
- RAFT (local)
- RAG + reranking (API)
- Function calling (API + outil `search_docs`)

> Note: le benchmark actuel combine des methodes API et locales. Pour une comparaison strictement equilibree, utiliser un backend d'inference unique.

## Structure du projet

```text
Article_scientifique/
├── 01_scraping.ipynb
├── 02_dataset_builder.ipynb
├── 02b_dataset_builder_async.ipynb
├── 03_baseline_rag.ipynb
├── 04_finetuning.ipynb
├── 05_raft.ipynb
├── 06_rag_rerank.ipynb
├── 07_function_calling.ipynb
├── 08_ft_plus_rag.ipynb
├── 09_evaluation.ipynb
├── 10_llm_as_judge.ipynb
├── 11_llm_judge_analysis.ipynb
├── data/
├── results/
├── scripts/
├── article_scientifique.tex
└── README.md
```

## Prerequis

- Python 3.10+
- GPU recommande pour les notebooks locaux (`04`, `05`, `08`)
- Cle API Groq pour les notebooks API (`03`, `06`, `07`)
- Optionnel: cle Anthropic pour `10_llm_as_judge.ipynb`

## Installation

Clone du depot:

```bash
git clone https://github.com/coulby45/llm-integration-pipeline.git
cd llm-integration-pipeline
```

Si vous utilisez un environnement local:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install jupyter notebook
```

Les dependances principales sont aussi installees dans les notebooks (cellules `!pip install ...`).

## Pipeline d'execution (ordre recommande)

1. `01_scraping.ipynb`  
   Collecte des sources brutes (Wikipedia, HAL, presse, code de la route).

2. `02_dataset_builder.ipynb` (ou `02b_dataset_builder_async.ipynb`)  
   Generation des paires Q/R, split train/test, metadonnees.

3. `03_baseline_rag.ipynb`  
   Baseline API + RAG API + index FAISS.

4. `04_finetuning.ipynb`  
   Fine-tuning LoRA (local).

5. `05_raft.ipynb`  
   Variante RAFT (local).

6. `06_rag_rerank.ipynb`  
   RAG + reranking (API).

7. `07_function_calling.ipynb`  
   Function calling (API).

8. `08_ft_plus_rag.ipynb`  
   Inference FT+RAG (local).

9. `09_evaluation.ipynb`  
   Aggregation de toutes les predictions, metriques, figures, export `final_report.json`.

10. `10_llm_as_judge.ipynb` puis `11_llm_judge_analysis.ipynb` (optionnel)  
    Evaluation subjective complementaire.

## Fichiers de sortie importants

Dans `results/`:
- `baseline_predictions.json`
- `rag_predictions.json`
- `finetuned_predictions.json`
- `ft_rag_predictions.json`
- `raft_predictions.json`
- `rerank_predictions.json`
- `function_calling_predictions.json`
- `final_report.json`

Et dans `results/plots/`:
- figures exportees par `09_evaluation.ipynb` (comparaisons globales, par domaine, latence, impact eco, etc.).

## Metriques suivies

- F1 token-level
- BERTScore
- ROUGE-L
- METEOR
- Fidelite au contexte (faithfulness)
- Proxy hallucination
- Accuracy@BERTScore (80/85/90)
- Latence moyenne
- Intervalles de confiance bootstrap (CI 95%)

## Reproductibilite

- Le rapport final est versionne dans `results/final_report.json`.
- Les notebooks fixent un maximum de parametres (seeds, chemins, seuils).
- Pour comparer des runs, conserver une copie des JSON de prediction et du `final_report.json`.

## Resultats (snapshot)

Sur la campagne reference (voir `results/final_report.json`):
- `Rerank` et `RAG` dominent les metriques globales.
- Les methodes locales fine-tunees sont plus couteuses en latence dans la configuration actuelle.

> Toujours relancer `09_evaluation.ipynb` avant de communiquer un tableau de resultats.

## Bonnes pratiques

- Ne jamais committer `.env` ou des cles API.
- Nommer chaque campagne (date + config) si vous comparez plusieurs runs.
- Eviter de modifier l'ordre des notebooks sans mettre a jour ce README.

## Limites connues

- Benchmark mixte API/local: potentielle source de biais de comparaison.
- Metriques automatiques utiles mais non suffisantes (d'ou l'option LLM-as-judge).
- Sensibilite aux prompts et aux hyperparametres.

## Feuille de route (roadmap)

- [ ] Benchmark API-only strict pour toutes les methodes comparables
- [ ] Ablation systematique retriever/reranker
- [ ] Evaluation humaine experte sur sous-ensemble
- [ ] Packaging script CLI (hors notebooks)

## Citation

Si vous reutilisez ce travail:

```bibtex
@misc{coulibaly2026llm_integration,
  title  = {Integration de nouvelles informations dans les LLMs},
  author = {Coulibaly, Abdoul Karim},
  year   = {2026},
  note   = {Projet CESI A4 Data Science et IA}
}
```

## Licence

Ajouter ici la licence retenue pour le depot (par ex. MIT).
