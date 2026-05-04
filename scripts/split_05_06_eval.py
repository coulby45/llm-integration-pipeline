# -*- coding: utf-8 -*-
"""Split 05_hybrid_eval.ipynb into hybrid-only 05 + 06_evaluation.ipynb."""
import json
from copy import deepcopy

ROOT = r"C:/Users/Utilisateur/Desktop/Idee_random/Article_scientifique"
P05 = f"{ROOT}/05_hybrid_eval.ipynb"
P06 = f"{ROOT}/06_evaluation.ipynb"

HYBRID_END = 19  # cells 0..18 = hybrid pipeline; 19 = "## 7. Évaluation"

CELL_05_TITLE = [
    "# Notebook 05 — Méthode hybride (LoRA + FAISS)\n",
    "\n",
    "**Objectif** : charger le modèle fine-tuné et l’index FAISS, exécuter l’inférence **hybride** sur le jeu de test, et sauvegarder `results/hybrid_predictions.json`.\n",
    "\n",
    "**Évaluation des 4 méthodes** (métriques, figures, `final_report.json`) : voir **`06_evaluation.ipynb`**.\n",
    "\n",
    "| # | Méthode       | Description |\n",
    "|---|---------------|-------------|\n",
    "| 4 | **Hybride**   | LLaMA 3.1 8B + LoRA + top-5 chunks FAISS (docs bruts) |\n",
    "\n",
    "**Corpus FAISS** : documents bruts chunkés (400 mots, overlap 100) — aligné sur `03_baseline_rag.ipynb`.\n",
    "\n",
    "> **GPU recommandé** — Aller dans `Exécution > Modifier le type d'exécution > GPU`\n",
    "\n",
    "**Prérequis sur Drive** :\n",
    "- `BASE_PATH/models/lora_adapter/` (sortie de `04_finetuning.ipynb`)\n",
    "- `BASE_PATH/models/faiss_index/` (sortie de `03_baseline_rag.ipynb`)\n",
    "- `BASE_PATH/data/processed/test.json`\n",
    "\n",
    "**Sortie** : `BASE_PATH/results/hybrid_predictions.json`\n",
]

POINTER_MD = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Suite : évaluation comparative\n",
        "\n",
        "Exécutez **`06_evaluation.ipynb`** pour charger les 4 fichiers de prédictions, calculer EM / F1 / BERTScore / ROUGE-L / hallucination, produire les figures et `final_report.json`.\n",
        "\n",
        "**Ordre conseillé du pipeline :** `03` → `04` → **`05` (hybride)** → **`06` (évaluation)**.\n",
    ],
    "id": "cell-md-pointer-06",
}

CELL_06_TITLE = [
    "# Notebook 06 — Évaluation comparative (4 méthodes)\n",
    "\n",
    "**Prérequis** (sur Drive, `BASE_PATH`) :\n",
    "- `data/processed/test.json`\n",
    "- `results/baseline_predictions.json`, `rag_predictions.json`, `finetuned_predictions.json`, **`hybrid_predictions.json`** (produit par `05_hybrid_eval.ipynb`)\n",
    "\n",
    "**Sorties** : `results/final_report.json`, PNG dans `results/plots/`.\n",
    "\n",
    "Peut tourner **sans GPU** (agrégation JSON + métriques). Les installs sont plus légères que pour `05` (pas d’Unsloth).\n",
]

IMPORTS_06 = """# Imports évaluation (sans Unsloth / LoRA / FAISS)
import os, json, time, string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter, defaultdict
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer as _rouge_scorer
from tqdm.notebook import tqdm

PROCESSED_PATH = os.path.join(BASE_PATH, 'data', 'processed')
MODELS_PATH    = os.path.join(BASE_PATH, 'models', 'lora_adapter')
FAISS_PATH     = os.path.join(BASE_PATH, 'models', 'faiss_index')
RESULTS_PATH   = os.path.join(BASE_PATH, 'results')
PLOTS_PATH     = os.path.join(BASE_PATH, 'results', 'plots')

for path in [RESULTS_PATH, PLOTS_PATH]:
    os.makedirs(path, exist_ok=True)

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K       = 5
MAX_SEQ_LEN = 2048

_ROUGE = _rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

METHODS_ORDER    = ["Baseline", "RAG", "Fine-tuné", "Hybride"]
STRATA_ORDER     = ["récent", "intermédiaire", "fondamental"]
QTYPES_ORDER     = ["factuel", "synthese", "comprehension"]
METHOD_COLORS    = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
STRATA_COLORS    = ['#e74c3c', '#f39c12', '#2ecc71']
QTYPE_COLORS     = ['#3498db', '#9b59b6', '#1abc9c']

BS_THRESHOLDS = [0.80, 0.85, 0.90]
BS_THRESHOLD  = 0.85

print("Configuration évaluation chargée (notebook 06).")
print(f"  Seuils accuracy : {[int(t*100) for t in BS_THRESHOLDS]}%  (principal : {int(BS_THRESHOLD*100)}%)")
print(f"  Résultats    : {RESULTS_PATH}")
"""

LOAD_DATA_06 = """# Chargement des fichiers JSON depuis Drive (inclut les prédictions hybrides)
def load_json(path, label=""):
    \"\"\"Charge un fichier JSON avec gestion d'erreur.\"\"\"
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  [OK] {label or path} ({len(data)} entrées)")
        return data
    except FileNotFoundError:
        print(f"  [MANQUANT] {path}")
        return []
    except json.JSONDecodeError as e:
        print(f"  [ERROR JSON] {path} : {e}")
        return []

print("Chargement des fichiers...")
test_data             = load_json(os.path.join(PROCESSED_PATH,  'test.json'),                   'test.json')
baseline_predictions  = load_json(os.path.join(RESULTS_PATH,    'baseline_predictions.json'),   'baseline_predictions.json')
rag_predictions       = load_json(os.path.join(RESULTS_PATH,    'rag_predictions.json'),        'rag_predictions.json')
finetuned_predictions = load_json(os.path.join(RESULTS_PATH,    'finetuned_predictions.json'),  'finetuned_predictions.json')
hybrid_predictions    = load_json(os.path.join(RESULTS_PATH,    'hybrid_predictions.json'),    'hybrid_predictions.json')

print(f"\\nTest set : {len(test_data)} questions")
"""


def source_to_lines(s: str):
    lines = s.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def main():
    with open(P05, encoding="utf-8") as f:
        nb05 = json.load(f)

    cells = nb05["cells"]

    # --- Trimmed 05 (hybrid only)
    hybrid_cells = cells[:HYBRID_END]
    hybrid_cells[0]["source"] = CELL_05_TITLE
    hybrid_cells.append(deepcopy(POINTER_MD))

    nb05_out = deepcopy(nb05)
    nb05_out["cells"] = hybrid_cells
    nb05_out.setdefault("metadata", {}).setdefault("colab", {})["name"] = "05_hybrid_eval.ipynb"

    with open(P05, "w", encoding="utf-8") as f:
        json.dump(nb05_out, f, ensure_ascii=False, indent=2)

    # --- 06 = intro + drive + pip (cell 7) + trimmed imports + load + eval tail
    eval_cells = []

    eval_cells.append({"cell_type": "markdown", "metadata": {}, "source": CELL_06_TITLE, "id": "cell06-title"})
    eval_cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## 0. Montage Google Drive"], "id": "cell06-md-drive"})
    eval_cells.append(deepcopy(cells[4]))

    eval_cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## 1. Dépendances (métriques & plots)"], "id": "cell06-md-deps"})
    eval_cells.append(deepcopy(cells[7]))

    eval_cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## 2. Imports et chemins"], "id": "cell06-md-imports"})
    eval_cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": source_to_lines(IMPORTS_06),
        "execution_count": None,
        "outputs": [],
        "id": "cell06-imports",
    })

    eval_cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## 3. Chargement test.json et des 4 prédictions"], "id": "cell06-md-load"})
    eval_cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": source_to_lines(LOAD_DATA_06),
        "execution_count": None,
        "outputs": [],
        "id": "cell06-load",
    })

    # Section header "## 7." -> "## 4." for evaluation block
    md_eval = deepcopy(cells[19])
    md_eval["source"] = ["## 4. Évaluation des 4 méthodes\n"]
    md_eval["id"] = "cell06-md-eval"
    eval_cells.append(md_eval)

    for c in cells[20:]:
        cc = deepcopy(c)
        # Summary: extend file list + notebook labels
        src = "".join(cc.get("source", []))
        if cc.get("id") == "cell-summary" or "RÉSUMÉ FINAL" in src:
            src = src.replace(
                '("05_hybrid_eval",     os.path.join(BASE_PATH, \'results\', \'hybrid_predictions.json\')),\n'
                '    ("05_hybrid_eval",     report_path),\n'
                '] + [("05_hybrid_eval", p) for p in [p1, p2, p3, p4, p5, p6]]',
                '("05_hybrid_eval",     os.path.join(BASE_PATH, \'results\', \'hybrid_predictions.json\')),\n'
                '    ("06_evaluation",    report_path),\n'
                '] + [("06_evaluation", p) for p in [p1, p2, p3, p4, p5, p6, p7, p8, p9]]',
            )
            src = src.replace(
                "# Récapitulatif complet de tous les fichiers produits sur Drive",
                "# Récapitulatif des artefacts d'évaluation (notebook 06)",
            )
            cc["source"] = source_to_lines(src) if "\n" in src else [src]

        # Markdown "## 8. Visualisations" -> "## 5."
        if cc.get("id") == "cell-md-plots":
            cc["source"] = ["## 5. Visualisations\n"]
        if cc.get("id") == "cell-md-crosstable":
            cc["source"] = ["## 5b. Tableau croisé méthodes × dataset_type\n"]
        if cc.get("id") == "cell-md-summary":
            cc["source"] = ["## 6. Résumé final\n"]

        eval_cells.append(cc)

    nb06 = {
        "nbformat": nb05["nbformat"],
        "nbformat_minor": nb05["nbformat_minor"],
        "metadata": deepcopy(nb05["metadata"]),
        "cells": eval_cells,
    }
    nb06.setdefault("metadata", {}).setdefault("colab", {})["name"] = "06_evaluation.ipynb"
    nb06["metadata"]["accelerator"] = "GPU"  # optional for Colab; metrics run on CPU

    with open(P06, "w", encoding="utf-8") as f:
        json.dump(nb06, f, ensure_ascii=False, indent=2)

    print("Wrote trimmed", P05, "cells:", len(hybrid_cells))
    print("Wrote", P06, "cells:", len(eval_cells))


if __name__ == "__main__":
    main()
