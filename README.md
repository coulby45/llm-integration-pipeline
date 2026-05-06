# Intégration de nouvelles informations dans les LLMs
### Guide de rédaction — Article scientifique

---

## Structure de l'article (plan recommandé)

```
1. Titre & Résumé (Abstract)
2. Introduction
3. État de l'art (Related Work)
4. Méthodologie
5. Expériences et résultats
6. Discussion
7. Conclusion
8. Références
```

## Mode rédaction rapide

- Rédiger section par section dans l’ordre du document (2 → 8) puis revenir sur le titre.
- Extraire d’abord les chiffres depuis `results/final_report.json`, puis écrire l’interprétation.
- Garder chaque sous-section sur un format court : **résultat**, **interprétation**, **limite**.
- Mettre à jour les tableaux et figures issus de `09_evaluation.ipynb` ; pour **FT+RAG**, vérifier aussi la présence de `ft_rag_predictions.json` (notebook **`08_ft_plus_rag.ipynb`**).

---

## 1. Titre

Proposition principale :

> **Intégration de nouvelles informations dans les LLMs : baseline, RAG, fine-tuning LoRA, FT+RAG, RAFT, reranking et function calling sur corpus français hétérogène**

Variantes :
- *Mettre à jour les LLMs sans ré-entraînement : RAG, reranking, function calling vs fine-tuning / RAFT sur données techniques, scientifiques et d'actualité*
- *Connaissances figées vs connaissances dynamiques : sept stratégies d'intégration dans LLaMA 3.1 8B (baseline, RAG, LoRA, **FT+RAG**, RAFT, rerank, function calling)*

---

## 2. Résumé (Abstract) — ~150–200 mots

**Structure** : Contexte → Problème → Méthode → Résultats → Conclusion

> Les LLMs ont une date de coupure (décembre 2023 pour LLaMA 3.1 8B). Nous comparons **sept** stratégies d’intégration d’information sur un socle **LLaMA 3.1 8B** : **baseline** (Groq sans contexte), **RAG** (FAISS + Groq), **fine-tuning LoRA** (local), **FT+RAG** (adaptateur `lora_adapter_ft` + même index FAISS que le RAG — notebook **`08_ft_plus_rag.ipynb`**), **RAFT** (pseudo-RAG au train + LoRA `lora_adapter_raft`), **RAG + reranking** (cross-encoder multilingue + Groq), et **function calling** (outil `search_docs` + Groq). Corpus **100 % français** (Wikipedia, HAL, actualités, code de la route). Une **notation subjective complémentaire** (LLM-as-judge 1–5 : cohérence, utilité, fidélité) est disponible via **`10_llm_as_judge.ipynb`** puis **`11_llm_judge_analysis.ipynb`**. L’agrégation **objective** (F1, BERTScore, ROUGE-L, **METEOR**, fidélité au contexte, **IC bootstrap 95 %**, hallucination, latence, confiance ; impact écologique et **Fig. 10**) se fait dans **`09_evaluation.ipynb`** après génération des `*_predictions.json` (notebooks `03`→`08`, incl. **`08_ft_plus_rag.ipynb`** si FT+RAG).

---

## 3. Introduction — ~400–600 mots

### 3.1 Contexte et motivation
- Les LLMs (GPT-4, Mistral, LLaMA…) ont une **date de coupure** : ignorent les événements postérieurs à leur entraînement
- LLaMA 3.1 8B : coupure décembre 2023 → ignore tous les articles de presse d'avril 2026 et les papiers Arxiv post-2024
- Problème pratique : un LLM ne connaît pas les dernières publications IA, les nouvelles lois, les articles de presse récents
- Question centrale : **comment intégrer efficacement de nouvelles connaissances dans un LLM sans le ré-entraîner entièrement ?**
- Plusieurs angles : contexte à l'inférence (RAG, reranking, outils), adaptation des poids (LoRA, RAFT), ou les combiner selon les contraintes (latence, coût GPU, fraîcheur des documents)

### 3.2 Verrou scientifique
- Les comparaisons existantes utilisent souvent un modèle différent par méthode → biais de comparaison
- La plupart des benchmarks sont en anglais → peu de résultats sur du français
- Les évaluations ignorent souvent la **nature du domaine** (technique vs actualités) et la **complexité des questions**

### 3.3 Contributions de l'article
1. Construction d'un **corpus 100 % français** multi-source (Wikipedia, HAL, actualités, code de la route) avec types de questions variés
2. **Protocole dataset** : génération Q&R via **Groq** (LLaMA 3.1 8B) ; **non indépendant** de l’évaluation → limite méthodologique assumée ou documentée
3. **Base commune** : toutes les méthodes s’appuient sur la famille **LLaMA 3.1 8B** (Groq API et/ou base quantifiée locale + adaptateurs LoRA) → comparaison équitable à discuter (API vs local)
4. **Évaluation multi-axe** : F1, BERTScore (CI Bootstrap), ROUGE-L, METEOR, **fidélité** (ROUGE-L prédit/contexte), hallucination, latence, écologie (Fig. 10)
5. **Analyse croisée** méthodes × domaines + comparaison simple vs multi-sauts (HAL)
6. Résultats empiriques : comparaison **sept méthodes** (baseline, RAG, LoRA, **FT+RAG**, RAFT, rerank, function calling) avec discussion **RAG vs rerank**, **FT+RAG vs RAG**, et **coût Groq vs GPU** (LoRA / RAFT / FT+RAG)
7. Pistes : ablations retrieveur, évaluation des **appels d’outils** (précision JSON), évaluation humaine
8. Recommandations pratiques pour projets IA en français avec contraintes de coût/latence/empreinte carbone

### 3.4 Plan de l'article
> La section 2 présente l'état de l'art. La section 3 décrit notre méthodologie et la constitution du corpus (**quatre** types de sources : technique, scientifique HAL, presse, **juridique / code de la route**). Les résultats sont présentés en section 4 et discutés en section 5.

---

## 4. État de l'art — ~700–1000 mots

### 4.1 Modèles de langage de grande taille
- Transformer (Vaswani et al., 2017)
- GPT-2/3 (Radford et al., 2019 ; Brown et al., 2020), BERT (Devlin et al., 2019)
- LLaMA 2/3 (Touvron et al., 2023 ; Meta AI, 2024), Mistral 7B (Jiang et al., 2023)
- **Notre modèle** : LLaMA 3.1 8B — architecture optimisée, context window 128k, date de coupure décembre 2023

### 4.2 Problème de la mise à jour des connaissances
- Knowledge cutoff et hallucination (Ji et al., 2023)
- Catastrophic forgetting lors du fine-tuning (McCloskey & Cohen, 1989 ; Kirkpatrick et al., 2017)
- Edit de connaissances (Meng et al., 2022 — ROME ; Mitchell et al., 2022 — SERAC)

### 4.3 Génération augmentée par récupération (RAG)
- Lewis et al. (2020) — article fondateur du RAG
- FAISS (Johnson et al., 2019) — indexation vectorielle efficace
- Sentence-Transformers (Reimers & Gurevych, 2019) — embeddings sémantiques
- **Notre choix** : `paraphrase-multilingual-MiniLM-L12-v2` pour le support FR/EN
- **Corpus indexé** : documents bruts chunkés (200 mots, overlap 50) depuis `wikipedia_technique.json` + `hal.json` + `lemonde.json` + **`code_route.json`** (segments issus du PDF *code de la route*, `dataset_type` **juridique**) — meilleur que les extraits courts
- Limitations : qualité du retriever, pertinence des chunks, latence ajoutée

### 4.4 Fine-tuning efficace (PEFT/LoRA)
- Hu et al. (2022) — LoRA : Low-Rank Adaptation
- Dettmers et al. (2023) — QLoRA : quantification 4-bit + LoRA
- **Notre configuration** : LoRA rank=32, alpha=32, dropout 0.05, 7 modules (attention + MLP), max_seq_len=1024, 5 époques, lr 1e-4 (voir §5.5)
- Format Alpaca (Stanford, 2023) pour le fine-tuning supervisé

### 4.5 Reranking, RAFT et agents à outils
- Reranking cross-encoder (Nogueira et Cho, 2019 ; modèles multilingues type ms-marco) pour affiner le top-$k$ après récupération dense
- RAFT (Hong et al., 2024) : entraînement avec contexte utile + distracteurs pour mieux exploiter un contexte injecté
- Function calling / tool use : le LLM décide quand interroger une base documentaire (recherche simulée côté client dans notre pipeline)

### 4.6 Évaluation des LLMs
- F1 token-level (Rajpurkar et al., 2016) ; fidélité au passage source (ROUGE-L vs `context` du jeu de test)
- BERTScore (Zhang et al., 2020) — `distilbert-base-multilingual-cased`
- ROUGE-L (Lin, 2004)
- Hallucination proxy : 1 − ROUGE-L(prédit, contexte source) — ✅ calculé en français pour **tous** les domaines du jeu de test (y compris **juridique / code de la route** ; HAL = FR, biais langue éliminé)

---

## 5. Méthodologie — ~900–1400 mots

### 5.1 Architecture du pipeline

```
Wikipedia FR (100 articles)      ──┐  dataset_type = "technique"
                                    │
HAL.science FR (100 résumés)       ─┼──► LLM Q&R (Groq llama-3.1-8b-instant) ──► train.json
                                    │   [indépendant de LLaMA 3.1 8B]        ──► test.json
                                    │
Actualités FR (100 articles)       ─┤  dataset_type = "temporel"
 (Le Monde / France Info)           │
                                    │
Code de la route (PDF → segments)  ─┘  dataset_type = "juridique"
 (`code_route.json`, notebook 01)

test.json ──► Baseline (Groq, sans contexte)              — `03` ──► baseline_predictions.json
         ──► RAG (FAISS + Groq)                           — `03` ──► rag_predictions.json
         ──► Fine-tuning LoRA                             — `04` ──► finetuned_predictions.json + `models/lora_adapter_ft/`
         ──► FT+RAG (LoRA fine-tuné + FAISS + inférence locale) — `08` ──► ft_rag_predictions.json
         ──► RAFT (pseudo-RAG au train, LoRA dédié)        — `05` ──► raft_predictions.json + `models/lora_adapter_raft/`
         ──► RAG + reranking (cross-encoder + Groq)       — `06` ──► rerank_predictions.json
         ──► Function calling (outil search_docs + Groq) — `07` ──► function_calling_predictions.json
                                                                              │
                                                         `09_evaluation.ipynb` (agrège les 7 JSON)
                                                                              ▼
                                              F1 / BERTScore / ROUGE-L / Fidélité / Hallucination / Confiance / Fig. 10 (écologie)
                                              Tableau croisé + figures + `final_report.json`
```

### 5.2 Constitution du corpus (Notebooks 01 + 02)

| Dataset | Source | Docs | Contenu | `dataset_type` | Questions/doc |
|---------|--------|------|---------|----------------|-----------|
| Technique | Wikipedia FR | 100 | Articles complets (~1600 mots) | `technique` | 5 (2 factuelles, 2 synthèse, 1 compréhension) |
| Multi-sauts | HAL.science FR (**quotas temporels** sur `submittedDateY_i` : ≤2021 / 2022–2023 / ≥2024, complément large si besoin) | 100 | Résumés enrichis (API `fl=*` + page hal.science, **cible ≥ 40k mots**) | `multisauts` | 5 (3 simples + 2 complexes) |
| Temporel | Le Monde / France Info | 100 | Article complet scraped (~300 mots) | `temporel` | 5 (2 factuelles, 2 synthèse, 1 compréhension) |
| **Juridique** | **Code de la route** (PDF officiel → segments ~1400 mots → `data/raw/code_route.json`) | Segments **≥ 20** requis pour quotas Q&R (cible **100 paires** 80/20 si PDF complet) | Texte légal FR, `source` = `code_route_pdf` | **`juridique`** | 5 (même répartition de types de questions que les autres sources) |

**Split 80/20 (seed 42)** : au sein de **chaque** source, la division train/test est **stratifiée sur `recency_category`** (répartition ~proportionnelle au pool, méthode du plus grand reste), puis mélange final des listes concaténées.

| Split actuel | Technique | Multisauts | Temporel | Juridique | Total |
|-------------|-----------|------------|----------|-----------|-------|
| train | 320 | 360 | 400 | 80 | **1160** |
| test  |  80 |  90 | 100 | 20 | **290** |

Les métriques agrégées dans `results/final_report.json` utilisent **n = 290** sur la campagne courante.

- Génération Q&R : **Groq** `llama-3.1-8b-instant` — **aligné** sur baseline/RAG ; **biais circulaire** possible vs fine-tuning/RAFT (même famille 8B)
- Prompt avec règle obligatoire : `context` = citation verbatim du texte source (20-150 mots)
- Champ `recency_category` déduit de la date ISO du document : `récent` (≥2024), `intermédiaire` (2022-2023), `fondamental` (<2022)
- Fallback automatique : 400 premiers caractères si contexte trop court (<30 caractères)
- Dataset 100 % français : HAL remplace Arxiv (articles en anglais) pour une cohérence linguistique totale
- **Code de la route** : corpus **juridique** distinct (réglementation / sanctions / définitions) — indexé dans le **même FAISS** que les autres documents pour le RAG / rerank / FC ; les questions **juridiques** du test servent notamment à mesurer l’effet du retrieveur sur du texte normatif

### 5.3 Méthode 1 — Baseline
- Modèle : LLaMA 3.1 8B via Groq API, **aucun contexte externe**
- Prompt minimal en français, réponse directe depuis les paramètres du modèle
- Représente la limite basse : ce que LLaMA 3.1 8B sait déjà (avant décembre 2023)

### 5.4 Méthode 2 — RAG
- Embedding : `paraphrase-multilingual-MiniLM-L12-v2` (dimension 384, FR + EN)
- Index : FAISS `IndexFlatIP` (cosine similarity sur vecteurs normalisés L2)
- **Corpus indexé** : documents bruts complets (`wikipedia_technique.json`, `hal.json`, `lemonde.json`, **`code_route.json`**) découpés en chunks de 200 mots avec chevauchement de 50 mots (le juridique provient des segments PDF *code de la route*)
- Récupération : top-5 chunks (titre + texte injectés dans le prompt)
- Génération : LLaMA 3.1 8B via Groq avec contexte enrichi

### 5.5 Méthode 3 — Fine-tuning LoRA
- Modèle de base : `unsloth/Meta-Llama-3.1-8B-bnb-4bit` (quantifié 4-bit, **GPU A100** recommandé ; profil auto T4/L4/A100 dans NB04 et NB05)
- LoRA : **rank=32**, **alpha=32**, **dropout=0.05**, 7 modules (`q/k/v/o_proj`, `gate/up/down_proj`) — passer rank 16 si OOM
- **Format SFT** : gabarit Alpaca (`### Instruction / ### Input / ### Response`) + option contexte gold (`### Context`) activable
- **Améliorations appliquées (NB04)** : split validation, `load_best_model_at_end`, **6 époques**, `lr=8e-5`, génération anti-répétition renforcée (`repetition_penalty=1.16`, `no_repeat_ngram_size=4`, `max_new_tokens=360`) + nettoyage de sortie (boucles / fins coupées / bruit éditorial)
- Entraînement : batch=2, gradient_acc=8 (effectif 16), cosine, warmup, max_grad_norm=1, AdamW 8-bit
- max_seq_length=1024, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- **Inférence test** : ajout d'un **proxy de confiance** token-level (`confidence`) stocké dans les JSON

#### 5.5.bis RAFT (pseudo-RAG au train)

Notebook **`05_raft.ipynb`** : pour chaque exemple de `train.json`, récupération FAISS top-k, construction du prompt (`Instruction + Context + Input + Response`), entraînement LoRA sauvegardé dans `models/lora_adapter_raft/`, puis inférence sur `test.json` → `raft_predictions.json`.

#### 5.5.ter Recherche d’hyperparamètres (LoRA / entraînement)

- **Ne pas** optimiser sur le jeu **test** final : réserver une petite **validation** (par ex. 10 % de `train.json`, seed fixe) ou une métrique **validation loss** pendant l’entraînement.
- **Grille simple** : faire varier **une dimension à la fois** — `lr` (ex. `5e-5`, `1e-4`, `2e-4`), **nombre d’époques** (3–8), **rank LoRA** (16 vs 32), **warmup** ; garder le reste identique (seed, batch effectif).
- **Critère** : pertinence du projet — perte de validation la plus basse, ou **BERTScore / F1 sur la validation** si vous régénérez des réponses sur ce sous-ensemble (plus coûteux).
- Une fois les meilleurs hyperparamètres fixés, **un seul** entraînement sur tout `train.json`, puis évaluation **une fois** sur `test.json` via **`09_evaluation.ipynb`** (après exécution des notebooks de prédictions `03`–`08` si FT+RAG est utilisé, sinon `03`–`07` suffisent avant l’agrégation).

### 5.6 Méthode 5 — RAG + reranking
- Même index FAISS que le RAG ; récupération **élargie** (top-M candidats) puis **cross-encoder** multilingue (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) pour ne garder que les **top-k** passages.
- Génération : Groq `llama-3.1-8b-instant` — sortie `rerank_predictions.json` (notebook **`06_rag_rerank.ipynb`**).

### 5.7 Méthode 6 — Function calling
- Groq avec schéma d’outil `search_docs` (recherche FAISS simulée côté client) puis réponse finale en français — sortie `function_calling_predictions.json` (notebook **`07_function_calling.ipynb`**).

### 5.8 Note sur l'infrastructure
> *Baseline, RAG, RAG+rerank et function calling passent par **Groq** ; fine-tuning et RAFT utilisent **Unsloth + LoRA** en local (**A100** prioritaire, fallback T4/L4).*

### 5.9 Métriques d'évaluation

| Métrique | Description | Niveau |
|----------|-------------|--------|
| **Fidélité (faithfulness)** | Moyenne de **ROUGE-L(prédit, contexte)** du `test.json` (repli sur `true_answer` si contexte vide), en % — complémentaire du proxy d’hallucination | Fidélité |
| **F1 token-level** | Overlap de tokens entre prédiction et **référence** | Lexical |
| **ROUGE-L** | Plus longue sous-séquence commune | Lexical |
| **METEOR** | F-mesure sur tokens + synonymes (NLTK WordNet) | Lexical |
| **BERTScore** | Similarité sémantique (distilbert-base-multilingual) | Sémantique |
| **IC 95 % (bootstrap)** | 1 000 réplicatas sur la **moyenne** des scores **par paire** (BERTScore, METEOR, F1, ROUGE-L) — champs `*_ci95_low` / `*_ci95_high` dans `final_report.json` | Sémantique / lexical |
| **Accuracy@BS** | % réponses avec BERTScore ≥ seuil (80/85/90%) | Sémantique |
| **Hallucination** | Moyenne du proxy 1 − ROUGE-L(prédit, contexte source) (en %) | Fidélité |
| **Accuracy@Hall≤k** | % réponses avec proxy ≤ k (seuils k = 15, 20, 25 % sur l'échelle [0,100]) | Fidélité |
| **Longueur rép.** | Nombre moyen de mots par réponse prédite | Qualité |
| **Taux vides (%)** | % réponses vides ou < 3 caractères | Qualité |
| **% Français HAL** | % réponses en français (langdetect) pour questions HAL | Langue |
| **Latence** | Temps moyen de génération (ms) | Efficacité |
| **Confiance** | Moyenne du proxy token-level (si champ `confidence` présent) | Fiabilité |
| **Énergie (Wh)** | Latence × TDP hardware × PUE datacenter / 3600 | Écologique |
| **CO2eq (gCO2eq)** | Énergie × intensité carbone (France : 52 / Mondial : 475 gCO2/kWh) | Écologique |

*(Tableau principal et agrégation des **sept** méthodes : `09_evaluation.ipynb` + les sept fichiers `*_predictions.json`, incluant `ft_rag_predictions.json`.)*

✅ *Avec HAL (articles 100 % français), le biais de la métrique d'hallucination présent avec Arxiv (contexte EN vs réponse FR) est éliminé. Le ROUGE-L(prédit, contexte source) est maintenant calculé dans la même langue pour tous les exemples.*

---

## 6. Résultats — ~500–700 mots

> **Synchronisation** : agrégation depuis **`results/final_report.json`** de la campagne courante, avec les **sept** JSON `results/*_predictions.json` (dont **`ft_rag_predictions.json`**). Les figures de §6.6 sont exportées en PNG sous **`results/plots/`** lors de l’exécution de **`09_evaluation.ipynb`**. Après chaque nouveau run, régénérer le rapport puis mettre à jour les tableaux.

### 6.1 Tableau comparatif global (toutes les méthodes)

Campagne **`final_report.json`** généré le **2026-05-06**, **n = 290**.

| Méthode | Fidélité (%) | F1 (%) | BERTScore (%) | BS CI 95 % | ROUGE-L (%) | METEOR (%) | Acc@BS 85 % | Hallucin. (%) | Latence (ms) |
|---------|:------------:|:------:|:-------------:|:----------:|:-----------:|:----------:|:-----------:|:---------------:|:------------:|
| Baseline | 16.5 | 16.3 | 81.4 | [81.11, 81.72] | 13.7 | 34.5 | 9.7 | 83.5 | **549** |
| RAG | 31.8 | 32.8 | 85.8 | [85.22, 86.23] | 28.7 | 52.4 | 54.5 | 68.2 | 599 |
| Fine-tuné | 12.7 | 14.8 | 81.4 | [81.08, 81.64] | 11.6 | 35.5 | 6.2 | 87.3 | 14 217 |
| **FT+RAG** | 13.1 | 14.6 | 81.0 | [80.69, 81.27] | 11.7 | 35.8 | 5.9 | 86.9 | **18 976** |
| RAFT | 11.6 | 11.9 | 81.2 | [80.94, 81.54] | 10.1 | 26.3 | 4.8 | **88.4** | **21 224** |
| **Rerank** | **33.8** | **36.5** | **86.6** | [86.05, 87.12] | **32.1** | **56.1** | **60.7** | **66.2** | 570 |
| Function calling | 24.2 | 23.9 | 83.4 | [82.86, 83.89] | 20.9 | 44.6 | 31.4 | 75.8 | 1 078 |

*(**Fidélité** : moyenne ROUGE-L(prédit, `context` du test) ; exportée dans `final_report.json`.)*

#### Faits saillants (même campagne)

- **Rerank** domine les métriques **lexicales / sémantiques automatiques** (F1, BERTScore, ROUGE-L, **METEOR**, Acc@85, fidélité, hallucination) avec une latence API **du même ordre que la baseline** (~570 ms).
- **RAG** reste **très proche** du rerank sur la plupart des axes (notamment **METEOR** et découpes par domaine) ; **FT+RAG** **ne rattrape pas** le RAG ni le rerank — scores proches du **fine-tuné seul**, avec une **latence encore plus élevée** (LoRA local + décodage long + retrieval).
- **Fine-tuné**, **FT+RAG** et **RAFT** : **BERTScore** plateau ~81 % (vs ~86 % rerank) ; **hallucination** (proxy) **supérieure à la baseline** ; inférence locale **×26–39×** plus lente que la baseline.
- **RAFT** : **METEOR** le plus bas des sept ; **F1** et **ROUGE-L** sous baseline — le protocole RAFT + décodeur Unsloth ne se traduit pas par un gain sur ce benchmark.
- **Function calling** : entre **baseline** et **RAG** ; latence modérée (~1,1 s) pour un flux **deux passages** Groq.

### 6.1 bis Patterns qualitatifs (exploration des `*_predictions.json`)

- **Ancrage lexical** : le **RAG** et le **Rerank** produisent souvent des formulations du type *« selon … »* ou des références explicites aux **extraits** (*« contexte »*, *« document »*), beaucoup plus que la **baseline** — cohérent avec l’injection de passages chunkés.
- **Longueur des réponses (Fig. 17)** : **FT+RAG** tend à produire les réponses **les plus longues** (médiane élevée) ; **Rerank** et **RAFT** restent **plus courts** — la longueur seule n’explique pas la qualité BERTScore (ex. rerank court mais meilleur score).
- **Hésitation** : préfixes du type *« je ne … »* (refus / manque d’information) plus fréquents en **baseline** qu’avec **RAG / rerank** — la récupération réduit les refus explicites sur ce jeu.
- **Aucune identité brute** : **0** paires où la réponse **RAG** est strictement identique à la **baseline** (les passages modifient systématiquement le texte généré sur les 290 items).

### 6.1 ter Article LaTeX

Un squelette **LaTeX** (résumé, **sept** méthodes — à synchroniser avec FT+RAG — tableau principal à compléter depuis `final_report.json`) est dans [`article_scientifique.tex`](article_scientifique.tex) ; à compiler avec `pdflatex` / `latexmk` et à enrichir (figures, bibliographie complète).

### 6.2 BERTScore par domaine × méthode (%)

*(Colonnes = `dataset_type` dans `test.json`, campagne 2026-05-06.)*

| Méthode | technique | multisauts | temporel | juridique | Moy. 4 domaines |
|---------|-----------|------------|----------|-----------|-----------------|
| Baseline | 82.0 | 81.8 | 80.5 | 81.7 | 81.5 |
| RAG | 85.9 | 86.9 | 84.9 | 84.5 | 85.6 |
| Fine-tuné | 82.0 | 81.4 | 81.1 | 80.2 | 81.2 |
| FT+RAG | 80.8 | 81.1 | 81.2 | 80.2 | 80.8 |
| RAFT | 81.3 | 81.6 | 81.1 | 80.0 | 81.0 |
| **Rerank** | **86.6** | **87.2** | **86.3** | **85.2** | **86.3** |
| Function calling | 83.5 | 83.7 | 83.0 | 83.6 | 83.5 |

### 6.3 BERTScore par type de question (%)

| Méthode | factuel | synthese | comprehension | simple (HAL) | complexe (HAL) |
|---------|---------|----------|---------------|--------------|----------------|
| Baseline | 82.0 | 80.7 | 80.9 | 81.9 | 81.7 |
| RAG | 86.4 | 83.7 | 85.8 | 88.5 | 84.9 |
| Fine-tuné | 81.3 | 81.6 | 80.9 | 81.0 | 81.8 |
| FT+RAG | 80.5 | 81.3 | 81.2 | 80.2 | 82.1 |
| RAFT | 81.0 | 81.4 | 80.9 | 82.1 | 80.9 |
| **Rerank** | **88.5** | **84.4** | **85.5** | **89.0** | **85.0** |
| Function calling | 84.5 | 82.0 | 83.2 | 83.9 | 83.4 |

> **HAL** : **Rerank** et **RAG** tirent encore parti du **multi-sauts** ; **FT+RAG** reste ~5 points sous **RAG** sur le **factuel**.

### 6.4 Hallucination (proxy) par domaine (%)

*(Plus bas = mieux ; 100 × (1 − ROUGE-L(prédit, contexte source)).)*

| Méthode | technique | temporel | multisauts | juridique |
|---------|-----------|----------|-------------|-----------|
| Baseline | 80.2 | 86.4 | 84.2 | 79.5 |
| RAG | 67.5 | 73.4 | 63.7 | 66.1 |
| Fine-tuné | 85.7 | 87.7 | 88.0 | **88.6** |
| FT+RAG | 86.5 | 86.8 | 87.2 | 87.9 |
| RAFT | 88.4 | 88.6 | 87.2 | **92.8** |
| **Rerank** | **63.0** | **71.5** | **63.5** | **64.3** |
| Function calling | 73.9 | 78.5 | 76.2 | 69.0 |

> **Rerank** minimise le proxy sur **technique** et **multisauts** ; le **juridique** reste difficile pour toutes les méthodes **hors retrieval** (fine-tuné, RAFT, FT+RAG).

### 6.5 Analyse latence

| Méthode | Latence moy. (ms) | Ratio vs Baseline |
|---------|-------------------|-------------------|
| Baseline | 549 | ×1.0 |
| **Rerank** | **570** | ×1.04 |
| RAG | 599 | ×1.09 |
| Function calling | 1 078 | ×2.0 |
| Fine-tuné | 14 217 | ×25.9 |
| FT+RAG | 18 976 | ×34.6 |
| RAFT | 21 224 | ×38.7 |

### 6.6 Figures produites par `09_evaluation.ipynb`

| Figure | Description |
|--------|-------------|
| Fig. 1 | Métriques globales (F1, BERTScore, ROUGE-L, METEOR, fidélité) — barres groupées |
| Fig. 2 | BERTScore par strate temporelle × méthode |
| Fig. 3 | BERTScore par type de question × méthode |
| Fig. 4 | Latences (moy + p95) par méthode |
| Fig. 5 | Taux d'hallucination — global + par domaine |
| Fig. 6 | Trade-off qualité vs latence (scatter) |
| Fig. 7 | Heatmap BERTScore méthodes × dataset_type |
| Fig. 8 | Questions simples vs multi-sauts (HAL) |
| Fig. 9 | **Accuracy par seuil BERTScore** (≥80%, ≥85%, ≥90%) |
| Fig. 10 | **Impact écologique** — énergie (Wh), CO2 France vs mondial, CO2/prédiction |

### 6.7 Évaluation subjective — LLM-as-judge (Anthropic)

**Protocole** : notebook **`10_llm_as_judge.ipynb`** ; agrégation **`results/llm_judge_anthropic_all_methods.json`** ; figures **`results/plots/fig_judge_*.png`** et tableau **`results/llm_judge_summary_table.csv`** (voir aussi **`11_llm_judge_analysis.ipynb`**).

**Juge** : `claude-sonnet-4-6` ; scores **1–5** (cohérence, utilité, fidélité au contexte). Moyennes **sur n = 290** par fichier de prédictions :

| Rang | Méthode (label fichier) | Cohérence | Utilité | Fidélité | Composite* |
|:----:|-------------------------|:---------:|:-------:|:--------:|:------------:|
| 1 | `rerank` | **3.93** | **3.66** | **3.57** | **~3.72** |
| 2 | `rag` | 3.68 | 3.31 | 3.29 | ~3.43 |
| 3 | `function_calling` | 3.21 | 2.74 | 2.48 | ~2.81 |
| 4 | `baseline` | 2.46 | 1.72 | 1.37 | ~1.85 |
| 5 | `ft_rag` | 2.00 | 1.79 | 1.55 | ~1.78 |
| 6 | `finetuned` | 2.02 | 1.61 | 1.25 | ~1.63 |
| 7 | `raft` | 1.63 | 1.53 | 1.28 | ~1.48 |

\*Composite = moyenne arithmétique des trois critères (non présente telle quelle dans le JSON).

**Observations juge vs métriques auto** : le classement **rerank > RAG > function calling > baseline** est **aligné** avec le tableau §6.1. **FT+RAG** se situe **au niveau de la baseline** en composite juge — cohérent avec un **BERTScore** proche du fine-tuné **sans** atteindre le niveau **RAG/rerank**. **RAFT** est **dernier** des deux côtés.

**Limite** : lorsque le juge renvoie du JSON **entouré de balises Markdown**, certaines lignes peuvent être marquées `parse_error` dans le détail — prévoir un post-traitement ou des retries si l’on durcit le protocole.

---

## 7. Discussion — ~500–600 mots

### 7.1 Quelle méthode pour quel domaine ? *(croisé avec §6.2–6.4, campagne 2026-05-06)*

- **Rerank** et **RAG** : meilleurs compromis **qualité automatique / hallucination (proxy)** ; le **rerank** l’emporte sur **F1, BERTScore, ROUGE-L, METEOR** et **Acc@85** dans ce run.
- **FT+RAG** : **ne combine pas** ici les forces du fine-tuné et du RAG — scores proches du **fine-tuné seul**, **BERTScore** nettement sous **RAG**, **latence** parmi les plus élevées (GPU + retrieval + décodage). **Pistes** : gabarit prompt / distribution train vs prompts RAG, calibration du retrieveur avec le décodeur LoRA, ou conflit entre style SFT et format « passages injectés ».
- **Fine-tuné** : **ne surpasse pas la baseline** sur F1 / hallucination proxy dans ce run — sensibilité au format Alpaca, à la quantification locale et au décodage vs API Groq.
- **RAFT** : **F1** et **METEOR** les plus bas des sept ; l’hypothèse « contexte + distracteurs » ne se traduit pas par un gain sur ce protocole.
- **Function calling** : qualité **entre** baseline et RAG ; utile pour étudier un **flux agentif** au prix d’une latence **~2×** baseline.
- **Baseline** : référence sans documents ; utile comme **plancher de latence** et de complexité.

### 7.2 Impact de la complexité des questions

- Les questions **HAL complexes** restent favorisées par le **RAG** et le **Rerank** en BERTScore vs les **simples** (§6.3) — le retrieveur dense + re-classement semble aider le raisonnement multi-passages.
- Le **fine-tuning seul** sans contexte à l’inférence peine sur le raisonnement croisé si la réponse n’a pas été « mémorisée » dans les poids ; **RAFT** n’améliore pas cette situation sur les métriques rapportées.
- **FT+RAG** : sur le **factuel**, le **BERTScore** reste **plus bas** que celui du **RAG** — ajouter des passages ne compense pas ici l’écart de distribution **SFT vs prompt Groq** ; pistes : ré-écrire le gabarit d’inférence pour coller au train Alpaca, ou **fusionner** scores rerank + logits adaptateur (hors scope actuel).

### 7.3 Pourquoi le RAG sur documents bruts est supérieur
- **Hétérogénéité** : l’index couvre Wikipédia, HAL, presse **et** **Code de la route** (`juridique`) — le retrieveur doit donc aligner questions factuelles, scientifiques, d’actualité et **normatives** ; la colonne **juridique** des tableaux §6.2–6.4 en rend compte.
- Corpus FAISS sur extraits trop courts → retrieval **insuffisant** pour répondre → RAG peut dégrader vs baseline.
- Corpus FAISS sur documents complets chunkés (**200 mots, overlap 50**, compromis issu d’une **étude d’ablation** sur tailles de chunks) → meilleure couverture que des fenêtres trop petites.
- La qualité du corpus indexé est le facteur déterminant de l'efficacité du RAG

### 7.4 Limites méthodologiques
- Dataset généré par **Groq / LLaMA 3.1 8B** : **pas d’indépendance** vis-à-vis de l’inférence baseline/RAG ; biais de **style** et de **couverture** du générateur.
- Taille **1 160 / 290** sur ce dépôt — classes rares encore peu représentées pour certains croisements méthode × domaine.
- **Alignement train / inférence** : pour LoRA et RAFT, tout écart de gabarit prompt entre entraînement et test peut dégrader les scores ; documenter les gabarits dans l’article.
- ✅ HAL (FR) : métrique d’hallucination **sans** décalage linguistique anglais/français.

### 7.5 Limites techniques
- Fine-tuning/RAFT recommandés sur Colab **A100** pour stabilité/temps de run ; si fallback T4 : réduire `MAX_SEQ_LEN` et `MAX_CONTEXT_CHARS` (profil auto dans `04_finetuning.ipynb` et `05_raft.ipynb`)
- ar5iv.org peut être instable → certains papiers sans introduction (enrichissement partiel)
- Groq `llama-3.1-8b-instant` vs `unsloth/Meta-Llama-3.1-8B-bnb-4bit` : quantifications différentes

### 7.6 Rapport coût-bénéfice pratique

| Méthode | Coût infra | Mise à jour | Recommandé si... |
|---------|-----------|-------------|-----------------|
| Baseline | Très faible | Instantanée | Données stables pré-coupure |
| RAG | Faible | Instantanée (réindexation) | Données fraîches changeantes |
| Fine-tuné | Élevé (GPU) | Coûteuse (ré-entraîner) | Domaine spécialisé stable |
| **FT+RAG** | **Très élevé (GPU + index)** | Réindexation + ré-entraînement LoRA | Expérimental — à valider si synergie réelle prompt/train |
| RAFT | Élevé (GPU) | Coûteuse | Quand le format « contexte + distracteurs » est central |
| RAG + rerank | Moyen (API + rerank local) | Réindexation + même rerank | Qualité retrieveur limite le RAG seul |
| Function calling | Moyen (API, boucles possibles) | Réindexation + schéma outil | Contrôle explicite des recherches |

---

## 8. Conclusion — ~200–300 mots

1. **Rappel** : comparer **sept** stratégies (baseline, RAG, LoRA, **FT+RAG**, RAFT, rerank, function calling) sur un socle **LLaMA 3.1 8B** et un jeu **100 % français**.
2. **Résultats (§6.1, campagne 2026-05-06, n = 290)** :
   - **Rerank** domine les métriques **automatiques** (dont **METEOR**) ; **RAG** reste le **second meilleur** compromis global.
   - **FT+RAG** : **pas de synergie observée** — performances proches du **fine-tuné**, nettement sous **RAG/rerank**, **latence** très élevée (GPU + retrieval).
   - **Fine-tuning LoRA** et **RAFT** : scores lexicaux et fidélité **sous** baseline sur plusieurs indicateurs ; latence **~26–39×** baseline.
   - **Function calling** : **entre** baseline et RAG ; utile pour modéliser un **agent** avec recherche conditionnelle.
3. **LLM-as-judge (§6.7)** : ordre **rerank > rag > function calling > baseline ≈ ft_rag > finetuned > raft**, cohérent avec les métriques **BERTScore / F1**.
4. **Perspectives** : harmoniser **prompt/gabarit** pour FT+RAG, évaluation humaine, ablations retrieveur, gestion stricte des **réponses JSON** du juge, retrieveurs plus riches (DPR/ColBERT).

---

## 9. Références (format IEEE)

```
[1]  A. Vaswani et al., "Attention is all you need," NeurIPS, 2017.
[2]  P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," NeurIPS, 2020.
[3]  E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," ICLR, 2022.
[4]  T. Dettmers et al., "QLoRA: Efficient finetuning of quantized LLMs," NeurIPS, 2023.
[5]  N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," EMNLP, 2019.
[6]  J. Johnson, M. Douze, and H. Jégou, "Billion-scale similarity search with GPUs," IEEE Trans. Big Data, 2021.
[7]  Z. Ji et al., "Survey of hallucination in natural language generation," ACM CSUR, 2023.
[8]  H. Touvron et al., "LLaMA 2: Open foundation and fine-tuned chat models," arXiv:2307.09288, 2023.
[9]  Meta AI, "Meta LLaMA 3," arXiv:2407.21783, 2024.
[10] T. Zhang et al., "BERTScore: Evaluating text generation with BERT," ICLR, 2020.
[11] C.-Y. Lin, "ROUGE: A package for automatic evaluation of summaries," ACL Workshop, 2004.
[12] P. Rajpurkar et al., "SQuAD: 100,000+ questions for machine comprehension of text," EMNLP, 2016.
[13] S. Shi et al., "REPLUG: Retrieval-augmented language model pre-training," NAACL, 2023.
[14] K. Guu et al., "REALM: Retrieval augmented language model pre-training," ICML, 2020.
[15] R. McCloskey and N. Cohen, "Catastrophic interference in connectionist networks," Psychology of Learning and Motivation, 1989.
[16] J. Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks," PNAS, 2017.
```

---

## Conseils de rédaction

- **Longueur totale** : viser 8–12 pages (police 11pt, double colonne IEEE) ou 5000–7000 mots
- **Langage** : présent de vérité générale pour l'état de l'art, passé composé pour vos expériences
- **Chiffres** : arrondir à 1 décimale, mettre en **gras** le meilleur résultat par colonne
- **Code** : référencer les notebooks en annexe ou sur GitHub (rendre public avant soumission)
- **Reproductibilité** : mentionner seed=42, GPU **A100 Colab** (ou fallback T4/L4 avec profil auto), Groq Developer `llama-3.1-8b-instant`
- **Biais à mentionner** : dataset généré par le même modèle que celui évalué (LLaMA 3.1 8B)

---

## Correspondance Notebooks ↔ Sections de l'article

| Notebook | Section article | Produit |
|----------|-----------------|---------|
| `01_scraping.ipynb` | §5.2 Constitution du corpus | `wikipedia_technique.json`, `hal.json`, `lemonde.json`, **`code_route.json`** (PDF *code de la route* → segments, **juridique**) |
| `02_dataset_builder.ipynb` | §5.2 Tableau statistiques | `train.json`, `test.json` — Groq `llama-3.1-8b-instant` |
| `03_baseline_rag.ipynb` | §5.3–5.4 Baseline + RAG + FAISS | `baseline_predictions.json`, `rag_predictions.json`, `models/faiss_index/` |
| `04_finetuning.ipynb` | §5.5 Fine-tuning LoRA | `finetuned_predictions.json`, `models/lora_adapter_ft/` |
| `05_raft.ipynb` | §5.5.bis RAFT | `raft_predictions.json`, `models/lora_adapter_raft/` |
| `06_rag_rerank.ipynb` | §5.6 RAG + reranking | `rerank_predictions.json` |
| `07_function_calling.ipynb` | §5.7 Function calling | `function_calling_predictions.json` |
| `08_ft_plus_rag.ipynb` | §5 FT+RAG | `ft_rag_predictions.json` (LoRA `lora_adapter_ft` + FAISS) |
| `09_evaluation.ipynb` | §6 Résultats | `final_report.json`, figures PNG (**7 méthodes**) |
| `10_llm_as_judge.ipynb` | §6.7 | `llm_judge_<provider>_*.json`, `fig_judge_compare_*.png` |
| `11_llm_judge_analysis.ipynb` | §6.7 | `llm_judge_summary_table.csv`, `fig_judge_*.png` |
| `article_scientifique.tex` | §6.1 bis + rédaction complète | PDF article (compilation `pdflatex`) |

---

## État du pipeline

- [ ] `01` — Scraping : Wikipedia (100), HAL.science FR (100 notices en **3 quotas d’année de dépôt** + complément), Actualités FR (100), **Code de la route** (PDF → `code_route.json`, segments **juridiques**)
- [ ] `02` — Dataset builder Groq : train/test (contextes verbatim, json_repair)
- [ ] `03` — Baseline + RAG avec FAISS sur docs bruts chunkés 200 mots (**index incluant** Wikipedia + HAL + presse + **`code_route.json`**)
- [ ] `04` — Fine-tuning → `finetuned_predictions.json` + `lora_adapter_ft/`
- [ ] `05` — RAFT → `raft_predictions.json` + `lora_adapter_raft/`
- [ ] `06` — RAG + reranking → `rerank_predictions.json`
- [ ] `07` — Function calling → `function_calling_predictions.json`
- [ ] `08` — FT+RAG → `ft_rag_predictions.json` (adapter §04 + index §03)
- [ ] `09` — Évaluation agrégée (**7 méthodes**) → `final_report.json` + figures
- [ ] `10`–`11` — (optionnel) LLM-as-judge + analyses figures dédiées
- [ ] Figures 1–10 sauvegardées et vérifiées
- [ ] Rédaction article : sections 1–9 complétées

---

## Green AI — Impact écologique estimé

### Hypothèses de calcul

| Paramètre | Baseline / RAG / rerank / FC (Groq + rerank local) | Fine-tuné / FT+RAG / RAFT (Colab, GPU) |
|---|---|---|
| Hardware | H100 SXM (partagé) | NVIDIA A100 (Colab) |
| TDP effectif | ~50W (~700W / 14 slots) | ~150W (A100 partagé, hypothèse conservative) |
| PUE datacenter | 1.10 (Google infra) | 1.15 (Google Colab) |
| Intensité carbone | 52 gCO2/kWh (France) | 52 gCO2/kWh (France) |

> *Formule : Énergie (Wh) = latence_s × TDP_W × PUE / 3600 · CO2 (g) = Énergie_kWh × intensité*

### Estimations (jeu de test — campagne **2026-05-06**, **n = 290** par méthode)

Énergie estimée par Σ(latence\_s × TDP × PUE / 3600) avec **API** : TDP 50 W, PUE 1,10 · **GPU local** : TDP 150 W, PUE 1,15 (hypothèses §Green AI, alignées sur **`09_evaluation.ipynb`** Fig. 10).

| Méthode | Énergie (Wh) | CO₂ France (g)\* | Ratio énergie vs Baseline |
|---------|--------------|-------------------|---------------------------|
| Baseline | **2.43** | **0.13** | ×1.0 |
| Rerank | 2.52 | 0.13 | ×1.04 |
| RAG | 2.65 | 0.14 | ×1.09 |
| Function calling | 4.77 | 0.25 | ×2.0 |
| Fine-tuné | **197.6** | **10.3** | ×81 |
| FT+RAG | **263.7** | **13.7** | ×109 |
| RAFT | **294.9** | **15.3** | ×121 |

\*CO₂ (g) ≈ Wh/1000 × 52 g/kWh (mix France).

> **Conclusion Green AI** : **FT+RAG**, **RAFT** et **fine-tuné** concentrent l’essentiel de l’empreinte **Wh** (latences GPU cumulées). Les stratégies **Groq + retrieval** (baseline, RAG, rerank) restent **deux ordres de grandeur** plus légères sur ce proxy — au prix d’une dépendance API.

### À mentionner dans la section Discussion
- Mesure directe avec `codecarbon` sur l'inférence GPU locale (fine-tuné, RAFT)
- Estimation par formule TDP × latence pour les appels API (Baseline/RAG)
- Biais : Groq utilise de l'énergie renouvelable → CO2 réel probablement inférieur
- Référence : Strubell et al. (2019), "Energy and Policy Considerations for Deep Learning in NLP"

---

## Résultats clés à retenir pour la rédaction

| Observation | Chiffre (run **2026-05-06**, n=290) | Section |
|-------------|--------------------------------------|---------|
| **Rerank = meilleure méthode** sur F1 / ROUGE / **METEOR** / BS / Acc@85 / fidélité | F1 **36.5 %**, BS **86.6 %**, Acc@85 **60.7 %**, hall. proxy **66.2 %** | §6.1 |
| **RAG** = **second** ; meilleur écart vs baseline sur retrieval | F1 **32.8 %**, METEOR **52.4 %** | §6.1 |
| **FT+RAG** ≈ **fine-tuné** en BS (~81 %), **loin sous RAG** ; latence **~19 s** | BS **81.0 %**, F1 **14.6 %** | §6.1, §7.1 |
| **RAFT** : **METEOR** minimal (**26.3 %**) ; halluc. proxy **88 %** | À traiter comme **négatif empirique** sur ce setup | §6.1 |
| **LLM juge** (Claude Sonnet 4.6) : ordre **rerank > rag > FC > baseline ≈ ft_rag** | Composite ~**3.72** (rerank) vs ~**1.78** (FT+RAG) | §6.7 |
| Dataset 100 % FR + HAL | pas de biais hallu. cross-langue contexte/réponse | §5.2 |
| **Code de la route** indexé + test **juridique** | Voir **Fig. 7** / §6.2 | §5.1–5.4 |
| Source LaTeX pour papier | [`article_scientifique.tex`](article_scientifique.tex) — **mettre à jour pour 7 méthodes** | §6.1 bis |
| Empreinte **Wh** : FT+RAG / RAFT **×100×** vs baseline (proxy §Green AI) | Table §Green AI | §6.5 |
