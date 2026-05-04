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

---

## 1. Titre

Proposition principale :

> **Intégration de nouvelles informations dans les LLMs : baseline, RAG, fine-tuning LoRA, RAFT, reranking et function calling sur corpus français hétérogène**

Variantes :
- *Mettre à jour les LLMs sans ré-entraînement : RAG, reranking, function calling vs fine-tuning / RAFT sur données techniques, scientifiques et d'actualité*
- *Connaissances figées vs connaissances dynamiques : six stratégies d'intégration dans LLaMA 3.1 8B (baseline, RAG, LoRA, RAFT, rerank, function calling)*

---

## 2. Résumé (Abstract) — ~150–200 mots

**Structure** : Contexte → Problème → Méthode → Résultats → Conclusion

> Les LLMs ont une date de coupure (décembre 2023 pour LLaMA 3.1 8B). Nous comparons **six** stratégies d’intégration d’information sur un socle **LLaMA 3.1 8B** : **baseline** (Groq sans contexte), **RAG** (FAISS + Groq), **fine-tuning LoRA** (local), **RAFT** (pseudo-RAG au train + LoRA `lora_adapter_raft`), **RAG + reranking** (cross-encoder multilingue + Groq), et **function calling** (outil `search_docs` + Groq). Corpus **100 % français** (Option B : typiquement **1 200 train / 300 test** sans juridique, ou **1 160 / 290** lorsque le jeu **code_route** est actif — voir §5.2), généré via **Groq** — **biais circulaire** possible à discuter. L’agrégation des métriques (EM, F1, BERTScore, ROUGE-L, **METEOR**, **IC bootstrap 95 %** sur BERTScore / METEOR / F1 / ROUGE-L, hallucination, latence, confiance) se fait dans **`08_evaluation.ipynb`** après exécution ordonnée des notebooks `01`→`08`.

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
1. Construction d'un **corpus 100 % français** (Option B : 1200 train + 300 test) sur 3 domaines distincts avec types de questions variés
2. **Protocole dataset** : génération Q&R via **Groq** (LLaMA 3.1 8B) ; **non indépendant** de l’évaluation → limite méthodologique assumée ou documentée
3. **Base commune** : toutes les méthodes s’appuient sur la famille **LLaMA 3.1 8B** (Groq API et/ou base quantifiée locale + adaptateurs LoRA) → comparaison équitable à discuter (API vs local)
4. **Évaluation multi-axe** : EM, F1, BERTScore (CI Bootstrap), ROUGE-L, METEOR, hallucination, latence, écologie
5. **Analyse croisée** méthodes × domaines + comparaison simple vs multi-sauts (HAL)
6. Résultats empiriques : comparaison **six méthodes** (baseline, RAG, LoRA, RAFT, rerank, function calling) avec discussion **RAG vs rerank** et **coût Groq vs GPU** (LoRA/RAFT)
7. Pistes : ablations retrieveur, évaluation des **appels d’outils** (précision JSON), évaluation humaine
8. Recommandations pratiques pour projets IA en français avec contraintes de coût/latence/empreinte carbone

### 3.4 Plan de l'article
> La section 2 présente l'état de l'art. La section 3 décrit notre méthodologie et les 3 datasets. Les résultats sont présentés en section 4 et discutés en section 5.

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
- **Corpus indexé** : documents bruts chunkés (400 mots, overlap 100) depuis `wikipedia_technique.json` + `hal.json` + `lemonde.json` — meilleur que les extraits courts
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
- Exact Match, F1 token-level (Rajpurkar et al., 2016)
- BERTScore (Zhang et al., 2020) — `distilbert-base-multilingual-cased`
- ROUGE-L (Lin, 2004)
- Hallucination proxy : 1 − ROUGE-L(prédit, contexte source) — ✅ calculé en français pour les 3 datasets (HAL = FR, biais éliminé)

---

## 5. Méthodologie — ~900–1400 mots

### 5.1 Architecture du pipeline

```
Wikipedia FR (100 articles)  ──┐  dataset_type = "technique"
                                │
HAL.science FR (100 résumés)  ─┼──► LLM Q&R (Groq llama-3.1-8b-instant) ──► train.json (1200)
                                │   [indépendant de LLaMA 3.1 8B]        ──► test.json  (300)
                                │
Actualités FR (100 articles)  ─┘  dataset_type = "temporel"
 (Le Monde / France Info)

test.json ──► Baseline (Groq, sans contexte)              — `03` ──► baseline_predictions.json
         ──► RAG (FAISS + Groq)                           — `03` ──► rag_predictions.json
         ──► Fine-tuning LoRA                             — `04` ──► finetuned_predictions.json
         ──► RAFT (pseudo-RAG au train, LoRA dédié)        — `05` ──► raft_predictions.json + `models/lora_adapter_raft/`
         ──► RAG + reranking (cross-encoder + Groq)       — `06` ──► rerank_predictions.json
         ──► Function calling (outil search_docs + Groq) — `07` ──► function_calling_predictions.json
                                                                              │
                                                         `08_evaluation.ipynb` (agrège les 6 JSON)
                                                                              ▼
                                              EM / F1 / BERTScore / ROUGE-L / Hallucination / Confiance
                                              Tableau croisé + figures + `final_report.json`
```

### 5.2 Constitution du corpus (Notebooks 01 + 02)

| Dataset | Source | Docs | Contenu | `dataset_type` | Questions/doc |
|---------|--------|------|---------|----------------|-----------|
| Technique | Wikipedia FR | 100 | Articles complets (~1600 mots) | `technique` | 5 (2 factuelles, 2 synthèse, 1 compréhension) |
| Multi-sauts | HAL.science FR (**quotas temporels** sur `submittedDateY_i` : ≤2021 / 2022–2023 / ≥2024, complément large si besoin) | 100 | Résumés enrichis (API `fl=*` + page hal.science, **cible ≥ 40k mots**) | `multisauts` | 5 (3 simples + 2 complexes) |
| Temporel | Le Monde / France Info | 100 | Article complet scraped (~300 mots) | `temporel` | 5 (2 factuelles, 2 synthèse, 1 compréhension) |

**Split figé Option B — 80/20 (seed 42)** : au sein de **chaque** source (Wikipedia, HAL simple, HAL complexe, Le Monde), la division train/test est **stratifiée sur `recency_category`** (répartition ~proportionnelle au pool, méthode du plus grand reste), puis mélange final des listes concaténées.

| Split | Technique | HAL simple | HAL complexe | Temporel | Total |
|-------|-----------|------------|--------------|----------|-------|
| train | 400 | 200 | 200 | 400 | **1200** |
| test  | 100 |  50 |  50 | 100 |  **300** |

**Run actuel (juridique `code_route` inclus, seed 42)** : `train.json` **1 160** paires (320 technique + 360 multisauts + 400 temporel + 80 juridique) · `test.json` **290** paires (80 + 90 + 100 + 20). Les métriques agrégées dans `results/final_report.json` utilisent **n = 290**.

- Génération Q&R : **Groq** `llama-3.1-8b-instant` — **aligné** sur baseline/RAG ; **biais circulaire** possible vs fine-tuning/RAFT (même famille 8B)
- Prompt avec règle obligatoire : `context` = citation verbatim du texte source (20-150 mots)
- Champ `recency_category` déduit de la date ISO du document : `récent` (≥2024), `intermédiaire` (2022-2023), `fondamental` (<2022)
- Fallback automatique : 400 premiers caractères si contexte trop court (<30 caractères)
- Dataset 100 % français : HAL remplace Arxiv (articles en anglais) pour une cohérence linguistique totale

### 5.3 Méthode 1 — Baseline
- Modèle : LLaMA 3.1 8B via Groq API, **aucun contexte externe**
- Prompt minimal en français, réponse directe depuis les paramètres du modèle
- Représente la limite basse : ce que LLaMA 3.1 8B sait déjà (avant décembre 2023)

### 5.4 Méthode 2 — RAG
- Embedding : `paraphrase-multilingual-MiniLM-L12-v2` (dimension 384, FR + EN)
- Index : FAISS `IndexFlatIP` (cosine similarity sur vecteurs normalisés L2)
- **Corpus indexé** : documents bruts complets (`wikipedia_technique.json`, `hal.json`, `lemonde.json`) découpés en chunks de 400 mots avec chevauchement de 100 mots
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
- Une fois les meilleurs hyperparamètres fixés, **un seul** entraînement sur tout `train.json`, puis évaluation **une fois** sur `test.json` via **`08_evaluation.ipynb`** (après exécution des notebooks `03`–`07`).

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
| **Exact Match (EM)** | % réponses identiques à la référence (normalisées) | Lexical |
| **F1 token-level** | Overlap de tokens entre prédiction et référence | Lexical |
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

*(Tableau principal et agrégation des six méthodes : `08_evaluation.ipynb` + les six fichiers `*_predictions.json`.)*

✅ *Avec HAL (articles 100 % français), le biais de la métrique d'hallucination présent avec Arxiv (contexte EN vs réponse FR) est éliminé. Le ROUGE-L(prédit, contexte source) est maintenant calculé dans la même langue pour tous les exemples.*

---

## 6. Résultats — ~500–700 mots

> **Synchronisation (campagne du 2026-05-04)** : agrégation **`results/final_report.json`** (`generated_at`: 2026-05-04T07:31:49) sur **n = 290** échantillons de test, avec les six JSON `results/*_predictions.json` présents. Les figures §6.6 sont exportées en PNG sous **`results/plots/`** (`fig1_global_metrics.png` … `fig10_ecological_impact.png`) lors de l’exécution de **`08_evaluation.ipynb`**. Après une nouvelle campagne, régénérer le rapport et recopier les tableaux ci-dessous si besoin.

### 6.1 Tableau comparatif global (toutes les méthodes)

| Méthode | EM (%) | F1 (%) | BERTScore (%) | BS CI 95 % | ROUGE-L (%) | METEOR (%) | Acc@BS 85 % | Hallucin. (%) | Latence (ms) | Vides (%) |
|---------|:------:|:------:|:-------------:|:----------:|:-----------:|:----------:|:-----------:|:---------------:|:------------:|:---------:|
| Baseline | 0.0 | 9.3 | 79.7 | [79.31, 80.08] | 8.8 | **22.8** | 8.3 | 85.1 | 565 | 0.0 |
| RAG | 0.0 | 18.3 | 83.1 | [82.57, 83.60] | 17.0 | 39.0 | 36.6 | 74.2 | 939 | 0.0 |
| Fine-tuné | 0.0 | 8.0 | 79.4 | [78.99, 79.68] | 7.3 | 21.8 | 4.1 | 90.2 | **17 661** | 0.0 |
| RAFT | 0.0 | 6.0 | 80.2 | [79.81, 80.52] | 6.0 | 15.8 | 5.2 | **91.0** | **20 625** | **1.4** |
| **Rerank** | 0.0 | **19.4** | **83.6** | [83.11, 84.10] | **18.2** | 38.0 | **37.9** | **73.9** | 621 | 0.0 |
| Function calling | 0.0 | 13.5 | 81.2 | [80.61, 81.74] | 12.6 | 30.8 | 21.7 | 80.2 | 927 | 0.0 |

*(**Vides** : part des réponses vides ou de moins de 3 caractères, mesurée sur les JSON de prédictions ; quasi nulle sauf RAFT sur ce run.)*

#### Faits saillants (même campagne)

- **Rerank** devance légèrement le **RAG** sur **F1**, **BERTScore**, **ROUGE-L** et le proxy d’**hallucination** ; le **RAG** reste meilleur sur **METEOR** (~+1 pt). Latence **Rerank &lt; RAG** (~621 ms vs ~939 ms) sur ce matériel — prompts probablement plus courts après re-classement des passages.
- **Baseline** : METEOR le plus haut hors méthodes retrieval — paradoxe fréquent quand les réponses longues et génériques chevauchent le vocabulaire de la référence sans être factuellement proches (**F1 / ROUGE** faibles).
- **Fine-tuné** et **RAFT** : **F1** et **ROUGE** sous baseline, **hallucination** la plus élevée, latences **×30–36** vs baseline (inférence locale + décodage long). **RAFT** : **METEOR** le plus bas des six ; quelques sorties vides.
- **Function calling** : scores intermédiaires ; **36 / 290** réponses avec `tool_native_failed_or_skipped` (repli FAISS sur la question sans outil Groq valide au tour 1, ou exception au tour 2 — voir `07_function_calling.ipynb`).

### 6.1 bis Patterns qualitatifs (exploration des `*_predictions.json`)

- **Ancrage lexical** : le **RAG** et le **Rerank** produisent souvent des formulations du type *« selon … »* ou des références explicites aux **extraits** (*« contexte »*, *« document »*), beaucoup plus que la **baseline** — cohérent avec l’injection de passages chunkés.
- **Longueur moyenne (mots / réponse)** : **Fine-tuné ~144** &gt; **Function calling ~130** &gt; **Baseline ~118** &gt; **RAG ~110** &gt; **Rerank ~74** &gt; **RAFT ~70** — les deux méthodes **rerank** et **RAFT** sont les plus courtes ; le **RAFT** combine des réponses très courtes avec quelques **vides**.
- **Hésitation** : préfixes du type *« je ne … »* (refus / manque d’information) plus fréquents en **baseline** qu’avec **RAG / rerank** — la récupération réduit les refus explicites sur ce jeu.
- **Aucune identité brute** : **0** paires où la réponse **RAG** est strictement identique à la **baseline** (les passages modifient systématiquement le texte généré sur les 290 items).

### 6.1 ter Article LaTeX

Un squelette **LaTeX** (résumé, six méthodes, tableau principal à compléter depuis `final_report.json`) est dans [`article_scientifique.tex`](article_scientifique.tex) ; à compiler avec `pdflatex` / `latexmk` et à enrichir (figures, bibliographie complète).

### 6.2 BERTScore par domaine × méthode (%)

*(Champ `dataset_type` : **juridique** inclus dans ce run.)*

| Méthode | technique | multisauts | temporel | juridique | Moy. 4 domaines |
|---------|-----------|------------|----------|-----------|-----------------|
| Baseline | 80.6 | 80.5 | 78.3 | 79.4 | 79.7 |
| RAG | 83.3 | 84.2 | 82.3 | 81.0 | 82.7 |
| Fine-tuné | 79.4 | 79.7 | 79.1 | 79.0 | 79.3 |
| RAFT | 81.0 | 80.4 | 79.5 | 79.3 | 80.0 |
| Rerank | 82.5 | 85.1 | 83.0 | 84.1 | 83.7 |
| Function calling | 80.4 | 82.6 | 80.3 | 82.2 | 81.4 |

### 6.3 BERTScore par type de question (%)

| Méthode | factuel | synthese | comprehension | simple (HAL) | complexe (HAL) |
|---------|---------|----------|---------------|--------------|----------------|
| Baseline | 79.5 | 79.2 | 79.1 | 78.5 | 83.0 |
| RAG | 83.1 | 82.6 | 81.7 | 82.1 | **86.8** |
| Fine-tuné | 78.2 | 79.7 | 80.4 | 77.5 | 82.3 |
| RAFT | 79.7 | 80.2 | 80.6 | 80.8 | 79.9 |
| Rerank | 82.9 | 83.0 | 82.7 | 83.5 | 87.1 |
| Function calling | 81.2 | 80.1 | 80.1 | 81.2 | 84.2 |

> **HAL** : pour le **RAG** et le **Rerank**, le **complexe** dépasse encore le **simple** (BERTScore) ; le **RAFT** inverse l’écart (complexe plus bas que simple).

### 6.4 Hallucination (proxy) par domaine (%)

| Méthode | technique | temporel | multisauts | juridique |
|---------|-----------|----------|-------------|-----------|
| Baseline | 80.5 | 88.1 | 86.4 | 82.9 |
| RAG | 71.6 | 76.6 | 74.1 | 72.9 |
| Fine-tuné | 90.2 | 91.2 | 89.6 | 87.6 |
| RAFT | 90.9 | 91.6 | 91.2 | 87.9 |
| Rerank | 73.2 | 77.8 | 72.0 | **65.8** |
| Function calling | 78.8 | 81.7 | 81.5 | 72.9 |

> Le **juridique** obtient le proxy d’hallucination **le plus bas** en **Rerank** sur ce run — effet probable d’une meilleure adéquation passage–question après cross-encoder.

### 6.5 Analyse latence

| Méthode | Latence moy. (ms) | Ratio vs Baseline |
|---------|-------------------|-------------------|
| Baseline | 565 | ×1.0 |
| Rerank | 621 | ×1.1 |
| Function calling | 927 | ×1.6 |
| RAG | 939 | ×1.7 |
| Fine-tuné | 17 661 | ×31.3 |
| RAFT | 20 625 | ×36.5 |

### 6.6 Figures produites par `08_evaluation.ipynb`

| Figure | Description |
|--------|-------------|
| Fig. 1 | Métriques globales (EM, F1, BERTScore, ROUGE-L) — barres groupées |
| Fig. 2 | BERTScore par strate temporelle × méthode |
| Fig. 3 | BERTScore par type de question × méthode |
| Fig. 4 | Latences (moy + p95) par méthode |
| Fig. 5 | Taux d'hallucination — global + par domaine |
| Fig. 6 | Trade-off qualité vs latence (scatter) |
| Fig. 7 | Heatmap BERTScore méthodes × dataset_type |
| Fig. 8 | Questions simples vs multi-sauts (HAL) |
| Fig. 9 | **Accuracy par seuil BERTScore** (≥80%, ≥85%, ≥90%) |
| Fig. 10 | **Impact écologique** — énergie (Wh), CO2 France vs mondial, CO2/prédiction |

---

## 7. Discussion — ~500–600 mots

### 7.1 Quelle méthode pour quel domaine ? *(croisé avec §6.2–6.4, campagne 2026-05-04)*

- **Rerank** et **RAG** : meilleurs compromis **qualité / hallucination** ; le **rerank** l’emporte sur **F1, BERTScore, ROUGE-L** et latence, le **RAG** sur **METEOR**.
- **Fine-tuné** : **ne surpasse pas la baseline** sur F1 / ROUGE / hallucination dans ce run — sensibilité au format, au décodage et à la quantification locale vs API.
- **RAFT** : **F1** et **METEOR** les plus bas des six ; hypothèse « contexte + distracteurs » **non** reflétée par un gain sur ce protocole et ce décodeur.
- **Function calling** : qualité **entre** baseline et RAG ; surveiller le **taux de repli** sans outil natif (~12 % sur ce run) et la latence vs RAG direct.
- **Baseline** : référence sans documents ; **METEOR** élevé malgré **F1** faible (voir §6.1 bis).

### 7.2 Impact de la complexité des questions

- Les questions **HAL complexes** restent favorisées par le **RAG** et le **Rerank** en BERTScore vs les **simples** (§6.3) — le retrieveur dense + re-classement semble aider le raisonnement multi-passages.
- Le **fine-tuning seul** sans contexte à l’inférence peine sur le raisonnement croisé si la réponse n’a pas été « mémorisée » dans les poids ; **RAFT** n’améliore pas cette situation sur les métriques rapportées.

### 7.3 Pourquoi le RAG sur documents bruts est supérieur
- Corpus FAISS sur extraits courts (50-100 mots) → chunks insuffisants pour répondre → RAG pire que baseline
- Corpus FAISS sur documents complets chunkés (400 mots) → RAG +2.60 pts BERTScore vs baseline (+4.22 pts sur temporel)
- La qualité du corpus indexé est le facteur déterminant de l'efficacité du RAG

### 7.4 Limites méthodologiques
- Dataset généré par **Groq / LLaMA 3.1 8B** : **pas d’indépendance** vis-à-vis de l’inférence baseline/RAG ; biais de **style** et de **couverture** du générateur.
- Taille **1 160 / 290** (Option B + juridique sur ce dépôt) — classes rares encore peu représentées pour certains croisements méthode × domaine.
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
| RAFT | Élevé (GPU) | Coûteuse | Quand le format « contexte + distracteurs » est central |
| RAG + rerank | Moyen (API + rerank local) | Réindexation + même rerank | Qualité retrieveur limite le RAG seul |
| Function calling | Moyen (API, boucles possibles) | Réindexation + schéma outil | Contrôle explicite des recherches |

---

## 8. Conclusion — ~200–300 mots

1. **Rappel** : comparer **six** stratégies (baseline, RAG, LoRA, RAFT, rerank, function calling) sur un socle **LLaMA 3.1 8B** et un jeu **100 % français** (Option B).
2. **Résultats (§6.1, campagne 2026-05-04, n = 290)** :
   - **Rerank** et **RAG** dominent sur **F1, BERTScore, ROUGE-L** et **hallucination** ; **METEOR** maximal encore côté **RAG** (et partiellement **baseline**).
   - **Fine-tuning LoRA** et **RAFT** : scores lexicaux et fidélité **sous** baseline, latence **~30–35×** ; RAFT avec **sorties vides** ponctuelles.
   - **Function calling** : intermédiaire ; repli sans outil natif sur **~12 %** des questions.
3. **Recommandations provisoires** :
   - Priorité **qualité + fraîcheur documentaire** → **RAG + rerank** si le cross-encoder est acceptable, sinon **RAG** seul (légèrement meilleur METEOR).
   - **Function calling** pour analyser un **flux agentif** ou des politiques de recherche conditionnelle, au prix d’une qualité moindre que le RAG direct sur ce benchmark.
4. **Perspectives** : évaluation humaine, ablations retrieveur, mesure du **taux d’erreurs d’outil**, retrieveurs plus riches (DPR/ColBERT).

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
| `01_scraping.ipynb` | §5.2 Constitution du corpus | `wikipedia_technique.json`, `hal.json`, `lemonde.json` |
| `02_dataset_builder.ipynb` | §5.2 Tableau statistiques | `train.json` (1200), `test.json` (300) — Groq `llama-3.1-8b-instant` |
| `03_baseline_rag.ipynb` | §5.3–5.4 Baseline + RAG + FAISS | `baseline_predictions.json`, `rag_predictions.json`, `models/faiss_index/` |
| `04_finetuning.ipynb` | §5.5 Fine-tuning LoRA | `finetuned_predictions.json`, `models/lora_adapter_ft/` |
| `05_raft.ipynb` | §5.5.bis RAFT | `raft_predictions.json`, `models/lora_adapter_raft/` |
| `06_rag_rerank.ipynb` | §5.6 RAG + reranking | `rerank_predictions.json` |
| `07_function_calling.ipynb` | §5.7 Function calling | `function_calling_predictions.json` |
| `08_evaluation.ipynb` | §6 Résultats | `final_report.json`, figures PNG |
| `article_scientifique.tex` | §6.1 bis + rédaction complète | PDF article ( compilation `pdflatex` ) |

---

## État du pipeline — Option B

- [ ] `01` — Scraping : Wikipedia (100), HAL.science FR (100 notices en **3 quotas d’année de dépôt** + complément), Actualités FR (100)
- [ ] `02` — Dataset builder Groq : 1200 train + 300 test (contextes verbatim, json_repair)
- [ ] `03` — Baseline (300/300) + RAG avec FAISS sur docs bruts chunkés 400 mots
- [ ] `04` — Fine-tuning → `finetuned_predictions.json` + `lora_adapter_ft/`
- [ ] `05` — RAFT → `raft_predictions.json` + `lora_adapter_raft/`
- [ ] `06` — RAG + reranking → `rerank_predictions.json`
- [ ] `07` — Function calling → `function_calling_predictions.json`
- [ ] `08` — Évaluation agrégée (6 méthodes) → `final_report.json` + figures
- [ ] Figures 1–10 sauvegardées et vérifiées
- [ ] Rédaction article : sections 1–9 complétées

---

## Green AI — Impact écologique estimé

### Hypothèses de calcul

| Paramètre | Baseline / RAG / rerank / FC (Groq + rerank local) | Fine-tuné / RAFT (Colab, GPU) |
|---|---|---|
| Hardware | H100 SXM (partagé) | NVIDIA A100 (Colab) |
| TDP effectif | ~50W (~700W / 14 slots) | ~150W (A100 partagé, hypothèse conservative) |
| PUE datacenter | 1.10 (Google infra) | 1.15 (Google Colab) |
| Intensité carbone | 52 gCO2/kWh (France) | 52 gCO2/kWh (France) |

> *Formule : Énergie (Wh) = latence_s × TDP_W × PUE / 3600 · CO2 (g) = Énergie_kWh × intensité*

### Estimations (300 prédictions de test — Option B)

| Méthode | Latence tot. | Énergie (Wh) | CO2 France (g) | CO2/pred (mg) | Ratio vs Baseline |
|---|---|---|---|---|---|
| Baseline | ~152 s | ~2.32 | ~0.121 | ~0.40 | ×1.0 |
| RAG | ~244 s | ~3.72 | ~0.193 | ~0.64 | ×1.6 |
| Fine-tuné | ~2009 s | ~40.3 | ~2.09 | ~6.97 | ×17.4 |
| RAFT | *(à mesurer)* | — | — | — | — |
| RAG + rerank | *(à mesurer)* | — | — | — | — |
| Function calling | *(à mesurer)* | — | — | — | — |

> *(Estimations extrapolées × 2.5 depuis les 120 prédictions initiales → 300 prédictions Option B, pour les lignes Groq/fine-tuné historiques.)*

> **Conclusion Green AI** : les méthodes **GPU** (fine-tuné, RAFT) restent coûteuses en latence/énergie vs **API** pour baseline/RAG. Le **reranking** ajoute du calcul local ; le **function calling** peut multiplier les allers-retours API — **mesurer** après le run complet (`08_evaluation.ipynb`).

### À mentionner dans la section Discussion
- Mesure directe avec `codecarbon` sur l'inférence GPU locale (fine-tuné, RAFT)
- Estimation par formule TDP × latence pour les appels API (Baseline/RAG)
- Biais : Groq utilise de l'énergie renouvelable → CO2 réel probablement inférieur
- Référence : Strubell et al. (2019), "Energy and Policy Considerations for Deep Learning in NLP"

---

## Résultats clés à retenir pour la rédaction

| Observation | Chiffre (run Option B actuel) | Section |
|-------------|-------------------------------|---------|
| **RAG = meilleure méthode globale** sur F1 / ROUGE / METEOR / BS / Acc@85 | F1 **27.4**, BS **85.0**, Acc@85 **44.7 %** | §6.1 |
| RAG = meilleure **fidélité** (proxy hallucination le plus bas) | **72.7 %** vs 86.4 baseline | §6.1 |
| Fine-tuné sous baseline sur F1 / ROUGE | F1 12.3 vs 12.7 ; ROUGE 11.1 vs 12.0 | §6.1 |
| Rerank / RAFT / FC : compléter le tableau §6.1 | lignes « — » → `final_report.json` | §6.1 |
| Alignement gabarit LoRA / RAFT (train vs inférence) | §5.5 / §5.5.bis | §7.4 |
| Dataset 100 % FR + HAL | pas de biais hallu. cross-langue | §5.2 |
| Groq = même modèle que baseline/RAG (génération Q&R) | limite §Discussion | §5.2 |
| Source LaTeX pour papier | [`article_scientifique.tex`](article_scientifique.tex) | §6.1 bis |
| Latence / CO2 : ordres de grandeur §6.5 / §7.6 | à actualiser après dernier run **`08_evaluation.ipynb`** | §6.5 |
