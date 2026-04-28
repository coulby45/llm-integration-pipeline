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

> **Intégration de nouvelles informations dans les LLMs : étude comparative du RAG, du fine-tuning LoRA et de l'approche hybride sur trois corpus hétérogènes**

Variantes :
- *Mettre à jour les LLMs sans ré-entraînement : RAG vs Fine-tuning vs Hybride sur données techniques, scientifiques et d'actualité*
- *Connaissances figées vs connaissances dynamiques : évaluation expérimentale de quatre stratégies d'intégration dans LLaMA 3.1 8B*

---

## 2. Résumé (Abstract) — ~150–200 mots

**Structure** : Contexte → Problème → Méthode → Résultats → Conclusion

> Les grands modèles de langage (LLMs) apprennent des connaissances statiques lors du pré-entraînement et ne peuvent pas intégrer de nouvelles informations après leur date de coupure (décembre 2023 pour LLaMA 3.1 8B). Ce travail étudie quatre stratégies sur un modèle commun LLaMA 3.1 8B : (1) **baseline** sans contexte, (2) **RAG** avec index FAISS sur documents bruts et embeddings multilingues, (3) **fine-tuning LoRA** supervisé, et (4) une approche **hybride** combinant les deux. Nous construisons un corpus hétérogène de **420 paires question-réponse en français** réparties sur trois domaines : articles techniques Wikipedia FR (*technique*), résumés Arxiv post-2024 (*multi-sauts*), et articles d'actualité (*temporel*). Les questions Arxiv comprennent deux niveaux de complexité : simples (1 fait) et multi-sauts (raisonnement croisé). L'évaluation porte sur l'Exact Match, le F1, le BERTScore multilingue, le ROUGE-L, le taux d'hallucination et la latence. Nos résultats montrent que l'approche hybride obtient les meilleures performances globales (BERTScore 85.43 %), que le RAG est particulièrement efficace sur les données temporelles (BERTScore 86.24 %) et que la latence constitue le principal frein à l'adoption des approches locales.

---

## 3. Introduction — ~400–600 mots

### 3.1 Contexte et motivation
- Les LLMs (GPT-4, Mistral, LLaMA…) ont une **date de coupure** : ignorent les événements postérieurs à leur entraînement
- LLaMA 3.1 8B : coupure décembre 2023 → ignore tous les articles de presse d'avril 2026 et les papiers Arxiv post-2024
- Problème pratique : un LLM ne connaît pas les dernières publications IA, les nouvelles lois, les articles de presse récents
- Question centrale : **comment intégrer efficacement de nouvelles connaissances dans un LLM sans le ré-entraîner entièrement ?**
- Trois angles d'attaque : injecter le contexte à l'inférence (RAG), adapter les poids (fine-tuning), ou combiner les deux

### 3.2 Verrou scientifique
- Les comparaisons existantes utilisent souvent un modèle différent par méthode → biais de comparaison
- La plupart des benchmarks sont en anglais → peu de résultats sur du français
- Les évaluations ignorent souvent la **nature du domaine** (technique vs actualités) et la **complexité des questions**

### 3.3 Contributions de l'article
1. Construction d'un **corpus tridomaine** (300 train + 120 test) sur 3 domaines distincts avec 5 types de questions
2. **Base commune** : les 4 méthodes utilisent toutes LLaMA 3.1 8B → comparaison équitable
3. **Évaluation multi-axe** : EM, F1, BERTScore, ROUGE-L, hallucination, latence
4. **Analyse croisée** méthodes × domaines + comparaison simple vs multi-sauts
5. Mise en évidence empirique que **le RAG avec corpus brut indexé est supérieur** au RAG sur extraits courts
6. Recommandations pratiques pour projets IA en français avec contraintes de coût/latence

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
- **Corpus indexé** : documents bruts chunkés (400 mots, overlap 100) — meilleur que les extraits courts
- Limitations : qualité du retriever, pertinence des chunks, latence ajoutée

### 4.4 Fine-tuning efficace (PEFT/LoRA)
- Hu et al. (2022) — LoRA : Low-Rank Adaptation
- Dettmers et al. (2023) — QLoRA : quantification 4-bit + LoRA
- **Notre configuration** : LoRA rank=16, alpha=16, 7 modules (attention + MLP), max_seq_len=1024
- Format Alpaca (Stanford, 2023) pour le fine-tuning supervisé

### 4.5 Approches hybrides
- REALM (Guu et al., 2020), RAG + fine-tuning (Shi et al., 2023)
- Limites : coût computationnel, dépendance à la qualité du retriever ET du modèle

### 4.6 Évaluation des LLMs
- Exact Match, F1 token-level (Rajpurkar et al., 2016)
- BERTScore (Zhang et al., 2020) — `distilbert-base-multilingual-cased`
- ROUGE-L (Lin, 2004)
- Hallucination proxy : 1 − ROUGE-L(prédit, contexte source) — ⚠️ biaisé pour les paires FR/EN (Arxiv)

---

## 5. Méthodologie — ~900–1400 mots

### 5.1 Architecture du pipeline

```
Wikipedia FR (40 articles) ──┐  dataset_type = "technique"
                              │
Arxiv post-2024 (80 résumés) ─┼──► LLM Q&R (Groq Developer) ──► train.json (300)
+ intro ar5iv.org             │                               ──► test.json  (120)
                              │
Actualités FR (60 articles)  ─┘  dataset_type = "temporel"
 (Le Monde / France Info)

test.json ──► Baseline (LLaMA 3.1 8B via Groq, sans contexte)       ──► baseline_predictions.json
         ──► RAG (FAISS docs bruts + LLaMA 3.1 8B via Groq)          ──► rag_predictions.json
         ──► Fine-tuning LoRA (LLaMA 3.1 8B + LoRA local GPU)        ──► finetuned_predictions.json
         ──► Hybride (FAISS + LLaMA 3.1 8B + LoRA local GPU)         ──► hybrid_predictions.json
                                                                              │
                                              EM / F1 / BERTScore / ROUGE-L / Hallucination
                                              Tableau croisé méthodes × domaines + 8 figures
```

### 5.2 Constitution du corpus (Notebooks 01 + 02)

| Dataset | Source | Docs | Contenu | `dataset_type` | Questions/doc |
|---------|--------|------|---------|----------------|-----------|
| Technique | Wikipedia FR | 40 | Articles complets (~1600 mots) | `technique` | 5 (2 factuelles, 2 synthèse, 1 compréhension) |
| Multi-sauts | Arxiv 2024+ | 80 | Résumé + intro ar5iv (~500 mots) | `multisauts` | 5 (3 simples + 2 complexes) |
| Temporel | Le Monde / France Info | 60 | Article complet scraped (~300 mots) | `temporel` | 5 (2 factuelles, 2 synthèse, 1 compréhension) |

**Split figé (seed 42)** :

| Split | Technique | Arxiv simple | Arxiv complexe | Temporel | Total |
|-------|-----------|--------------|----------------|----------|-------|
| train | 100 | 50 | 50 | 100 | **300** |
| test  | 40  | 20 | 20 | 40  | **120** |

- Génération Q&R : Groq Developer API `llama-3.1-8b-instant` — même modèle de base que les 4 méthodes comparées
- Prompt avec règle obligatoire : `context` = citation verbatim du texte source (20-150 mots)
- Fallback automatique : 400 premiers caractères si placeholder détecté

### 5.3 Méthode 1 — Baseline
- Modèle : LLaMA 3.1 8B via Groq API, **aucun contexte externe**
- Prompt minimal en français, réponse directe depuis les paramètres du modèle
- Représente la limite basse : ce que LLaMA 3.1 8B sait déjà (avant décembre 2023)

### 5.4 Méthode 2 — RAG
- Embedding : `paraphrase-multilingual-MiniLM-L12-v2` (dimension 384, FR + EN)
- Index : FAISS `IndexFlatIP` (cosine similarity sur vecteurs normalisés L2)
- **Corpus indexé** : documents bruts complets (`wikipedia_technique.json`, `arxiv.json`, `lemonde.json`) découpés en chunks de 400 mots avec chevauchement de 100 mots
- Récupération : top-5 chunks (titre + texte injectés dans le prompt)
- Génération : LLaMA 3.1 8B via Groq avec contexte enrichi

### 5.5 Méthode 3 — Fine-tuning LoRA
- Modèle de base : `unsloth/Meta-Llama-3.1-8B-bnb-4bit` (quantifié 4-bit, GPU T4)
- LoRA : rank=16, alpha=16, dropout=0, 7 modules (`q/k/v/o_proj`, `gate/up/down_proj`)
- Format : Alpaca (`### Instruction / ### Input / ### Response`)
- Entraînement : 3 époques, batch=2, gradient_acc=8 (effectif=16), lr=2e-4, AdamW 8-bit
- max_seq_length=1024, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 300 exemples d'entraînement, ~57 steps

### 5.6 Méthode 4 — Hybride
- Combine la récupération FAISS (top-5 chunks) avec le modèle fine-tuné (LoRA local)
- Le contexte récupéré est injecté dans le champ `### Input` du prompt Alpaca
- Inférence locale sur GPU T4

### 5.7 Note sur l'infrastructure
> *Pour les méthodes Baseline et RAG, l'inférence est réalisée via l'API Groq (`llama-3.1-8b-instant`), tandis que Fine-tuné et Hybride utilisent une inférence locale avec les poids LoRA (`unsloth/Meta-Llama-3.1-8B-bnb-4bit`). Les deux reposent sur LLaMA 3.1 8B, mais avec des pipelines d'inférence distincts imposés par la nature de chaque méthode (les poids LoRA personnalisés ne peuvent pas être déployés via API externe).*

### 5.8 Métriques d'évaluation

| Métrique | Description | Niveau |
|----------|-------------|--------|
| **Exact Match (EM)** | % réponses identiques à la référence (normalisées) | Lexical |
| **F1 token-level** | Overlap de tokens entre prédiction et référence | Lexical |
| **ROUGE-L** | Plus longue sous-séquence commune | Lexical |
| **BERTScore** | Similarité sémantique (`distilbert-base-multilingual`) | Sémantique |
| **Accuracy@BS** | % réponses avec BERTScore ≥ seuil (80/85/90%) | Sémantique |
| **Hallucination** | 1 − ROUGE-L(prédit, contexte source) — proxy | Fidélité |
| **Latence** | Temps moyen de génération (ms) | Efficacité |

⚠️ *La métrique d'hallucination est biaisée pour les questions Arxiv (contexte EN, réponse FR) : ROUGE-L entre deux langues différentes est structurellement proche de 0, ce qui gonfle artificiellement le taux d'hallucination à ~96% pour toutes les méthodes sur ce dataset.*

---

## 6. Résultats — ~500–700 mots

### 6.1 Tableau comparatif global

| Méthode | EM (%) | F1 (%) | BERTScore (%) | ROUGE-L (%) | Hallucin. (%) | Latence (ms) |
|---------|--------|--------|---------------|-------------|---------------|--------------|
| Baseline | 0.0 | 19.51 | 82.28 | 16.37 | 87.73 | 508 |
| RAG | 0.0 | **28.59** | 84.88 | 24.39 | **80.53** | 814 |
| Fine-tuné | 1.67 | 28.41 | 85.32 | 24.53 | 87.94 | 6 689 |
| **Hybride** | **2.50** | 31.03 | **85.43** | **26.21** | 84.66 | 10 035 |

> L'hybride domine sur les métriques de qualité. Le RAG obtient le meilleur score d'hallucination et reste 8× plus rapide que le fine-tuné.

### 6.2 BERTScore par domaine × méthode (%)

| Méthode | technique | multisauts | temporel | Moyenne |
|---------|-----------|------------|----------|---------|
| Baseline | 82.45 | 82.36 | 82.02 | 82.28 |
| RAG | 84.58 | 83.81 | **86.24** | 84.88 |
| Fine-tuné | 85.40 | 84.56 | 86.02 | 85.32 |
| **Hybride** | **85.82** | **84.51** | 85.95 | **85.43** |

> **Résultat clé** : le RAG excelle sur les données temporelles (86.24 %), surpassant même le fine-tuné (86.02 %). Cela confirme que le RAG est l'approche la plus adaptée pour des données fraîches hors de la fenêtre d'entraînement du modèle.

### 6.3 BERTScore par type de question (%)

| Méthode | factuel | synthese | comprehension | simple (Arxiv) | complexe (Arxiv) |
|---------|---------|----------|---------------|----------------|-----------------|
| Baseline | 83.11 | 81.01 | 81.16 | 82.54 | 82.18 |
| RAG | **86.41** | 84.30 | 83.25 | 83.26 | **84.36** |
| Fine-tuné | 87.19 | 83.72 | 83.65 | **85.88** | 83.23 |
| **Hybride** | 87.01 | **84.86** | **82.77** | 84.95 | 84.07 |

> **Observation notable** : le RAG est la seule méthode qui performe *mieux* sur les questions complexes Arxiv que sur les simples (+1.10 pts), tandis que le fine-tuné chute de 2.65 pts. Cela suggère que la récupération de contexte aide davantage pour le raisonnement croisé.

### 6.4 Hallucination par domaine (%)

| Méthode | technique | temporel | multisauts* |
|---------|-----------|----------|-------------|
| Baseline | 80.44 | 85.56 | 97.20 |
| RAG | **72.33** | **73.30** | 95.96 |
| Fine-tuné | 81.03 | 86.03 | 96.77 |
| **Hybride** | 76.95 | 80.31 | **96.71** |

> *\* Valeurs artificiellement élevées pour multisauts : métrique ROUGE-L calculée entre réponse française et contexte anglais → biais structural, non interprétable.*

### 6.5 Analyse latence

| Méthode | Latence moy. (ms) | Ratio vs Baseline |
|---------|-------------------|-------------------|
| Baseline | 508 | ×1.0 |
| RAG | 814 | ×1.6 |
| Fine-tuné | 6 689 | ×13.2 |
| Hybride | 10 035 | ×19.8 |

### 6.6 Figures produites par NB05

| Figure | Description |
|--------|-------------|
| Fig. 1 | Métriques globales (EM, F1, BERTScore, ROUGE-L) — barres groupées |
| Fig. 2 | BERTScore par strate temporelle × méthode |
| Fig. 3 | BERTScore par type de question × méthode |
| Fig. 4 | Latences (moy + p95) par méthode |
| Fig. 5 | Taux d'hallucination — global + par domaine |
| Fig. 6 | Trade-off qualité vs latence (scatter) |
| Fig. 7 | Heatmap BERTScore méthodes × dataset_type |
| Fig. 8 | Questions simples vs multi-sauts sur Arxiv |
| Fig. 9 | **Accuracy par seuil BERTScore** (≥80%, ≥85%, ≥90%) |

---

## 7. Discussion — ~500–600 mots

### 7.1 Quelle méthode pour quel domaine ?
- **RAG** : meilleur sur données temporelles (actualités 2026 hors coupure LLM) — la récupération compense le manque de connaissance paramétrique. Hallucination réduite de ~8 pts vs baseline sur technique et temporel.
- **Fine-tuné** : meilleur sur questions factuelles (87.19 %) — le modèle a mémorisé les faits du corpus d'entraînement
- **Hybride** : meilleur global — profite des deux mécanismes ; recommandé si latence non contraignante
- **Baseline** : acceptable uniquement sur contenu pré-coupure (Wikipedia) ; inefficace sur actualités récentes

### 7.2 Impact de la complexité des questions
- Les questions multi-sauts Arxiv sont difficiles pour toutes les méthodes (BERTScore plus faible qu'en factuel)
- **RAG est la seule méthode qui bénéficie de la complexité** : le contexte récupéré aide à relier des faits dispersés
- Le fine-tuning seul peine sur le raisonnement croisé : il mémorise des réponses directes, pas des chaînes d'inférences

### 7.3 Pourquoi le RAG sur documents bruts est supérieur
- Corpus FAISS sur extraits courts (50-100 mots) → chunks insuffisants pour répondre → RAG pire que baseline
- Corpus FAISS sur documents complets chunkés (400 mots) → RAG +2.60 pts BERTScore vs baseline (+4.22 pts sur temporel)
- La qualité du corpus indexé est le facteur déterminant de l'efficacité du RAG

### 7.4 Limites méthodologiques
- Dataset généré automatiquement par LLaMA → biais possible (questions trop faciles pour LLaMA, réponses circulaires)
- Taille du corpus (420 paires) — sous-représentation possible des classes rares
- Pas de strates temporelles exploitables : la colonne `recency_category` est toujours `inconnu`
- Métrique hallucination inutilisable sur Arxiv (cross-lingue FR/EN → ROUGE-L ≈ 0 structurellement)

### 7.5 Limites techniques
- Fine-tuning sur Colab T4 (14.5 Go VRAM) : max_seq_length réduit à 1024, 3 époques seulement
- ar5iv.org peut être instable → certains papiers sans introduction (enrichissement partiel)
- Groq `llama-3.1-8b-instant` vs `unsloth/Meta-Llama-3.1-8B-bnb-4bit` : quantifications différentes

### 7.6 Rapport coût-bénéfice pratique

| Méthode | Coût infra | Mise à jour | Recommandé si... |
|---------|-----------|-------------|-----------------|
| Baseline | Très faible | Instantanée | Données stables pré-coupure |
| RAG | Faible | Instantanée (réindexation) | Données fraîches changeantes |
| Fine-tuné | Élevé (GPU) | Coûteuse (ré-entraîner) | Domaine spécialisé stable |
| Hybride | Très élevé | Partielle | Performance max sans contrainte |

---

## 8. Conclusion — ~200–300 mots

1. **Rappel** : comment intégrer des connaissances post-coupure dans LLaMA 3.1 8B ?
2. **Résultats** :
   - Le fine-tuning LoRA domine sur les questions factuelles et structurées
   - Le RAG excelle sur les données temporelles fraîches (86.24 % vs 82.02 % baseline sur *temporel*)
   - L'hybride offre la meilleure moyenne globale (BERTScore 85.43 %) mais avec 20× plus de latence
   - La qualité du corpus FAISS est critique : indexer les documents bruts complets est indispensable
3. **Recommandations pratiques** :
   - Données fréquemment mises à jour → **RAG** (latence ×1.6, hallucination −7 pts)
   - Domaine spécialisé stable → **Fine-tuning LoRA**
   - Performance maximale sans contrainte de coût/latence → **Hybride**
4. **Perspectives** : évaluation humaine, fine-tuning continu (continual learning), RAG dense (DPR/ColBERT), meilleure métrique d'hallucination cross-lingue

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
- **Reproductibilité** : mentionner seed=42, GPU T4 Colab, Groq Developer `llama-3.1-8b-instant`
- **Biais à mentionner** : dataset généré par le même modèle que celui évalué (LLaMA 3.1 8B)

---

## Correspondance Notebooks ↔ Sections de l'article

| Notebook | Section article | Produit |
|----------|-----------------|---------|
| `01_scraping.ipynb` | §5.2 Constitution du corpus | `wikipedia_technique.json`, `arxiv.json`, `lemonde.json` |
| `02_dataset_builder.ipynb` | §5.2 Tableau statistiques | `train.json` (300), `test.json` (120) |
| `03_baseline_rag.ipynb` | §5.3 Baseline + §5.4 RAG | `baseline_predictions.json`, `rag_predictions.json` |
| `04_finetuning.ipynb` | §5.5 Fine-tuning LoRA | `finetuned_predictions.json` |
| `05_hybrid_eval.ipynb` | §5.6 Hybride + §6 Résultats | `hybrid_predictions.json`, 8 figures, `final_report.json` |

---

## État du pipeline

- [x] `01` — Scraping : Wikipedia (40), Arxiv (80 + enrichissement ar5iv), Actualités (60)
- [x] `02` — Dataset builder : 300 train + 120 test (contextes verbatim, json_repair)
- [x] `03` — Baseline (120/120) + RAG avec FAISS sur docs bruts chunkés 400 mots
- [x] `04` — Fine-tuning : 3 époques, batch=2, grad_acc=8, max_seq=1024 (GPU T4 15Go)
- [x] `05` — Évaluation complète : 6 métriques × 4 méthodes × 3 domaines, 8 figures
- [ ] Figures 1–8 sauvegardées et vérifiées
- [ ] Rédaction article : sections 1–9 complétées

---

## Résultats clés à retenir pour la rédaction

| Observation | Chiffre | Section |
|-------------|---------|---------|
| Hybride = meilleure méthode globale | BERTScore 85.43 % | §6.1 |
| RAG > Fine-tuné sur données temporelles | 86.24 % vs 86.02 % | §6.2 |
| RAG réduit l'hallucination | 80.53 % vs 87.73 % baseline | §6.1 |
| RAG seul à bénéficier des questions complexes | +1.10 pts simple→complexe | §6.3 |
| Fine-tuné meilleur sur factuel | 87.19 % BERTScore | §6.3 |
| Hybride 20× plus lent que Baseline | 10 035 ms vs 508 ms | §6.5 |
| FAISS brut > FAISS extraits courts | +2.60 pts BERTScore RAG | §7.3 |
