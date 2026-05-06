# Rapport d'Initiation à la Recherche (PGE)
## Intégration de nouvelles informations dans les LLMs : étude comparative de sept stratégies sur corpus français

**Auteur :** Abdoul Karim Coulibaly  
**Formation :** CESI — FISE A4 Data Science and IA (2025–2026)  
**Projet :** Initiation à la Recherche  
**Version :** Mai 2026  

---

## Résumé

L’actualisation des connaissances des grands modèles de langage constitue un enjeu central pour leur usage en contexte professionnel. Un modèle figé à sa date de coupure peut produire des réponses obsolètes ou insuffisamment fiables dès que les questions portent sur des faits récents, des normes évolutives ou des contenus métier spécifiques. Ce travail propose une comparaison expérimentale de sept stratégies d’intégration de connaissances sur un socle commun LLaMA 3.1 8B : baseline sans contexte, RAG dense, fine-tuning LoRA, FT+RAG, RAFT, RAG avec reranking cross-encoder, et function calling piloté par outil documentaire.

L’évaluation est réalisée sur un corpus intégralement francophone multi-source (Wikipedia, HAL, presse, code de la route), avec 1450 paires question-réponse et 290 échantillons de test pour la campagne de référence. Le protocole combine des métriques lexicales (F1, ROUGE-L, METEOR), sémantiques (BERTScore), de fidélité au contexte, de proxy d’hallucination, de latence et des intervalles de confiance bootstrap. Les résultats montrent une domination nette des stratégies retrieval-centric, en particulier RAG+Rerank (F1 = 36,47 ; BERTScore = 86,56), devant RAG dense (F1 = 32,82 ; BERTScore = 85,75). Les méthodes de fine-tuning local restent en retrait dans la configuration étudiée et présentent des latences élevées.

Au-delà du classement de méthodes, ce rapport met en évidence les compromis qualité/coût, discute les limites méthodologiques et propose des recommandations pour un déploiement réaliste en contexte francophone.

**Mots-clés :** LLM, RAG, LoRA, RAFT, reranking, function calling, hallucination, évaluation, français, IA appliquée.

---

## Abstract (EN)

Keeping large language models up-to-date is a major challenge for real-world deployments. Models frozen at training cutoff dates can become outdated and produce less reliable answers on recent or domain-evolving topics. This report presents a controlled comparison of seven knowledge integration strategies built on a common LLaMA 3.1 8B backbone: baseline (no external context), dense RAG, LoRA fine-tuning, FT+RAG, RAFT, RAG with cross-encoder reranking, and tool-based function calling.

Experiments are conducted on a fully French heterogeneous corpus (Wikipedia, HAL, news, traffic law), with 1,450 QA pairs and 290 test samples for the reference campaign. Evaluation combines lexical metrics (F1, ROUGE-L, METEOR), semantic similarity (BERTScore), context faithfulness, hallucination proxy, latency, and bootstrap confidence intervals. Results show strong dominance of retrieval-centric approaches, especially RAG+Rerank (F1 = 36.47; BERTScore = 86.56), followed by dense RAG (F1 = 32.82; BERTScore = 85.75). Local fine-tuning variants underperform in this setup and incur significantly higher latency.

The contribution is not limited to ranking methods: this report emphasizes quality/cost trade-offs, methodological limitations, and deployment-oriented recommendations for French-language AI systems.

---

## Table des matières

1. Introduction  
2. Contexte scientifique et état de l’art  
3. Problématique, hypothèses et contributions  
4. Données, corpus et préparation  
5. Méthodes comparées  
6. Protocole d’évaluation et reproductibilité  
7. Résultats expérimentaux  
8. Discussion  
9. Implications opérationnelles et recommandations  
10. Limites et menaces à la validité  
11. Perspectives de recherche  
12. Conclusion générale  
13. Références (sélection)  
14. Annexes (figures, tableaux, checklists)

---

## 1. Introduction

### 1.1 Motivation

Les grands modèles de langage ont franchi un cap en termes de polyvalence, mais leur usage en production reste confronté à une limite structurante : la coupure temporelle des connaissances. Un assistant basé uniquement sur ses poids internes peut répondre avec fluidité tout en restant incorrect dès que la question dépasse son horizon d’entraînement. Dans les organisations, ce phénomène a des conséquences concrètes : erreurs d’information, perte de confiance, et coûts de vérification humaine.

Dans ce contexte, intégrer des connaissances nouvelles sans réentraînement complet est devenu un axe majeur de recherche appliquée. Plusieurs familles de solutions existent, mais elles sont rarement comparées à cadre expérimental constant sur des données francophones hétérogènes. Ce rapport répond précisément à ce besoin.

### 1.2 Contexte du projet

Le projet s’inscrit dans le cadre d’une initiation à la recherche orientée ingénierie. L’objectif n’est pas uniquement de proposer une implémentation technique, mais de construire une démarche scientifique complète : question de recherche explicite, hypothèses formulées, protocole reproductible, mesure multi-critères, discussion des limites et recommandations fondées.

### 1.3 Question de recherche

La question centrale est la suivante :

> Quelle stratégie d’intégration de connaissances offre le meilleur compromis entre qualité de réponse, fidélité documentaire, coût et latence sur un corpus francophone hétérogène, à socle modèle constant ?

### 1.4 Objectifs du rapport

Ce document vise quatre objectifs :

- formaliser rigoureusement la démarche méthodologique ;
- rapporter les résultats quantitatifs de manière interprétable ;
- expliciter les compromis et limites ;
- fournir une base académique longue, convertible en `.docx`, compatible avec un rendu de 15–20 pages après insertion de figures et annexes.

---

## 2. Contexte scientifique et état de l’art

### 2.1 LLMs et date de coupure

Les architectures Transformer ont permis l’essor des LLMs modernes. Cependant, ces modèles ne « voient » pas le monde en temps réel : leur connaissance est arrêtée à la date de fin de préentraînement. Cette propriété explique la difficulté à traiter correctement des questions d’actualité, de réglementation, ou de veille scientifique.

### 2.2 Hallucination et fiabilité

Le phénomène d’hallucination désigne la génération d’énoncés plausibles mais factuellement incorrects. Il est amplifié lorsque le modèle répond sans support documentaire externe. D’un point de vue applicatif, la réduction d’hallucination est aussi importante que l’amélioration des scores de similarité.

### 2.3 RAG et mémoire externe

Le paradigme Retrieval-Augmented Generation délègue la mémoire à une base documentaire indexée et réserve au LLM le rôle de synthèse/génération. En pratique, le pipeline type est : requête -> embedding -> recherche top-k -> assemblage de contexte -> génération. L’intérêt majeur est la mise à jour rapide de la connaissance par réindexation, sans réentraînement du modèle générateur.

### 2.4 Fine-tuning efficace : LoRA et RAFT

LoRA permet d’adapter un modèle via un nombre réduit de paramètres. RAFT prolonge cette logique en entraînant le modèle avec des contextes récupérés (et distracteurs), pour mieux apprendre l’utilisation de la preuve documentaire. Ces approches sont séduisantes mais sensibles à la qualité des données, aux formats de prompt et à la stabilité de l’inférence.

### 2.5 Reranking cross-encoder

Le reranking affine les candidats issus d’un retrieveur dense en évaluant finement la pertinence requête-passage. Il augmente le coût de récupération, mais améliore souvent la qualité de contexte injecté, donc la qualité de réponse.

### 2.6 Function calling et architectures agentiques

Le function calling permet au modèle de décider d’appeler un outil (ici `search_docs`) avant de répondre. Cette logique est proche des architectures agentiques : raisonnement, sélection d’action, exécution d’outil, puis synthèse finale. Elle apporte de la flexibilité mais introduit des risques supplémentaires (erreurs de schéma, appels inutiles, oubli d’appel).

### 2.7 Lacunes identifiées dans la littérature appliquée

Dans les pratiques de benchmark appliqué, trois points reviennent souvent :

- comparaisons non homogènes (modèles différents) ;
- faible couverture francophone ;
- analyses trop globales, peu ventilées par type de question/domaine.

Ce rapport cherche à réduire ces biais par un protocole unifié et des analyses détaillées.

---

## 3. Problématique, hypothèses et contributions

### 3.1 Problématique opérationnelle

À budget contraint, faut-il prioriser un pipeline retrieval (RAG/Rerank), investir dans du fine-tuning local, ou combiner les deux ? La réponse dépend du compromis entre qualité et coût d’exploitation.

### 3.2 Hypothèses de travail

- **H1** : les méthodes retrieval (RAG, Rerank) améliorent significativement la fidélité et réduisent l’hallucination par rapport à la baseline.
- **H2** : le reranking apporte un gain mesurable vs RAG dense seul.
- **H3** : les méthodes de fine-tuning local peuvent rester sous-optimales si l’alignement train/inférence est imparfait.
- **H4** : le function calling atteint une performance intermédiaire, avec potentiel d’amélioration via meilleure orchestration outil.

### 3.3 Contributions principales

1. **Corpus français hétérogène** structuré en quatre domaines.
2. **Comparaison de sept méthodes** sur socle LLaMA 3.1 8B.
3. **Évaluation multi-axe** (qualité, fidélité, hallucination, latence, CI bootstrap).
4. **Discussion scientifique et opérationnelle** orientée décision de déploiement.

---

## 4. Données, corpus et préparation

### 4.1 Sources de données

Le corpus est composé de contenus de natures complémentaires :

- Wikipedia FR (connaissances techniques/encyclopédiques) ;
- HAL (contenus scientifiques, raisonnement multi-sauts) ;
- presse francophone (dimension temporelle) ;
- code de la route (dimension juridique/normative).

### 4.2 Construction des paires QA

Les paires question-réponse sont générées et nettoyées selon un protocole homogène, avec stratification des types de questions (factuelles, simples, complexes, synthèse, compréhension). Le jeu total atteint 1450 paires, dont 290 en test pour la campagne de référence.

### 4.3 Qualité des données

Le pipeline applique un contrôle de format et de cohérence des champs. La qualité des données reste un facteur limitant majeur, en particulier pour les méthodes fine-tunées qui sont très sensibles au bruit de supervision.

### 4.4 Métadonnées analytiques

Les champs `dataset_type` et `recency_category` permettent des analyses croisées essentielles : ce ne sont pas des détails techniques, mais des variables explicatives pour interpréter les écarts de performance.

### 4.5 Limites data-centric

- hétérogénéité de style des sources ;
- risque de paraphrase proche des réponses attendues ;
- équilibre de difficulté non parfaitement uniforme.

Ces limites sont explicitement prises en compte dans la discussion.

---

## 5. Méthodes comparées

### 5.1 Baseline (sans contexte)

Référence minimale. Le modèle répond à partir de ses poids internes uniquement.

### 5.2 RAG dense

Recherche top-k dans l’index FAISS à partir d’un embedding de requête, puis génération conditionnée. C’est la première méthode retrieval de référence.

### 5.3 Fine-tuning LoRA

Adaptation locale du modèle avec LoRA. Objectif : internaliser des patterns de réponse ciblés sur le corpus.

### 5.4 FT+RAG

Combinaison d’un modèle fine-tuné et d’une récupération documentaire à l’inférence. Théoriquement, cette approche peut cumuler spécialisation et grounding.

### 5.5 RAFT

Entraînement avec contexte récupéré et distracteurs pour apprendre à exploiter une preuve documentaire en situation bruitée.

### 5.6 RAG + Reranking

Après récupération dense initiale, un cross-encodeur réordonne les passages. Cette méthode vise à injecter un contexte plus pertinent et plus compact.

### 5.7 Function calling

Le modèle peut appeler l’outil `search_docs` selon le besoin. Le pipeline évalue la qualité finale obtenue après ce cycle outil.

### 5.8 Justification de comparaison

Les sept méthodes couvrent les grandes familles de stratégies connues : mémoire interne seule, mémoire externe, adaptation paramétrique, combinaison hybride et orchestration agentique. Leur confrontation dans un protocole commun donne une base de décision robuste.

---

## 6. Protocole d’évaluation et reproductibilité

### 6.1 Métriques retenues

- **F1** : recouvrement lexical ;
- **BERTScore** : proximité sémantique contextualisée ;
- **ROUGE-L** : structure textuelle ;
- **METEOR** : alignement lexical/sémantique ;
- **Fidélité** : proximité réponse/contexte ;
- **Hallucination proxy** : 100 - fidélité (ou formulation équivalente du pipeline) ;
- **Latence** : coût temporel moyen.

### 6.2 Fiabilisation statistique

Le protocole utilise un bootstrap (1000 tirages, seed 42) pour produire des intervalles de confiance à 95 %. Cette étape limite l’interprétation abusive d’écarts faibles.

### 6.3 Granularité d’analyse

Les résultats sont ventilés :

- par méthode ;
- par domaine (`technique`, `multisauts`, `temporel`, `juridique`) ;
- par type de question ;
- par seuils BERTScore (80, 85, 90).

### 6.4 Artefacts et traçabilité

La campagne est consolidée dans `results/final_report.json` et peut être régénérée via l’ordre des notebooks. Cette structuration répond à une exigence clé de recherche appliquée : pouvoir reproduire et auditer les conclusions.

---

## 7. Résultats expérimentaux

### 7.1 Résultats globaux (campagne du 2026-05-06)

| Méthode | F1 | BERTScore | ROUGE-L | METEOR | Fidélité | Hallucination | Latence ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 16,26 | 81,43 | 13,71 | 34,53 | 16,46 | 83,54 | 548,8 |
| RAG | 32,82 | 85,75 | 28,67 | 52,35 | 31,77 | 68,23 | 598,7 |
| Fine-tuné | 14,75 | 81,36 | 11,56 | 35,49 | 12,72 | 87,28 | 14 217,3 |
| FT+RAG | 14,61 | 80,97 | 11,66 | 35,77 | 13,08 | 86,92 | 18 975,8 |
| RAFT | 11,88 | 81,24 | 10,08 | 26,30 | 11,62 | 88,38 | 21 224,4 |
| Rerank | **36,47** | **86,56** | **32,11** | **56,14** | **33,83** | **66,17** | 569,6 |
| Function-calling | 23,89 | 83,38 | 20,88 | 44,55 | 24,17 | 75,83 | 1 077,5 |

### 7.2 Lecture des performances

Le résultat principal est la supériorité de **Rerank**, suivi de **RAG**. La baseline confirme que la mémoire interne seule est insuffisante sur ce benchmark. Le function calling se situe entre baseline et retrieval classique.

### 7.3 Résultats contre-intuitifs

Les méthodes fine-tunées locales ne dépassent pas les pipelines retrieval dans cette configuration. Ce point est important scientifiquement : il évite une conclusion dogmatique selon laquelle « fine-tuning > RAG ». Le résultat dépend du protocole, du dataset, et des réglages.

### 7.4 Robustesse et significativité

Les CI bootstrap soutiennent la stabilité de l’ordre des meilleures méthodes sur les indicateurs majeurs. Pour des écarts plus faibles entre méthodes proches, une campagne additionnelle serait utile.

### 7.5 Analyse par domaine

Les ventilations par domaine montrent que les approches retrieval maintiennent une meilleure stabilité inter-domaines. Le domaine temporel reste généralement plus difficile (documents évolutifs, formulations plus variées).

### 7.6 Analyse par type de question

Les questions de synthèse/compréhension exposent davantage les limites des méthodes faibles en grounding documentaire. Les approches retrieval atténuent ces difficultés mais n’éliminent pas totalement la perte de performance.

### 7.7 Hallucination et fidélité

La relation est cohérente : meilleure fidélité contextuelle -> hallucination proxy plus faible. Rerank fournit ici le meilleur équilibre.

### 7.8 Latence et coût

Les méthodes locales (Fine-tuné, FT+RAG, RAFT) ont des latences très supérieures. Dans un contexte temps réel, cet écart peut être rédhibitoire sans optimisation matérielle majeure.

---

## 8. Discussion

### 8.1 Validation des hypothèses

- **H1 validée** : RAG et Rerank dépassent nettement la baseline.
- **H2 validée** : Rerank améliore RAG sur les métriques principales.
- **H3 plausible** : les performances faibles des méthodes fine-tunées suggèrent un enjeu d’alignement train/inférence.
- **H4 partiellement validée** : function calling apporte un gain, mais pas au niveau de RAG/Rerank dans ce setup.

### 8.2 Pourquoi retrieval domine ici

Le retrieval apporte une mise à jour de connaissance explicite et contrôlable. Le reranking améliore encore la pertinence du contexte. Dans un corpus multi-source et non trivial, ce mécanisme semble plus robuste que l’absorption paramétrique seule.

### 8.3 Pourquoi fine-tuning peut échouer malgré son potentiel

- sensibilité au format des données ;
- possible décalage instruction/contexte entre train et test ;
- volume d’entraînement possiblement insuffisant ;
- coût d’itération élevé à cause de la latence locale.

### 8.4 Place du function calling

Le function calling n’est pas une simple variante de RAG : c’est une logique décisionnelle. Son efficacité dépend de la qualité de gouvernance des appels d’outils. Il devient plus intéressant dans des architectures multi-outils et des agents planificateurs.

### 8.5 Impact pédagogique et scientifique

Le projet démontre une vraie démarche de recherche appliquée : questionnement, protocole, comparaison, interprétation critique, et projection opérationnelle.

---

## 9. Implications opérationnelles et recommandations

### 9.1 Scénario entreprise : priorité qualité/réactivité

Recommandation : **RAG + Reranking**.  
Justification : meilleur score global, latence maîtrisée, maintenance par réindexation.

### 9.2 Scénario entreprise : priorité simplicité

Recommandation : **RAG dense**.  
Justification : excellent compromis performance/complexité.

### 9.3 Scénario agentique

Recommandation : **Function calling + observabilité renforcée**.  
Justification : flexible mais nécessite garde-fous et tests de robustesse.

### 9.4 Scénario spécialisation métier lourde

Recommandation : **Fine-tuning ciblé**, uniquement si :

- dataset de supervision de haute qualité,
- budget GPU et itérations de calibration,
- protocole d’évaluation strict post-déploiement.

### 9.5 Gouvernance qualité à mettre en place

- monitoring des métriques en continu ;
- audits d’hallucination sur cas critiques ;
- versionnement des index et prompts ;
- stratégie de rollback.

---

## 10. Limites et menaces à la validité

### 10.1 Limites internes

- dépendance à une campagne de référence ;
- biais possible lié au générateur de données ;
- sensibilité aux hyperparamètres non entièrement explorés.

### 10.2 Limites externes

- généralisation à d’autres langues à confirmer ;
- généralisation à d’autres familles de modèles à tester ;
- transposabilité secteur régulé nécessitant validation additionnelle.

### 10.3 Menaces de mesure

- les métriques automatiques ne capturent pas toute la vérité factuelle ;
- la corrélation entre BERTScore et utilité métier n’est pas parfaite ;
- le proxy d’hallucination reste une approximation.

---

## 11. Perspectives de recherche

1. Ablation retrieveur/reranker (impact top-k, chunking, overlap).
2. Alignement prompt train/inférence pour FT et RAFT.
3. Évaluation humaine experte sur sous-ensemble critique.
4. Multi-agent function calling avec politiques d’appel.
5. Étude coût-carbone plus fine par architecture matérielle.
6. Réplication cross-lingue pour mesurer l’effet langue.

---

## 12. Conclusion générale

Cette étude confirme, dans le cadre expérimental retenu, que les stratégies retrieval-centric sont les plus performantes et pragmatiques pour intégrer des connaissances nouvelles dans un LLM francophone sans réentraînement complet. Le reranking apporte un gain concret sur le RAG dense. Les approches de fine-tuning local, bien que théoriquement pertinentes, requièrent des conditions méthodologiques plus strictes pour devenir compétitives ici.

Le résultat le plus important n’est pas uniquement le classement des méthodes, mais la consolidation d’une démarche de recherche appliquée reproductible et argumentée. Cette base est immédiatement exploitable pour une soutenance, un rapport long, et des itérations futures orientées industrialisation.

---

## 13. Références (sélection)

- Vaswani et al., 2017 — *Attention Is All You Need*.
- Lewis et al., 2020 — *Retrieval-Augmented Generation*.
- Johnson et al., 2019 — *FAISS*.
- Reimers & Gurevych, 2019 — *Sentence-BERT*.
- Hu et al., 2022 — *LoRA*.
- Dettmers et al., 2023 — *QLoRA*.
- Hong et al., 2024 — *RAFT*.
- Zhang et al., 2020 — *BERTScore*.
- Ji et al., 2023 — Hallucination in NLG.

*(La bibliographie complète est disponible dans `article_scientifique.tex`.)*

---

## 14. Annexes prêtes à compléter

### Annexe A — Figures à insérer

- Figure A1 : architecture complète du pipeline.
- Figure A2 : tableau radar multi-métriques par méthode.
- Figure A3 : performances par domaine.
- Figure A4 : performances par type de question.
- Figure A5 : hallucination et fidélité comparées.
- Figure A6 : latence comparative API vs local.
- Figure A7 : impact écologique estimé.

### Annexe B — Tableaux détaillés

- Tableau B1 : CI95 par métrique et par méthode.
- Tableau B2 : scores par seuil BERTScore.
- Tableau B3 : comparaison domaines x méthodes.

### Annexe C — Checklist soutenance (PGE)

- Problématique claire en 30 secondes.
- Explication des 7 méthodes en 2 minutes.
- Résultats clés (Rerank > RAG > FC > baseline > FT variants).
- Limites assumées et pistes concrètes.
- Lien explicite avec projet professionnel.

---

## Note de formatage pour la version `.docx`

Avec ce volume textuel, une mise en page académique standard (police 11 ou 12, interligne 1,15 à 1,5, marges 2,5 cm) donne déjà un document substantiel. L’ajout des figures, tableaux détaillés, légendes interprétatives et annexes méthodologiques doit permettre d’atteindre naturellement la plage **15–20 pages**, voire davantage selon la densité graphique.
