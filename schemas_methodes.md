# Schémas des paradigmes d’intégration d’information

Document autonome pour le projet **Article_scientifique** : corpus français (Option B), base génératrice **LLaMA 3.1 8B** alignée sur **`llama-3.1-8b-instant`** (Groq) pour les méthodes API, modèle quantifié + adaptateurs **LoRA** (Unsloth) pour l’adaptation locale.

Les **sept** stratégies du README se regroupent ici en **cinq familles** :

1. **Baseline** — aucune récupération.  
2. **RAG (+ reranking)** — récupération à l’inférence (dense puis optionnellement cross-encoder).  
3. **Fine-tuning LoRA (+ RAFT)** — adaptation des poids (train supervisé ; RAFT = variante « contexte + distracteurs » au train).  
4. **FT+RAG** — **combinaison explicite** : même index FAISS que le RAG + **inférence** avec l’adaptateur `lora_adapter_ft` (notebook **`08_ft_plus_rag.ipynb`**).  
5. **Function calling** — décision structurée d’appeler la recherche documentaire avant synthèse.

L’**agrégation des métriques** sur les **sept** fichiers `*_predictions.json` (dont **`ft_rag_predictions.json`**) se fait dans **`09_evaluation.ipynb`**. Une **évaluation subjective** complémentaire (LLM-as-judge) est décrite dans **`10_llm_as_judge.ipynb`** / **`11_llm_judge_analysis.ipynb`**.

---

## 1. Baseline (sans récupération)

**Notebook :** `03_baseline_rag.ipynb` (branche baseline)  
**Entrée :** `data/processed/test.json` (question seule)  
**Sortie :** `results/baseline_predictions.json`

```mermaid
flowchart LR
  subgraph Entrées
    T["test.json<br/>question"]
  end
  subgraph Inférence
    G["Groq API<br/>llama-3.1-8b-instant"]
  end
  subgraph Sortie
    P["baseline_predictions.json<br/>predicted_answer"]
  end
  T --> G --> P
```

**Idée :** aucun passage documentaire ; le modèle ne s’appuie que sur ses poids figés (cut-off connu).

---

## 2. RAG dense (+ extension reranking)

**Notebooks :** `03_baseline_rag.ipynb` (RAG) · `06_rag_rerank.ipynb` (rerank)  
**Index :** `models/faiss_index/` (`index.faiss`, `metadata.json`)  
**Embeddings :** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`  
**Sorties :** `results/rag_predictions.json` · `results/rerank_predictions.json`

```mermaid
flowchart TB
  subgraph Corpus_indexé
    M["metadata.json<br/>chunks texte + titres"]
    I["FAISS index.faiss"]
  end
  subgraph RAG_dense_03
    Q["Question test.json"]
    E["Bi-encodeur<br/>MiniLM multilingue"]
    K["Top-k voisins<br/>FAISS"]
    G1["Groq<br/>prompt + passages"]
    R1["rag_predictions.json"]
    Q --> E --> K
    M --> K
    I --> K
    K --> G1 --> R1
  end
  subgraph Extension_rerank_06
    K2["Top-M candidats<br/>FAISS"]
    CE["Cross-encoder<br/>mmarco-mMiniLMv2-L12"]
    K3["Top-k après score<br/>paire query-passage"]
    G2["Groq<br/>mêmes passages rerankés"]
    R2["rerank_predictions.json"]
    Q --> E --> K2
    M --> K2
    I --> K2
    K2 --> CE --> K3 --> G2 --> R2
  end
```

**Idée :** le RAG injecte des extraits récupérés **à l’inférence** ; le rerank **re-classe** les passages avant génération pour réduire le bruit du retrieval dense.

---

## 3. Fine-tuning LoRA (+ variante RAFT)

**Notebooks :** `04_finetuning.ipynb` (LoRA standard) · `05_raft.ipynb` (RAFT)  
**Entrée train :** `data/processed/train.json`  
**Sorties modèles :** `models/lora_adapter_ft/` · `models/lora_adapter_raft/`  
**Inférence :** locale (Unsloth + base quantifiée LLaMA 3.1 8B)  
**Sorties prédictions :** `results/finetuned_predictions.json` · `results/raft_predictions.json`

```mermaid
flowchart TB
  subgraph Entraînement
    TR["train.json<br/>Q / R / contexte"]
    U["Unsloth + LoRA<br/>sur base 8B"]
    AD1["Adaptateur FT<br/>lora_adapter_ft/"]
    AD2["Adaptateur RAFT<br/>lora_adapter_raft/"]
    TR --> U
    U --> AD1
    U -.->|"données style RAFT<br/>pseudo-documents"| AD2
  end
  subgraph Inférence_locale
    Q["Question test.json"]
    B["Base LLaMA 3.1 8B<br/>quantifiée"]
    B --> AD1
    B --> AD2
    AD1 --> GEN1["Génération<br/>finetuned_predictions.json"]
    AD2 --> GEN2["Génération<br/>raft_predictions.json"]
    Q --> GEN1
    Q --> GEN2
  end
```

**Idée :** les connaissances cibles sont **incorporées dans les poids** (adaptateurs) plutôt que rappelées par index ; le **RAFT** entraîne avec des contextes proches du train pour imiter un comportement RAG au moment de l’entraînement.

---

## 4. FT+RAG — LoRA fine-tuné **+** retrieval FAISS (inférence)

**Notebook :** `08_ft_plus_rag.ipynb`  
**Index :** identique au RAG (`models/faiss_index/`) — chunks **200 / 50**.  
**Modèle :** charge **`lora_adapter_ft`** (produit par **`04_finetuning.ipynb`**).  
**Sortie :** `results/ft_rag_predictions.json`

```mermaid
flowchart LR
  subgraph Retrieval
    Q["Question"]
    E["MiniLM + FAISS"]
    C["Top-k passages"]
  end
  subgraph Gen_GPU
    L["Base 8B quantifiée<br/>+ lora_adapter_ft"]
    O["Décodage Alpaca<br/>Instruction / Context / Input"]
  end
  Q --> E --> C --> O
  L --> O
  O --> R["ft_rag_predictions.json"]
```

**Idée :** combiner **passages récupérés** (comme le RAG) avec un **générateur adapté par LoRA**. Ce schéma est **distinct** du RAFT : ici le retrieveur est celui du pipeline RAG à l’inférence, pas un entraînement « pseudo-RAG » isolé.

---

## 5. Function calling (outil de recherche)

**Notebook :** `07_function_calling.ipynb`  
**Même index que le RAG :** FAISS + `metadata.json` + bi-encodeur  
**Sortie :** `results/function_calling_predictions.json`

```mermaid
sequenceDiagram
  participant U as Utilisateur / test.json
  participant L as Groq LLM tour 1
  participant C as Client Python
  participant F as FAISS + embeddings
  participant L2 as Groq LLM tour 2

  U->>L: question + schéma outil search_docs
  L->>C: tool_call search_docs query=...
  C->>F: encodage requête + recherche top-k
  F->>C: extraits texte
  C->>L2: historique + résultat tool sans redéclarer tools
  L2->>U: réponse finale predicted_answer
```

**Équivalent flux batch (vue pipeline) :**

```mermaid
flowchart LR
  Q["Question"]
  G1["Groq + tools<br/>décision search_docs"]
  EXE["Python : tool_search_docs"]
  F["FAISS + MiniLM"]
  G2["Groq synthèse<br/>sans tools"]
  OUT["function_calling_predictions.json"]
  Q --> G1 --> EXE --> F
  F --> EXE
  EXE --> G2 --> OUT
```

**Idée :** le modèle **décide** (en théorie) quand lancer une recherche et avec quelle requête ; l’exécution reste **côté Python** comme pour le RAG, mais le contrôle est **structuré** (appel d’outil) plutôt qu’un prompt unique « passages collés ».

---

## Tableau de correspondance rapide

| Schéma | Fichiers / artefacts clés | Rôle du corpus à l’inférence |
|--------|---------------------------|------------------------------|
| 1. Baseline | Groq seul | Aucun |
| 2. RAG (+ rerank) | FAISS, metadata, Groq | Passages récupérés (dense puis optionnellement rerankés) |
| 3. LoRA / RAFT | train.json, adaptateurs `lora_adapter_ft` / `_raft` | Souvent question seule en éval (pas de FAISS) |
| **4. FT+RAG** | **`08_ft_plus_rag.ipynb`**, `lora_adapter_ft`, FAISS | Passages **puis** génération LoRA locale |
| 5. Function calling | FAISS + outil `search_docs` | Passages après décision d’outil |

Pour l’**agrégation des métriques** sur **les sept stratégies**, utiliser **`09_evaluation.ipynb`** et les JSON listés dans le README (incluant **`ft_rag_predictions.json`**).
