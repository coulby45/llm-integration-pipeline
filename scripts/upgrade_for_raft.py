import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(r"C:/Users/Utilisateur/Desktop/Idee_random/Article_scientifique")


def load_nb(name: str):
    with open(ROOT / name, "r", encoding="utf-8") as f:
        return json.load(f)


def save_nb(name: str, nb):
    with open(ROOT / name, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)


def set_cell_source(nb, cell_id: str, source_text: str):
    lines = source_text.split("\n")
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    for c in nb["cells"]:
        if c.get("id") == cell_id:
            c["source"] = src
            if c["cell_type"] == "code":
                c["execution_count"] = None
                c["outputs"] = []
            return
    raise ValueError(f"Cell id not found: {cell_id}")


def add_cell_before(nb, before_id: str, cell: dict):
    for i, c in enumerate(nb["cells"]):
        if c.get("id") == before_id:
            nb["cells"].insert(i, cell)
            return
    raise ValueError(f"before_id not found: {before_id}")


def add_cell_after(nb, after_id: str, cell: dict):
    for i, c in enumerate(nb["cells"]):
        if c.get("id") == after_id:
            nb["cells"].insert(i + 1, cell)
            return
    raise ValueError(f"after_id not found: {after_id}")


def patch_04():
    nb = load_nb("04_finetuning.ipynb")

    set_cell_source(
        nb,
        "cell-imports",
        """# Imports de toutes les bibliothèques nécessaires
import os
import json
import time
import re
import string
import numpy as np
import torch
from tqdm.notebook import tqdm
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

# Chemins Drive
PROCESSED_PATH = os.path.join(BASE_PATH, 'data', 'processed')
MODELS_PATH    = os.path.join(BASE_PATH, 'models', 'lora_adapter_ft')
RESULTS_PATH   = os.path.join(BASE_PATH, 'results')

for path in [MODELS_PATH, RESULTS_PATH]:
    os.makedirs(path, exist_ok=True)

# LLaMA 3.1 8B — même modèle que Baseline/RAG → comparaison équitable
BASE_MODEL   = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
LORA_RANK    = 32
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
TARGET_MODS  = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_SEQ_LEN  = 1024

# Hyperparamètres robustes
NUM_EPOCHS   = 6
BATCH_SIZE   = 2
LR           = 8e-5
VAL_RATIO    = 0.10
SEED         = 42

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print("Configuration :")
print(f"  Modèle de base   : {BASE_MODEL}")
print(f"  LoRA rank/alpha  : {LORA_RANK}/{LORA_ALPHA}")
print(f"  Époques          : {NUM_EPOCHS}")
print(f"  Learning rate    : {LR}")
print(f"  Validation ratio : {VAL_RATIO}")
print(f"  LoRA output path : {MODELS_PATH}")""",
    )

    set_cell_source(
        nb,
        "cell-format",
        """# Formatage pour le SFTTrainer
ALPACA_TEMPLATE = (
    "### Instruction: Réponds à cette question en français, de façon concise et fidèle.\\n"
    "### Input: {question}\\n"
    "### Response: {answer}"
)

# Pour améliorer le fine-tuning seul, on active le contexte gold en entraînement.
# L'inférence 04 reste sans retrieval externe (pas de FAISS), donc méthode "fine-tunée seule".
USE_CONTEXT_IN_TRAINING = True
MAX_CONTEXT_CHARS = 1200

CONTEXT_TEMPLATE = (
    "### Instruction: Réponds à cette question en te basant sur le contexte ci-dessous.\\n"
    "### Context: {context}\\n"
    "### Input: {question}\\n"
    "### Response: {answer}"
)

def format_training_example(item):
    q = item.get('question', '').strip()
    a = item.get('answer', '').strip()
    ctx = str(item.get('context', '')).strip()
    if USE_CONTEXT_IN_TRAINING and len(ctx) >= 40:
        ctx = ctx[:MAX_CONTEXT_CHARS]
        text = CONTEXT_TEMPLATE.format(context=ctx, question=q, answer=a)
    else:
        text = ALPACA_TEMPLATE.format(question=q, answer=a)
    return {"text": text}

formatted_train = [format_training_example(item) for item in train_data if item.get('question') and item.get('answer')]
hf_dataset = Dataset.from_list(formatted_train)

split = hf_dataset.train_test_split(test_size=VAL_RATIO, seed=SEED, shuffle=True)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"Dataset SFT prêt : total={len(hf_dataset)} train={len(train_dataset)} val={len(eval_dataset)}")
print(f"  USE_CONTEXT_IN_TRAINING = {USE_CONTEXT_IN_TRAINING}")
print("\\nAperçu du premier exemple :")
print(train_dataset[0]['text'][:500])""",
    )

    set_cell_source(
        nb,
        "cell-trainer",
        """# Configuration du SFTTrainer avec validation (checkpoint meilleur modèle)
training_args = TrainingArguments(
    output_dir="/content/tmp_checkpoints_ft",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=8,
    warmup_steps=30,
    max_grad_norm=1.0,
    learning_rate=LR,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=SEED,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    dataset_num_proc=2,
    args=training_args,
)

print("Trainer configuré. Démarrage de l'entraînement...")""",
    )

    set_cell_source(
        nb,
        "cell-inference-setup",
        """# Passage en mode inférence
FastLanguageModel.for_inference(model)

_ALPACA_STOP_MARKERS = ["\\n### Instruction:", "\\n### Input:", "\\n### Response:", "\\n### Context:"]
_NOISE_PATTERNS = [
    r"pour sauvegarder cet article.*$",
    r"connectez-vous.*$",
    r"abonnez-vous.*$",
    r"cookies?.*$",
]

def _cleanup_answer(text):
    out = (text or "").strip()
    for marker in _ALPACA_STOP_MARKERS:
        if marker in out:
            out = out.split(marker)[0].strip()
    for p in _NOISE_PATTERNS:
        out = re.sub(p, "", out, flags=re.IGNORECASE | re.MULTILINE).strip()
    out = re.sub(r"(\\b.{1,40}?\\b)(\\s+\\1){2,}", r"\\1", out, flags=re.IGNORECASE)
    if len(out) > 40 and out[-1] not in ".!?":
        cut = max(out.rfind("."), out.rfind("!"), out.rfind("?"))
        if cut > 40:
            out = out[:cut+1]
    return out.strip()

def _confidence_from_scores(gen_outputs):
    # score proxy: moyenne des probas max de chaque token généré
    if not getattr(gen_outputs, "scores", None):
        return None
    probs = [torch.softmax(s[0], dim=-1).max().item() for s in gen_outputs.scores]
    if not probs:
        return None
    return round(float(np.mean(probs)), 4)

def generate_answer(question, max_new_tokens=360):
    prompt = (
        "### Instruction: Réponds à cette question en français, de façon concise et fidèle.\\n"
        f"### Input: {question}\\n"
        "### Response:"
    )
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        start = time.time()
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=20,
                do_sample=False,
                repetition_penalty=1.16,
                no_repeat_ngram_size=4,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        latency_ms = round((time.time() - start) * 1000)
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = gen.sequences[0][prompt_len:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        answer = _cleanup_answer(answer)
        confidence = _confidence_from_scores(gen)
        truncated = len(new_tokens) >= max_new_tokens
        return answer, latency_ms, confidence, truncated
    except Exception as e:
        print(f"  [ERROR] Génération : {e}")
        return "", 0, None, False

print("Modèle en mode inférence. Lancement sur le test set...")""",
    )

    set_cell_source(
        nb,
        "cell-inference-run",
        """# Inférence sur tout le test set avec barre de progression
finetuned_predictions = []

for item in tqdm(test_data, desc="Inférence fine-tuné"):
    question = item.get('question', '')
    true_answer = item.get('answer', '')

    predicted, latency, confidence, truncated = generate_answer(question)

    finetuned_predictions.append({
        "pair_id": item.get('pair_id', ''),
        "question": question,
        "predicted_answer": predicted,
        "true_answer": true_answer,
        "latency_ms": latency,
        "confidence": confidence,
        "truncated": truncated,
        "method": "finetuned"
    })

latencies = [p['latency_ms'] for p in finetuned_predictions if p['latency_ms'] > 0]
trunc_rate = 100 * np.mean([p.get("truncated", False) for p in finetuned_predictions]) if finetuned_predictions else 0.0
conf_vals = [p["confidence"] for p in finetuned_predictions if p.get("confidence") is not None]
print(f"\\nInférence terminée : {len(finetuned_predictions)} prédictions")
print(f"Latence moyenne    : {np.mean(latencies):.0f} ms" if latencies else "Latence : N/A")
print(f"Taux troncature    : {trunc_rate:.1f}%")
print(f"Confiance moyenne  : {np.mean(conf_vals):.3f}" if conf_vals else "Confiance : N/A")""",
    )

    add_cell_after(
        nb,
        "cell-save-predictions",
        {
            "cell_type": "code",
            "id": "cell-ft-quick-metrics",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                ln + "\n"
                for ln in """# Mini-évaluation immédiate (utile quand on exécute les notebooks séparément)
_rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

def _norm(t):
    t = (t or "").lower().strip()
    t = t.translate(str.maketrans('', '', string.punctuation))
    return " ".join(t.split())

def _f1(p, g):
    pt, gt = _norm(p).split(), _norm(g).split()
    if not pt or not gt:
        return 0.0
    inter = len(set(pt) & set(gt))
    if inter == 0:
        return 0.0
    pr, rc = inter / len(pt), inter / len(gt)
    return 2 * pr * rc / (pr + rc)

preds = [x.get("predicted_answer","") for x in finetuned_predictions]
refs  = [x.get("true_answer","") for x in finetuned_predictions]
em = np.mean([int(_norm(p) == _norm(g)) for p, g in zip(preds, refs)]) * 100
f1 = np.mean([_f1(p, g) for p, g in zip(preds, refs)]) * 100
rl = np.mean([_rouge.score(g if g else " ", p if p else " ")["rougeL"].fmeasure for p, g in zip(preds, refs)]) * 100
try:
    _, _, F = bert_score_fn(preds if preds else [" "], refs if refs else [" "], lang="fr",
                            model_type="distilbert-base-multilingual-cased", batch_size=32, verbose=False)
    bs = float(F.mean()) * 100
except Exception as e:
    print(f"[WARN] BERTScore indisponible: {e}")
    bs = 0.0
print("\\n--- Mini-évaluation Fine-tuné (NB04) ---")
print(f"EM={em:.1f}% | F1={f1:.1f}% | ROUGE-L={rl:.1f}% | BERTScore={bs:.1f}%")
print(f"Résultat sauvegardé: {finetuned_path}")""".split("\n")
            ][:-1]
            + [
                """# Mini-évaluation immédiate (utile quand on exécute les notebooks séparément)
_rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

def _norm(t):
    t = (t or "").lower().strip()
    t = t.translate(str.maketrans('', '', string.punctuation))
    return " ".join(t.split())

def _f1(p, g):
    pt, gt = _norm(p).split(), _norm(g).split()
    if not pt or not gt:
        return 0.0
    inter = len(set(pt) & set(gt))
    if inter == 0:
        return 0.0
    pr, rc = inter / len(pt), inter / len(gt)
    return 2 * pr * rc / (pr + rc)

preds = [x.get("predicted_answer","") for x in finetuned_predictions]
refs  = [x.get("true_answer","") for x in finetuned_predictions]
em = np.mean([int(_norm(p) == _norm(g)) for p, g in zip(preds, refs)]) * 100
f1 = np.mean([_f1(p, g) for p, g in zip(preds, refs)]) * 100
rl = np.mean([_rouge.score(g if g else " ", p if p else " ")["rougeL"].fmeasure for p, g in zip(preds, refs)]) * 100
try:
    _, _, F = bert_score_fn(preds if preds else [" "], refs if refs else [" "], lang="fr",
                            model_type="distilbert-base-multilingual-cased", batch_size=32, verbose=False)
    bs = float(F.mean()) * 100
except Exception as e:
    print(f"[WARN] BERTScore indisponible: {e}")
    bs = 0.0
print("\\n--- Mini-évaluation Fine-tuné (NB04) ---")
print(f"EM={em:.1f}% | F1={f1:.1f}% | ROUGE-L={rl:.1f}% | BERTScore={bs:.1f}%")
print(f"Résultat sauvegardé: {finetuned_path}")""",
            ],
        },
    )

    set_cell_source(
        nb,
        "cell-summary",
        """# Affichage complet du résumé de ce qui a été produit
latencies = [p['latency_ms'] for p in finetuned_predictions if p['latency_ms'] > 0]
conf_vals = [p["confidence"] for p in finetuned_predictions if p.get("confidence") is not None]

print("=" * 65)
print("RÉSUMÉ — Notebook 04 : Fine-tuning")
print("=" * 65)
print(f"\\nModèle de base      : {BASE_MODEL}")
print(f"LoRA rank / alpha   : {LORA_RANK} / {LORA_ALPHA}")
print(f"Modules cibles      : {TARGET_MODS}")
print(f"Époques             : {NUM_EPOCHS}")
print(f"Exemples entraînem. : train={len(train_dataset)} / val={len(eval_dataset)}")
try:
    print(f"Loss finale         : {trainer_stats.training_loss:.4f}")
    print(f"Durée entraînem.    : {train_duration/60:.1f} min")
except Exception:
    pass
print(f"Prédictions test    : {len(finetuned_predictions)}")
print(f"Latence moy. infér. : {np.mean(latencies):.0f} ms" if latencies else "Latence : N/A")
print(f"Confiance moyenne   : {np.mean(conf_vals):.3f}" if conf_vals else "Confiance : N/A")

print(f"\\nFichiers produits :")
for fpath in [finetuned_path]:
    try:
        print(f"  {fpath}  ({os.path.getsize(fpath)/1024:.1f} Ko)")
    except Exception:
        print(f"  {fpath}")
print(f"  {MODELS_PATH}/  (adaptateur LoRA)")

print("\\n✔ Notebook 04 terminé. Lancez 07_raft.ipynb (LoRA hybride spécialisé) puis 05_hybrid_eval.ipynb.")
print("=" * 65)""",
    )

    save_nb("04_finetuning.ipynb", nb)


def patch_05():
    nb = load_nb("05_hybrid_eval.ipynb")

    set_cell_source(
        nb,
        "cell-md-title",
        """# Notebook 05 — Méthode hybride (LoRA spécialisé + FAISS)

**Objectif** : charger l'adaptateur LoRA **spécialisé hybride** (sortie de `07_raft.ipynb`), exécuter l'inférence hybride sur le test set, puis sauvegarder `results/hybrid_predictions.json`.

Ce notebook n'entraîne pas de LoRA : il applique le LoRA hybride sur le retrieval FAISS.
L'évaluation globale 4/5 méthodes est dans `06_evaluation.ipynb`.""",
    )

    set_cell_source(
        nb,
        "cell-imports",
        """# Imports nécessaires
import os, json, time, string, re
import numpy as np
import pandas as pd
import faiss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter, defaultdict
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer as _rouge_scorer
from sentence_transformers import SentenceTransformer
from unsloth import FastLanguageModel
from tqdm.notebook import tqdm
import torch

PROCESSED_PATH = os.path.join(BASE_PATH, 'data', 'processed')
MODELS_PATH    = os.path.join(BASE_PATH, 'models', 'lora_adapter_hybrid')
FAISS_PATH     = os.path.join(BASE_PATH, 'models', 'faiss_index')
RESULTS_PATH   = os.path.join(BASE_PATH, 'results')
PLOTS_PATH     = os.path.join(BASE_PATH, 'results', 'plots')

for path in [RESULTS_PATH, PLOTS_PATH]:
    os.makedirs(path, exist_ok=True)

BASE_MODEL  = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K       = 5
MAX_SEQ_LEN = 2048

print("Configuration chargée.")
print(f"  LoRA hybride : {MODELS_PATH}")
print(f"  Index FAISS  : {FAISS_PATH}")
print(f"  Résultats    : {RESULTS_PATH}")""",
    )

    set_cell_source(
        nb,
        "cell-load-lora",
        """# Chargement du modèle de base + adaptateur LoRA hybride
print(f"Chargement du modèle de base : {BASE_MODEL}")
print("(Téléchargement si nécessaire, peut prendre 5-10 min)")

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    print("Modèle de base chargé.")
except Exception as e:
    raise RuntimeError(f"Échec chargement modèle : {e}")

try:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, MODELS_PATH)
    print(f"Adaptateur LoRA hybride chargé depuis : {MODELS_PATH}")
except Exception as e:
    raise RuntimeError(
        f"Échec chargement LoRA hybride : {e}\\n"
        "→ Exécutez d'abord 07_raft.ipynb pour entraîner l'adaptateur spécialisé."
    )

FastLanguageModel.for_inference(model)
print("Modèle en mode inférence.")""",
    )

    set_cell_source(
        nb,
        "cell-hybrid-fn",
        """# Fonctions de récupération FAISS et de génération hybride (version robuste)
HYBRID_TEMPLATE = (
    "### Instruction: Réponds à cette question en te basant sur le contexte fourni. "
    "Ignore les éléments non éditoriaux (paywall, connexion, cookies, navigation).\\n"
    "### Context: {context_block}\\n"
    "### Input: {question}\\n"
    "### Response:"
)
_STOP_MARKERS = ["\\n### Instruction:", "\\n### Input:", "\\n### Context:", "\\n### Response:"]
_NOISE_PATTERNS = [
    r"pour sauvegarder cet article.*$",
    r"connectez-vous.*$",
    r"abonnez-vous.*$",
    r"cookies?.*$",
    r"créez un compte.*$",
    r"menu\\s+principal.*$",
]

def _clean_context_text(text):
    out = (text or "").strip()
    for pat in _NOISE_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE | re.MULTILINE).strip()
    out = re.sub(r"\\s+", " ", out)
    return out

def _clean_answer(text):
    out = (text or "").strip()
    for marker in _STOP_MARKERS:
        if marker in out:
            out = out.split(marker)[0].strip()
    for pat in _NOISE_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE | re.MULTILINE).strip()
    out = re.sub(r"(\\b.{1,40}?\\b)(\\s+\\1){2,}", r"\\1", out, flags=re.IGNORECASE)
    if len(out) > 40 and out[-1] not in ".!?":
        cut = max(out.rfind("."), out.rfind("!"), out.rfind("?"))
        if cut > 40:
            out = out[:cut + 1]
    return out.strip()

def _confidence_from_scores(gen_outputs):
    if not getattr(gen_outputs, "scores", None):
        return None
    probs = [torch.softmax(s[0], dim=-1).max().item() for s in gen_outputs.scores]
    return round(float(np.mean(probs)), 4) if probs else None

def retrieve_top_k(question, k=TOP_K):
    q_emb = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_emb, k)
    chunks = [corpus_meta[i] for i in indices[0] if i < len(corpus_meta)]
    return chunks, scores[0].tolist()

def generate_hybrid(question, max_new_tokens=320):
    try:
        chunks, scores = retrieve_top_k(question, k=TOP_K)
        context_lines = []
        for c in chunks:
            raw = c.get('text', c.get('context', ''))
            cleaned = _clean_context_text(raw)[:700]
            if len(cleaned) >= 30:
                context_lines.append(f"[{c.get('title','')[:60]}] {cleaned}")
        context_block = "\\n\\n".join(context_lines) if context_lines else "Contexte indisponible."

        prompt = HYBRID_TEMPLATE.format(context_block=context_block, question=question)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        start = time.time()
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=24,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        latency_ms = round((time.time() - start) * 1000)
        prompt_len = inputs["input_ids"].shape[1]
        out_tokens = gen.sequences[0][prompt_len:]
        answer = tokenizer.decode(out_tokens, skip_special_tokens=True).strip()
        answer = _clean_answer(answer)
        confidence = _confidence_from_scores(gen)
        truncated = len(out_tokens) >= max_new_tokens

        chunk_ids = [f"{c.get('doc_id',c.get('pair_id',''))}#{c.get('chunk_idx','')}" for c in chunks]
        return answer, latency_ms, chunk_ids, scores, confidence, truncated
    except Exception as e:
        print(f"  [ERROR] generate_hybrid : {e}")
        return "", 0, [], [], None, False

print("Fonctions hybrides prêtes.")""",
    )

    set_cell_source(
        nb,
        "cell-hybrid-run",
        """# Exécution de la méthode hybride sur tout le test set
hybrid_predictions = []

for item in tqdm(test_data, desc="Hybride (LoRA spécialisé + FAISS)"):
    question = item.get('question', '')
    true_answer = item.get('answer', '')
    predicted, latency, chunk_ids, scores, confidence, truncated = generate_hybrid(question)

    hybrid_predictions.append({
        "pair_id": item.get('pair_id', ''),
        "question": question,
        "predicted_answer": predicted,
        "true_answer": true_answer,
        "latency_ms": latency,
        "retrieved_chunks": chunk_ids,
        "retrieval_scores": scores,
        "confidence": confidence,
        "truncated": truncated,
        "method": "hybrid"
    })

latencies = [p['latency_ms'] for p in hybrid_predictions if p['latency_ms'] > 0]
conf_vals = [p["confidence"] for p in hybrid_predictions if p.get("confidence") is not None]
trunc_rate = 100 * np.mean([p.get("truncated", False) for p in hybrid_predictions]) if hybrid_predictions else 0.0
print(f"\\nHybride terminé : {len(hybrid_predictions)} prédictions")
print(f"Latence moyenne  : {np.mean(latencies):.0f} ms" if latencies else "Latence : N/A")
print(f"Taux troncature  : {trunc_rate:.1f}%")
print(f"Confiance moyenne: {np.mean(conf_vals):.3f}" if conf_vals else "Confiance : N/A")""",
    )

    add_cell_after(
        nb,
        "cell-hybrid-save",
        {
            "cell_type": "code",
            "id": "cell-hybrid-quick-metrics",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Mini-évaluation immédiate de l'hybride\n",
                "from rouge_score import rouge_scorer\n",
                "_rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)\n",
                "def _norm(t):\n",
                "    t = (t or '').lower().strip()\n",
                "    t = t.translate(str.maketrans('', '', string.punctuation))\n",
                "    return ' '.join(t.split())\n",
                "def _f1(p,g):\n",
                "    pt, gt = _norm(p).split(), _norm(g).split()\n",
                "    if not pt or not gt: return 0.0\n",
                "    inter = len(set(pt)&set(gt))\n",
                "    if inter == 0: return 0.0\n",
                "    pr, rc = inter/len(pt), inter/len(gt)\n",
                "    return 2*pr*rc/(pr+rc)\n",
                "preds = [x.get('predicted_answer','') for x in hybrid_predictions]\n",
                "refs  = [x.get('true_answer','') for x in hybrid_predictions]\n",
                "em = np.mean([int(_norm(p)==_norm(g)) for p,g in zip(preds,refs)])*100\n",
                "f1 = np.mean([_f1(p,g) for p,g in zip(preds,refs)])*100\n",
                "rl = np.mean([_rouge.score(g if g else ' ', p if p else ' ')['rougeL'].fmeasure for p,g in zip(preds,refs)])*100\n",
                "try:\n",
                "    _,_,F = bert_score_fn(preds if preds else [' '], refs if refs else [' '], lang='fr', model_type='distilbert-base-multilingual-cased', batch_size=32, verbose=False)\n",
                "    bs = float(F.mean())*100\n",
                "except Exception as e:\n",
                "    print(f'[WARN] BERTScore indisponible: {e}')\n",
                "    bs = 0.0\n",
                "print('\\n--- Mini-évaluation Hybride (NB05) ---')\n",
                "print(f'EM={em:.1f}% | F1={f1:.1f}% | ROUGE-L={rl:.1f}% | BERTScore={bs:.1f}%')\n",
            ],
        },
    )

    set_cell_source(
        nb,
        "cell-md-pointer-06",
        """## Suite : évaluation comparative

Exécutez **`06_evaluation.ipynb`** pour agréger Baseline / RAG / Fine-tuné / Hybride (+ RAFT optionnel), calculer les métriques et générer `final_report.json`.

**Ordre conseillé :** `03` → `04` (fine-tuning simple) → `07` (RAFT / LoRA hybride spécialisé) → `05` (hybride inférence) → `06`.""",
    )

    save_nb("05_hybrid_eval.ipynb", nb)


def make_07():
    nb04 = load_nb("04_finetuning.ipynb")
    nb07 = deepcopy(nb04)
    nb07["metadata"]["colab"]["name"] = "07_raft.ipynb"

    # Title
    set_cell_source(
        nb07,
        "cell-md-title",
        """# Notebook 07 — RAFT (Retrieval-Augmented Fine-Tuning)

**Objectif** : entraîner un LoRA **spécialisé hybride** avec pseudo-RAG (contexts FAISS au train), puis évaluer rapidement sur le test.

Sorties :
- `models/lora_adapter_hybrid/` (adaptateur LoRA pour la méthode hybride)
- `results/raft_predictions.json` (approche RAFT)
- résumé métriques en fin de notebook""",
    )

    # installs
    set_cell_source(
        nb07,
        "cell-install-rest",
        """# Dépendances RAFT
!pip install -q trl peft transformers accelerate bitsandbytes sentence-transformers faiss-cpu rouge-score bert-score""",
    )

    set_cell_source(
        nb07,
        "cell-imports",
        """import os, json, time, re, string
import numpy as np
import torch
import faiss
from tqdm.notebook import tqdm
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

PROCESSED_PATH = os.path.join(BASE_PATH, 'data', 'processed')
MODELS_PATH    = os.path.join(BASE_PATH, 'models', 'lora_adapter_hybrid')
FAISS_PATH     = os.path.join(BASE_PATH, 'models', 'faiss_index')
RESULTS_PATH   = os.path.join(BASE_PATH, 'results')
for p in [MODELS_PATH, RESULTS_PATH]:
    os.makedirs(p, exist_ok=True)

BASE_MODEL   = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
EMBED_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K        = 5
MAX_SEQ_LEN  = 1024
LORA_RANK    = 32
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
TARGET_MODS  = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
NUM_EPOCHS   = 5
BATCH_SIZE   = 2
LR           = 1e-4
SEED         = 42

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
print("Configuration RAFT prête.")
print(f"  LoRA output : {MODELS_PATH}")
print(f"  FAISS path  : {FAISS_PATH}")""",
    )

    set_cell_source(
        nb07,
        "cell-load",
        """def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  [OK] {path} ({len(data)} entrées)")
        return data
    except Exception as e:
        print(f"  [ERROR] {path}: {e}")
        return []

print("Chargement train/test + FAISS metadata...")
train_data = load_json(os.path.join(PROCESSED_PATH, 'train.json'))
test_data  = load_json(os.path.join(PROCESSED_PATH, 'test.json'))
corpus_meta = load_json(os.path.join(FAISS_PATH, 'metadata.json'))
index = faiss.read_index(os.path.join(FAISS_PATH, 'index.faiss'))
embed_model = SentenceTransformer(EMBED_MODEL)
print(f"Train={len(train_data)} | Test={len(test_data)} | FAISS={index.ntotal}")""",
    )

    set_cell_source(
        nb07,
        "cell-format",
        """# Construction pseudo-RAG pour l'entraînement (RAFT)
RAFT_TEMPLATE = (
    "### Instruction: Réponds à cette question en te basant uniquement sur le contexte fourni.\\n"
    "### Context: {context}\\n"
    "### Input: {question}\\n"
    "### Response: {answer}"
)
NOISE_PATTERNS = [
    r"pour sauvegarder cet article.*$",
    r"connectez-vous.*$",
    r"abonnez-vous.*$",
    r"cookies?.*$",
]

def clean_text(txt):
    out = (txt or "").strip()
    for p in NOISE_PATTERNS:
        out = re.sub(p, "", out, flags=re.IGNORECASE | re.MULTILINE).strip()
    out = re.sub(r"\\s+", " ", out)
    return out

def retrieve_context(question, k=TOP_K):
    q_emb = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, idxs = index.search(q_emb, k)
    chunks = [corpus_meta[i] for i in idxs[0] if i < len(corpus_meta)]
    lines = []
    for c in chunks:
        txt = clean_text(c.get('text', c.get('context', '')))[:700]
        if len(txt) >= 30:
            lines.append(f"[{c.get('title','')[:60]}] {txt}")
    return "\\n\\n".join(lines) if lines else "Contexte indisponible."

def format_raft_example(item):
    q = item.get('question', '').strip()
    a = item.get('answer', '').strip()
    if not q or not a:
        return None
    ctx = retrieve_context(q, TOP_K)
    return {"text": RAFT_TEMPLATE.format(context=ctx, question=q, answer=a)}

formatted = []
for it in tqdm(train_data, desc="Pseudo-RAG train build"):
    row = format_raft_example(it)
    if row:
        formatted.append(row)

hf_dataset = Dataset.from_list(formatted)
split = hf_dataset.train_test_split(test_size=0.1, seed=SEED, shuffle=True)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"RAFT dataset: total={len(hf_dataset)} train={len(train_dataset)} val={len(eval_dataset)}")
print(train_dataset[0]["text"][:500])""",
    )

    set_cell_source(
        nb07,
        "cell-trainer",
        """training_args = TrainingArguments(
    output_dir="/content/tmp_checkpoints_raft",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=8,
    warmup_steps=30,
    max_grad_norm=1.0,
    learning_rate=LR,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=SEED,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    dataset_num_proc=2,
    args=training_args,
)
print("Trainer RAFT prêt.")""",
    )

    set_cell_source(
        nb07,
        "cell-save-predictions",
        """# Sauvegarde des prédictions RAFT
raft_path = os.path.join(RESULTS_PATH, 'raft_predictions.json')
try:
    with open(raft_path, 'w', encoding='utf-8') as f:
        json.dump(finetuned_predictions, f, ensure_ascii=False, indent=2)
    print(f"Prédictions RAFT sauvegardées : {raft_path}")
except Exception as e:
    print(f"[ERROR] Sauvegarde RAFT : {e}")""",
    )

    set_cell_source(
        nb07,
        "cell-md-summary",
        "## 10. Résumé final RAFT",
    )
    set_cell_source(
        nb07,
        "cell-summary",
        """# Résumé final RAFT
latencies = [p['latency_ms'] for p in finetuned_predictions if p['latency_ms'] > 0]
print("=" * 70)
print("RÉSUMÉ — Notebook 07 : RAFT")
print("=" * 70)
print(f"LoRA hybride sauvegardé dans : {MODELS_PATH}")
print(f"Prédictions RAFT             : {os.path.join(RESULTS_PATH, 'raft_predictions.json')}")
print(f"Nombre prédictions test      : {len(finetuned_predictions)}")
print(f"Latence moyenne              : {np.mean(latencies):.0f} ms" if latencies else "Latence : N/A")
print("✔ Exécutez ensuite 05_hybrid_eval.ipynb puis 06_evaluation.ipynb")
print("=" * 70)""",
    )

    # Rename finetuned method label to raft in run cell
    set_cell_source(
        nb07,
        "cell-inference-run",
        """# Inférence RAFT sur tout le test set
finetuned_predictions = []

for item in tqdm(test_data, desc="Inférence RAFT"):
    question = item.get('question', '')
    true_answer = item.get('answer', '')
    predicted, latency, confidence, truncated = generate_answer(question)
    finetuned_predictions.append({
        "pair_id": item.get('pair_id', ''),
        "question": question,
        "predicted_answer": predicted,
        "true_answer": true_answer,
        "latency_ms": latency,
        "confidence": confidence,
        "truncated": truncated,
        "method": "raft"
    })

latencies = [p['latency_ms'] for p in finetuned_predictions if p['latency_ms'] > 0]
print(f"\\nInférence RAFT terminée : {len(finetuned_predictions)} prédictions")
print(f"Latence moyenne         : {np.mean(latencies):.0f} ms" if latencies else "Latence : N/A")""",
    )

    save_nb("07_raft.ipynb", nb07)


def patch_06():
    nb = load_nb("06_evaluation.ipynb")

    set_cell_source(
        nb,
        "cell06-title",
        """# Notebook 06 — Évaluation comparative (Baseline / RAG / Fine-tuné / Hybride + RAFT optionnel)

**Prérequis** :
- `data/processed/test.json`
- `results/baseline_predictions.json`, `rag_predictions.json`, `finetuned_predictions.json`, `hybrid_predictions.json`
- optionnel : `results/raft_predictions.json`

**Sorties** : `results/final_report.json`, figures PNG dans `results/plots/`.""",
    )

    set_cell_source(
        nb,
        "cell06-load",
        """# Chargement des fichiers JSON depuis Drive
def load_json(path, label=""):
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
test_data             = load_json(os.path.join(PROCESSED_PATH,  'test.json'),                  'test.json')
baseline_predictions  = load_json(os.path.join(RESULTS_PATH,    'baseline_predictions.json'),  'baseline_predictions.json')
rag_predictions       = load_json(os.path.join(RESULTS_PATH,    'rag_predictions.json'),       'rag_predictions.json')
finetuned_predictions = load_json(os.path.join(RESULTS_PATH,    'finetuned_predictions.json'), 'finetuned_predictions.json')
hybrid_predictions    = load_json(os.path.join(RESULTS_PATH,    'hybrid_predictions.json'),    'hybrid_predictions.json')
raft_predictions      = load_json(os.path.join(RESULTS_PATH,    'raft_predictions.json'),      'raft_predictions.json (optionnel)')

print(f"\\nTest set : {len(test_data)} questions")""",
    )

    set_cell_source(
        nb,
        "cell-metrics-fn",
        """# Métriques d'évaluation : EM, F1, BERTScore, ROUGE-L, hallucination + confiance
def normalize_text(text):
    text = (text or "").lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop = {'a','an','the','le','la','les','un','une','des'}
    return ' '.join(t for t in text.split() if t not in stop)

def exact_match(pred, gold):
    return int(normalize_text(pred) == normalize_text(gold))

def f1_token(pred, gold):
    pred_tok = normalize_text(pred).split()
    gold_tok = normalize_text(gold).split()
    if not pred_tok or not gold_tok:
        return 0.0
    common = Counter(pred_tok) & Counter(gold_tok)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(pred_tok)
    r = n / len(gold_tok)
    return 2 * p * r / (p + r)

def rouge_l(pred, gold):
    if not (pred or "").strip() or not (gold or "").strip():
        return 0.0
    return _ROUGE.score(gold, pred)['rougeL'].fmeasure

def compute_bert_score(preds, refs, batch_size=32):
    try:
        _, _, F = bert_score_fn(preds, refs, lang="fr",
                                model_type="distilbert-base-multilingual-cased",
                                batch_size=batch_size, verbose=False)
        return F.tolist()
    except Exception as e:
        print(f"  [ERROR] BERTScore : {e}")
        return [0.0] * len(preds)

def hallucination_score(pred, context):
    if not (pred or "").strip() or not (context or "").strip():
        return 1.0
    return 1.0 - rouge_l(pred, context)

def evaluate_method(predictions, method_name, test_lookup=None):
    if not predictions:
        print(f"  [WARN] Aucune prédiction pour '{method_name}'")
        return {"method": method_name, "n": 0, "exact_match": 0.0, "f1": 0.0, "bertscore": 0.0,
                "rouge_l": 0.0, "hallucination": 0.0, "latency_ms": 0.0, "confidence_mean": None}

    em_list, f1_list, rl_list, hall_list, lat_list = [], [], [], [], []
    conf_list = []
    preds_list, refs_list = [], []
    by_recency = defaultdict(lambda: {"preds": [], "refs": [], "hall": []})
    by_qtype   = defaultdict(lambda: {"preds": [], "refs": [], "hall": []})
    by_dstype  = defaultdict(lambda: {"preds": [], "refs": [], "hall": []})

    for p in predictions:
        pred, gold = p.get('predicted_answer', '') or '', p.get('true_answer', '') or ''
        pair_id = p.get('pair_id', '')
        recency, qtype, context = 'inconnu', 'inconnu', ''
        item = None
        if test_lookup and pair_id in test_lookup:
            item = test_lookup[pair_id]
            recency = item.get('recency_category', 'inconnu')
            qtype = item.get('question_type', 'inconnu')
            context = item.get('context', '')

        em_list.append(exact_match(pred, gold))
        f1_list.append(f1_token(pred, gold))
        rl_list.append(rouge_l(pred, gold))
        hall_list.append(hallucination_score(pred, context if context else gold))
        preds_list.append(pred if pred else " ")
        refs_list.append(gold if gold else " ")

        by_recency[recency]["preds"].append(preds_list[-1]); by_recency[recency]["refs"].append(refs_list[-1]); by_recency[recency]["hall"].append(hall_list[-1])
        by_qtype[qtype]["preds"].append(preds_list[-1]); by_qtype[qtype]["refs"].append(refs_list[-1]); by_qtype[qtype]["hall"].append(hall_list[-1])
        dstype = item.get("dataset_type", "inconnu") if item else "inconnu"
        by_dstype[dstype]["preds"].append(preds_list[-1]); by_dstype[dstype]["refs"].append(refs_list[-1]); by_dstype[dstype]["hall"].append(hall_list[-1])

        if p.get('latency_ms', 0) > 0:
            lat_list.append(p['latency_ms'])
        if p.get("confidence") is not None:
            conf_list.append(float(p.get("confidence")))

    print(f"  Calcul BERTScore pour '{method_name}' ({len(preds_list)} paires)...")
    bs_list = compute_bert_score(preds_list, refs_list)
    bs_by_recency = {cat: round(np.mean(compute_bert_score(d['preds'], d['refs'])) * 100, 2) if d["preds"] else 0.0 for cat, d in by_recency.items()}
    bs_by_qtype = {qt: round(np.mean(compute_bert_score(d['preds'], d['refs'])) * 100, 2) if d["preds"] else 0.0 for qt, d in by_qtype.items()}
    hall_by_recency = {k: round(np.mean(v["hall"]) * 100, 2) for k, v in by_recency.items()}
    hall_by_qtype = {k: round(np.mean(v["hall"]) * 100, 2) for k, v in by_qtype.items()}
    accuracy_by_threshold = {f"acc_bs{int(t*100)}": round(sum(s >= t for s in bs_list) / len(bs_list) * 100, 2) for t in BS_THRESHOLDS}

    return {
        "method": method_name,
        "exact_match": round(np.mean(em_list) * 100, 2),
        "f1": round(np.mean(f1_list) * 100, 2),
        "bertscore": round(np.mean(bs_list) * 100, 2),
        "rouge_l": round(np.mean(rl_list) * 100, 2),
        "hallucination": round(np.mean(hall_list) * 100, 2),
        "latency_ms": round(np.mean(lat_list), 1) if lat_list else 0,
        "confidence_mean": round(np.mean(conf_list), 4) if conf_list else None,
        "n": len(predictions),
        **accuracy_by_threshold,
        "bs_by_recency": bs_by_recency,
        "bs_by_qtype": bs_by_qtype,
        "hall_by_recency": hall_by_recency,
        "hall_by_qtype": hall_by_qtype,
        "bs_by_dstype": {k: round(np.mean(compute_bert_score(v["preds"], v["refs"])) * 100, 2) if v["preds"] else 0.0 for k, v in by_dstype.items()},
        "hall_by_dstype": {k: round(np.mean(v["hall"]) * 100, 2) for k, v in by_dstype.items()},
    }

print("Fonctions d'évaluation prêtes (inclut confiance moyenne si disponible).")""",
    )

    set_cell_source(
        nb,
        "cell-eval-run",
        """# Calcul des métriques
test_lookup = {item.get('pair_id', ''): item for item in test_data}

methods_to_eval = [
    ("Baseline", baseline_predictions),
    ("RAG", rag_predictions),
    ("Fine-tuné", finetuned_predictions),
    ("Hybride", hybrid_predictions),
]
if raft_predictions:
    methods_to_eval.append(("RAFT", raft_predictions))

print("Évaluation en cours...\\n")
all_results = []
for method_name, preds in methods_to_eval:
    result = evaluate_method(preds, method_name, test_lookup=test_lookup)
    all_results.append(result)
    acc85 = result.get('acc_bs85', 0)
    conf = result.get("confidence_mean")
    conf_txt = f"{conf:.3f}" if conf is not None else "N/A"
    print(
        f"  {method_name:<12} EM={result['exact_match']:5.1f}%  F1={result['f1']:5.1f}%  "
        f"BERTScore={result['bertscore']:5.1f}%  ROUGE-L={result['rouge_l']:5.1f}%  "
        f"Acc@85%={acc85:5.1f}%  Halluc.={result['hallucination']:5.1f}%  "
        f"Latence={result['latency_ms']:.0f}ms  Conf={conf_txt}"
    )

print("\\nÉvaluation terminée.")""",
    )

    set_cell_source(
        nb,
        "cell-eval-table",
        """# Tableau comparatif pandas
df = pd.DataFrame([{
    "Méthode": r['method'],
    "Exact Match (%)": r['exact_match'],
    "F1 Token (%)": r['f1'],
    "BERTScore (%)": r['bertscore'],
    "ROUGE-L (%)": r['rouge_l'],
    "Acc@85% (%)": r.get('acc_bs85', 0),
    "Hallucination (%)": r['hallucination'],
    "Latence moy. (ms)": r['latency_ms'],
    "Confiance moy.": r.get('confidence_mean', None),
    "N": r['n'],
} for r in all_results]).set_index("Méthode")

print("=" * 90)
print("TABLEAU COMPARATIF — Méthodes disponibles")
print("=" * 90)
display(
    df.style
      .highlight_max(subset=["Exact Match (%)","F1 Token (%)","BERTScore (%)","ROUGE-L (%)","Acc@85% (%)"], color='lightgreen')
      .highlight_min(subset=["Hallucination (%)","Latence moy. (ms)"], color='lightblue')
      .format(precision=3)
)
print("\\nConfiance moy. : moyenne du proxy de confiance token-level (si présent dans les prédictions).")""",
    )

    save_nb("06_evaluation.ipynb", nb)


def main():
    patch_04()
    patch_05()
    make_07()
    patch_06()
    print("Patched 04, 05, created 07, patched 06.")


if __name__ == "__main__":
    main()

