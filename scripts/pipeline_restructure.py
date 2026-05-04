# -*- coding: utf-8 -*-
"""Remove hybrid, rename notebooks to execution order, add rerank + function calling, rebuild 08 evaluation."""
import json
import os
import re
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_nb(name: str):
    with open(ROOT / name, encoding="utf-8") as f:
        return json.load(f)


def save_nb(name: str, nb):
    with open(ROOT / name, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)


def replace_in_nb(nb, old: str, new: str):
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", [])
        if isinstance(src, str):
            c["source"] = src.replace(old, new)
        else:
            c["source"] = [ln.replace(old, new) for ln in src]


def cell_md(text: str, cid: str):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": [text + "\n"]}


def cell_code(lines: list, cid: str):
    return {
        "cell_type": "code",
        "id": cid,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [ln if ln.endswith("\n") else ln + "\n" for ln in lines],
    }


def rm_if_exists(p: Path):
    if p.exists():
        p.unlink()


def patch_03_summary(nb):
    for c in nb["cells"]:
        if c.get("id") == "cell-md-title":
            c["source"] = [
                "# Notebook 03 — Baseline Groq & RAG avec FAISS\n",
                "\n",
                "**Ordre pipeline :** `01` → `02` → **`03`** → `04` → `05` → `06` → `07` → `08`.\n",
                "\n",
                "**Objectif** :\n",
                "1. **Baseline** : Groq sans contexte\n",
                "2. **RAG** : Groq + top-k FAISS (bi-encodeur)\n",
                "\n",
                "**Outputs** : `baseline_predictions.json`, `rag_predictions.json`, index `models/faiss_index/`.\n",
            ]
            return


def patch_04(nb):
    replace_in_nb(nb, "format hybride", "format avec contexte gold (optionnel)")
    replace_in_nb(
        nb,
        "Lancez 07_raft.ipynb (LoRA hybride spécialisé) puis 05_hybrid_eval.ipynb.",
        "Lancez `05_raft.ipynb`, puis `06_rag_rerank.ipynb`, `07_function_calling.ipynb`, enfin `08_evaluation.ipynb`.",
    )


def raft_to_05():
    p_old = ROOT / "07_raft.ipynb"
    if not p_old.exists():
        raise FileNotFoundError("07_raft.ipynb manquant")
    nb = load_nb("07_raft.ipynb")
    for c in nb["cells"]:
        if isinstance(c.get("source"), list):
            c["source"] = [
                re.sub(r"Notebook 07", "Notebook 05", ln)
                .replace("lora_adapter_hybrid", "lora_adapter_raft")
                .replace("LoRA hybride", "LoRA RAFT")
                .replace("méthode hybride", "méthode RAFT")
                for ln in c["source"]
            ]
    if "metadata" in nb and "colab" in nb["metadata"]:
        nb["metadata"]["colab"]["name"] = "05_raft.ipynb"
    save_nb("05_raft.ipynb", nb)
    rm_if_exists(p_old)


def set_cell_lines(nb, cell_id: str, lines: list[str]):
    for c in nb["cells"]:
        if c.get("id") == cell_id:
            c["source"] = [ln if ln.endswith("\n") else ln + "\n" for ln in lines]
            c["execution_count"] = None
            c["outputs"] = []
            return
    raise KeyError(cell_id)


def eval_to_08():
    p_old = ROOT / "06_evaluation.ipynb"
    if not p_old.exists():
        raise FileNotFoundError("06_evaluation.ipynb manquant")
    nb = load_nb("06_evaluation.ipynb")

    set_cell_lines(
        nb,
        "cell06-title",
        [
            "# Notebook 08 — Évaluation comparative (6 méthodes)",
            "",
            "**Prérequis** : `test.json` + les JSON dans `results/` :",
            "- `baseline_predictions.json`, `rag_predictions.json`, `finetuned_predictions.json`",
            "- `raft_predictions.json`, `rerank_predictions.json`, `function_calling_predictions.json`",
            "",
            "**Sorties** : `final_report.json`, figures dans `results/plots/`.",
        ],
    )
    set_cell_lines(nb, "cell06-md-load", ["## 3. Chargement des fichiers de prédictions"])
    set_cell_lines(
        nb,
        "cell06-load",
        [
            "# Chargement des fichiers JSON depuis Drive",
            'def load_json(path, label=""):',
            "    try:",
            "        with open(path, 'r', encoding='utf-8') as f:",
            "            data = json.load(f)",
            '        print(f"  [OK] {label or path} ({len(data)} entrées)")',
            "        return data",
            "    except FileNotFoundError:",
            '        print(f"  [MANQUANT] {path}")',
            "        return []",
            "    except json.JSONDecodeError as e:",
            '        print(f"  [ERROR JSON] {path} : {e}")',
            "        return []",
            "",
            'print("Chargement des fichiers...")',
            "test_data             = load_json(os.path.join(PROCESSED_PATH,  'test.json'),                   'test.json')",
            "baseline_predictions  = load_json(os.path.join(RESULTS_PATH,    'baseline_predictions.json'),   'baseline_predictions.json')",
            "rag_predictions       = load_json(os.path.join(RESULTS_PATH,    'rag_predictions.json'),        'rag_predictions.json')",
            "finetuned_predictions = load_json(os.path.join(RESULTS_PATH,    'finetuned_predictions.json'),  'finetuned_predictions.json')",
            "raft_predictions      = load_json(os.path.join(RESULTS_PATH,    'raft_predictions.json'),        'raft_predictions.json')",
            "rerank_predictions    = load_json(os.path.join(RESULTS_PATH,    'rerank_predictions.json'),      'rerank_predictions.json')",
            "fc_predictions        = load_json(os.path.join(RESULTS_PATH,    'function_calling_predictions.json'), 'function_calling_predictions.json')",
            "",
            'print(f"\\nTest set : {len(test_data)} questions")',
        ],
    )
    set_cell_lines(nb, "cell06-md-eval", ["## 4. Évaluation des 6 méthodes"])
    set_cell_lines(
        nb,
        "cell-eval-run",
        [
            "# Calcul des métriques pour les 6 méthodes",
            "test_lookup = {item.get('pair_id', ''): item for item in test_data}",
            "",
            'print("Évaluation en cours...\\n")',
            "methods_to_eval = [",
            '    ("Baseline", baseline_predictions),',
            '    ("RAG", rag_predictions),',
            '    ("Fine-tuné", finetuned_predictions),',
            '    ("RAFT", raft_predictions),',
            '    ("Rerank", rerank_predictions),',
            '    ("Function-calling", fc_predictions),',
            "]",
            "all_results = []",
            "for method_name, preds in methods_to_eval:",
            "    result = evaluate_method(preds, method_name, test_lookup=test_lookup)",
            "    all_results.append(result)",
            "    acc85 = result.get('acc_bs85', 0)",
            '    conf = result.get("confidence_mean")',
            '    conf_txt = f"{conf:.3f}" if conf is not None else "N/A"',
            '    print(f"  {method_name:<18} EM={result[\'exact_match\']:5.1f}%  F1={result[\'f1\']:5.1f}%  "',
            '          f"BERTScore={result[\'bertscore\']:5.1f}%  ROUGE-L={result[\'rouge_l\']:5.1f}%  "',
            '          f"Acc@85%={acc85:5.1f}%  Halluc.={result[\'hallucination\']:5.1f}%  "',
            '          f"Latence={result[\'latency_ms\']:.0f}ms  Conf={conf_txt}")',
            "",
            'print("\\nÉvaluation terminée.")',
            'METHODS_ORDER = [r["method"] for r in all_results]',
            "METHOD_COLORS = ['#4C72B0','#55A868','#C44E52','#8172B2','#CCB974','#8c564b'][:len(METHODS_ORDER)]",
        ],
    )

    # Imports cell: extend colors + notebook label
    for c in nb["cells"]:
        if c.get("id") == "cell06-imports":
            src = "".join(c["source"])
            src = src.replace(
                'METHODS_ORDER    = ["Baseline", "RAG", "Fine-tuné", "Hybride"]',
                'METHODS_ORDER    = ["Baseline", "RAG", "Fine-tuné", "RAFT", "Rerank", "Function-calling"]',
            )
            src = src.replace(
                "METHOD_COLORS    = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']",
                "METHOD_COLORS    = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#8c564b']",
            )
            src = src.replace("notebook 06", "notebook 08")
            lines = src.split("\n")
            c["source"] = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])

    # Summary cell: replace hybrid refs
    for c in nb["cells"]:
        if c.get("id") == "cell-summary":
            src = "".join(c["source"])
            src = src.replace("05_hybrid_eval", "07_function_calling")
            src = src.replace("hybrid_predictions.json", "function_calling_predictions.json")
            src = src.replace("hybrid_predictions", "rerank_predictions")
            if "rerank_predictions.json" in src and "rerank_predictions" in src:
                pass
            lines = src.split("\n")
            c["source"] = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])

    for c in nb["cells"]:
        if c.get("id", "").startswith("cell06-"):
            c["id"] = c["id"].replace("cell06-", "cell08-")

    if "metadata" in nb and "colab" in nb["metadata"]:
        nb["metadata"]["colab"]["name"] = "08_evaluation.ipynb"

    save_nb("08_evaluation.ipynb", nb)
    rm_if_exists(p_old)


def make_06_rerank():
    lines_code_imports = """# Chemins et constantes reranking
import os, json, time, getpass
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm.notebook import tqdm

RAW_PATH       = os.path.join(BASE_PATH, 'data', 'raw')
PROCESSED_PATH = os.path.join(BASE_PATH, 'data', 'processed')
RESULTS_PATH   = os.path.join(BASE_PATH, 'results')
FAISS_PATH     = os.path.join(BASE_PATH, 'models', 'faiss_index')
os.makedirs(RESULTS_PATH, exist_ok=True)

GROQ_MODEL       = "llama-3.1-8b-instant"
EMBED_MODEL      = "paraphrase-multilingual-MiniLM-L12-v2"
CROSS_MODEL      = "cross-encoder/ms-marco-Multilingual-MiniLM-L-12-v2"
TOP_M            = 30   # candidats bi-encodeur avant rerank
TOP_K_FINAL      = 5    # chunks après cross-encoder
THROTTLE_S       = 0.5
print("Rerank : bi-encodeur + cross-encoder + Groq")""".split(
        "\n"
    )

    lines_load = """def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] {path}: {e}")
        return []

print("Chargement FAISS + test...")
test_data   = load_json(os.path.join(PROCESSED_PATH, 'test.json'))
corpus_meta = load_json(os.path.join(FAISS_PATH, 'metadata.json'))
index       = faiss.read_index(os.path.join(FAISS_PATH, 'index.faiss'))
embed_model = SentenceTransformer(EMBED_MODEL)
reranker    = CrossEncoder(CROSS_MODEL)

api_key = getpass.getpass("Clé Groq API : ")
groq_client = Groq(api_key=api_key)
print(f"Index: {index.ntotal} vecteurs | Test: {len(test_data)}")""".split(
        "\n"
    )

    lines_rag = """RAG_PROMPT_TEMPLATE = \"\"\"Tu es un assistant expert. Réponds EN FRANÇAIS, de façon concise et factuelle,
en t'appuyant sur les extraits ci-dessous (ignore le bruit type paywall / navigation).

Contexte :
{context_block}

Question : {question}\"\"\"

def retrieve_wide(question, m=TOP_M):
    q_emb = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, idxs = index.search(q_emb, m)
    return [corpus_meta[i] for i in idxs[0] if i < len(corpus_meta)], scores[0].tolist()

def rerank_chunks(question, chunks):
    texts = [c.get('text', c.get('context', ''))[:1200] for c in chunks]
    pairs = [[question, t] for t in texts]
    ce_scores = reranker.predict(pairs, show_progress_bar=False)
    order = np.argsort(-np.array(ce_scores))
    ranked = [chunks[i] for i in order[:TOP_K_FINAL]]
    return ranked, ce_scores.tolist() if hasattr(ce_scores, 'tolist') else list(map(float, ce_scores))

def call_groq_rerank(question, chunks, retries=4):
    context_block = "\\n\\n".join([
        f"[{i+1} — {c.get('title','')[:60]}] {c.get('text', '')[:900]}"
        for i, c in enumerate(chunks)
    ])
    prompt = RAG_PROMPT_TEMPLATE.format(context_block=context_block, question=question)
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            lat = round((time.time() - t0) * 1000)
            ans = r.choices[0].message.content.strip()
            tin = r.usage.prompt_tokens if r.usage else 0
            tout = r.usage.completion_tokens if r.usage else 0
            return ans, lat, tin, tout
        except Exception as e:
            err = str(e)
            if '429' in err or 'rate_limit' in err.lower():
                time.sleep(5 * (2 ** attempt))
            else:
                time.sleep(1)
    return "", 0, 0, 0

print("Prêt : rerank + Groq")""".split(
        "\n"
    )

    lines_run = """rerank_predictions = []
for item in tqdm(test_data, desc="Rerank (FAISS+M+CE -> Groq)"):
    q = item.get('question', '')
    gold = item.get('answer', '')
    chunks, _ = retrieve_wide(q, TOP_M)
    top_chunks, _ = rerank_chunks(q, chunks)
    pred, lat, tin, tout = call_groq_rerank(q, top_chunks)
    rerank_predictions.append({
        "pair_id": item.get('pair_id', ''),
        "question": q,
        "predicted_answer": pred,
        "true_answer": gold,
        "latency_ms": lat,
        "tokens_in": tin,
        "tokens_out": tout,
        "method": "rerank",
        "dataset_type": item.get("dataset_type", ""),
        "question_type": item.get("question_type", ""),
    })
    time.sleep(THROTTLE_S)

outp = os.path.join(RESULTS_PATH, 'rerank_predictions.json')
with open(outp, 'w', encoding='utf-8') as f:
    json.dump(rerank_predictions, f, ensure_ascii=False, indent=2)
print(f"Sauvegardé : {outp} ({len(rerank_predictions)} lignes)")""".split(
        "\n"
    )

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "name": "06_rag_rerank.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": [
            cell_md(
                "# Notebook 06 — RAG + reranking (cross-encoder)\n\n"
                "**Prérequis :** exécuter `03` (index FAISS).\n\n"
                "**Sortie :** `results/rerank_predictions.json`\n\n"
                "**Étape suivante :** `07_function_calling.ipynb`",
                "cell06-md-title",
            ),
            cell_md("## 0. Montage Google Drive", "cell06-md-drive"),
            cell_code(
                [
                    "from google.colab import drive",
                    "drive.mount('/content/drive')",
                    "BASE_PATH = '/content/drive/MyDrive/llm-integration-study/'",
                ],
                "cell06-drive",
            ),
            cell_md("## 1. Installation", "cell06-md-install"),
            cell_code(["!pip install -q groq sentence-transformers faiss-cpu"], "cell06-pip"),
            cell_md("## 2. Imports", "cell06-md-import"),
            cell_code(lines_code_imports, "cell06-imports"),
            cell_md("## 3. Chargement", "cell06-md-load"),
            cell_code(lines_load, "cell06-load"),
            cell_md("## 4. Inférence rerank + Groq", "cell06-md-run"),
            cell_code(lines_rag, "cell06-rag-fn"),
            cell_code(lines_run, "cell06-run"),
        ],
    }
    save_nb("06_rag_rerank.ipynb", nb)


def make_07_fc():
    code = """import os, json, time, getpass
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm

PROCESSED_PATH = os.path.join(BASE_PATH, 'data', 'processed')
RESULTS_PATH   = os.path.join(BASE_PATH, 'results')
FAISS_PATH     = os.path.join(BASE_PATH, 'models', 'faiss_index')
os.makedirs(RESULTS_PATH, exist_ok=True)

GROQ_MODEL  = "llama-3.1-8b-instant"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K       = 5
THROTTLE_S  = 0.5

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Recherche sémantique dans le corpus indexé (FAISS). Retourne des extraits pertinents.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Requête de recherche"}},
            "required": ["query"],
        },
    }
}]

def load_json(p):
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(e)
        return []

test_data   = load_json(os.path.join(PROCESSED_PATH, 'test.json'))
corpus_meta = load_json(os.path.join(FAISS_PATH, 'metadata.json'))
index       = faiss.read_index(os.path.join(FAISS_PATH, 'index.faiss'))
embed_model = SentenceTransformer(EMBED_MODEL)

api_key = getpass.getpass("Clé Groq API : ")
client = Groq(api_key=api_key)

def tool_search_docs(query: str) -> str:
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    _, idxs = index.search(q_emb, TOP_K)
    chunks = [corpus_meta[i] for i in idxs[0] if i < len(corpus_meta)]
    parts = []
    for j, c in enumerate(chunks):
        t = (c.get('text', '') or '')[:800]
        parts.append(f"[{j+1} | {c.get('title','')[:50]}] {t}")
    return "\\n".join(parts) if parts else "(aucun document)"

def answer_with_tools(question: str):
    messages = [
        {"role": "system", "content": "Tu es un assistant. Tu DOIS appeler l'outil search_docs une fois avec une requête courte dérivée de la question, puis résumer la réponse finale EN FRANÇAIS."},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    r1 = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "search_docs"}},
    )
    msg = r1.choices[0].message
    if not getattr(msg, "tool_calls", None):
        lat = round((time.time()-t0)*1000)
        return (msg.content or "").strip(), lat, []
    args = json.loads(msg.tool_calls[0].function.arguments)
    q = args.get("query", question)[:500]
    doc_txt = tool_search_docs(q)
    messages.append({"role": "assistant", "tool_calls": msg.tool_calls})
    messages.append({"role": "tool", "tool_call_id": msg.tool_calls[0].id, "content": doc_txt})
    r2 = client.chat.completions.create(model=GROQ_MODEL, messages=messages)
    lat = round((time.time()-t0)*1000)
    return r2.choices[0].message.content.strip(), lat, msg.tool_calls

fc_predictions = []
for item in tqdm(test_data, desc="Function-calling + RAG"):
    q = item.get('question', '')
    gold = item.get('answer', '')
    pred, lat, tcalls = answer_with_tools(q)
    fc_predictions.append({
        "pair_id": item.get('pair_id', ''),
        "question": q,
        "predicted_answer": pred,
        "true_answer": gold,
        "latency_ms": lat,
        "method": "function_calling",
        "dataset_type": item.get("dataset_type", ""),
        "question_type": item.get("question_type", ""),
        "tool_calls": [{"name": tc.function.name, "arguments": tc.function.arguments} for tc in tcalls] if tcalls else [],
    })
    time.sleep(THROTTLE_S)

outp = os.path.join(RESULTS_PATH, 'function_calling_predictions.json')
with open(outp, 'w', encoding='utf-8') as f:
    json.dump(fc_predictions, f, ensure_ascii=False, indent=2)
print("Sauvegardé", outp, len(fc_predictions))""".split(
        "\n"
    )

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"name": "07_function_calling.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": [
            cell_md(
                "# Notebook 07 — Function calling + retrieval (Groq)\n\n"
                "**Prérequis :** `03` (FAISS).\n\n"
                "**Sortie :** `results/function_calling_predictions.json`\n\n"
                "**Suite :** `08_evaluation.ipynb`",
                "cell07-md-title",
            ),
            cell_md("## 0. Drive", "cell07-md-drive"),
            cell_code(
                [
                    "from google.colab import drive",
                    "drive.mount('/content/drive')",
                    "BASE_PATH = '/content/drive/MyDrive/llm-integration-study/'",
                ],
                "cell07-drive",
            ),
            cell_md("## 1. Install", "cell07-md-install"),
            cell_code(["!pip install -q groq sentence-transformers faiss-cpu"], "cell07-pip"),
            cell_md("## 2. Run", "cell07-md-run"),
            cell_code(code, "cell07-main"),
        ],
    }
    save_nb("07_function_calling.ipynb", nb)


def main():
    rm_if_exists(ROOT / "05_hybrid_eval.ipynb")

    nb03 = load_nb("03_baseline_rag.ipynb")
    patch_03_summary(nb03)
    save_nb("03_baseline_rag.ipynb", nb03)

    nb04 = load_nb("04_finetuning.ipynb")
    patch_04(nb04)
    save_nb("04_finetuning.ipynb", nb04)

    raft_to_05()
    make_06_rerank()
    make_07_fc()
    eval_to_08()

    print("OK: hybrid removed, 05_raft, 06_rag_rerank, 07_function_calling, 08_evaluation")


if __name__ == "__main__":
    main()
