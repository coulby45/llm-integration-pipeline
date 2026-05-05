"""
LLM-as-judge : notation des réponses (1–5) via l'API OpenAI (GPT-4 / GPT-4o).

Critères : cohérence, utilité, fidélité au contexte de référence.
Peut être importé depuis un notebook ou exécuté en CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError as e:  # pragma: no cover
    raise ImportError("Installez le paquet openai : pip install openai") from e

DEFAULT_MODEL = "gpt-4o"

JUDGE_SYSTEM = """Tu es un évaluateur neutre pour des réponses en français.
Tu dois renvoyer UNIQUEMENT un objet JSON valide avec les clés :
  "coherence" : entier de 1 à 5 (cohérence interne et clarté de la réponse),
  "utilite"   : entier de 1 à 5 (à quel point la réponse répond utilement à la question),
  "fidelite"  : entier de 1 à 5 (respect du contexte fourni ; pénalise les inventions contraires au contexte),
  "commentaire_bref" : chaîne courte (optionnelle, une phrase).

Échelle : 1 = très mauvais, 3 = acceptable, 5 = excellent.
Les trois scores doivent être des entiers entre 1 et 5 inclus."""


def build_user_content(
    question: str,
    context: str,
    reference_answer: str,
    predicted: str,
) -> str:
    ctx = (context or "").strip()
    if len(ctx) > 8000:
        ctx = ctx[:8000] + "\n[… contexte tronqué …]"
    ref = (reference_answer or "").strip()
    pred = (predicted or "").strip()
    q = (question or "").strip()
    return f"""Question :
{q}

Contexte de référence (source documentaire ; sert à juger la fidélité) :
{ctx}

Réponse de référence (résumé attendu, indication de qualité) :
{ref}

Réponse du système à évaluer :
{pred}

Évalue uniquement la « Réponse du système ». Renvoie le JSON demandé."""


def judge_one(
    client: Any,
    *,
    question: str,
    context: str,
    reference_answer: str,
    predicted: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Un appel juge → dict avec scores + clé raw si besoin."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": build_user_content(
                    question, context, reference_answer, predicted
                ),
            },
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    text = (resp.choices[0].message.content or "").strip()
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        out = {
            "coherence": None,
            "utilite": None,
            "fidelite": None,
            "commentaire_bref": "parse_error",
            "raw": text,
        }
        return out
    for k in ("coherence", "utilite", "fidelite"):
        if k in out and out[k] is not None:
            try:
                out[k] = int(out[k])
            except (TypeError, ValueError):
                pass
    return out


def load_test_index(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        pid = row.get("pair_id") or row.get("id")
        if pid:
            by_id[str(pid)] = row
    return by_id


def run_judge_on_predictions(
    *,
    predictions_path: str,
    test_json_path: str,
    out_path: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    max_items: Optional[int] = None,
    sleep_s: float = 0.3,
    predictions_label: str = "",
) -> List[Dict[str, Any]]:
    """
    Joint prédictions et test.json par pair_id, appelle le juge pour chaque ligne.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Clé OpenAI manquante : définissez OPENAI_API_KEY ou passez api_key=")

    client = OpenAI(api_key=key)

    with open(predictions_path, "r", encoding="utf-8") as f:
        preds = json.load(f)
    test_by_id = load_test_index(test_json_path)

    results: List[Dict[str, Any]] = []
    for i, p in enumerate(preds):
        if max_items is not None and i >= max_items:
            break
        pid = str(p.get("pair_id", ""))
        t = test_by_id.get(pid, {})
        question = p.get("question") or t.get("question", "")
        predicted = p.get("predicted_answer", "")
        ref = p.get("true_answer") or t.get("answer", "")
        ctx = t.get("context", "")

        scores = judge_one(
            client,
            question=question,
            context=ctx,
            reference_answer=ref,
            predicted=predicted,
            model=model,
        )
        row = {
            "pair_id": pid,
            "predictions_file": predictions_path,
            "predictions_label": predictions_label or os.path.basename(predictions_path),
            "model_judge": model,
            **scores,
        }
        results.append(row)
        time.sleep(sleep_s)

    summary = {
        "predictions_file": predictions_path,
        "test_file": test_json_path,
        "judge_model": model,
        "n": len(results),
        "mean_coherence": _mean_int(results, "coherence"),
        "mean_utilite": _mean_int(results, "utilite"),
        "mean_fidelite": _mean_int(results, "fidelite"),
    }

    payload = {"summary": summary, "details": results}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return results


def _mean_int(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append(float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)


def main() -> None:
    p = argparse.ArgumentParser(description="LLM-as-judge (OpenAI) pour réponses FR.")
    p.add_argument("--predictions", required=True, help="Chemin vers *_predictions.json")
    p.add_argument("--test", required=True, help="Chemin vers test.json")
    p.add_argument("--out", required=True, help="Sortie JSON (summary + details)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Défaut : {DEFAULT_MODEL}")
    p.add_argument("--max-items", type=int, default=None)
    p.add_argument("--sleep", type=float, default=0.3)
    args = p.parse_args()

    run_judge_on_predictions(
        predictions_path=args.predictions,
        test_json_path=args.test,
        out_path=args.out,
        model=args.model,
        max_items=args.max_items,
        sleep_s=args.sleep,
    )
    print(f"Écrit : {args.out}")


if __name__ == "__main__":
    main()
