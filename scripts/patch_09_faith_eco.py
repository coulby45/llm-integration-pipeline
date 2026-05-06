# -*- coding: utf-8 -*-
"""Patch 09_evaluation.ipynb: remove EM, add faithfulness + fig10 ecology."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "09_evaluation.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))


def get_cell(cid):
    for c in nb["cells"]:
        if c.get("id") == cid:
            return c
    raise KeyError(cid)


def set_src(cid, text):
    c = get_cell(cid)
    c["source"] = text.splitlines(keepends=True)


# --- cell-metrics-fn ---
cell = get_cell("cell-metrics-fn")
src = "".join(cell["source"])
src = src.replace(
    "# Métriques : EM, F1, ROUGE-L, BERTScore, METEOR, hallucination, latence, confiance\n",
    "# Métriques : F1, ROUGE-L, BERTScore, METEOR, fidélité (faithfulness), hallucination, latence, confiance\n",
)
src = re.sub(
    r"def exact_match\(pred, gold\):\n    return int\(normalize_text\(pred\) == normalize_text\(gold\)\)\n\n",
    "",
    src,
    count=1,
)
insert_after = (
    "def hallucination_score(pred, context):\n"
    "    if not (pred or \"\").strip() or not (context or \"\").strip():\n"
    "        return 1.0\n"
    "    return 1.0 - rouge_l(pred, context)\n\n"
)
faith_fn = (
    "def faithfulness_score(pred, context, gold):\n"
    "    \"\"\"Fidélité au passage source : ROUGE-L(prédit, contexte). Si contexte absent, repli sur gold.\"\"\"\n"
    "    ref = (context or \"\").strip() or (gold or \"\").strip()\n"
    "    if not ref:\n"
    "        return 0.0\n"
    "    return float(rouge_l(pred, ref))\n\n"
)
if insert_after not in src:
    raise SystemExit("hallucination_score block not found")
src = src.replace(insert_after, insert_after + faith_fn)

src = src.replace(
    '_EMPTY_METRICS = {\n    "exact_match": 0.0, "f1": 0.0',
    '_EMPTY_METRICS = {\n    "f1": 0.0',
)
src = src.replace(
    '"hallucination": 0.0, "latency_ms": 0.0',
    '"faithfulness": 0.0, "hallucination": 0.0, "latency_ms": 0.0',
)

src = src.replace(
    "    em_list, f1_list, rl_list, hall_list, lat_list = [], [], [], [], []\n",
    "    f1_list, rl_list, faith_list, hall_list, lat_list = [], [], [], [], []\n",
)
src = src.replace(
    "        em_list.append(exact_match(pred, gold))\n        f1_list.append(f1_token(pred, gold))\n",
    "        f1_list.append(f1_token(pred, gold))\n",
)
src = src.replace(
    "        hall_list.append(hallucination_score(pred, context if context else gold))\n",
    "        hall_list.append(hallucination_score(pred, context if context else gold))\n"
    "        faith_list.append(faithfulness_score(pred, context, gold))\n",
)

src = src.replace(
    '        "exact_match": round(np.mean(em_list) * 100, 2),\n        "f1":',
    '        "f1":',
)
src = src.replace(
    '        "hallucination": round(np.mean(hall_list) * 100, 2),\n',
    '        "faithfulness": round(np.mean(faith_list) * 100, 2),\n'
    '        "hallucination": round(np.mean(hall_list) * 100, 2),\n',
)

src = src.replace(
    "print(\"Fonctions d'évaluation prêtes : METEOR + IC bootstrap 95 % (BERTScore, METEOR, F1, ROUGE-L).\")",
    "print(\"Fonctions d'évaluation prêtes : fidélité + METEOR + IC bootstrap 95 % (BERTScore, METEOR, F1, ROUGE-L).\")",
)
set_src("cell-metrics-fn", src)

# --- cell-eval-run ---
src = "".join(get_cell("cell-eval-run")["source"])
src = src.replace(
    "    print(f\"  {method_name:<18} EM={result['exact_match']:5.1f}%  F1={result['f1']:5.1f}%  \"\n",
    "    print(f\"  {method_name:<18} Fid={result.get('faithfulness',0):5.1f}%  F1={result['f1']:5.1f}%  \"\n",
)
set_src("cell-eval-run", src)

# --- cell-eval-table ---
src = "".join(get_cell("cell-eval-table")["source"])
src = src.replace(
    '    "Exact Match (%)\": r[\'exact_match\'],\n    "F1 (%)\":',
    '    "Fidélité (%)\": r.get(\'faithfulness\', 0),\n    "F1 (%)\":',
)
src = src.replace(
    '.highlight_max(subset=["Exact Match (%)","F1 (%)","BERTScore (%)","ROUGE-L (%)","METEOR (%)","Acc@85% (%)"], color=\'lightgreen\')\n',
    '.highlight_max(subset=["Fidélité (%)","F1 (%)","BERTScore (%)","ROUGE-L (%)","METEOR (%)","Acc@85% (%)"], color=\'lightgreen\')\n',
)
src = src.replace(
    '          subset=["Exact Match (%)","F1 (%)","BERTScore (%)","ROUGE-L (%)","METEOR (%)","Acc@85% (%)","Hallucination (%)","Latence moy. (ms)","Confiance moy.","N"],\n',
    '          subset=["Fidélité (%)","F1 (%)","BERTScore (%)","ROUGE-L (%)","METEOR (%)","Acc@85% (%)","Hallucination (%)","Latence moy. (ms)","Confiance moy.","N"],\n',
)
set_src("cell-eval-table", src)

# --- cell-plot-metrics ---
src = "".join(get_cell("cell-plot-metrics")["source"])
src = src.replace(
    "# ── Figure 1 : Métriques globales (EM / F1 / BERTScore / ROUGE-L) ──────────────\n",
    "# ── Figure 1 : Métriques globales (F1, BERTScore, ROUGE-L, METEOR, Fidélité) ───\n",
)
old_metrics = (
    "metrics  = {\n"
    '    "Exact Match (%)\": [r[\'exact_match\'] for r in all_results],\n'
    '    "F1 (%)\":            [r[\'f1\']          for r in all_results],\n'
    '    "BERTScore (%)\":   [r[\'bertscore\']   for r in all_results],\n'
    '    "ROUGE-L (%)\":     [r[\'rouge_l\']     for r in all_results],\n'
    '    "METEOR (%)\":      [r.get(\'meteor\', 0) for r in all_results],\n'
    "}\n"
)
new_metrics = (
    "metrics  = {\n"
    '    "F1 (%)\":            [r[\'f1\']          for r in all_results],\n'
    '    "BERTScore (%)\":   [r[\'bertscore\']   for r in all_results],\n'
    '    "ROUGE-L (%)\":     [r[\'rouge_l\']     for r in all_results],\n'
    '    "METEOR (%)\":      [r.get(\'meteor\', 0) for r in all_results],\n'
    '    "Fidélité (%)\":    [r.get(\'faithfulness\', 0) for r in all_results],\n'
    "}\n"
)
if old_metrics not in src:
    raise SystemExit("metrics dict pattern not found in fig1 cell")
src = src.replace(old_metrics, new_metrics)
src = src.replace(
    "offset = (i - 2) * width\n",
    "offset = (i - (len(metrics) - 1) / 2) * width\n",
)
src = src.replace(
    "ax1.set_title('Figure 1 — Métriques globales par méthode\\n(Exact Match, F1, BERTScore, ROUGE-L, METEOR)',\n",
    "ax1.set_title('Figure 1 — Métriques globales par méthode\\n(F1, BERTScore, ROUGE-L, METEOR, fidélité au contexte)',\n",
)
set_src("cell-plot-metrics", src)

# --- cell-summary : replace EM best, add p10 to list ---
src = "".join(get_cell("cell-summary")["source"])
src = src.replace(
    "] + [(\"09_evaluation\", p) for p in [p1, p2, p3, p4, p5, p6, p7, p8, p9]]\n",
    "] + [(\"09_evaluation\", p) for p in [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]]\n",
)
src = src.replace(
    "best_em   = max(all_results, key=lambda x: x['exact_match'])\n",
    "best_fid  = max(all_results, key=lambda x: x.get('faithfulness', 0))\n",
)
src = src.replace(
    "print(f\"  Exact Match      : {best_em['method']:<12} {best_em['exact_match']:.1f}%\")\n",
    "print(f\"  Fidélité (contexte): {best_fid['method']:<12} {best_fid.get('faithfulness',0):.1f}%\")\n",
)

# Insert fig10 cell before markdown cell-md-summary
fig10_source = r'''# ── Figure 10 : Impact écologique estimé (Wh, CO₂) ─────────────────────────────
# Hypothèses (README § Green AI) : API Groq ~ 50 W / PUE 1,10 ; inférence locale GPU ~ 150 W / PUE 1,15
# Énergie (Wh) = Σ_i (latence_ms_i / 1000) × TDP × PUE / 3600
TDP_API_W, PUE_API = 50.0, 1.10
TDP_GPU_W, PUE_GPU = 150.0, 1.15
G_CO2_PER_KWH_FR = 52.0
G_CO2_PER_KWH_WORLD = 475.0
API_METHODS = {"Baseline", "RAG", "Rerank", "Function-calling"}

def total_energy_wh(predictions, tdp_w, pue):
    tot = 0.0
    for p in predictions:
        lat = float(p.get("latency_ms") or 0)
        if lat <= 0:
            continue
        tot += (lat / 1000.0) * tdp_w * pue / 3600.0
    return tot

eco_rows = []
for method_name, preds in methods_to_eval:
    if not preds:
        continue
    tdp, pue = (TDP_API_W, PUE_API) if method_name in API_METHODS else (TDP_GPU_W, PUE_GPU)
    wh = total_energy_wh(preds, tdp, pue)
    kwh = wh / 1000.0
    co2_fr = kwh * G_CO2_PER_KWH_FR
    co2_wo = kwh * G_CO2_PER_KWH_WORLD
    n = len(preds)
    eco_rows.append(
        {"method": method_name, "Wh": wh, "CO2_FR_g": co2_fr, "CO2_world_g": co2_wo,
         "mg_per_pred_FR": (co2_fr * 1000.0 / n) if n else 0.0}
    )

eco_methods = [r["method"] for r in eco_rows]
wh_vals = [r["Wh"] for r in eco_rows]
co2_fr_vals = [r["CO2_FR_g"] for r in eco_rows]
co2_wo_vals = [r["CO2_world_g"] for r in eco_rows]

fig10, axes10 = plt.subplots(1, 2, figsize=(14, 5))

ax_a = axes10[0]
bars_a = ax_a.bar(eco_methods, wh_vals, color=METHOD_COLORS, alpha=0.87, width=0.55)
mxw = max(wh_vals) if wh_vals else 1.0
for bar, val in zip(bars_a, wh_vals):
    ax_a.text(bar.get_x() + bar.get_width()/2, val + mxw * 0.02,
              f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax_a.set_ylabel('Énergie estimée (Wh)')
ax_a.set_title('10a — Énergie totale (TDP×PUE×latence cumulée)', fontsize=11, fontweight='bold')
ax_a.set_xticklabels(eco_methods, rotation=18, ha='right')
ax_a.grid(axis='y', alpha=0.3)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

ax_b = axes10[1]
x10 = np.arange(len(eco_methods))
w10 = 0.35
ax_b.bar(x10 - w10/2, co2_fr_vals, w10, label='France (52 g/kWh)', color='#2ecc71', alpha=0.85)
ax_b.bar(x10 + w10/2, co2_wo_vals, w10, label='Mix mondial (475 g/kWh)', color='#95a5a6', alpha=0.85)
ax_b.set_xticks(x10)
ax_b.set_xticklabels(eco_methods, rotation=18, ha='right')
ax_b.set_ylabel('CO₂ équivalent (g)')
ax_b.set_title('10b — CO₂ à partir des Wh', fontsize=11, fontweight='bold')
ax_b.legend(fontsize=9)
ax_b.grid(axis='y', alpha=0.3)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

fig10.suptitle(
    'Figure 10 — Impact écologique estimé (hypothèses documentées dans le README)',
    fontsize=12, fontweight='bold',
)
plt.tight_layout()
p10 = os.path.join(PLOTS_PATH, 'fig10_ecological_impact.png')
fig10.savefig(p10, dpi=150, bbox_inches='tight')
plt.show()
print(f"Figure 10 sauvegardée : {p10}")
print("Estimation : Wh = Σ(lat_s × TDP × PUE / 3600) ; CO₂ (g) = (Wh/1000) × g/kWh.")
'''

insert_idx = None
for i, c in enumerate(nb["cells"]):
    if c.get("id") == "cell-md-summary":
        insert_idx = i
        break
if insert_idx is None:
    raise SystemExit("cell-md-summary not found")

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "cell-plot-ecology",
    "metadata": {},
    "outputs": [],
    "source": fig10_source.splitlines(keepends=True),
}
nb["cells"].insert(insert_idx, new_cell)

set_src("cell-summary", src)

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Patched", NB_PATH)
