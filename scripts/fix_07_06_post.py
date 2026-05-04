import json
from pathlib import Path

ROOT = Path(r"C:/Users/Utilisateur/Desktop/Idee_random/Article_scientifique")


def to_src(text: str):
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def patch_07():
    p = ROOT / "07_raft.ipynb"
    nb = json.load(open(p, encoding="utf-8"))
    new_code = """# Inférence RAFT (retrieval au test, aligné avec l'entraînement)
FastLanguageModel.for_inference(model)
_STOP = ["\\n### Instruction:", "\\n### Input:", "\\n### Response:", "\\n### Context:"]

def _cleanup_answer(text):
    out = (text or "").strip()
    for m in _STOP:
        if m in out:
            out = out.split(m)[0].strip()
    if len(out) > 40 and out[-1] not in ".!?":
        cut = max(out.rfind('.'), out.rfind('!'), out.rfind('?'))
        if cut > 40:
            out = out[:cut+1]
    return out.strip()

def _conf(gen):
    if not getattr(gen, 'scores', None):
        return None
    probs = [torch.softmax(s[0], dim=-1).max().item() for s in gen.scores]
    return round(float(np.mean(probs)), 4) if probs else None

def generate_answer(question, max_new_tokens=320):
    ctx = retrieve_context(question, TOP_K)
    prompt = (
        "### Instruction: Réponds à cette question en te basant uniquement sur le contexte fourni.\\n"
        f"### Context: {ctx}\\n"
        f"### Input: {question}\\n"
        "### Response:"
    )
    try:
        inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
        start = time.time()
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=20,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        latency_ms = round((time.time() - start) * 1000)
        prompt_len = inputs['input_ids'].shape[1]
        out_tokens = gen.sequences[0][prompt_len:]
        answer = tokenizer.decode(out_tokens, skip_special_tokens=True).strip()
        answer = _cleanup_answer(answer)
        return answer, latency_ms, _conf(gen), len(out_tokens) >= max_new_tokens
    except Exception as e:
        print(f"[ERROR] génération RAFT: {e}")
        return "", 0, None, False

print('Mode inférence RAFT prêt.')"""
    for c in nb["cells"]:
        if c.get("id") == "cell-inference-setup":
            c["source"] = to_src(new_code)
            c["execution_count"] = None
            c["outputs"] = []
            break
    json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def patch_06():
    p = ROOT / "06_evaluation.ipynb"
    nb = json.load(open(p, encoding="utf-8"))
    for c in nb["cells"]:
        if c.get("id") == "cell-eval-run":
            txt = "".join(c["source"])
            if "METHODS_ORDER = [r['method'] for r in all_results]" not in txt:
                txt += "\nMETHODS_ORDER = [r['method'] for r in all_results]\nMETHOD_COLORS = ['#4C72B0','#55A868','#C44E52','#8172B2','#CCB974'][:len(METHODS_ORDER)]\n"
            c["source"] = to_src(txt)
            c["execution_count"] = None
            c["outputs"] = []
            break
    json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    patch_07()
    patch_06()
    print("patched 07 and 06")

