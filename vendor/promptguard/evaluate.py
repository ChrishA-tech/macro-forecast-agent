"""
Guardrail evaluation: original v1 rules vs v2, strict and lenient thresholds.

Datasets
  dev      tests/fixtures/dev_prompts.jsonl - 60 hand-written prompts used while building v2
           (IN-SAMPLE: the rules were written with these in view)
  deepset  deepset/prompt-injections on Hugging Face (662 labelled prompts, train+test splits,
           English and German). Not used to build the rules: HELD-OUT estimate.
           Downloaded via the Hugging Face datasets-server API and cached in data/.

    python -m promptguard.evaluate                 # dev + deepset (downloads on first run)
    python -m promptguard.evaluate --datasets dev  # offline

Writes results/eval.json, results/eval.md and the README results section.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List

from .guardrails import evaluate_prompt_safety as guard_v2
from .guardrails_v1 import evaluate_prompt_safety as guard_v1

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
HF_ROWS = "https://datasets-server.huggingface.co/rows?dataset={ds}&config=default&split={split}&offset={off}&length=100"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def load_dev() -> List[Dict]:
    return [json.loads(l) for l in (ROOT / "tests" / "fixtures" / "dev_prompts.jsonl").read_text().splitlines() if l.strip()]


def load_deepset(refresh: bool = False) -> List[Dict]:
    cache = DATA / "deepset_prompt_injections.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    rows: List[Dict] = []
    for split in ("train", "test"):
        off, total = 0, None
        while total is None or off < total:
            url = HF_ROWS.format(ds=urllib.parse.quote("deepset/prompt-injections", safe=""), split=split, off=off)
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "promptguard-eval"}), timeout=30) as r:
                page = json.loads(r.read())
            total = page["num_rows_total"]
            rows += [{"text": x["row"]["text"], "label": int(x["row"]["label"]), "split": split} for x in page["rows"]]
            off += 100
    DATA.mkdir(exist_ok=True)
    cache.write_text(json.dumps(rows, ensure_ascii=False))
    return rows


def score(rows: List[Dict], guard: Callable, strict: bool) -> Dict:
    tp = fp = fn = tn = 0
    fn_ex, fp_ex = [], []
    t0 = time.perf_counter()
    for r in rows:
        blocked = not guard(r["text"], strict_mode=strict).is_safe
        if r["label"] == 1 and blocked:
            tp += 1
        elif r["label"] == 1:
            fn += 1
            fn_ex.append(r["text"])
        elif blocked:
            fp += 1
            fp_ex.append(r["text"])
        else:
            tn += 1
    ms = (time.perf_counter() - t0) * 1000 / max(len(rows), 1)
    p = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(rows), "attacks": tp + fn, "benign": fp + tn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(p, 3), "recall": round(rec, 3), "f1": round(2 * p * rec / (p + rec), 3) if p + rec else 0.0,
            "false_positive_rate": round(fp / (fp + tn), 3) if fp + tn else 0.0,
            "accuracy": round((tp + tn) / max(len(rows), 1), 3), "ms_per_prompt": round(ms, 3),
            "missed_examples": [t[:120] for t in fn_ex[:8]], "false_positive_examples": [t[:120] for t in fp_ex[:8]]}


def to_markdown(res: Dict) -> str:
    lines = [f"Run {res['run_at'][:10]}. A prompt counts as flagged when the guardrail blocks it.", "",
             "| Dataset | Guardrail | Precision | Recall | F1 | False-positive rate | Accuracy |",
             "|---|---|---|---|---|---|---|"]
    names = {"dev": "Dev set (60, in-sample)", "deepset": "deepset/prompt-injections (662, held-out)"}
    for ds, by in res["datasets"].items():
        for g, s in by.items():
            bold = "**" if ds == "deepset" and g == "v2 strict" else ""
            lines.append(f"| {names.get(ds, ds)} | {g} | {s['precision']:.3f} | {bold}{s['recall']:.3f}{bold} | "
                         f"{bold}{s['f1']:.3f}{bold} | {s['false_positive_rate']:.3f} | {s['accuracy']:.3f} |")
    if "deepset" in res["datasets"]:
        s = res["datasets"]["deepset"]["v2 strict"]
        lines += ["", f"Held-out (deepset, v2 strict): {s['tp']}/{s['attacks']} attacks flagged, "
                  f"{s['fp']}/{s['benign']} benign prompts wrongly flagged, {s['ms_per_prompt']} ms per prompt. "
                  "Rule-based screening misses many paraphrased and non-English injections; see results/eval.md "
                  "for examples."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["dev", "deepset"], choices=["dev", "deepset"])
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args(argv)
    loaders = {"dev": load_dev, "deepset": lambda: load_deepset(a.refresh)}
    res = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "datasets": {}}
    failed = []
    for ds in a.datasets:
        try:
            rows = loaders[ds]()
        except Exception as e:   # e.g. no network: keep going with the other datasets
            print(f"!! Could not load {ds}: {type(e).__name__}: {e}")
            failed.append(ds)
            continue
        res["datasets"][ds] = {"v1 (original)": score(rows, guard_v1, True),
                               "v2 strict": score(rows, guard_v2, True),
                               "v2 lenient": score(rows, guard_v2, False)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "eval.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    md = to_markdown(res)
    ex = ["", "## Error analysis (v2 strict)", ""]
    for ds, by in res["datasets"].items():
        s = by["v2 strict"]
        ex += [f"**{ds}: missed attacks (first 8)**", ""] + [f"- `{t}`" for t in s["missed_examples"]] + [""]
        ex += [f"**{ds}: false positives (first 8)**", ""] + [f"- `{t}`" for t in s["false_positive_examples"]] + [""]
    (RESULTS / "eval.md").write_text("# Guardrail evaluation\n\n" + md + "\n".join(ex))
    readme = ROOT / "README.md"
    s = readme.read_text(encoding="utf-8")
    if START in s and "deepset" in res["datasets"]:
        readme.write_text(s[:s.index(START) + len(START)] + "\n" + md + s[s.index(END):], encoding="utf-8")
    print(md)
    if failed:
        raise SystemExit(f"Dataset(s) not evaluated: {', '.join(failed)} (results for the others were saved).")
    return res


if __name__ == "__main__":
    main()
