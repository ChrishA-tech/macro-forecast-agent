"""
Guardrail evaluation: original v1 rules vs v2, strict and lenient thresholds.

Datasets
  dev      tests/fixtures/dev_prompts.jsonl - 60 hand-written prompts used while building v2
           (IN-SAMPLE: the rules were written with these in view)
  deepset  deepset/prompt-injections on Hugging Face (662 labelled prompts, train+test splits,
           English and German). Not used to build the rules: HELD-OUT estimate for the rules.
           Downloaded via the Hugging Face datasets-server API and cached in data/.

Learned classifier (promptguard.classifier): trained on the deepset TRAIN split only (C and threshold by
cross-validation inside it), saved to models/, then scored once on the deepset TEST split (116 prompts) and
on the dev set (a different, hand-written dataset). The rules are scored on the same test split, so all
guardrails are compared on identical prompts. Recall carries a 95% Wilson interval: the test split is small.

    python -m promptguard.evaluate                 # dev + deepset (downloads on first run)
    python -m promptguard.evaluate --datasets dev  # offline

Writes results/eval.json, results/eval.md and the README results section.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List

from .classifier import DEFAULT_MODEL, InjectionClassifier, hybrid_assess
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


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def score(rows: List[Dict], guard: Callable, strict: bool) -> Dict:
    return score_fn(rows, lambda t: not guard(t, strict_mode=strict).is_safe)


def score_fn(rows: List[Dict], is_blocked: Callable[[str], bool]) -> Dict:
    tp = fp = fn = tn = 0
    fn_ex, fp_ex = [], []
    t0 = time.perf_counter()
    for r in rows:
        blocked = is_blocked(r["text"])
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
            "precision": round(p, 3), "recall": round(rec, 3), "recall_ci95": wilson(tp, tp + fn), "f1": round(2 * p * rec / (p + rec), 3) if p + rec else 0.0,
            "false_positive_rate": round(fp / (fp + tn), 3) if fp + tn else 0.0,
            "accuracy": round((tp + tn) / max(len(rows), 1), 3), "ms_per_prompt": round(ms, 3),
            "missed_examples": [t[:120] for t in fn_ex[:8]], "false_positive_examples": [t[:120] for t in fp_ex[:8]]}


NAMES = {"dev": "Dev set (60)", "deepset_test": "deepset test split (116, held-out)",
         "deepset": "deepset all (662, held-out for rules)"}
HEADLINE = ("deepset_test", "Hybrid (v2 strict + ML)")


def to_markdown(res: Dict) -> str:
    lines = [f"Run {res['run_at'][:10]}. A prompt counts as flagged when the guardrail blocks it. "
             "Dev set: in-sample for the rules (written with it in view), out-of-sample for the classifier.", "",
             "| Dataset | Guardrail | Precision | Recall (95% CI) | F1 | False-positive rate | Accuracy | ms/prompt |",
             "|---|---|---|---|---|---|---|---|"]
    for ds, by in res["datasets"].items():
        for g, s in by.items():
            b = "**" if (ds, g) == HEADLINE else ""
            lo, hi = s.get("recall_ci95", (0, 0))
            lines.append(f"| {NAMES.get(ds, ds)} | {b}{g}{b} | {s['precision']:.3f} | {b}{s['recall']:.3f}{b} "
                         f"({lo:.2f}-{hi:.2f}) | {b}{s['f1']:.3f}{b} | {s['false_positive_rate']:.3f} | "
                         f"{s['accuracy']:.3f} | {s['ms_per_prompt']} |")
    t = res["datasets"].get("deepset_test", {})
    if "Hybrid (v2 strict + ML)" in t:
        h, r = t["Hybrid (v2 strict + ML)"], t["v2 strict"]
        lines += ["", f"Held-out (deepset test split): rules alone flag {r['tp']}/{r['attacks']} attacks; rules + "
                  f"classifier flag {h['tp']}/{h['attacks']} with {h['fp']}/{h['benign']} benign prompts wrongly "
                  f"flagged. The classifier was trained on the train split only "
                  f"(C={res['classifier']['C']}, threshold={res['classifier']['threshold']}, chosen by CV)."]
    elif "deepset" in res["datasets"]:
        s = res["datasets"]["deepset"]["v2 strict"]
        lines += ["", f"Held-out (deepset, v2 strict): {s['tp']}/{s['attacks']} attacks flagged, "
                  f"{s['fp']}/{s['benign']} benign prompts wrongly flagged."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["dev", "deepset"], choices=["dev", "deepset"])
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args(argv)
    res = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "datasets": {}}
    failed = []
    dev = load_dev()
    deep = None
    if "deepset" in a.datasets:
        try:
            deep = load_deepset(a.refresh)
        except Exception as e:   # e.g. no network: keep going with the dev set
            print(f"!! Could not load deepset: {type(e).__name__}: {e}")
            failed.append("deepset")

    clf = None
    if deep:
        train = [r for r in deep if r["split"] == "train"]
        clf = InjectionClassifier.train([r["text"] for r in train], [r["label"] for r in train],
                                        data_name="deepset/prompt-injections train split")
        clf.save(DEFAULT_MODEL)
        res["classifier"] = clf.card
    else:
        clf = InjectionClassifier.load_default()
        if clf:
            res["classifier"] = clf.card

    def guards(include_ml: bool):
        g = {"v1 (original)": lambda t: not guard_v1(t, strict_mode=True).is_safe,
             "v2 strict": lambda t: not guard_v2(t, strict_mode=True).is_safe,
             "v2 lenient": lambda t: not guard_v2(t, strict_mode=False).is_safe}
        if include_ml and clf:
            g["ML classifier"] = clf.is_attack
            g["Hybrid (v2 strict + ML)"] = lambda t: not hybrid_assess(t, clf, True).is_safe
        return g

    if "dev" in a.datasets:
        res["datasets"]["dev"] = {n: score_fn(dev, f) for n, f in guards(True).items()}
    if deep:
        test = [r for r in deep if r["split"] == "test"]
        res["datasets"]["deepset_test"] = {n: score_fn(test, f) for n, f in guards(True).items()}
        res["datasets"]["deepset"] = {n: score_fn(deep, f) for n, f in guards(False).items()}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "eval.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    md = to_markdown(res)
    ex = ["", "## Error analysis", ""]
    for ds, by in res["datasets"].items():
        g = "Hybrid (v2 strict + ML)" if "Hybrid (v2 strict + ML)" in by else "v2 strict"
        s = by[g]
        ex += [f"**{ds} ({g}): missed attacks (first 8)**", ""] + [f"- `{t}`" for t in s["missed_examples"]] + [""]
        ex += [f"**{ds} ({g}): false positives (first 8)**", ""] + [f"- `{t}`" for t in s["false_positive_examples"]] + [""]
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
