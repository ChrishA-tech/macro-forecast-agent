"""
Copy the three component packages (and their cached data/results) from sibling repos:
  ../oil-price-forecasting/oilcast, ../uk-cpi-forecasting/ukcpi, ../langgraph-prompt-guard/promptguard
Run after updating any of them:  python sync_vendor.py
"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = {"oilcast": "oil-price-forecasting", "ukcpi": "uk-cpi-forecasting", "promptguard": "langgraph-prompt-guard"}
DATA = {"oil-price-forecasting": ["data/brent_monthly.csv", "results/backtest.json"],
        "uk-cpi-forecasting": ["data/uk_macro_monthly.csv", "data/sources.json", "results/backtest.json"],
        "langgraph-prompt-guard": ["results/eval.json", "models/prompt_injection_clf.joblib",
                                   "models/prompt_injection_clf.json"]}

for pkg, repo in SRC.items():
    src = HERE.parent / repo / pkg
    if not src.exists():
        raise SystemExit(f"Missing {src}")
    dst = HERE / "vendor" / pkg
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    for rel in DATA[repo]:
        f = HERE.parent / repo / rel
        if f.exists():
            out = HERE / "vendor_data" / repo / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
    print(f"synced {pkg} from {repo}")
