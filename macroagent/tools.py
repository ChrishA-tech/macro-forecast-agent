"""
Agent tools. Each returns a dict of FACTS (numbers with labels and a source), which are the
only numbers the answer is allowed to use (see grounding.py).

Data comes from a DataProvider: real cached data synced from the component repos
(vendor_data/), or an injected provider (tests use synthetic data).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from . import VENDOR_DATA  # noqa: F401  (also puts vendor/ on sys.path)
from oilcast.models import MODEL_BY_NAME, Naive
from ukcpi.forecast import forecast_now, oil_scenario


@dataclass
class DataProvider:
    brent: pd.DataFrame                       # date, brent_usd
    macro: pd.DataFrame                       # date, cpi_yoy, bank_rate, brent_usd
    oil_backtest: Optional[Dict] = None       # oilcast results/backtest.json
    cpi_backtest: Optional[Dict] = None       # ukcpi results/backtest.json
    sources: Dict[str, str] = field(default_factory=dict)
    synthetic: bool = False

    @classmethod
    def from_vendor_data(cls, root: Path = None) -> "DataProvider":
        from oilcast.data import load_brent
        from ukcpi.data import load_macro
        root = root or VENDOR_DATA
        oil_dir, cpi_dir = root / "oil-price-forecasting", root / "uk-cpi-forecasting"
        if not (oil_dir / "data" / "brent_monthly.csv").exists() or not (cpi_dir / "data" / "uk_macro_monthly.csv").exists():
            raise FileNotFoundError("No real data in vendor_data/: run the component repos, then `python sync_vendor.py`.")
        macro = load_macro(data_dir=cpi_dir / "data")

        def j(p):
            return json.loads(p.read_text()) if p.exists() else None
        return cls(brent=load_brent(cache=oil_dir / "data" / "brent_monthly.csv", start="2000-01-01"), macro=macro,
                   oil_backtest=j(oil_dir / "results" / "backtest.json"), cpi_backtest=j(cpi_dir / "results" / "backtest.json"),
                   sources={"brent": "FRED POILBREUSDM", **macro.attrs.get("sources", {})})


def _oil_model_name(p: DataProvider) -> str:
    if p.oil_backtest and p.oil_backtest.get("best_model_h1") in MODEL_BY_NAME:
        return p.oil_backtest["best_model_h1"]
    return Naive.name


def latest_data(p: DataProvider) -> Dict[str, Any]:
    m, b = p.macro.iloc[-1], p.brent.iloc[-1]
    return {"tool": "latest_data", "source": "data" if not p.synthetic else "SYNTHETIC test data",
            "facts": {"latest Brent month": b["date"].strftime("%Y-%m"), "latest Brent price (USD/bbl)": round(float(b["brent_usd"]), 2),
                      "latest CPI month": m["date"].strftime("%Y-%m"), "latest CPI inflation (%)": round(float(m["cpi_yoy"]), 1),
                      "latest Bank Rate (%)": round(float(m["bank_rate"]), 2)},
            "sources": p.sources}


def oil_forecast(p: DataProvider, horizon: int = 12) -> Dict[str, Any]:
    horizon = int(min(max(horizon, 1), 24))
    name = _oil_model_name(p)
    y = p.brent["brent_usd"].to_numpy(dtype=float)
    fc = MODEL_BY_NAME[name]().fit(y).forecast(horizon)
    last = p.brent["date"].iloc[-1]
    target = (last + pd.DateOffset(months=horizon)).strftime("%Y-%m")
    facts = {"model": name, "latest Brent price (USD/bbl)": round(float(y[-1]), 2), "latest month": last.strftime("%Y-%m"),
             "forecast month": target, f"Brent forecast for {target} (USD/bbl)": round(float(fc[-1]), 2),
             "forecast change vs latest (%)": round(100 * (fc[-1] / y[-1] - 1), 1)}
    acc = (p.oil_backtest or {}).get("backtest", {}).get("models", {}).get(name, {}).get("horizons", {})
    h_key = min(acc, key=lambda h: abs(int(h) - horizon)) if acc else None
    if h_key:
        facts[f"backtest RMSE at {h_key}-month horizon (USD/bbl)"] = acc[h_key]["rmse"]
        if "rel_rmse_vs_naive" in acc[h_key]:
            facts[f"RMSE relative to naive at {h_key} months"] = acc[h_key]["rel_rmse_vs_naive"]
    return {"tool": "oil_forecast", "facts": facts, "path": [round(float(v), 2) for v in fc],
            "source": "oilcast model zoo (best 1-month model from rolling backtest)"}


def cpi_forecast(p: DataProvider, horizon: int = 12) -> Dict[str, Any]:
    h = int(min(max(horizon, 1), 24))
    fc = forecast_now(p.macro, (h,))[str(h)]
    facts = {"forecast month": fc["target_month"], "latest CPI inflation (%)": round(float(p.macro["cpi_yoy"].iloc[-1]), 1),
             "ARX forecast of CPI inflation (%)": fc["ARX (+ Bank Rate, Brent)"],
             "AR forecast of CPI inflation (%)": fc["AR (CPI lags)"], "naive forecast of CPI inflation (%)": fc["Naive (no change)"]}
    rows = (p.cpi_backtest or {}).get("backtest", {}).get("models", {})
    for m, label in (("ARX (+ Bank Rate, Brent)", "ARX"), ("Naive (no change)", "naive")):
        r = rows.get(m, {})
        hk = min(r, key=lambda k: abs(int(k) - h)) if r else None
        if hk:
            facts[f"{label} backtest RMSE at {hk} months (pp)"] = r[hk]["rmse"]
    return {"tool": "cpi_forecast", "facts": facts, "source": "ukcpi direct AR/ARX regressions (ONS, BoE, FRED data)"}


def oil_cpi_scenario(p: DataProvider, oil_change_pct: float) -> Dict[str, Any]:
    sc = oil_scenario(p.macro, oil_change_pct)
    facts = {"assumed Brent change (%)": round(float(oil_change_pct), 1)}
    logb = np.log(p.macro["brent_usd"].to_numpy(dtype=float))
    max_seen = float(np.nanmax(np.abs(100 * (logb[12:] - logb[:-12])))) if len(logb) > 12 else 0.0
    if abs(100 * np.log1p(oil_change_pct / 100)) > max_seen:
        facts["extrapolation warning"] = f"largest 12-month Brent move in the data is {max_seen:.0f} log points"
    for h, v in sc.items():
        facts[f"CPI inflation change after {h} months (pp)"] = v["cpi_change_pp"]
        facts[f"95% CI after {h} months (pp)"] = v["ci95"]
    return {"tool": "oil_cpi_scenario", "facts": facts,
            "source": "ARX regression coefficients on Brent (associational, not causal)"}


def oil_forecast_to_cpi(p: DataProvider, horizon: int = 12) -> Dict[str, Any]:
    """Chain: oil forecast -> implied % change -> CPI scenario."""
    o = oil_forecast(p, horizon)
    pct = o["facts"]["forecast change vs latest (%)"]
    s = oil_cpi_scenario(p, pct if abs(pct) > 0.05 else 0.0)
    return {"tool": "oil_forecast_to_cpi", "facts": {**o["facts"], **s["facts"]},
            "source": o["source"] + " + " + s["source"]}


def model_accuracy(p: DataProvider) -> Dict[str, Any]:
    facts = {}
    for label, bt in (("oil", p.oil_backtest), ("CPI", p.cpi_backtest)):
        rows = (bt or {}).get("backtest", {}).get("models", {})
        for m, v in rows.items():
            hs = v.get("horizons", v)
            if "1" in hs and "rel_rmse_vs_naive" in hs["1"]:
                facts[f"{label}: {m} 1-month RMSE relative to naive"] = hs["1"]["rel_rmse_vs_naive"]
    return {"tool": "model_accuracy", "facts": facts or {"note": "no backtest results available"},
            "source": "rolling-origin backtests (results/backtest.json)"}


def price_to_pct(p: DataProvider, target_price: float) -> float:
    last = float(p.brent["brent_usd"].iloc[-1])
    return round(100 * (target_price / last - 1), 1)


TOOLS = {"oil_forecast": oil_forecast, "cpi_forecast": cpi_forecast, "oil_cpi_scenario": oil_cpi_scenario,
         "oil_forecast_to_cpi": oil_forecast_to_cpi, "model_accuracy": model_accuracy, "latest_data": latest_data}
