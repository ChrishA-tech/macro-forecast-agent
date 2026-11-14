"""SYNTHETIC demo data provider (tests and offline demo only; clearly labelled in all outputs)."""
import numpy as np
import pandas as pd

from macroagent.tools import DataProvider


def synthetic_provider(n: int = 300, seed: int = 0, with_backtests: bool = True) -> DataProvider:
    rng = np.random.default_rng(seed)
    logb = np.log(60) + np.cumsum(rng.normal(0, 0.07, n))
    cpi, rate = np.zeros(n), np.zeros(n)
    cpi[:13], rate[:13] = 2.0, 3.0
    for t in range(13, n):
        cpi[t] = 0.3 + 0.88 * cpi[t - 1] + 0.006 * 100 * (logb[t - 1] - logb[t - 13]) - 0.03 * (rate[t - 1] - 3) + rng.normal(0, 0.18)
        rate[t] = np.clip(rate[t - 1] + 0.05 * (cpi[t - 1] - 2) + rng.normal(0, 0.05), 0.1, 8)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    macro = pd.DataFrame({"date": dates, "cpi_yoy": cpi.round(2), "bank_rate": rate.round(3), "brent_usd": np.exp(logb).round(2)})
    oil_bt = cpi_bt = None
    if with_backtests:
        oil_bt = {"best_model_h1": "Naive (random walk)", "backtest": {"models": {
            "Naive (random walk)": {"horizons": {"1": {"rmse": 6.1}, "12": {"rmse": 18.2}}},
            "AR(12) on levels (original)": {"horizons": {"1": {"rmse": 6.4, "rel_rmse_vs_naive": 1.05}, "12": {"rmse": 19.0, "rel_rmse_vs_naive": 1.04}}}}}}
        cpi_bt = {"backtest": {"models": {
            "Naive (no change)": {"1": {"rmse": 0.31}, "12": {"rmse": 2.1}},
            "ARX (+ Bank Rate, Brent)": {"1": {"rmse": 0.29, "rel_rmse_vs_naive": 0.94}, "12": {"rmse": 1.8, "rel_rmse_vs_naive": 0.86}}}}}
    return DataProvider(brent=macro[["date", "brent_usd"]].copy(), macro=macro, oil_backtest=oil_bt,
                        cpi_backtest=cpi_bt, sources={"cpi_yoy": "synthetic"}, synthetic=True)
