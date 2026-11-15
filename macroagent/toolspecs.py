"""
Single definition of the agent's tools: name, description, JSON-schema parameters and implementation.
Used by the MCP server (macroagent.mcp_server) and by the in-process toolbox, so the two cannot drift.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from . import tools as T

ToolFn = Callable[[T.DataProvider, Dict[str, Any]], Dict[str, Any]]

HORIZON = {"type": "integer", "minimum": 1, "maximum": 24,
           "description": "Months ahead, 1-24. 'Next year' = 12, 'next quarter' = 3."}

SPECS: List[Dict[str, Any]] = [
    {"name": "latest_data",
     "description": "Latest observed Brent crude price (USD/bbl), UK CPI inflation (%) and Bank Rate (%), with their months.",
     "parameters": {"type": "object", "properties": {}},
     "fn": lambda p, a: T.latest_data(p)},
    {"name": "oil_forecast",
     "description": "Forecast the Brent crude oil price (USD/bbl) a number of months ahead, with the model's backtest error.",
     "parameters": {"type": "object", "properties": {"horizon": HORIZON}, "required": ["horizon"]},
     "fn": lambda p, a: T.oil_forecast(p, int(a.get("horizon", 12)))},
    {"name": "cpi_forecast",
     "description": "Forecast UK CPI inflation (annual %, ARX/AR/naive models) a number of months ahead, with backtest errors.",
     "parameters": {"type": "object", "properties": {"horizon": HORIZON}, "required": ["horizon"]},
     "fn": lambda p, a: T.cpi_forecast(p, int(a.get("horizon", 12)))},
    {"name": "oil_cpi_scenario",
     "description": ("Estimated change in UK CPI inflation (percentage points, with 95% CIs, after 1/3/6/12 months) for a "
                     "HYPOTHETICAL Brent move stated by the user. Pass oil_change_pct for a percentage move (e.g. 20, -15) "
                     "OR target_price_usd for a price level (e.g. 100 for '$100 a barrel'); the tool converts a price "
                     "using the latest Brent price itself. For the effect of the model's own oil FORECAST, use "
                     "oil_forecast_to_cpi instead."),
     "parameters": {"type": "object", "properties": {
         "oil_change_pct": {"type": "number", "description": "Sustained change in Brent, percent (e.g. 20 or -15)."},
         "target_price_usd": {"type": "number", "description": "Target Brent price in USD per barrel (e.g. 100)."}},
         "required": [], "one_of": ["oil_change_pct", "target_price_usd"]},
     "fn": lambda p, a: T.oil_cpi_scenario(p, a.get("oil_change_pct"), a.get("target_price_usd"))},
    {"name": "oil_forecast_to_cpi",
     "description": ("What the oil price FORECAST implies for UK CPI inflation, in one call: forecasts Brent for the horizon, "
                     "then applies the oil-to-CPI effect of the forecast change. Use this for 'what does the oil forecast "
                     "imply for inflation'; do not chain oil_forecast and oil_cpi_scenario yourself."),
     "parameters": {"type": "object", "properties": {"horizon": HORIZON}, "required": ["horizon"]},
     "fn": lambda p, a: T.oil_forecast_to_cpi(p, int(a.get("horizon", 12)))},
    {"name": "model_accuracy",
     "description": "Backtest accuracy of the oil and CPI models: 1-month RMSE relative to the naive benchmark (below 1 = better).",
     "parameters": {"type": "object", "properties": {}},
     "fn": lambda p, a: T.model_accuracy(p)},
]
BY_NAME = {s["name"]: s for s in SPECS}


def coerce_args(name: str, args: Any) -> Dict[str, Any]:
    """Models sometimes send numbers as strings or extra keys: keep declared params, cast to schema types."""
    if not isinstance(args, dict):
        args = {}
    props = BY_NAME[name]["parameters"].get("properties", {})
    out = {}
    for k, schema in props.items():
        if k not in args or args[k] in (None, ""):
            continue
        v = args[k]
        try:
            if schema["type"] == "integer":
                v = int(round(float(str(v).replace("%", "").strip())))
                v = max(schema.get("minimum", v), min(schema.get("maximum", v), v))
            elif schema["type"] == "number":
                v = float(str(v).replace("%", "").replace("+", "").strip())
        except (TypeError, ValueError):
            continue
        out[k] = v
    return out


def run_tool(provider: T.DataProvider, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name not in BY_NAME:
        return {"error": f"unknown tool {name!r}; available: {sorted(BY_NAME)}"}
    a = coerce_args(name, args)
    params = BY_NAME[name]["parameters"]
    missing = [k for k in params.get("required", []) if k not in a]
    if missing:
        return {"error": f"missing required argument(s) {missing} for {name}"}
    one_of = params.get("one_of")
    if one_of and sum(k in a for k in one_of) != 1:
        return {"error": f"{name} needs exactly one of {one_of}"}
    try:
        return BY_NAME[name]["fn"](provider, a)
    except Exception as e:  # tool failure is reported to the model, not raised
        return {"error": f"{type(e).__name__}: {e}"}
