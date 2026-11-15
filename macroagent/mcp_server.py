"""
MCP server exposing the forecasting tools over stdio.

    python -m macroagent.mcp_server                 # real data from vendor_data/
    MACROAGENT_SYNTHETIC=1 python -m macroagent.mcp_server   # synthetic data (tests, demos)

Any MCP client can use it, e.g. Claude Desktop config:
    {"mcpServers": {"macro-forecast": {"command": "python", "args": ["-m", "macroagent.mcp_server"],
                                       "cwd": "/path/to/macro-forecast-agent"}}}
"""
from __future__ import annotations

import json
import os

from mcp.server.fastmcp import FastMCP

from .toolspecs import BY_NAME, run_tool

server = FastMCP("macro-forecast", log_level=os.environ.get("FASTMCP_LOG_LEVEL", "WARNING"))
_provider = None


def provider():
    global _provider
    if _provider is None:
        if os.environ.get("MACROAGENT_SYNTHETIC") == "1":
            from .demo import synthetic_provider
            _provider = synthetic_provider()
        else:
            from .tools import DataProvider
            _provider = DataProvider.from_vendor_data()
    return _provider


def _call(name, **args) -> str:
    return json.dumps(run_tool(provider(), name, args), default=str)


@server.tool(description=BY_NAME["latest_data"]["description"])
def latest_data() -> str:
    return _call("latest_data")


@server.tool(description=BY_NAME["oil_forecast"]["description"])
def oil_forecast(horizon: int = 12) -> str:
    return _call("oil_forecast", horizon=horizon)


@server.tool(description=BY_NAME["cpi_forecast"]["description"])
def cpi_forecast(horizon: int = 12) -> str:
    return _call("cpi_forecast", horizon=horizon)


@server.tool(description=BY_NAME["oil_cpi_scenario"]["description"])
def oil_cpi_scenario(oil_change_pct: float) -> str:
    return _call("oil_cpi_scenario", oil_change_pct=oil_change_pct)


@server.tool(description=BY_NAME["oil_forecast_to_cpi"]["description"])
def oil_forecast_to_cpi(horizon: int = 12) -> str:
    return _call("oil_forecast_to_cpi", horizon=horizon)


@server.tool(description=BY_NAME["model_accuracy"]["description"])
def model_accuracy() -> str:
    return _call("model_accuracy")


if __name__ == "__main__":
    server.run()
