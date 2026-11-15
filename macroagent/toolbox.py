"""Two ways for the agent to reach the tools, behind one async interface: in-process, or over MCP (stdio)."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

from . import ROOT
from .toolspecs import SPECS, run_tool


class LocalToolbox:
    transport = "in-process"

    def __init__(self, provider):
        self.provider = provider

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def list_tools(self) -> List[Dict[str, Any]]:
        return [{"name": s["name"], "description": s["description"], "parameters": s["parameters"]} for s in SPECS]

    async def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return run_tool(self.provider, name, args)


class MCPToolbox:
    """Starts `python -m macroagent.mcp_server` and talks to it with the official MCP client."""
    transport = "MCP (stdio)"

    def __init__(self, synthetic: bool = False):
        self.synthetic = synthetic

    async def __aenter__(self):
        from contextlib import AsyncExitStack
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        env.setdefault("FASTMCP_LOG_LEVEL", "WARNING")      # keep the server's per-request logs out of the run log
        if self.synthetic:
            env["MACROAGENT_SYNTHETIC"] = "1"
        params = StdioServerParameters(command=sys.executable, args=["-m", "macroagent.mcp_server"], env=env,
                                       cwd=str(ROOT))
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        return self

    async def __aexit__(self, *exc):
        await self._stack.aclose()
        return False

    async def list_tools(self) -> List[Dict[str, Any]]:
        res = await self.session.list_tools()
        return [{"name": t.name, "description": t.description or "", "parameters": t.inputSchema} for t in res.tools]

    async def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        res = await self.session.call_tool(name, args)
        text = "".join(getattr(c, "text", "") for c in res.content)
        if res.isError:
            return {"error": text or "tool error"}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"error": f"non-JSON tool output: {text[:200]}"}
