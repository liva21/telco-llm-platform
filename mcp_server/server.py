"""
MCP Server - Model Context Protocol uygulamasi.
Fine-tuned modelin cagirabilecegi araclari (tools) yayinlar.

Hem resmi MCP SDK (stdio/JSON-RPC) hem de basit HTTP modu sunar.
Uretim icin: mcp.server (stdio)
Geliştirme/test icin: --http ile FastAPI tabanli HTTP mod

Kullanim:
    python mcp_server/server.py              # MCP stdio modu
    python mcp_server/server.py --http       # HTTP test modu :8765
"""
import argparse
import asyncio
import json
import sys
from typing import Any, Dict

from mcp_server.tools import (
    check_network_status,
    get_customer_info,
    query_knowledge_base,
)

TOOL_REGISTRY = {
    "query_knowledge_base": {
        "fn": query_knowledge_base,
        "description": "Telekom bilgi tabanında ilgili dokumanlari arar.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Arama sorgusu"},
                "top_k": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
    },
    "get_customer_info": {
        "fn": get_customer_info,
        "description": "Telefon numarasina gore musteri bilgilerini getirir.",
        "input_schema": {
            "type": "object",
            "properties": {"phone": {"type": "string"}},
            "required": ["phone"],
        },
    },
    "check_network_status": {
        "fn": check_network_status,
        "description": "Belirtilen bolgedeki sebeke durumunu kontrol eder.",
        "input_schema": {
            "type": "object",
            "properties": {"region": {"type": "string"}},
            "required": ["region"],
        },
    },
}


def call_tool(name: str, args: Dict[str, Any]) -> Any:
    if name not in TOOL_REGISTRY:
        return {"error": f"bilinmeyen tool: {name}"}
    try:
        return TOOL_REGISTRY[name]["fn"](**args)
    except TypeError as e:
        return {"error": f"gecersiz parametre: {e}"}
    except Exception as e:
        return {"error": f"tool hatasi: {e}"}


def list_tools_spec():
    return [
        {"name": n, "description": v["description"], "inputSchema": v["input_schema"]}
        for n, v in TOOL_REGISTRY.items()
    ]


# ----------------------- HTTP mod (dev/test) -----------------------
def run_http(port: int = 8765):
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="Telco MCP Server (HTTP dev mode)")

    class ToolCall(BaseModel):
        name: str
        arguments: Dict[str, Any] = {}

    @app.get("/tools")
    def tools():
        return {"tools": list_tools_spec()}

    @app.post("/call")
    def call(req: ToolCall):
        return {"name": req.name, "result": call_tool(req.name, req.arguments)}

    @app.get("/health")
    def health():
        return {"status": "ok", "tools": list(TOOL_REGISTRY.keys())}

    uvicorn.run(app, host="0.0.0.0", port=port)


# ----------------------- MCP stdio mode -----------------------
async def run_stdio():
    """
    Resmi MCP SDK ile stdio uzerinden JSON-RPC.
    mcp paketi kurulu degilse, basit fallback JSON line protokolu kullanilir.
    """
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
    except ImportError:
        print("[warn] mcp paketi bulunamadi, fallback stdio JSON-lines aktif", file=sys.stderr)
        await _fallback_stdio()
        return

    server = Server("telco-llm-mcp")

    @server.list_tools()
    async def _list():
        return [
            Tool(name=n, description=v["description"], inputSchema=v["input_schema"])
            for n, v in TOOL_REGISTRY.items()
        ]

    @server.call_tool()
    async def _call(name: str, arguments: dict):
        result = call_tool(name, arguments or {})
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


async def _fallback_stdio():
    """MCP SDK yoksa cok basit line-delimited JSON protokolu."""
    loop = asyncio.get_event_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        try:
            req = json.loads(line)
        except Exception:
            continue
        method = req.get("method")
        if method == "tools/list":
            resp = {"id": req.get("id"), "result": {"tools": list_tools_spec()}}
        elif method == "tools/call":
            p = req.get("params", {})
            resp = {"id": req.get("id"),
                    "result": call_tool(p.get("name"), p.get("arguments", {}))}
        else:
            resp = {"id": req.get("id"), "error": f"unknown method {method}"}
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="HTTP dev modunda calistir")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.http:
        run_http(args.port)
    else:
        asyncio.run(run_stdio())


if __name__ == "__main__":
    main()