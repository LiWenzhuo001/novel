"""MCP 服务（Model Context Protocol）——对外暴露小说知识库检索/列文件工具。

基于已安装的 mcp 2.0 低层 Server API（stdio 传输），无需新增依赖。可作为独立进程
运行，供支持 MCP 的客户端（Claude Desktop、IDE 插件等）按需接入小说知识库。

运行：
    cd backend
    ../.venv/Scripts/python.exe -m app.mcp_server

工具：
    - search_kb(query, top_k=5): 混合检索小说原文，返回 [来源, 片段, 分数]
    - list_kb_files():            列出已入库的知识库文件
"""

import asyncio
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    TextContent,
    Tool,
)

from app.core.rag import similarity_search
from app.services import kb_service


server = Server("novel-agent-kb")


async def handle_list_tools(request: ListToolsRequest) -> ListToolsResult:
    tools = [
        Tool(
            name="search_kb",
            description="在小说知识库中混合检索（向量+全文+RRF+重排），返回最相关的原文片段及其来源文件、章节与相似度分数。用于回答人物关系、情节发展、时间线、章节定位等问题。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索问句，如『主角第一次见到反派是在哪一章』"},
                    "top_k": {"type": "integer", "description": "返回条数，默认 5", "default": 5},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_kb_files",
            description="列出当前知识库中已入库（已索引完成）的文件清单，含文件名与切片数。",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]
    return ListToolsResult(tools=tools)


async def handle_call_tool(request: CallToolRequest) -> CallToolResult:
    name = request.params.name
    arguments = request.params.arguments or {}
    try:
        if name == "search_kb":
            query = arguments.get("query", "")
            top_k = int(arguments.get("top_k", 5) or 5)
            if not query.strip():
                raise ValueError("search_kb 需要提供 query 参数")
            docs = await similarity_search(query, k=top_k)
            payload = [
                {
                    "source": d.metadata.get("source", "未知"),
                    "score": d.metadata.get("score", 0.0),
                    "snippet": d.page_content[:400],
                }
                for d in docs
            ]
            text = json.dumps(payload, ensure_ascii=False)
        elif name == "list_kb_files":
            files = await kb_service.list_files()
            text = json.dumps(files, ensure_ascii=False)
        else:
            raise ValueError(f"未知工具：{name}")
        return CallToolResult(content=[TextContent(type="text", text=text)], is_error=False)
    except Exception as e:  # noqa: BLE001
        return CallToolResult(content=[TextContent(type="text", text=f"工具执行失败：{e}")], is_error=True)


server.add_request_handler("tools/list", ListToolsRequest, handle_list_tools)
server.add_request_handler("tools/call", CallToolRequest, handle_call_tool)


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
