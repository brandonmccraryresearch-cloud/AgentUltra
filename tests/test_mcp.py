"""Tests for the MCP client infrastructure."""

import asyncio

from agent_ultra.mcp.client import (
    MCPRequest,
    MCPResponse,
    MCPServerRegistry,
    MCPServerType,
    StubMCPClient,
)


class TestMCPRequest:
    def test_to_json(self):
        req = MCPRequest(method="solve", params={"equation": "x^2 - 1"})
        import json
        data = json.loads(req.to_json())
        assert data["jsonrpc"] == "2.0"
        assert data["method"] == "solve"
        assert data["params"]["equation"] == "x^2 - 1"


class TestMCPResponse:
    def test_from_json(self):
        raw = '{"jsonrpc":"2.0","id":"abc","result":{"x":42}}'
        resp = MCPResponse.from_json(raw)
        assert resp.request_id == "abc"
        assert resp.result == {"x": 42}
        assert not resp.is_error

    def test_error_response(self):
        raw = '{"jsonrpc":"2.0","id":"abc","error":{"code":-1,"message":"fail"}}'
        resp = MCPResponse.from_json(raw)
        assert resp.is_error


class TestStubMCPClient:
    def test_connect_disconnect(self):
        client = StubMCPClient(MCPServerType.MATH)
        assert not client.is_connected
        asyncio.get_event_loop().run_until_complete(client.connect())
        assert client.is_connected
        asyncio.get_event_loop().run_until_complete(client.disconnect())
        assert not client.is_connected

    def test_call(self):
        client = StubMCPClient(
            MCPServerType.MATH,
            responses={"solve": {"x": [1, -1]}},
        )
        asyncio.get_event_loop().run_until_complete(client.connect())
        result = asyncio.get_event_loop().run_until_complete(
            client.call("solve", equation="x^2 - 1")
        )
        assert result == {"x": [1, -1]}


class TestMCPServerRegistry:
    def test_register_and_get(self):
        registry = MCPServerRegistry()
        client = StubMCPClient(MCPServerType.MATH)
        registry.register(client)
        assert registry.get(MCPServerType.MATH) is client
        assert MCPServerType.MATH in registry.list_servers()

    def test_connect_all(self):
        registry = MCPServerRegistry()
        c1 = StubMCPClient(MCPServerType.MATH)
        c2 = StubMCPClient(MCPServerType.LEAN_LSP)
        registry.register(c1)
        registry.register(c2)
        asyncio.get_event_loop().run_until_complete(registry.connect_all())
        assert c1.is_connected
        assert c2.is_connected
