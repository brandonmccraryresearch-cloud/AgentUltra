"""MCP (Model Context Protocol) base client.

Provides the abstract base class and common infrastructure for communicating
with specialized MCP servers (math-mcp, lean-lsp-mcp, lattice-mcp, etc.)
using JSON-RPC 2.0.
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class MCPServerType(str, Enum):
    """Enumeration of supported MCP server types."""

    MATH = "math-mcp"
    QUANTUM = "quantum-mcp"
    MOLECULAR = "molecular-mcp"
    NEURAL = "neural-mcp"
    LEAN_LSP = "lean-lsp-mcp"
    LATTICE = "lattice-mcp"
    GROUP_THEORY = "group-theory-mcp"
    EXPERIMENTAL_DATA = "experimental-data-mcp"
    PHYS = "phys-mcp"
    SYMBOLICA = "symbolica-mcp"


@dataclass
class MCPRequest:
    """A JSON-RPC 2.0 request to an MCP server."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_json(self) -> str:
        """Serialize to JSON-RPC 2.0 format."""
        payload = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": self.method,
            "params": self.params,
        }
        return json.dumps(payload)


@dataclass
class MCPResponse:
    """A JSON-RPC 2.0 response from an MCP server."""

    request_id: str
    result: Any = None
    error: dict[str, Any] | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None

    @classmethod
    def from_json(cls, data: str) -> MCPResponse:
        """Deserialize from a JSON-RPC 2.0 response string."""
        payload = json.loads(data)
        return cls(
            request_id=payload.get("id", ""),
            result=payload.get("result"),
            error=payload.get("error"),
        )


class MCPClient(ABC):
    """Abstract base class for MCP server clients.

    Subclasses implement the transport layer (stdio, HTTP/SSE, etc.)
    while this class handles request/response serialization and
    method dispatch.
    """

    def __init__(self, server_type: MCPServerType, url: str = "") -> None:
        self.server_type = server_type
        self.url = url
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the MCP server."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the MCP server."""

    @abstractmethod
    async def send(self, request: MCPRequest) -> MCPResponse:
        """Send a request and receive a response."""

    async def call(self, method: str, **params: Any) -> Any:
        """High-level RPC call: send a request and return the result.

        Raises ``RuntimeError`` on JSON-RPC errors, and also when the
        client is not connected — an RPC must never silently succeed
        against a dead transport, or lifecycle bugs stay hidden.
        """
        if not self._connected:
            raise RuntimeError(
                f"Cannot call {self.server_type.value}.{method}: "
                "MCP client is not connected (call connect() first)"
            )
        req = MCPRequest(method=method, params=params)
        logger.debug("MCP call: %s.%s(%s)", self.server_type.value, method, params)
        resp = await self.send(req)
        if resp.is_error:
            raise RuntimeError(
                f"MCP error from {self.server_type.value}: {resp.error}"
            )
        return resp.result


class StubMCPClient(MCPClient):
    """In-memory stub MCP client for testing and development.

    Returns canned responses so the rest of the system can be exercised
    without running real MCP servers.
    """

    def __init__(
        self,
        server_type: MCPServerType,
        url: str = "",
        *,
        responses: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(server_type, url)
        self._responses: dict[str, Any] = responses or {}

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def send(self, request: MCPRequest) -> MCPResponse:
        if not self._connected:
            raise RuntimeError(
                f"Cannot send {request.method}: "
                "StubMCPClient is not connected (call connect() first)"
            )
        result = self._responses.get(request.method, {"status": "stub_response"})
        return MCPResponse(request_id=request.request_id, result=result)


class MCPServerRegistry:
    """Registry of available MCP server clients."""

    def __init__(self) -> None:
        self._clients: dict[MCPServerType, MCPClient] = {}

    def register(self, client: MCPClient) -> None:
        """Register an MCP client."""
        self._clients[client.server_type] = client

    def get(self, server_type: MCPServerType) -> MCPClient | None:
        """Retrieve a registered client."""
        return self._clients.get(server_type)

    def list_servers(self) -> list[MCPServerType]:
        """List all registered server types."""
        return list(self._clients.keys())

    async def connect_all(self) -> None:
        """Connect to all registered servers."""
        for client in self._clients.values():
            await client.connect()

    async def disconnect_all(self) -> None:
        """Disconnect from all registered servers."""
        for client in self._clients.values():
            await client.disconnect()
