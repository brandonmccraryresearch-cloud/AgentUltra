"""Specialized MCP server interfaces.

Provides typed wrappers around the generic MCP client for each of the
specialized computation servers described in the specification.
"""

from __future__ import annotations

from typing import Any

from agent_ultra.mcp.client import MCPClient, MCPServerType, StubMCPClient


async def _ensure_connected(client: MCPClient) -> None:
    """Explicit auto-connect policy for the typed wrappers.

    The wrappers own their client (defaulting to an in-memory stub), so
    they connect it on first use instead of requiring the caller to
    manage the lifecycle. ``MCPClient.call()`` itself still refuses to
    run while disconnected, so a client used directly can never silently
    succeed against a dead transport.
    """
    if not client.is_connected:
        await client.connect()


class MathMCPClient:
    """Interface to the math-mcp server for symbolic algebra operations."""

    def __init__(self, client: MCPClient | None = None) -> None:
        self._client = client or StubMCPClient(MCPServerType.MATH)

    async def solve(self, equation: str, variable: str) -> Any:
        await _ensure_connected(self._client)
        return await self._client.call("solve", equation=equation, variable=variable)

    async def differentiate(self, expression: str, variable: str) -> Any:
        await _ensure_connected(self._client)
        return await self._client.call(
            "differentiate", expression=expression, variable=variable
        )

    async def integrate(self, expression: str, variable: str, **kwargs: Any) -> Any:
        await _ensure_connected(self._client)
        return await self._client.call(
            "integrate", expression=expression, variable=variable, **kwargs
        )

    async def simplify(self, expression: str) -> Any:
        await _ensure_connected(self._client)
        return await self._client.call("simplify", expression=expression)

    async def fft(self, data: list[float]) -> Any:
        await _ensure_connected(self._client)
        return await self._client.call("fft", data=data)


class LeanLSPClient:
    """Interface to the lean-lsp-mcp server for formal verification."""

    def __init__(self, client: MCPClient | None = None) -> None:
        self._client = client or StubMCPClient(MCPServerType.LEAN_LSP)

    async def check_proof(self, lean_code: str) -> Any:
        await _ensure_connected(self._client)
        return await self._client.call("check_proof", code=lean_code)

    async def tactic_search(self, goal_state: str) -> Any:
        await _ensure_connected(self._client)
        return await self._client.call("tactic_search", goal=goal_state)

    async def library_search(self, query: str, *, method: str = "loogle") -> Any:
        await _ensure_connected(self._client)
        return await self._client.call(
            "library_search", query=query, method=method
        )

    async def find_counterexample(self, theorem: str) -> Any:
        await _ensure_connected(self._client)
        return await self._client.call("find_counterexample", theorem=theorem)


class LatticeMCPClient:
    """Interface to the lattice-mcp server for lattice field theory."""

    def __init__(self, client: MCPClient | None = None) -> None:
        self._client = client or StubMCPClient(MCPServerType.LATTICE)

    async def wilson_loop(self, **params: Any) -> Any:
        await _ensure_connected(self._client)
        return await self._client.call("wilson_loop", **params)

    async def propagator(self, **params: Any) -> Any:
        await _ensure_connected(self._client)
        return await self._client.call("propagator", **params)

    async def spectral_function(self, **params: Any) -> Any:
        await _ensure_connected(self._client)
        return await self._client.call("spectral_function", **params)


class GroupTheoryMCPClient:
    """Interface to the group-theory-mcp server for Lie algebra computations."""

    def __init__(self, client: MCPClient | None = None) -> None:
        self._client = client or StubMCPClient(MCPServerType.GROUP_THEORY)

    async def root_system(self, algebra_type: str, rank: int) -> Any:
        await _ensure_connected(self._client)
        return await self._client.call(
            "root_system", type=algebra_type, rank=rank
        )

    async def branching_rule(
        self, group_g: str, subgroup_h: str, representation: str
    ) -> Any:
        await _ensure_connected(self._client)
        return await self._client.call(
            "branching_rule", G=group_g, H=subgroup_h, rep=representation
        )

    async def casimir(self, group: str, representation: str) -> Any:
        await _ensure_connected(self._client)
        return await self._client.call(
            "casimir", group=group, representation=representation
        )

    async def weyl_group(self, algebra_type: str, rank: int) -> Any:
        await _ensure_connected(self._client)
        return await self._client.call(
            "weyl_group", type=algebra_type, rank=rank
        )


class ExperimentalDataMCPClient:
    """Interface to the experimental-data-mcp server (PDG, CODATA, etc.)."""

    def __init__(self, client: MCPClient | None = None) -> None:
        self._client = client or StubMCPClient(MCPServerType.EXPERIMENTAL_DATA)

    async def pdg_particle(self, name: str) -> Any:
        await _ensure_connected(self._client)
        return await self._client.call("pdg_particle", name=name)

    async def codata_constant(self, name: str) -> Any:
        await _ensure_connected(self._client)
        return await self._client.call("codata_constant", name=name)

    async def planck_parameter(self, name: str) -> Any:
        await _ensure_connected(self._client)
        return await self._client.call("planck_parameter", name=name)

    async def flag_average(self, observable: str) -> Any:
        await _ensure_connected(self._client)
        return await self._client.call("flag_average", observable=observable)
