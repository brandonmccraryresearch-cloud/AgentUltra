"""Agent configuration system.

Defines the YAML-based configuration schema for the Frontier Physics Research Agent,
including model settings, reasoning protocols, tool integrations, and evaluation criteria.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ReasoningProtocol(str, Enum):
    """Supported reasoning protocol modes."""

    FOUR_PHASE = "four_phase"
    SINGLE_PASS = "single_pass"


class FormalVerificationConfig(BaseModel):
    """Configuration for formal proof verification."""

    prover: str = "Lean 4"
    library: str = "Mathlib4"
    timeout_seconds: int = 300
    max_sorry_attempts: int = 10


class NumericalEngineConfig(BaseModel):
    """Configuration for the numerical computation engine."""

    framework: str = "numpy"
    accelerator: str = "cpu"
    precision: str = "float64"
    max_lattice_size: int = 64


class MCPTransport(str, Enum):
    """MCP transport protocols."""

    JSON_RPC = "json-rpc-2.0"
    STDIO = "stdio"
    SSE = "sse"


class MCPServerConfig(BaseModel):
    """Configuration for an individual MCP server."""

    name: str
    url: str = ""
    transport: MCPTransport = MCPTransport.JSON_RPC
    enabled: bool = True


class MCPIntegrationConfig(BaseModel):
    """Configuration for all MCP server integrations."""

    transport: MCPTransport = MCPTransport.JSON_RPC
    servers: list[MCPServerConfig] = Field(default_factory=list)


class EvaluationConfig(BaseModel):
    """Configuration for the evaluation and scoring system."""

    auto_grade: bool = True
    detect_tautology: bool = True
    calibration_tracking: bool = True
    parsimony_computation: bool = True
    falsifiability_assessment: bool = True


class ReasoningConfig(BaseModel):
    """Configuration for reasoning behaviour."""

    protocol: ReasoningProtocol = ReasoningProtocol.FOUR_PHASE
    epistemic_hygiene: bool = True
    failure_detection: bool = True
    self_correction: bool = True


class ToolsConfig(BaseModel):
    """Configuration for computational tools."""

    symbolic: list[str] = Field(default_factory=lambda: ["sympy"])
    numerical: list[str] = Field(default_factory=lambda: ["numpy", "scipy"])
    formal_proof: list[str] = Field(default_factory=lambda: ["lean4"])
    visualization: list[str] = Field(default_factory=lambda: ["matplotlib"])
    data: list[str] = Field(default_factory=list)


class KnowledgeConfig(BaseModel):
    """Configuration for the knowledge base."""

    physics: list[str] = Field(
        default_factory=lambda: ["qft", "gr", "sm", "cosmology", "condensed_matter"]
    )
    mathematics: list[str] = Field(
        default_factory=lambda: [
            "lie_theory",
            "topology",
            "differential_geometry",
            "number_theory",
        ]
    )
    experimental: list[str] = Field(default_factory=lambda: ["pdg_2024", "codata_2018"])


class AgentConfig(BaseModel):
    """Root configuration for the Frontier Physics Research Agent."""

    name: str = "Frontier Physics Research Agent"
    version: str = "1.0"
    model: str = "claude-opus-4.6"
    context_window: int = 200_000
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    formal_verification: FormalVerificationConfig = Field(
        default_factory=FormalVerificationConfig
    )
    numerical_engine: NumericalEngineConfig = Field(default_factory=NumericalEngineConfig)
    mcp_integration: MCPIntegrationConfig = Field(default_factory=MCPIntegrationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data.get("agent_configuration", data))

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Serialize configuration to YAML."""
        payload = {"agent_configuration": self.model_dump(mode="json")}
        text = yaml.dump(payload, default_flow_style=False, sort_keys=False)
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def default(cls) -> AgentConfig:
        """Return the default agent configuration."""
        return cls()


def load_config(path: str | Path | None = None) -> AgentConfig:
    """Load or create a default agent configuration.

    Parameters
    ----------
    path:
        Optional path to a YAML configuration file.  When *None*, the
        default configuration is returned.
    """
    if path is not None:
        return AgentConfig.from_yaml(path)
    return AgentConfig.default()
