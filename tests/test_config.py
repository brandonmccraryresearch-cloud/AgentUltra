"""Tests for the configuration system."""

import tempfile
from pathlib import Path

from agent_ultra.core.config import (
    AgentConfig,
    ReasoningProtocol,
    load_config,
)


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig.default()
        assert cfg.name == "Frontier Physics Research Agent"
        assert cfg.version == "1.0"
        assert cfg.reasoning.protocol == ReasoningProtocol.FOUR_PHASE
        assert cfg.reasoning.epistemic_hygiene is True
        assert cfg.evaluation.auto_grade is True

    def test_yaml_round_trip(self):
        cfg = AgentConfig.default()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name
            cfg.to_yaml(path)

        loaded = AgentConfig.from_yaml(path)
        assert loaded.name == cfg.name
        assert loaded.reasoning.protocol == cfg.reasoning.protocol
        Path(path).unlink()

    def test_to_yaml_string(self):
        cfg = AgentConfig.default()
        text = cfg.to_yaml()
        assert "agent_configuration" in text
        assert "Frontier Physics Research Agent" in text


class TestLoadConfig:
    def test_default_config(self):
        cfg = load_config()
        assert isinstance(cfg, AgentConfig)

    def test_from_file(self):
        cfg = AgentConfig(name="Test Agent")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name
            cfg.to_yaml(path)

        loaded = load_config(path)
        assert loaded.name == "Test Agent"
        Path(path).unlink()
