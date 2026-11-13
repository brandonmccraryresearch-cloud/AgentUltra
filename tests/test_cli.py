"""Tests for the CLI."""

from agent_ultra.cli.main import main


class TestCLI:
    def test_version(self, capsys):
        try:
            main(["--version"])
        except SystemExit:
            pass
        captured = capsys.readouterr()
        assert "1.0.0" in captured.out

    def test_config_yaml(self, capsys):
        main(["config"])
        captured = capsys.readouterr()
        assert "agent_configuration" in captured.out

    def test_config_json(self, capsys):
        main(["config", "--format", "json"])
        captured = capsys.readouterr()
        assert "Frontier Physics Research Agent" in captured.out

    def test_constants(self, capsys):
        main(["constants", "c"])
        captured = capsys.readouterr()
        assert "299792458" in captured.out

    def test_constants_unknown(self):
        code = main(["constants", "nonexistent"])
        assert code == 1

    def test_evaluate(self, capsys):
        main(["evaluate", "E = mc^2"])
        captured = capsys.readouterr()
        assert "claim_id" in captured.out
        assert "session" in captured.out

    def test_no_command(self, capsys):
        main([])
        captured = capsys.readouterr()
        assert "agent-ultra" in captured.out
