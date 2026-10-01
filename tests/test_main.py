from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from mimir_rag import main as cli
from mimir_rag.errors import ProviderError


@pytest.mark.parametrize("prefix", [[], ["--db", "library.sqlite3"], ["--db=library.sqlite3"]])
def test_alias_only_changes_command_position(monkeypatch, prefix):
    captured: list[argparse.Namespace] = []

    async def capture(args: argparse.Namespace) -> int:
        captured.append(args)
        return 0

    monkeypatch.setattr(cli, "_run", capture)
    assert cli.main([*prefix, "/ask", "Explain", "/ingest", "/ask", "$(echo secret)"]) == 0
    assert captured[0].command == "ask"
    assert captured[0].question == ["Explain", "/ingest", "/ask", "$(echo secret)"]


def test_provider_failure_has_controlled_exit(monkeypatch, capsys):
    async def fail(args: argparse.Namespace) -> int:
        raise ProviderError("Provider unavailable after bounded retries.")

    monkeypatch.setattr(cli, "_run", fail)
    assert cli.main(["ask", "question"]) == 2
    assert "bounded retries" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["doctor", "list", "graph", "delete"])
def test_local_commands_need_no_provider_key(tmp_path: Path, monkeypatch, capsys, command):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    private = tmp_path / "library"
    args = ["--db", str(private / "library.sqlite3"), command]
    if command == "delete":
        args.append("missing-document")
    assert cli.main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    if command == "doctor":
        assert payload["openai_key_present"] is False
        assert payload["anthropic_key_present"] is False
    elif command == "list":
        assert payload == []
    elif command == "delete":
        assert payload["deleted"] is False
    else:
        assert payload["nodes"] == []


def test_abstention_exit_and_json(tmp_path: Path, monkeypatch, capsys):
    from mimir_rag.models import Answer

    class EmptyGenerator:
        def __init__(self, settings, store, provider):
            pass

        async def ask(self, question, *, top_k=None):
            return Answer(markdown="Insufficient evidence.", abstained=True, source_ids=[])

    monkeypatch.setattr(cli, "Generator", EmptyGenerator)
    assert cli.main(["--db", str(tmp_path / "private" / "db"), "ask", "unknown", "--json"]) == 3
    assert json.loads(capsys.readouterr().out)["abstained"] is True
