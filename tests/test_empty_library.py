from __future__ import annotations

import json
from pathlib import Path

from mimir_rag.main import main


def test_empty_library_abstains_without_credentials_or_cloud(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = main(
        ["--db", str(tmp_path / "private" / "library.sqlite3"), "ask", "unknown", "--json"]
    )
    assert result == 3
    answer = json.loads(capsys.readouterr().out)
    assert answer["abstained"] is True
    assert answer["reason"] == "insufficient_evidence"
    assert answer["source_ids"] == []
