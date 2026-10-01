from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import httpx
import pytest

from mimir_rag import main as cli
from mimir_rag.config import Settings
from mimir_rag.providers import ProviderClient


class OfflineWire:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.embedding_failure = False
        self.forge_source = False

    @property
    def embedding_calls(self) -> list[dict[str, Any]]:
        return [body for path, body in self.calls if path == "/v1/embeddings"]

    @property
    def completion_calls(self) -> list[dict[str, Any]]:
        return [body for path, body in self.calls if path in {"/v1/responses", "/v1/messages"}]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.scheme == "https"
        assert request.url.host in {"api.openai.com", "api.anthropic.com"}
        body = json.loads(request.content)
        self.calls.append((request.url.path, body))
        if request.url.path == "/v1/embeddings":
            assert request.url.host == "api.openai.com"
            assert body["model"] == "text-embedding-3-small"
            assert body["dimensions"] == 3
            assert body["encoding_format"] == "float"
            if self.embedding_failure:
                return httpx.Response(503, json={"error": {"message": "offline failure"}})
            data = []
            for index, text in enumerate(body["input"]):
                if "jupiter" in text.casefold():
                    vector = [0.0, 1.0, 0.0]
                elif "feedback" in text.casefold():
                    vector = [1.0, 0.0, 0.0]
                else:
                    vector = [0.0, 0.0, 1.0]
                data.append({"index": index, "embedding": vector})
            return httpx.Response(200, json={"data": list(reversed(data))})
        if request.url.path == "/v1/responses":
            assert request.url.host == "api.openai.com"
            assert body["store"] is False
            assert body["text"]["format"]["type"] == "json_schema"
            envelope = json.loads(body["input"])
        else:
            assert request.url.path == "/v1/messages"
            assert request.url.host == "api.anthropic.com"
            assert body["output_config"]["format"]["type"] == "json_schema"
            envelope = json.loads(body["messages"][0]["content"])
        if body["model"] == "offline-review":
            assert "proposal" in envelope
            by_id = {record["source_id"]: record for record in envelope["evidence"]}
            proposal = envelope["proposal"]
            for item in [*proposal["claims"], *proposal["connections"]]:
                assert all(
                    reference["quote"] in by_id[reference["source_id"]]["text"]
                    for reference in item["evidence"]
                )
            result = {
                "approved": True,
                "claim_verdicts": [
                    {"id": item["id"], "supported": True} for item in proposal["claims"]
                ],
                "connection_verdicts": [
                    {"id": item["id"], "supported": True} for item in proposal["connections"]
                ],
            }
        else:
            assert body["model"] == "offline-synthesis"
            assert "proposal" not in envelope
            evidence = next(
                record for record in envelope["evidence"] if "feedback loop" in record["concepts"]
            )
            assert {"feedback loop", "control theory"} <= set(evidence["concepts"])
            observed = "observed outcomes" in evidence["text"]
            phrase = "observed outcomes" if observed else "recorded results"
            quote = f"adjusts action using {phrase}"
            assert quote in evidence["text"]
            references = [
                {
                    "source_id": "forged-source" if self.forge_source else evidence["source_id"],
                    "quote": quote,
                }
            ]
            result = {
                "answerable": True,
                "claims": [
                    {
                        "id": "claim-1",
                        "text": f"The handbook uses {phrase} to guide the next adjustment.",
                        "evidence": references,
                    }
                ],
                "connections": [
                    {
                        "id": "connection-1",
                        "concept_a": "feedback loop",
                        "concept_b": "control theory",
                        "relation": "co_occurs",
                        "text": "These concepts appear together in the feedback discussion.",
                        "evidence": references,
                    }
                ],
            }
        if request.url.path == "/v1/responses":
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": json.dumps(result)}],
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": json.dumps(result)}],
            },
        )


def configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthesis: str) -> OfflineWire:
    for key in list(os.environ):
        if key.startswith("MIMIR_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-offline-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-offline-key")
    monkeypatch.setenv("MIMIR_DB_PATH", str(tmp_path / "private" / "library.sqlite3"))
    monkeypatch.setenv("MIMIR_EMBEDDING_DIMENSIONS", "3")
    monkeypatch.setenv("MIMIR_SYNTHESIS_PROVIDER", synthesis)
    monkeypatch.setenv("MIMIR_SYNTHESIS_MODEL", "offline-synthesis")
    monkeypatch.setenv("MIMIR_VERIFICATION_MODEL", "offline-review")
    monkeypatch.setenv("MIMIR_API_MAX_RETRIES", "0")
    wire = OfflineWire()

    def provider(settings: Settings) -> ProviderClient:
        return ProviderClient(settings, transport=httpx.MockTransport(wire))

    monkeypatch.setattr(cli, "ProviderClient", provider)
    return wire


def manuscript(phrase: str = "observed outcomes") -> str:
    return (
        '---\ntitle: "Systems [Handbook]"\nauthor: "Synthetic Author"\n---\n'
        "# Chapter 2 / Feedback\n\n"
        f"The **Feedback Loop** framework adjusts action using {phrase}. "
        "The **Control Theory** principle studies feedback.\n\n"
        "# Appendix / Gardening\n\n"
        "A **Garden Method** recommends placing basil plants near sunlight.\n"
    )


def invoke(capsys: pytest.CaptureFixture[str], arguments: list[str], code: int = 0) -> Any:
    assert cli.main(arguments) == code
    output = capsys.readouterr()
    assert output.err == ""
    return json.loads(output.out)


@pytest.mark.parametrize("synthesis", ["openai", "anthropic"])
def test_real_cli_library_lifecycle_with_precise_citations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    synthesis: str,
) -> None:
    wire = configure(tmp_path, monkeypatch, synthesis)
    path = tmp_path / "systems (handbook).md"
    text = manuscript()
    path.write_text(text, encoding="utf-8")
    indexed = invoke(capsys, ["/ingest", str(path), "--rights", "Synthetic test manuscript"])
    assert indexed["status"] == "indexed"
    assert indexed["chunk_count"] >= 2
    document_id = indexed["document_id"]
    documents = invoke(capsys, ["list"])
    assert len(documents) == 1
    assert documents[0]["id"] == document_id
    assert documents[0]["title"] == "Systems [Handbook]"
    assert documents[0]["author"] == "Synthetic Author"
    assert "heuristic" in documents[0]["metadata_origin"]
    question = "How does the handbook describe feedback?"
    answer = invoke(capsys, ["/ask", question, "--json"])
    assert answer["abstained"] is False
    assert answer["markdown"].startswith("- [Systems \\[Handbook\\], Chapter 2 / Feedback, lines ")
    assert "Concept mapping:" in answer["markdown"]
    assert "feedback loop ↔ control theory (co-occurrence)" in answer["markdown"]
    assert answer["markdown"].index(")") < answer["markdown"].index("The handbook uses")
    assert [body["model"] for body in wire.completion_calls] == [
        "offline-synthesis",
        "offline-review",
    ]
    synthesis_body = wire.completion_calls[0]
    envelope = json.loads(
        synthesis_body["input"]
        if synthesis == "openai"
        else synthesis_body["messages"][0]["content"]
    )
    evidence = {record["source_id"]: record for record in envelope["evidence"]}
    assert set(answer["source_ids"]) <= set(evidence)
    for record in evidence.values():
        assert text[record["char_start"] : record["char_end"]] == record["text"]
        assert record["line_start"] == text[: record["char_start"]].count("\n") + 1
        assert {"Framework", "Core Theory"} <= set(record["tags"])
    links = re.findall(r"\]\((file:[^)]+)\)", answer["markdown"])
    assert links and all("%20%28handbook%29.md" in link for link in links)
    for link in links:
        parsed = urlsplit(link)
        fragment = parse_qs(parsed.fragment)
        record = evidence[fragment["chunk"][0]]
        assert Path(unquote(parsed.path)) == path.resolve()
        assert fragment["chars"] == [f"{record['char_start']}-{record['char_end']}"]
        assert fragment["lines"] == [f"{record['line_start']}-{record['line_end']}"]
    embedded = len(wire.embedding_calls)
    repeated = invoke(capsys, ["ingest", str(path), "--rights", "Synthetic test manuscript"])
    assert repeated["status"] == "unchanged"
    assert len(wire.embedding_calls) == embedded
    graph = invoke(capsys, ["graph", "--document-id", document_id])
    assert any(edge["type"] == "co_occurrence" for edge in graph["edges"])
    path.write_text(manuscript("recorded results"), encoding="utf-8")
    replaced = invoke(capsys, ["ingest", str(path), "--rights", "Synthetic test manuscript"])
    assert replaced["document_id"] == document_id
    assert replaced["status"] == "indexed"
    updated = invoke(capsys, ["ask", question, "--json"])
    assert updated["abstained"] is False
    assert "recorded results" in updated["markdown"]
    assert set(answer["source_ids"]).isdisjoint(updated["source_ids"])
    latest = invoke(capsys, ["list"])
    assert len(latest) == 1
    assert latest[0]["generation"] != documents[0]["generation"]
    assert latest[0]["content_hash"] != documents[0]["content_hash"]
    completed = len(wire.completion_calls)
    unrelated = invoke(capsys, ["ask", "What is the orbital eccentricity of Jupiter?", "--json"], 3)
    assert unrelated["abstained"] is True
    assert unrelated["source_ids"] == []
    assert unrelated["reason"] == "insufficient_evidence"
    assert len(wire.completion_calls) == completed
    requests = len(wire.calls)
    assert invoke(capsys, ["delete", document_id])["deleted"] is True
    assert invoke(capsys, ["list"]) == []
    assert invoke(capsys, ["graph"]) == {"nodes": [], "edges": []}
    assert len(wire.calls) == requests
    empty = invoke(capsys, ["/ask", question, "--json"], 3)
    assert empty["abstained"] is True and empty["source_ids"] == []
    assert len(wire.calls) == requests
    assert len(wire.completion_calls) == completed


def test_http_embedding_failure_preserves_last_cli_document_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wire = configure(tmp_path, monkeypatch, "openai")
    path = tmp_path / "synthetic.md"
    path.write_text(manuscript(), encoding="utf-8")
    invoke(capsys, ["ingest", str(path)])
    previous = invoke(capsys, ["list"])
    path.write_text(manuscript("recorded results"), encoding="utf-8")
    wire.embedding_failure = True
    assert cli.main(["ingest", str(path)]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "HTTP 503" in output.err
    assert "recorded results" not in output.err
    assert invoke(capsys, ["list"]) == previous
    wire.embedding_failure = False
    old_answer = invoke(capsys, ["ask", "How does the handbook describe feedback?", "--json"])
    assert old_answer["abstained"] is False
    assert "observed outcomes" in old_answer["markdown"]


def test_http_forged_source_cannot_reach_cli_answer_or_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wire = configure(tmp_path, monkeypatch, "openai")
    path = tmp_path / "synthetic.md"
    path.write_text(manuscript(), encoding="utf-8")
    invoke(capsys, ["ingest", str(path)])
    wire.forge_source = True
    answer = invoke(capsys, ["/ask", "How does the handbook describe feedback?", "--json"], 3)
    assert answer["abstained"] is True
    assert answer["reason"] == "grounding_validation_failed"
    assert answer["source_ids"] == []
    assert "forged-source" not in answer["markdown"]
    assert [body["model"] for body in wire.completion_calls] == ["offline-synthesis"]
