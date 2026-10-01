from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .config import Settings
from .errors import MimirError
from .generator import Generator
from .ingestor import Ingestor
from .providers import ProviderClient
from .vector_store import VectorStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mimir-rag",
        description=(
            "Personal-library search with verified citations. Cloud inference sends source text."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--db", type=Path, help="Library path; overrides MIMIR_DB_PATH.")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Index authorized PDF, UTF-8 text or Markdown.")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--title")
    ingest.add_argument("--author")
    ingest.add_argument("--rights", default="user-authorized personal processing")
    ask = commands.add_parser(
        "ask", help="Retrieve, synthesize and independently review an answer."
    )
    ask.add_argument("question", nargs="+")
    ask.add_argument("--top-k", type=int)
    ask.add_argument("--json", action="store_true", help="Output answer and source IDs as JSON.")
    commands.add_parser("list", help="List indexed documents without provider calls.")
    graph = commands.add_parser("graph", help="Inspect source-backed concepts and co-occurrence.")
    graph.add_argument("--document-id")
    delete = commands.add_parser("delete", help="Remove a document and its searchable descendants.")
    delete.add_argument("document_id")
    commands.add_parser("doctor", help="Show local capabilities and configuration; no cloud calls.")
    return parser


async def _run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.db is not None:
        settings = Settings.model_validate(
            {
                **settings.model_dump(),
                "db_path": args.db,
                "openai_api_key": settings.openai_api_key,
                "anthropic_api_key": settings.anthropic_api_key,
            }
        )
    store = VectorStore(settings)
    await store.initialize()
    payload: Any
    if args.command == "doctor":
        payload = {
            "version": __version__,
            "python": sys.version.split()[0],
            "sqlite": sqlite3.sqlite_version,
            "database": str(settings.db_path),
            "embedding_model": settings.embedding_model,
            "embedding_dimensions": settings.embedding_dimensions,
            "synthesis_provider": settings.synthesis_provider,
            "synthesis_model": settings.synthesis_model,
            "openai_key_present": bool(settings.openai_api_key),
            "anthropic_key_present": bool(settings.anthropic_api_key),
            "privacy": "Local plaintext storage; cloud inference sends selected text.",
        }
    elif args.command == "list":
        payload = [document.model_dump() for document in await store.list_documents()]
    elif args.command == "graph":
        payload = await store.concept_graph(args.document_id)
    elif args.command == "delete":
        payload = {
            "document_id": args.document_id,
            "deleted": await store.delete_document(args.document_id),
        }
    else:
        async with ProviderClient(settings) as provider:
            if args.command == "ingest":
                result = await Ingestor(settings, store, provider).ingest(
                    args.path,
                    title=args.title,
                    author=args.author,
                    rights=args.rights,
                )
                payload = result.model_dump()
            else:
                answer = await Generator(settings, store, provider).ask(
                    " ".join(args.question),
                    top_k=args.top_k,
                )
                if not args.json:
                    print(answer.markdown)
                    return 3 if answer.abstained else 0
                payload = answer.model_dump()
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return 3 if answer.abstained else 0
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    command_index = 0
    while command_index < len(arguments):
        if arguments[command_index] == "--db":
            command_index += 2
        elif arguments[command_index].startswith("--db="):
            command_index += 1
        else:
            break
    if command_index < len(arguments) and arguments[command_index] in {"/ingest", "/ask"}:
        arguments[command_index] = arguments[command_index].removeprefix("/")
    args = _parser().parse_args(arguments)
    try:
        return asyncio.run(_run(args))
    except MimirError as exc:
        print(f"mimir-rag: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("mimir-rag: interrupted; uncommitted ingestion was not published.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
