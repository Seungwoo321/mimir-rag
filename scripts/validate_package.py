from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def validate(root: Path = ROOT) -> list[str]:
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    manifests = [
        json.loads((root / directory / "plugin.json").read_text())
        for directory in (".claude-plugin", ".codex-plugin")
    ]
    for manifest in manifests:
        require(manifest["name"] == project["name"], "Plugin name differs from package name.")
        require(manifest["version"] == project["version"], "Plugin version differs from package.")
        require(manifest["license"] == project["license"], "Plugin license differs from package.")
    require(manifests[1]["skills"] == "./skills/", "Codex must expose the shared skill folder.")
    claude_market = json.loads((root / ".claude-plugin/marketplace.json").read_text())
    codex_market = json.loads((root / ".agents/plugins/marketplace.json").read_text())
    require(claude_market["name"] == codex_market["name"], "Marketplace names differ.")
    require(claude_market["plugins"][0]["source"] == "./", "Claude source must be local root.")
    require(
        codex_market["plugins"][0]["source"] == {"source": "local", "path": "./"},
        "Codex source must be the local repository root.",
    )
    for marketplace in (claude_market, codex_market):
        require(
            marketplace["plugins"][0]["name"] == project["name"],
            "Marketplace plugin name differs from package.",
        )
    skill = (root / "skills/mimir-rag/SKILL.md").read_text()
    match = re.match(r"\A---\n(.*?)\n---\n", skill, re.S)
    require(match is not None, "Shared skill has no YAML front matter.")
    if match:
        metadata = yaml.safe_load(match.group(1))
        require(metadata.get("name") == project["name"], "Skill name differs from package.")
        require(bool(metadata.get("description")), "Skill description is missing.")
    for command in ("ingest", "ask"):
        body = (root / "commands" / f"{command}.md").read_text()
        require("$ARGUMENTS" in body, f"{command} command must preserve the input arguments.")
        require("mimir-rag" in body, f"{command} command must call the shared runtime.")
    vocabulary = root / "src/mimir_rag/data/cl100k_base.tiktoken"
    require(
        hashlib.sha256(vocabulary.read_bytes()).hexdigest()
        == "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        "Bundled tokenizer checksum differs from the authoritative vocabulary.",
    )
    require((root / "LICENSE").is_file(), "Project license is missing.")
    require((root / "THIRD_PARTY_NOTICES.md").is_file(), "Tokenizer attribution is missing.")
    ignored = (root / ".gitignore").read_text().splitlines()
    for pattern in (".local/", ".venv/", "*.sqlite3", "*.sqlite3-*", ".env"):
        require(pattern in ignored, f"Missing local-data ignore pattern: {pattern}")
    for source in (root / "src").rglob("*.py"):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and node.value.value is Ellipsis
            ):
                failures.append(
                    f"Production placeholder in {source.relative_to(root)}:{node.lineno}"
                )
        require(
            not re.search(r"(?im)^\s*#\s*(TODO|FIXME|implement here)\b", source.read_text()),
            f"Unfinished production comment in {source.relative_to(root)}",
        )
    return failures


def main() -> int:
    try:
        failures = validate()
    except (OSError, KeyError, ValueError, SyntaxError, TypeError, yaml.YAMLError):
        print("Package validation could not read a required artifact.", file=sys.stderr)
        return 1
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("Package contracts passed: both hosts, versions, skills, licenses and offline assets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
