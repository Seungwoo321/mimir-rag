from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install the shared skill without changing host settings."
    )
    parser.add_argument(
        "--target", type=Path, default=Path.home() / ".agents" / "skills" / "mimir-rag"
    )
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "skills" / "mimir-rag"
    target = args.target.expanduser().absolute()
    if target.is_symlink():
        parser.error("Target is a symlink; select a new skill directory.")
    if target.exists():
        existing = target / "SKILL.md"
        if existing.is_file() and existing.read_bytes() == (source / "SKILL.md").read_bytes():
            print(f"Skill already installed: {target}")
            return 0
        parser.error(
            "Target already exists with different content; preserve it and select a new target."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mimir-rag-install-", dir=target.parent) as temporary:
        staged = Path(temporary) / "mimir-rag"
        shutil.copytree(source, staged)
        staged.rename(target)
    print(f"Installed skill: {target}")
    print("Install the shared mimir-rag CLI and restart the host to discover the skill.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
