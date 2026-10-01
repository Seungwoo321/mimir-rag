from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_shared_plugin_contracts():
    path = ROOT / "scripts/validate_package.py"
    spec = importlib.util.spec_from_file_location("validate_package", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.validate(ROOT) == []


def test_skill_install_is_idempotent_and_preserves_different_content(tmp_path: Path):
    target = tmp_path / "skills" / "mimir-rag"
    command = [sys.executable, str(ROOT / "scripts/install_skill.py"), "--target", str(target)]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "Installed skill" in first.stdout
    expected = (ROOT / "skills/mimir-rag/SKILL.md").read_bytes()
    assert (target / "SKILL.md").read_bytes() == expected
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "already installed" in second.stdout
    (target / "SKILL.md").write_text("User-authored skill; preserve me.")
    third = subprocess.run(command, capture_output=True, text=True)
    assert third.returncode == 2
    assert (target / "SKILL.md").read_text() == "User-authored skill; preserve me."


def test_skill_installer_rejects_symlink_target(tmp_path: Path):
    existing = tmp_path / "existing"
    existing.mkdir()
    target = tmp_path / "linked"
    target.symlink_to(existing, target_is_directory=True)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/install_skill.py"), "--target", str(target)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert list(existing.iterdir()) == []
