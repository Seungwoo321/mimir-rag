from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

import pytest

from mimir_rag import claude_code
from mimir_rag.claude_code import ClaudeCodeClient
from mimir_rag.config import Settings
from mimir_rag.errors import ConfigurationError, ProviderError

SUCCESS = {"is_error": False, "subtype": "success", "structured_output": {"ok": True}}


def stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> Path:
    script = tmp_path / "fake_claude.py"
    script.write_text("import json,os,sys,time,signal,subprocess\n" + body)
    real_exec = asyncio.create_subprocess_exec

    async def execute(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
        return await real_exec(sys.executable, str(script), *args[1:], **kwargs)

    monkeypatch.setattr(claude_code.shutil, "which", lambda _: str(script))
    monkeypatch.setattr(claude_code.asyncio, "create_subprocess_exec", execute)
    return script


async def test_literal_stdin_safe_tools_schema_and_clean_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = tmp_path / "capture.json"
    monkeypatch.setenv("MIMIR_CAPTURE", str(capture))
    forbidden = [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDECODE",
    ]
    for key in forbidden:
        monkeypatch.setenv(key, "do-not-forward")
    stub(
        tmp_path,
        monkeypatch,
        "data={'args':sys.argv[1:],'stdin':sys.stdin.read(),'cwd':os.getcwd(),"
        "'env':{k:v for k,v in os.environ.items() if k.startswith(('OPENAI','ANTHROPIC_BASE',"
        "'ANTHROPIC_AUTH','CLAUDE_CODE_USE')) or k=='CLAUDECODE'}}\n"
        "open(os.environ['MIMIR_CAPTURE'],'w').write(json.dumps(data))\n"
        f"print({json.dumps(json.dumps(SUCCESS))})\n",
    )
    question = "untrusted $(command) `command`\n한국어 source"
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    result = await ClaudeCodeClient(
        Settings(verification_model="claude-review-model")
    ).complete_json("trusted-system", question, schema, role="verification")
    assert result == {"ok": True}
    data = json.loads(capture.read_text())
    args = data["args"]
    assert data["stdin"] == question
    assert data["env"] == {}
    assert data["cwd"] != str(tmp_path)
    for flag in [
        "--print",
        "--safe-mode",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--strict-mcp-config",
    ]:
        assert flag in args
    for flag, value in [
        ("tools", ""),
        ("setting-sources", ""),
        ("mcp-config", '{"mcpServers":{}}'),
        ("system-prompt", "trusted-system"),
        ("model", "claude-review-model"),
        ("output-format", "json"),
    ]:
        assert args[args.index("--" + flag) + 1] == value
    assert json.loads(args[args.index("--json-schema") + 1]) == schema
    assert not await asyncio.to_thread(Path(data["cwd"]).exists)


@pytest.mark.parametrize(
    "output",
    [
        "not-json",
        "[]",
        "{}",
        json.dumps({"is_error": True, "subtype": "error", "structured_output": {"ok": True}}),
        json.dumps({"is_error": False, "subtype": "success", "structured_output": []}),
    ],
)
async def test_malformed_refused_and_unstructured_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    stub(tmp_path, monkeypatch, f"print({output!r})\n")
    with pytest.raises(ProviderError):
        await ClaudeCodeClient(Settings()).complete_json("system", "source", {})


async def test_nonzero_exit_does_not_leak_source_or_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub(tmp_path, monkeypatch, "sys.stderr.write('private-source-secret');sys.exit(7)\n")
    with pytest.raises(ProviderError) as error:
        await ClaudeCodeClient(Settings()).complete_json("system", "source", {})
    assert "private-source-secret" not in str(error.value)


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
async def test_output_bound_on_each_pipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: str
) -> None:
    monkeypatch.setattr(claude_code, "MAX_OUTPUT_BYTES", 1024)
    stub(
        tmp_path, monkeypatch, f"sys.{stream}.write('x'*2048);sys.{stream}.flush();time.sleep(30)\n"
    )
    with pytest.raises(ProviderError, match="byte limit"):
        await ClaudeCodeClient(Settings(api_timeout_seconds=2)).complete_json(
            "system", "source", {}
        )


async def test_missing_executable_fails_before_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_code.shutil, "which", lambda _: None)
    with pytest.raises(ConfigurationError, match="Install Claude Code"):
        await ClaudeCodeClient(Settings()).complete_json("system", "source", {})


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group contract")
@pytest.mark.parametrize("mode", ["timeout", "cancel", "exited-parent"])
async def test_failure_closes_entire_owned_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    pid_path = tmp_path / "child.pid"
    monkeypatch.setenv("MIMIR_CHILD_PID", str(pid_path))
    child = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"
    body = (
        f"child=subprocess.Popen([sys.executable,'-c',{child!r}])\n"
        "open(os.environ['MIMIR_CHILD_PID'],'w').write(str(child.pid))\n"
        + ("sys.exit(0)\n" if mode == "exited-parent" else "time.sleep(30)\n")
    )
    stub(tmp_path, monkeypatch, body)
    task = asyncio.create_task(
        ClaudeCodeClient(Settings(api_timeout_seconds=2)).complete_json("system", "source", {})
    )
    try:
        for _ in range(500):
            if pid_path.exists():
                break
            await asyncio.sleep(0.01)
        assert pid_path.exists()
        pid = int(pid_path.read_text())
        if mode == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(ProviderError, match="deadline"):
                await task
        for _ in range(100):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("Owned inference descendant survives failure cleanup")
    finally:
        if pid_path.exists():
            try:
                os.kill(int(pid_path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, ProviderError):
            pass


async def test_startup_failure_is_secret_free(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_code.shutil, "which", lambda _: "/missing/claude")

    async def failed(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
        raise OSError("private-source-sensitive-path")

    monkeypatch.setattr(claude_code.asyncio, "create_subprocess_exec", failed)
    with pytest.raises(ProviderError) as error:
        await ClaudeCodeClient(Settings()).complete_json("system", "source", {})
    assert "private-source-sensitive-path" not in str(error.value)


@pytest.mark.parametrize("role", ["synthesis", "verification"])
async def test_non_claude_model_rejected_before_process(
    monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    monkeypatch.setattr(claude_code.shutil, "which", lambda _: "/unused/claude")

    async def forbidden(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
        raise AssertionError("No process should start for a prohibited model")

    monkeypatch.setattr(claude_code.asyncio, "create_subprocess_exec", forbidden)
    settings = Settings.model_construct(synthesis_model="gpt-test", verification_model="gpt-test")
    with pytest.raises(ConfigurationError, match="explicit Claude model"):
        await ClaudeCodeClient(settings).complete_json("system", "source", {}, role=role)
