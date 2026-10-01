from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import tempfile
from typing import Any, Literal

from .config import Settings
from .errors import ConfigurationError, ProviderError

MAX_OUTPUT_BYTES = 16 * 1024 * 1024


class ClaudeCodeClient:
    """Tool-free Claude Code structured inference using its own supported authentication."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def complete_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        role: Literal["synthesis", "verification"] = "synthesis",
    ) -> dict[str, Any]:
        executable = shutil.which("claude")
        if executable is None:
            raise ConfigurationError("Install Claude Code and run claude auth login for synthesis.")
        model = self.settings.synthesis_model
        if role == "verification" and self.settings.verification_model:
            model = self.settings.verification_model
        if not model.startswith("claude-"):
            raise ConfigurationError("Claude Code inference requires an explicit Claude model.")
        environment = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "CLAUDECODE",
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
                "ANTHROPIC_BASE_URL",
                "ANTHROPIC_AUTH_TOKEN",
                "CLAUDE_CODE_USE_BEDROCK",
                "CLAUDE_CODE_USE_VERTEX",
                "CLAUDE_CODE_USE_FOUNDRY",
            }
        }
        arguments = [
            executable,
            "--print",
            "--safe-mode",
            "--tools",
            "",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--setting-sources",
            "",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema, separators=(",", ":")),
            "--system-prompt",
            system,
            "--model",
            model,
        ]
        with tempfile.TemporaryDirectory(prefix="mimir-synthesis-") as directory:
            try:
                process = await asyncio.create_subprocess_exec(
                    *arguments,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=directory,
                    env=environment,
                    start_new_session=os.name == "posix",
                )
            except OSError:
                raise ProviderError("Could not start the Claude Code inference process.") from None
            try:
                async with asyncio.timeout(self.settings.api_timeout_seconds):
                    assert process.stdin is not None
                    assert process.stdout is not None
                    assert process.stderr is not None

                    async def read(stream: asyncio.StreamReader) -> bytes:
                        body = bytearray()
                        while part := await stream.read(65536):
                            if len(body) + len(part) > MAX_OUTPUT_BYTES:
                                raise ProviderError("Claude Code output exceeds its byte limit.")
                            body.extend(part)
                        return bytes(body)

                    async def write() -> None:
                        assert process.stdin is not None
                        process.stdin.write(user.encode())
                        await process.stdin.drain()
                        process.stdin.close()

                    async with asyncio.TaskGroup() as group:
                        output_task = group.create_task(read(process.stdout))
                        group.create_task(read(process.stderr))
                        group.create_task(write())
                        group.create_task(process.wait())
                    output = output_task.result()
                if process.returncode != 0:
                    raise ProviderError(
                        "Claude Code inference failed; check its login, model access and limits."
                    )
                try:
                    envelope = json.loads(output)
                except (ValueError, RecursionError):
                    raise ProviderError("Claude Code returned malformed JSON.") from None
                if (
                    not isinstance(envelope, dict)
                    or envelope.get("is_error") is not False
                    or envelope.get("subtype") != "success"
                    or not isinstance(envelope.get("structured_output"), dict)
                ):
                    raise ProviderError("Claude Code did not complete a structured answer.")
                return dict(envelope["structured_output"])
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                raise ProviderError(
                    "Claude Code inference exceeded its bounded deadline."
                ) from None
            except ExceptionGroup:
                raise ProviderError(
                    "Claude Code stream failed or exceeded its byte limit."
                ) from None
            finally:
                cleanup = asyncio.create_task(self._stop(process))
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        continue
                cleanup.result()

    @staticmethod
    async def _stop(process: asyncio.subprocess.Process) -> None:
        def send(kind: signal.Signals) -> None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, kind)
                elif kind == signal.SIGTERM:
                    process.terminate()
                else:
                    process.kill()
            except ProcessLookupError:
                pass

        send(signal.SIGTERM)
        try:
            async with asyncio.timeout(2):
                await process.wait()
        except TimeoutError:
            send(signal.SIGKILL)
            await process.wait()
        # A reaped parent can leave descendants holding inherited output pipes.
        if os.name == "posix":
            send(signal.SIGKILL)
