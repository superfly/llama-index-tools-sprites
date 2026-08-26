"""Sprites tool specification for LlamaIndex agents."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from llama_index.core.tools.tool_spec.base import BaseToolSpec

from sprites import ListOptions, SpriteConfig, SpritesClient
from sprites.exec import run as _run_command

_DEFAULT_BASE_URL = "https://api.sprites.dev"
_DEFAULT_MAX_OUTPUT_CHARS = 20_000
_MAX_COMMAND_TIMEOUT_SECONDS = 300.0


class SpritesToolSpec(BaseToolSpec):
    """Give a LlamaIndex agent controlled access to Sprites.

    The tool spec authenticates with a Sprites API token passed to the
    constructor or read from ``SPRITE_TOKEN``. Command execution uses an argv
    list directly; shell syntax is not interpreted unless the caller explicitly
    invokes a shell such as ``bash -lc``.
    """

    spec_functions = [
        "list_sprites",
        "get_sprite",
        "create_sprite",
        "run_command",
        "delete_sprite",
    ]

    def __init__(
        self,
        token: str | None = None,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 30.0,
        max_output_chars: int = _DEFAULT_MAX_OUTPUT_CHARS,
        client: SpritesClient | None = None,
    ) -> None:
        """Initialize the Sprites tools.

        Args:
            token: Sprites API token. Falls back to ``SPRITE_TOKEN``.
            base_url: Sprites API base URL.
            timeout: Default SDK request timeout in seconds.
            max_output_chars: Maximum stdout and stderr characters returned to
                an agent for one command. Longer streams retain their beginning
                and end with a truncation marker between them.
            client: Optional preconfigured Sprites client. Primarily useful for
                dependency injection and tests.
        """
        if max_output_chars < 256:
            raise ValueError("max_output_chars must be at least 256")

        if client is None:
            resolved_token = token or os.environ.get("SPRITE_TOKEN")
            if not resolved_token:
                raise ValueError(
                    "A Sprites API token is required. Pass token=... or set "
                    "SPRITE_TOKEN."
                )
            client = SpritesClient(
                token=resolved_token,
                base_url=base_url,
                timeout=timeout,
            )

        self.client = client
        self.max_output_chars = max_output_chars

    def close(self) -> None:
        """Close the underlying Sprites client."""
        self.client.close()

    def list_sprites(
        self,
        prefix: str | None = None,
        max_results: int = 50,
        continuation_token: str | None = None,
    ) -> dict[str, Any]:
        """List Sprites available to the authenticated account.

        Use this before creating a Sprite to avoid duplicate resources and to
        discover an existing Sprite by name.

        Args:
            prefix: Optional Sprite-name prefix filter.
            max_results: Maximum results to return, from 1 through 100.
            continuation_token: Token returned by an earlier call when
                ``has_more`` is true.

        Returns:
            A dictionary with ``sprites``, ``has_more``, and a possible
            ``next_continuation_token``.
        """
        if not 1 <= max_results <= 100:
            raise ValueError("max_results must be between 1 and 100")

        result = self.client.list_sprites(
            ListOptions(
                prefix=prefix,
                max_results=max_results,
                continuation_token=continuation_token,
            )
        )
        return {
            "sprites": [self._sprite_info_to_dict(sprite) for sprite in result.sprites],
            "has_more": result.has_more,
            "next_continuation_token": result.next_continuation_token,
        }

    def get_sprite(self, name: str) -> dict[str, Any]:
        """Get the current details for one Sprite.

        Args:
            name: Exact Sprite name.

        Returns:
            The Sprite's name, ID, status, URL, region, organization, and labels
            when those values are available.
        """
        self._require_nonempty("name", name)
        return self._sprite_info_to_dict(self.client.get_sprite(name))

    def create_sprite(
        self,
        name: str,
        region: str | None = None,
        ram_mb: int | None = None,
        cpus: int | None = None,
        storage_gb: int | None = None,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a Sprite for isolated command execution.

        List existing Sprites first when resource reuse is acceptable. Creation
        allocates a persistent remote resource that remains until deleted.

        Args:
            name: Unique Sprite name.
            region: Optional Fly region code.
            ram_mb: Optional memory allocation in MiB.
            cpus: Optional CPU count.
            storage_gb: Optional persistent storage allocation in GiB.
            labels: Optional labels to attach to the Sprite.

        Returns:
            A confirmation containing the created Sprite name.
        """
        self._require_nonempty("name", name)
        self._require_positive_optional("ram_mb", ram_mb)
        self._require_positive_optional("cpus", cpus)
        self._require_positive_optional("storage_gb", storage_gb)

        config_values = (region, ram_mb, cpus, storage_gb)
        config = None
        if any(value is not None for value in config_values):
            config = SpriteConfig(
                region=region,
                ram_mb=ram_mb,
                cpus=cpus,
                storage_gb=storage_gb,
            )

        sprite = self.client.create_sprite(name, config=config, labels=labels)
        return {"created": True, "name": sprite.name}

    def run_command(
        self,
        sprite_name: str,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Run a non-interactive command on a Sprite and capture its result.

        ``command`` and ``args`` are passed directly as an argv list. Shell
        operators, substitutions, and redirects are not interpreted. Invoke a
        shell explicitly only when shell behavior is required.

        Args:
            sprite_name: Exact name of an existing Sprite.
            command: Executable name or path, for example ``python3``.
            args: Individual command arguments, for example
                ``["-c", "print('hello')"]``.
            cwd: Optional working directory inside the Sprite.
            env: Optional environment variables for the command.
            timeout: Command timeout in seconds, from 0.1 through 300.

        Returns:
            Exit code, stdout, stderr, and truncation flags. Non-zero command
            exits are returned normally so the agent can inspect and recover.
        """
        self._require_nonempty("sprite_name", sprite_name)
        self._require_nonempty("command", command)
        if not 0.1 <= timeout <= _MAX_COMMAND_TIMEOUT_SECONDS:
            raise ValueError("timeout must be between 0.1 and 300 seconds")

        command_args = args or []
        if not all(isinstance(arg, str) for arg in command_args):
            raise TypeError("every command argument must be a string")
        if env is not None and not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in env.items()
        ):
            raise TypeError("environment variable names and values must be strings")

        sprite = self.client.sprite(sprite_name)
        result = _run_command(
            sprite,
            command,
            *command_args,
            capture_output=True,
            timeout=timeout,
            check=False,
            cwd=cwd,
            env=env,
        )
        stdout, stdout_truncated = self._decode_and_truncate(result.stdout)
        stderr, stderr_truncated = self._decode_and_truncate(result.stderr)
        return {
            "exit_code": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }

    def delete_sprite(self, name: str) -> dict[str, Any]:
        """Permanently delete a Sprite.

        This destroys the named remote resource and its persistent state. Use
        it only when deletion is explicitly intended, typically to clean up a
        Sprite created for a completed task.

        Args:
            name: Exact Sprite name to delete.

        Returns:
            A confirmation containing the deleted Sprite name.
        """
        self._require_nonempty("name", name)
        self.client.delete_sprite(name)
        return {"deleted": True, "name": name}

    def _decode_and_truncate(self, value: bytes | None) -> tuple[str, bool]:
        text = (value or b"").decode("utf-8", errors="replace")
        if len(text) <= self.max_output_chars:
            return text, False

        marker_template = "\n... truncated {omitted} characters ...\n"
        marker = marker_template.format(omitted=0)
        retained = self.max_output_chars - len(marker)
        head_size = retained // 2
        tail_size = retained - head_size
        omitted = len(text) - head_size - tail_size
        marker = marker_template.format(omitted=omitted)

        # Account for the number of digits in the final omitted count.
        retained = self.max_output_chars - len(marker)
        head_size = retained // 2
        tail_size = retained - head_size
        omitted = len(text) - head_size - tail_size
        marker = marker_template.format(omitted=omitted)
        return f"{text[:head_size]}{marker}{text[-tail_size:]}", True

    @staticmethod
    def _sprite_info_to_dict(sprite: Any) -> dict[str, Any]:
        values = {
            "name": getattr(sprite, "name", None),
            "id": getattr(sprite, "id", None),
            "status": getattr(sprite, "status", None),
            "url": getattr(sprite, "url", None),
            "primary_region": getattr(sprite, "primary_region", None),
            "organization": getattr(
                sprite,
                "organization",
                getattr(sprite, "organization_name", None),
            ),
            "labels": getattr(sprite, "labels", None),
            "created_at": SpritesToolSpec._serialize_datetime(
                getattr(sprite, "created_at", None)
            ),
            "updated_at": SpritesToolSpec._serialize_datetime(
                getattr(sprite, "updated_at", None)
            ),
        }
        return {key: value for key, value in values.items() if value is not None}

    @staticmethod
    def _serialize_datetime(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    @staticmethod
    def _require_nonempty(field: str, value: str) -> None:
        if not value.strip():
            raise ValueError(f"{field} must not be empty")

    @staticmethod
    def _require_positive_optional(field: str, value: int | None) -> None:
        if value is not None and value <= 0:
            raise ValueError(f"{field} must be greater than zero")
