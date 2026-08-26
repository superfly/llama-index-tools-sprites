from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from sprites import ListOptions, SpriteConfig

from llama_index.tools.sprites import SpritesToolSpec


@dataclass
class FakeSprite:
    name: str
    id: str | None = None
    status: str | None = None
    url: str | None = None
    primary_region: str | None = None
    organization_name: str | None = None
    labels: list[str] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.closed = False
        self.list_result = SimpleNamespace(
            sprites=[],
            has_more=False,
            next_continuation_token=None,
        )

    def close(self) -> None:
        self.closed = True

    def list_sprites(self, options: ListOptions) -> Any:
        self.calls.append(("list_sprites", options))
        return self.list_result

    def get_sprite(self, name: str) -> FakeSprite:
        self.calls.append(("get_sprite", name))
        return FakeSprite(
            name=name,
            id="sprite-123",
            status="running",
            url=f"https://{name}.sprites.app",
            primary_region="ord",
            organization_name="example-org",
            labels=["agent"],
            created_at=datetime(2026, 8, 26, tzinfo=timezone.utc),
        )

    def create_sprite(
        self,
        name: str,
        config: SpriteConfig | None = None,
        labels: list[str] | None = None,
    ) -> FakeSprite:
        self.calls.append(("create_sprite", name, config, labels))
        return FakeSprite(name=name)

    def sprite(self, name: str) -> FakeSprite:
        self.calls.append(("sprite", name))
        return FakeSprite(name=name)

    def delete_sprite(self, name: str) -> None:
        self.calls.append(("delete_sprite", name))


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def tool_spec(fake_client: FakeClient) -> SpritesToolSpec:
    return SpritesToolSpec(client=fake_client)  # type: ignore[arg-type]


def test_is_llama_index_tool_spec(tool_spec: SpritesToolSpec) -> None:
    assert isinstance(tool_spec, BaseToolSpec)
    assert [tool.metadata.name for tool in tool_spec.to_tool_list()] == [
        "list_sprites",
        "get_sprite",
        "create_sprite",
        "run_command",
        "delete_sprite",
    ]


def test_requires_token_without_injected_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPRITE_TOKEN", raising=False)

    with pytest.raises(ValueError, match="Sprites API token"):
        SpritesToolSpec()


def test_uses_token_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("SPRITE_TOKEN", "test-token")
    monkeypatch.setattr("llama_index.tools.sprites.base.SpritesClient", Client)

    SpritesToolSpec(base_url="https://example.test", timeout=12.0)

    assert captured == {
        "token": "test-token",
        "base_url": "https://example.test",
        "timeout": 12.0,
    }


def test_lists_sprites_with_pagination(
    tool_spec: SpritesToolSpec, fake_client: FakeClient
) -> None:
    fake_client.list_result = SimpleNamespace(
        sprites=[
            SimpleNamespace(
                name="agent-one",
                id="sprite-1",
                status="running",
                url="https://agent-one.sprites.app",
                primary_region="ord",
                organization="example-org",
            )
        ],
        has_more=True,
        next_continuation_token="next-page",
    )

    result = tool_spec.list_sprites(
        prefix="agent-", max_results=25, continuation_token="current-page"
    )

    options = fake_client.calls[0][1]
    assert options.prefix == "agent-"
    assert options.max_results == 25
    assert options.continuation_token == "current-page"
    assert result == {
        "sprites": [
            {
                "name": "agent-one",
                "id": "sprite-1",
                "status": "running",
                "url": "https://agent-one.sprites.app",
                "primary_region": "ord",
                "organization": "example-org",
            }
        ],
        "has_more": True,
        "next_continuation_token": "next-page",
    }


@pytest.mark.parametrize("max_results", [0, 101])
def test_rejects_invalid_page_size(
    tool_spec: SpritesToolSpec, max_results: int
) -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        tool_spec.list_sprites(max_results=max_results)


def test_gets_and_serializes_sprite(
    tool_spec: SpritesToolSpec, fake_client: FakeClient
) -> None:
    result = tool_spec.get_sprite("agent-one")

    assert fake_client.calls == [("get_sprite", "agent-one")]
    assert result == {
        "name": "agent-one",
        "id": "sprite-123",
        "status": "running",
        "url": "https://agent-one.sprites.app",
        "primary_region": "ord",
        "organization": "example-org",
        "labels": ["agent"],
        "created_at": "2026-08-26T00:00:00+00:00",
    }


def test_creates_sprite_with_resource_config(
    tool_spec: SpritesToolSpec, fake_client: FakeClient
) -> None:
    result = tool_spec.create_sprite(
        "agent-one",
        region="ord",
        ram_mb=2048,
        cpus=2,
        storage_gb=20,
        labels=["llama-index"],
    )

    _, name, config, labels = fake_client.calls[0]
    assert name == "agent-one"
    assert config == SpriteConfig(
        region="ord", ram_mb=2048, cpus=2, storage_gb=20
    )
    assert labels == ["llama-index"]
    assert result == {"created": True, "name": "agent-one"}


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("ram_mb", {"ram_mb": 0}),
        ("cpus", {"cpus": -1}),
        ("storage_gb", {"storage_gb": 0}),
    ],
)
def test_rejects_invalid_resource_config(
    tool_spec: SpritesToolSpec, field: str, kwargs: dict[str, int]
) -> None:
    with pytest.raises(ValueError, match=field):
        tool_spec.create_sprite("agent-one", **kwargs)


def test_runs_command_without_shell_interpretation(
    monkeypatch: pytest.MonkeyPatch,
    tool_spec: SpritesToolSpec,
    fake_client: FakeClient,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(sprite: Any, command: str, *args: str, **kwargs: Any) -> Any:
        captured.update(
            sprite=sprite,
            command=command,
            args=args,
            kwargs=kwargs,
        )
        return SimpleNamespace(
            returncode=7,
            stdout=b"standard output\n",
            stderr=b"standard error\n",
        )

    monkeypatch.setattr("llama_index.tools.sprites.base._run_command", fake_run)

    result = tool_spec.run_command(
        "agent-one",
        "python3",
        ["-c", "print('hello')"],
        cwd="/workspace",
        env={"EXAMPLE": "value"},
        timeout=45,
    )

    assert fake_client.calls == [("sprite", "agent-one")]
    assert captured["command"] == "python3"
    assert captured["args"] == ("-c", "print('hello')")
    assert captured["kwargs"] == {
        "capture_output": True,
        "timeout": 45,
        "check": False,
        "cwd": "/workspace",
        "env": {"EXAMPLE": "value"},
    }
    assert result == {
        "exit_code": 7,
        "stdout": "standard output\n",
        "stderr": "standard error\n",
        "stdout_truncated": False,
        "stderr_truncated": False,
    }


def test_truncates_large_command_output(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeClient
) -> None:
    tool_spec = SpritesToolSpec(
        client=fake_client,  # type: ignore[arg-type]
        max_output_chars=256,
    )

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return SimpleNamespace(returncode=0, stdout=b"a" * 400, stderr=None)

    monkeypatch.setattr("llama_index.tools.sprites.base._run_command", fake_run)

    result = tool_spec.run_command("agent-one", "print-a-lot")

    assert result["stdout_truncated"] is True
    assert "truncated" in result["stdout"]
    assert result["stdout"].startswith("a")
    assert result["stdout"].endswith("a")
    assert len(result["stdout"]) == 256
    assert result["stderr"] == ""


@pytest.mark.parametrize("timeout", [0, 301])
def test_rejects_invalid_command_timeout(
    tool_spec: SpritesToolSpec, timeout: float
) -> None:
    with pytest.raises(ValueError, match="between 0.1 and 300"):
        tool_spec.run_command("agent-one", "true", timeout=timeout)


def test_deletes_sprite(tool_spec: SpritesToolSpec, fake_client: FakeClient) -> None:
    result = tool_spec.delete_sprite("agent-one")

    assert fake_client.calls == [("delete_sprite", "agent-one")]
    assert result == {"deleted": True, "name": "agent-one"}


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("get_sprite", ("  ",)),
        ("create_sprite", ("",)),
        ("run_command", ("sprite", "")),
        ("delete_sprite", ("\t",)),
    ],
)
def test_rejects_empty_names_and_commands(
    tool_spec: SpritesToolSpec, method: str, args: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        getattr(tool_spec, method)(*args)


def test_closes_client(tool_spec: SpritesToolSpec, fake_client: FakeClient) -> None:
    tool_spec.close()

    assert fake_client.closed is True
