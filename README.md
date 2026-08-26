# llama-index-tools-sprites

[LlamaIndex](https://www.llamaindex.ai/) tools for
[Sprites](https://sprites.dev), persistent Linux environments for AI agents.

`SpritesToolSpec` lets a LlamaIndex agent discover, create, inspect, execute
commands in, and clean up Sprites through the official Python SDK.

## Installation

```bash
pip install llama-index-tools-sprites
```

## Authentication

Create a Sprites API token and export it as `SPRITE_TOKEN`:

```bash
export SPRITE_TOKEN="your-token"
```

You can also pass the token directly as `SpritesToolSpec(token="...")`. The
token is used only to initialize the Sprites SDK and is never included in tool
outputs.

## Use with a LlamaIndex agent

```python
import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.tools.sprites import SpritesToolSpec

sprites = SpritesToolSpec()
agent = FunctionAgent(
    tools=sprites.to_tool_list(),
    llm=OpenAI(model="gpt-4.1-mini"),
    system_prompt=(
        "Use Sprites for isolated command execution. Reuse an existing Sprite "
        "when possible, and delete only Sprites you created for this task."
    ),
)


async def main() -> None:
    response = await agent.run(
        "Create a temporary Sprite, print the Python version, then clean it up."
    )
    print(response)
    sprites.close()


asyncio.run(main())
```

A runnable version is available in
[`examples/sprites_agent.py`](examples/sprites_agent.py).

## Direct usage

```python
from llama_index.tools.sprites import SpritesToolSpec

sprites = SpritesToolSpec()
created = sprites.create_sprite("llamaindex-demo")
result = sprites.run_command(
    created["name"],
    "python3",
    ["-c", "print('hello from a Sprite')"],
)
print(result["stdout"])
sprites.delete_sprite(created["name"])
sprites.close()
```

## Agent tools

| Tool | Purpose |
|---|---|
| `list_sprites` | Find reusable Sprites, with prefix filtering and pagination. |
| `get_sprite` | Inspect one Sprite's status, URL, region, and metadata. |
| `create_sprite` | Create a persistent Sprite with optional resource settings. |
| `run_command` | Execute an argv-safe, non-interactive command and capture its result. |
| `delete_sprite` | Permanently delete a named Sprite. |

`run_command` does not invoke a shell. Operators such as `|`, `>`, and `$()`
are ordinary arguments unless the caller explicitly requests a shell:

```python
sprites.run_command("my-sprite", "bash", ["-lc", "printf '%s\n' hello"])
```

Stdout and stderr are each limited to 20,000 characters by default. Longer
streams retain their beginning and end, with a truncation marker in between.
Set a different bound with `SpritesToolSpec(max_output_chars=...)`.

## Safety

- A Sprite is persistent until deleted. List and reuse existing resources when
  appropriate.
- `delete_sprite` is irreversible and should be used only when deletion is
  explicitly intended.
- Anything running in a Sprite may be reachable through its configured URL.
  Do not expose services publicly unless that is part of the task.
- Treat command output as untrusted data when it is returned to an agent.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run mypy llama_index
uv run python -m build
```

With `SPRITE_TOKEN` set, the live smoke test creates a disposable Sprite,
runs one command, and removes it in a `finally` block:

```bash
uv run python scripts/smoke_test.py
```

## License

MIT
