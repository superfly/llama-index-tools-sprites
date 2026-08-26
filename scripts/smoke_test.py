"""Create, exercise, and remove a disposable Sprite."""

from __future__ import annotations

from uuid import uuid4

from llama_index.tools.sprites import SpritesToolSpec


def main() -> None:
    name = f"llama-index-tools-smoke-{uuid4().hex[:10]}"
    sprites = SpritesToolSpec()
    created = False
    try:
        result = sprites.create_sprite(name, labels=["llama-index-tools-smoke"])
        created = result["created"] is True
        command = sprites.run_command(
            name,
            "python3",
            ["-c", "print('llama-index-tools-sprites smoke test')"],
            timeout=120,
        )
        assert command["exit_code"] == 0, command
        assert command["stdout"] == "llama-index-tools-sprites smoke test\n", command
        print(f"Smoke test passed on {name}")
    finally:
        if created:
            sprites.delete_sprite(name)
        sprites.close()


if __name__ == "__main__":
    main()
