"""Run a small LlamaIndex agent backed by Sprites."""

import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

from llama_index.tools.sprites import SpritesToolSpec


async def main() -> None:
    sprites = SpritesToolSpec()
    try:
        agent = FunctionAgent(
            tools=sprites.to_tool_list(),
            llm=OpenAI(model="gpt-4.1-mini"),
            system_prompt=(
                "Use Sprites for isolated command execution. Reuse existing "
                "resources when possible and never delete a Sprite unless it was "
                "created for the current task."
            ),
        )
        response = await agent.run(
            "Create a temporary Sprite, print its Python version, and clean it up."
        )
        print(response)
    finally:
        sprites.close()


if __name__ == "__main__":
    asyncio.run(main())
