import os, json, logging
from typing import Optional
from openai import OpenAI

from memory import MemoryStore
from screenshot import capture_fullscreen_b64


logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
memory = MemoryStore()


def respond_always_enhanced(
    session_id: str,
    question: str,
    *,
    include_web_search: bool = True,
    include_screenshot: bool = True,
    include_memory: bool = True,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.3,
) -> str:
    """
    Answer using Responses API, always attempting to include:
    - memory (long-term + last turns)
    - screenshot as an input_image
    - web_search tool (auto)

    Falls back gracefully if screenshot capture fails.
    """
    system_prompt = "You are a helpful voice-based butler/assistant."

    user_text_parts = [question]


    memory_text: Optional[str] = None
    if include_memory:
        try:
            # Ensure memory block is up to date and fetch compact window
            memory.recompute_summary(session_id)
            memory_text = memory.get_compact_memory(session_id, max_chars=4000)
            if memory_text:
                user_text_parts.append("\n\n[MEMORY]\n" + memory_text)
                logger.info(f"Added memory context ({len(memory_text)} chars)")
        except Exception as e:
            logger.error(f"Failed to compute or fetch memory: {e}")


    # Build input content for Responses API (after memory may have been appended)
    input_user_content = [
        {"type": "input_text", "text": "\n".join(user_text_parts)}
    ]
    if include_screenshot:
        try:
            b64 = capture_fullscreen_b64()
            data_url = f"data:image/png;base64,{b64}"
            input_user_content.append({"type": "input_image", "image_url": data_url})
            logger.info("Screenshot captured and appended to input content")
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")

    tools = []
    tool_choice = "none"
    if include_web_search:
        tools.append({"type": "web_search"})
        tool_choice = "auto"

    logger.info("Sending to OpenAI Responses API")
    logger.info(f"  Model: {model}")
    logger.info(f"  Include memory: {include_memory}")
    logger.info(f"  Include screenshot: {include_screenshot}")
    logger.info(f"  Web search: {include_web_search}")

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": input_user_content},
        ],
        tools=tools if tools else None,
        tool_choice=tool_choice,
        temperature=temperature,
    )

    text = (resp.output_text or "").strip()
    logger.info(f"OpenAI response: {text[:200]}...")
    return text


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    SESSION_ID = "default"
    q = sys.argv[1] if len(sys.argv) > 1 else "What's new in AI today? Also summarize our recent context."

    answer = respond_always_enhanced(
        session_id=SESSION_ID,
        question=q,
        include_web_search=True,
        include_screenshot=True,
        include_memory=True,
    )
    print("\nAnswer:\n", answer)

    # Store the new turn
    memory.add_turn(SESSION_ID, "user", q)
    memory.add_turn(SESSION_ID, "assistant", answer)
    memory.recompute_summary(SESSION_ID)


