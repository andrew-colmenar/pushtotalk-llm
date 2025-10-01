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
    memory_text: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
) -> str:
    """
    Answer using Responses API, always attempting to include:
    - memory (long-term + last turns)
    - screenshot as an input_image
    - web_search tool (auto)

    Falls back gracefully if screenshot capture fails.
    """
    system_prompt = (
        "You are a voice-based assistant with resources to answer the user's question to the best of your ability.\n"
        "Resources: Conversational history is stored under [CONVERSATION HISTORY] section and a screenshot of the user's screen at time of question is provided. You may also use web search to find relevant information.\n"
        "Be concise, avoid long responses unless a longer narrative is needed.\n"
        "IMPORTANT: Respond in a TTS-friendly way: Avoid filler, emojis, bold text, markdown. Use minimal formatting and no links, if referncing a source just name the site.\n"
        "IMPORTANT:When using web search results: Provide only the direct answer concisely as possible. Do not dump raw articles or long summaries. Reformat all outputs for TTS.\n"
    )


    #user_text_parts = [question]

    #memory_text: Optional[str] = None

    input_user_content = [{"type": "input_text", "text": "USER MESSAGE NEEDING RESPONSE: " + question}]

    if include_screenshot:
        try:
            b64 = capture_fullscreen_b64()
            data_url = f"data:image/png;base64,{b64}"
            input_user_content.append({"type": "input_image", "image_url": data_url})
            logger.info("Screenshot captured and appended")
        except Exception as e:
            logger.error(f"Screenshot skipped: {e}")

    if include_memory:
        try:
            if memory_text is None:
                memory_text = memory.get_compact_memory(session_id, max_chars=40000)
            if memory_text:
                input_user_content.append({"type": "input_text", "text": "[CONVERSATION HISTORY]\n" + memory_text})
                logger.info(f"Added memory context ({len(memory_text)} chars)")
        except Exception as e:
            logger.error(f"Failed to fetch memory: {e}")

    tools = []
    tool_choice = "none"
    if include_web_search:
        tools.append({"type": "web_search"})
        tool_choice = "auto"

    # logger.info("Sending to OpenAI Responses API")
    # logger.info(f"  Model: {model}")
    # logger.info(f"  Include memory: {include_memory}")
    # logger.info(f"  Include screenshot: {include_screenshot}")
    # logger.info(f"  Web search: {include_web_search}")

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
    # logger.info(f"OpenAI response: {text[:200]}...")

    #print(input_user_content)
    return text

# if __name__ == "__main__":
#     import sys
#     logging.basicConfig(level=logging.INFO)

#     SESSION_ID = "default"
#     turns = [
#         "What's new in AI today?",
#         "Summarize the key points in one paragraph.",
#         "Based on that, list 3 follow-up tasks for me.",
#         "Also, what do you see on my screen that might be relevant?",
#         "Please recap our ongoing plan from memory in 2 sentences.",
#     ]

#     for i, user_msg in enumerate(turns, start=1):
#         print(f"\n[Turn {i}] User:\n{user_msg}")

#         # Build memory block for debugging
#         mem_text = memory.get_compact_memory(SESSION_ID, max_chars=4000)

#         # Human-readable view
#         print("\n[Context Sent to LLM]")
#         if mem_text:
#             print("--- MEMORY BLOCK ---")
#             print(mem_text)
#         print("--- QUESTION ---")
#         print(user_msg)

#         # Exact payload view
#         system_prompt = "You are a voice-based assistant..."  # keep your real system prompt here
#         input_user_content = []
#         if mem_text:
#             input_user_content.append({"type": "input_text", "text": "[MEMORY]\n" + mem_text})
#         input_user_content.append({"type": "input_text", "text": user_msg})

#         payload = [
#             {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
#             {"role": "user", "content": input_user_content},
#         ]
#         print("\n[Exact API Payload Sent to OpenAI]")
#         print(json.dumps(payload, indent=2)[:2000])  # truncate if too long

#         # Now actually call your generator
#         answer = respond_always_enhanced(
#             session_id=SESSION_ID,
#             question=user_msg,
#             include_web_search=True,
#             include_screenshot=True,
#             include_memory=True,
#         )
#         print("\nAssistant:\n", answer)

#         # Update memory
#         memory.add_turn(SESSION_ID, "user", user_msg)
#         memory.add_turn(SESSION_ID, "assistant", answer)
#         if i % 4 == 0:
#             memory.recompute_summary(SESSION_ID)

#         print("\n[Next-call MEMORY] (len:", len(memory.get_compact_memory(SESSION_ID, max_chars=40000)), ")")
#         print(memory.get_compact_memory(SESSION_ID, max_chars=40000))






if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    SESSION_ID = "default"
    # Example multi-turn conversation to exercise memory + web_search + screenshot
    turns = [
        "What's new in AI today?",
        "Summarize the key points in one paragraph.",
        "Based on that, list 3 follow-up tasks for me.",
        "Also, what do you see on my screen that might be relevant?",
        "Please recap our ongoing plan from memory in 2 sentences.",
        "Add two constraints we should always remember going forward.",
        "Now give me one-sentence recap of our progress so far.",
        "Where was I last week?",
        "Who is Lebron's dad",
        "Tell me a story about Drew Colmenar"
    ]

    for i, user_msg in enumerate(turns, start=1):
        print("=" * 80)
        print(f"[Turn {i}] User:\n{user_msg}")

        # Build the compact memory that would be attached
        mem_text = memory.get_compact_memory(SESSION_ID, max_chars=4000)

        # Show what we’re about to send to the LLM
        # print("\n[Context Sent to LLM]")
        # if mem_text:
        #     print("--- MEMORY BLOCK ---")
        #     print(mem_text)
        # print("--- QUESTION ---")
        # print(user_msg)

        # Generate assistant answer (this call also reads current memory)
        answer = respond_always_enhanced(
            session_id=SESSION_ID,
            question=user_msg,
            include_web_search=True,
            include_screenshot=True,
            include_memory=True,
        )
        print("\n[Assistant Response]")
        print(answer)

        # Persist turn pair
        memory.add_turn(SESSION_ID, "user", user_msg)
        memory.add_turn(SESSION_ID, "assistant", answer)

        # Recompute summary every 4 turns
        if i % 4 == 0:
            print("\n(Recomputing memory summary at turn", i, ")")
            memory.recompute_summary(SESSION_ID)

        # Show updated memory that will be available for the next turn
        # print("\n[Next-call MEMORY] (len:", len(memory.get_compact_memory(SESSION_ID)), ")")
        # print(memory.get_compact_memory(SESSION_ID))



