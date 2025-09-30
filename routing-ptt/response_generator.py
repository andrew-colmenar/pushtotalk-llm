
import os, json
import logging
from openai import OpenAI
from plan_router import plan_tools
from screenshot import capture_fullscreen_b64

from memory import MemoryStore

memory = MemoryStore()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up logging
logger = logging.getLogger(__name__)


def execute_plan(
    question: str,
    plan: dict,
    memory_text: str = ""
) -> str:
    """
    Given the question and the plan dict from plan_tools(),
    attach memory/screenshot/web_search if requested,
    and return the assistant's final answer.
    """
    logger.info(f"execute_plan called with question: '{question}'")
    logger.info(f"Plan: {json.dumps(plan, indent=2)}")
    
    system_prompt = "You are a helpful voice-based butler/assistant."

    # Build the user message content
    user_content = question

    if plan.get("needs_memory") and memory_text:
        logger.info(f"Adding memory context ({len(memory_text)} chars)")
        user_content += f"\n\n[MEMORY]\n{memory_text}"

    # For images, we need to use the vision model and proper content format
    if plan.get("needs_image"):
        logger.info("Capturing screenshot...")
        try:
            b64 = capture_fullscreen_b64()
            data_url = f"data:image/png;base64,{b64}"
            
            # Use vision model for images
            model = "gpt-4o-mini"
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_content},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ]
            logger.info("Screenshot captured and added to content")
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            # Fall back to text-only
            model = "gpt-4o-mini"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
    else:
        # Text-only content
        model = "gpt-4o-mini"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    tools = []
    if plan.get("needs_websearch"):
        logger.info("Adding web search tool")
        tools.append({"type": "web_search"})

    # Log what we're sending to the LLM
    logger.info(f"Sending to OpenAI API:")
    logger.info(f"  Model: {model}")
    logger.info(f"  System prompt: {system_prompt}")
    logger.info(f"  User content: {user_content[:200]}...")
    logger.info(f"  Has image: {plan.get('needs_image', False)}")
    logger.info(f"  Tools: {tools if tools else 'None'}")
    logger.info(f"  Temperature: 0.3")

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools if tools else None,
            temperature=0.3,
        )
        
        response_text = resp.choices[0].message.content
        logger.info(f"OpenAI API response received: {response_text[:200]}...")
        return response_text
        
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise


if __name__ == "__main__":
    # simple session id; swap for per-user/per-hotkey if you want
    SESSION_ID = "default"

    q = "What time is it in Bali indonesia and In my code what is my plan print statement doing?"

    # 1) plan which tools/context to include
    plan = plan_tools(q)
    print("Plan:", json.dumps(plan, indent=2))

    # 2) recompute memory block from prior conversation, then fetch compact memory
    #    (the memory block includes LONG-TERM bullets + WINDOW summary; last 4 turns
    #     are appended by get_compact_memory)
    memory.recompute_summary(SESSION_ID)
    mem_text = memory.get_compact_memory(SESSION_ID, max_chars=4000) if plan.get("needs_memory") else ""

    # (debug) show memory length to confirm no local truncation
    if mem_text:
        print(f"[MEMORY length]: {len(mem_text)}")

    # 3) generate final answer with the selected tools/context
    answer = execute_plan(
        question=q,
        plan=plan,
        memory_text=mem_text
    )
    print("\nAnswer:\n", answer)

    # 4) record the new turn pair and refresh memory for the next cycle
    memory.add_turn(SESSION_ID, "user", q)
    memory.add_turn(SESSION_ID, "assistant", answer)
    memory.recompute_summary(SESSION_ID)

    # 5) (debug) show what we'd pass next time
    next_mem = memory.get_compact_memory(SESSION_ID, max_chars=4000)
    print("\n[Next-call MEMORY] (len:", len(next_mem), ")")
    print(next_mem)
