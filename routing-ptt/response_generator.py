
import os, json
from openai import OpenAI
from plan_router import plan_tools
from screenshot import capture_fullscreen_b64

from memory import MemoryStore

memory = MemoryStore()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
    system_prompt = "You are a helpful voice-based butler/assistant."

    # Build one user message with multiple content blocks
    content = [{"type": "input_text", "text": question}]

    if plan.get("needs_memory") and memory_text:
        content.append({"type": "input_text", "text": f"[MEMORY]\n{memory_text}"})

    if plan.get("needs_image"):
        b64 = capture_fullscreen_b64()
        data_url = f"data:image/png;base64,{b64}"
        content.append({
            "type": "input_image",
            "image_url": data_url,
        })

    tools = []
    if plan.get("needs_websearch"):
        tools.append({"type": "web_search"})

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": content}
        ],
        tools=tools,
        temperature=0.3,
    )
    return resp.output_text


if __name__ == "__main__":
    # simple session id; swap for per-user/per-hotkey if you want
    SESSION_ID = "default"

    q = "In my code what is my plan print statement doing?"

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
