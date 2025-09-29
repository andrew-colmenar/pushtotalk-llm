import os, json
from openai import OpenAI
from plan_router import plan_tools  
from screenshot import capture_fullscreen_b64

from memory import MemoryStore, summarize_fn
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
        temperature=0.2,
    )
    print(content)
    print(resp.output_text)
    return resp.output_text

# if __name__ == "__main__":
#     q = "What is happening?"
#     plan = plan_tools(q)
#     print("Plan:", json.dumps(plan, indent=2))
#     answer = execute_plan(q, plan, memory_text="user's name is Batman", image_b64="")
#     #print("Answer:", answer)

"""
if __name__ == "__main__":
    from plan_router import plan_tools


    q = "In my code what is my plan print statement doing?"

    plan = plan_tools(q)
    print("Plan:", json.dumps(plan, indent=2))

    # capture screenshot only if needed
    # image_b64 = ""
    # if plan.get("needs_image"):
    #     from screenshot import capture_fullscreen_b64
    #     image_b64 = capture_fullscreen_b64()

    answer = execute_plan(
        question=q,
        plan=plan,
        memory_text="User's name is Batman"
            )
    print("\nAnswer:\n", answer)


"""


if __name__ == "__main__":
    # simple session id; swap for per-user/per-hotkey if you want
    SESSION_ID = "default"

    q = "In my code what is my plan print statement doing?"
    plan = plan_tools(q)
    print("Plan:", json.dumps(plan, indent=2))

    # ✅ pull compact memory (even if plan says false we can pass empty)
    mem_text = memory.get_compact_memory(SESSION_ID, max_chars=800) if plan.get("needs_memory") else ""

    answer = execute_plan(
        question=q,
        plan=plan,
        memory_text=mem_text
    )
    print("\nAnswer:\n", answer)

    # ✅ record turns into memory
    memory.add_turn(SESSION_ID, "user", q)
    memory.add_turn(SESSION_ID, "assistant", answer)

    # ✅ optional: occasionally condense older turns
    memory.maybe_summarize(SESSION_ID, summarize_fn, keep_last=6, threshold=10)

    # (debug) show what we'd pass next time
    print("\n[Next-call MEMORY]")
    print(memory.get_compact_memory(SESSION_ID, max_chars=400))

