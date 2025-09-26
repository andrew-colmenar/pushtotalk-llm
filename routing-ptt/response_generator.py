import os, json
from typing import Optional, Dict, Any
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def execute_plan(
    question: str,
    plan: Dict[str, Any],
    *,
    memory_text: Optional[str] = None,     # e.g., "• last task …\n• prior answer …"
    image_b64: Optional[str] = None,       # base64 PNG if you captured a screenshot
    model: str = "gpt-4o-mini"             # vision-capable & inexpensive
) -> str:
    # Build message content
    content = [{"type": "input_text", "text": question}]
    if plan.get("needs_memory") and memory_text:
        content.append({"type": "input_text", "text": f"[MEMORY]\n{memory_text[:2000]}"} )
    if plan.get("needs_image") and image_b64:
        content.append({"type": "input_image", "image_data": image_b64, "mime_type": "image/png"})

    # Enable tools based on plan
    tools = []
    if plan.get("needs_websearch"):
        tools.append({"type": "web_search"})

    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        tools=tools,
        temperature=0.2,
    )
    return resp.output_text

# optional tiny CLI for testing:
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python action_executor.py '<question>' '{\"needs_websearch\":false,\"needs_image\":false,\"needs_memory\":false}'")
        raise SystemExit(1)
    q = sys.argv[1]
    plan = json.loads(sys.argv[2])
    print(execute_plan(q, plan))
import os, json
from openai import OpenAI
from plan_router import plan_tools   # import your planner

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def execute_plan(
    question: str,
    plan: dict,
    memory_text: str = "",
    image_b64: str = ""
) -> str:
    """
    Given the question and the plan dict from plan_tools(),
    attach memory/screenshot/web_search if requested,
    and return the assistant's final answer.
    """
    content = [{"type": "input_text", "text": question}]

    if plan.get("needs_memory") and memory_text:
        content.append({"type": "input_text", "text": f"[MEMORY]\n{memory_text}"})

    if plan.get("needs_image") and image_b64:
        content.append({"type": "input_image", "image_data": image_b64, "mime_type": "image/png"})

    tools = []
    if plan.get("needs_websearch"):
        tools.append({"type": "web_search"})

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": content}],
        tools=tools,
        temperature=0.2,
    )
    return resp.output_text

if __name__ == "__main__":
    # Quick manual test
    q = "What should I put here?"
    plan = plan_tools(q)
    print("Plan:", json.dumps(plan, indent=2))
    answer = execute_plan(q, plan, memory_text="last conversation summary", image_b64="")
    print("Answer:", answer)
