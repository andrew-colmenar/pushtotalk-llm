import os, json
from openai import OpenAI
from plan_router import plan_tools  
from screenshot import capture_fullscreen_b64



client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def execute_plan(
    question: str,
    plan: dict,
    memory_text: str = "",
    image_b64: str = "",

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
        image_b64 = capture_fullscreen_b64()  # capture once, pass to model
        content.append({"type": "input_image", "image_data": image_b64, "mime_type": "image/png"})

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

