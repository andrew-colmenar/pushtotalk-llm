# simple_plan_router.py
import os, sys, json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def plan_tools(question: str, model="gpt-4.1-mini") -> dict:
    prompt = f"""
Return ONLY valid JSON with exactly these keys:
  needs_websearch (true/false),
  needs_image (true/false),
  needs_memory (true/false),
  reason (string, one short sentence).

Example format:
{{"needs_websearch": false, "needs_image": true, "needs_memory": true, "reason": "short reason"}}

Decide what is needed to answer the user:

- needs_websearch = true only if fresh/verified info is likely required (news, prices, schedules, release dates, "today/now").
- needs_image = true if a screenshot/visual might help OR if the message references the screen/code/UI ("this", "here", "above", "on my screen", "what should I put here?") OR if it's ambiguous what they're pointing at.
- needs_memory = true if the message likely depends on prior turns OR is very short/elliptical ("yes", "no", "that", "same", "continue", "again") OR if meaning is unclear without context.

Be lenient: if uncertain, set needs_image/needs_memory to true.
User question: {question}
"""
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        temperature=0,
    )
    # just parse whatever JSON the model outputs
    return json.loads(resp.output_text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_plan_router.py 'your question here'")
        sys.exit(1)
    user_message = " ".join(sys.argv[1:])
    print(json.dumps(plan_tools(user_message), indent=2))

