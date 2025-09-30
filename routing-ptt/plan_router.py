import os, sys, json
import logging
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logger = logging.getLogger(__name__)

def plan_tools(question: str, model="gpt-4.1-mini") -> dict:
    logger.info(f"plan_tools called with question: '{question}'")
    logger.info(f"Using model: {model}")
    
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
- needs_image = Default to true, only set to false if it adds no value and there is no ambiguity. true if a screenshot/visual might help or if the message references the screen/code/UI ("this", "here", "above", "on my screen", "what should I put here?") OR if it's unclear what they're referencing.
- needs_memory = true if the message has any chance of depending on prior turns OR is very short/elliptical ("yes", "no", "that", "same", "continue", "again") OR if meaning is unclear without context. Default to true, only set to false if you are confident it adds no value and there is no ambiguity.

Be lenient: for any uncertainty set needs_image/needs_memory to true, for few word responses (yes, no, how about this, etc) also set both to true.
User question: {question}
"""
    
    logger.info(f"Sending planning prompt to OpenAI API")
    logger.info(f"Prompt preview: {prompt[:200]}...")
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        
        response_text = resp.choices[0].message.content
        logger.info(f"Planning API response: {response_text}")
        
        # just parse whatever JSON the model outputs
        plan = json.loads(response_text)
        logger.info(f"Parsed plan: {json.dumps(plan, indent=2)}")
        return plan
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response from planning API: {e}")
        logger.error(f"Raw response: {response_text}")
        raise
    except Exception as e:
        logger.error(f"Planning API call failed: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_plan_router.py 'your question here'")
        sys.exit(1)
    user_message = " ".join(sys.argv[1:])
    print(json.dumps(plan_tools(user_message), indent=2))

