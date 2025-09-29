# memory_test.py
# Run: python memory_test.py
import os
import time
from memory import MemoryStore, summarize_fn  # summarize_fn lives in your memory_store.py

def print_state(step, store, session_id="demo", max_chars=300):
    s = store.get(session_id)
    compact = store.get_compact_memory(session_id, max_chars=max_chars)
    print(f"\n=== STEP {step} ===")
    print("Summary:", (s.summary or "(none)"))
    print("Turns (count):", len(s.turns))
    print("Compact memory ->")
    print(compact)
    

if __name__ == "__main__":
   # Make sure OPENAI_API_KEY is set if summarize_fn uses OpenAI.
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set. summarize_fn may fail; set the key to test model-based summaries.\n")

    mem = MemoryStore()
    session_id = "demo"

    # Fake conversation (user/assistant pairs)
    script = [
        ("user", "Hey, I’m setting up a push-to-talk assistant. Where should I start?"),
        ("assistant", "Begin with input pipeline: hotkey → STT → router → executor."),
        ("user", "Cool. Router should decide web search, memory, or screenshot, right?"),
        ("assistant", "Yes—plan first, then attach only what’s needed."),
        ("user", "What model is cheap but good for this?"),
        ("assistant", "gpt-4o-mini for vision; 4.1-mini for pure text."),
        ("user", "Now I’m getting a 400 on image_data param."),
        ("assistant", "Use image_url with a data: URL (base64)."),
        ("user", "Got it. How do I add a system prompt?"),
        ("assistant", "Add a system message at the start of the input list."),
        ("user", "Can you recap our architecture so far?"),
        ("assistant", "Hotkey → STT → plan → capture/collect → execute (with tools) → TTS."),
        ("user", "Okay, let’s keep memory small but useful."),
        ("assistant", "We’ll keep last N turns and a rolling one-line summary."),
        ("user", "Thanks! What’s next to productionize?"),
        ("assistant", "Latency caps, error handling, and token budgeting."),
    ]

    # Feed turns, occasionally summarize old history
    for i, (role, text) in enumerate(script, start=1):
        mem.add_turn(session_id, role, text)

        # Occasionally condense (every 4th message here, just for demo)
        if i % 4 == 0:
            try:
                mem.maybe_summarize(session_id, summarize_fn, keep_last=6, threshold=8)
            except Exception as e:
                # If summarize_fn hits an issue (e.g., no API key), continue gracefully
                print(f"(summarize_fn skipped due to: {e})")

        print_state(i, mem, session_id=session_id, max_chars=400)
        # tiny sleep so the output is easier to read in some terminals
        time.sleep(0.05)