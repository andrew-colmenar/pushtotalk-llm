# memory_test.py
# Run: python memory_test.py
import os, time
from memory import MemoryStore, LAST_N

def print_state(step, store, session_id="demo", max_chars=700):
    s = store.get(session_id)
    compact = store.get_compact_memory(session_id, max_chars=max_chars)
    print(f"\n=== STEP {step} ===")
    print("[MEMORY BLOCK]:")
    print(s.memory_block if s.memory_block else "(none yet)")
    print(f"\n[LAST {LAST_N} TURNS VERBATIM]:")
    for t in list(s.turns)[-LAST_N:]:
        who = "User" if t.role == "user" else "Assistant"
        print(f"{who}: {t.text.strip()}")
    print("\n[COMPACT MEMORY SENT TO MODEL]\n" + compact)

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set; summarization will fail.\n")

    mem = MemoryStore()
    session_id = "demo"

    script = [
        ("user", "I’m setting up a push-to-talk assistant. Where should I start?"),
        ("assistant", "Begin with input pipeline: hotkey → STT → router → executor."),
        ("user", "Router should decide web search, memory, or screenshot, right?"),
        ("assistant", "Yes—plan first, then attach only what’s needed."),
        ("user", "What model is cheap but good for this?"),
        ("assistant", "gpt-4o-mini for vision; 4.1-mini for text."),
        ("user", "Now I’m getting a 400 on image_data param."),
        ("assistant", "Use image_url with a data: URL (base64)."),
        ("user", "Got it. How do I add a system prompt?"),
        ("assistant", "Add a system message at the start of the input list."),
        ("user", "Can you recap our architecture so far?"),
        ("assistant", "Hotkey → STT → plan → capture/collect → execute (with tools) → TTS."),
        ("user", "Okay, let’s keep memory small but useful."),
        ("assistant", "We’ll keep last 4 turns verbatim and summarize the 6 before."),
        ("user", "Thanks! What’s next to productionize?"),
        ("assistant", "Latency caps, error handling, and token budgeting."),
    ]

    for i, (role, text) in enumerate(script, start=1):
        mem.add_turn(session_id, role, text)
        # recompute plain-text memory block each step (simple & visible)
        try:
            mem.recompute_summary(session_id)
        except Exception:
            pass
        print_state(i, mem, session_id, max_chars=900)
        time.sleep(0.05)
