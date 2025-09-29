from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List

import os
from typing import List
from openai import OpenAI


@dataclass
class Turn:
    role: str   # "user" | "assistant"
    text: str

@dataclass
class SessionMemory:
    turns: Deque[Turn] = field(default_factory=lambda: deque(maxlen=12))  # recent turns
    summary: str = ""  # rolling 1–2 line summary

    def add_turn(self, role: str, text: str) -> None:
        self.turns.append(Turn(role, text))

    def compact_text(self, max_chars: int = 800) -> str:
        parts: List[str] = []
        if self.summary:
            parts.append(f"[SUMMARY] {self.summary}")
        for t in list(self.turns)[-8:]:  # last ~4 exchanges
            prefix = "U:" if t.role == "user" else "A:"
            parts.append(f"{prefix} {t.text.strip()}")
        out = "\n".join(parts)
        return (out[:max_chars-3] + "...") if len(out) > max_chars else out

    # NEW: condense older turns into a one-liner using a provided summarize() fn
    def update_summary(self, summarize_fn, keep_last: int = 6) -> None:
        """
        summarize_fn: callable(list_of_Turn, existing_summary:str) -> str
        keep_last: how many most-recent turns to retain verbatim after summarizing
        """
        if len(self.turns) <= keep_last:
            return  # nothing to condense
        old = list(self.turns)[:-keep_last]       # older context
        self.summary = summarize_fn(old, self.summary).strip()
        # drop old, keep last K
        recent = list(self.turns)[-keep_last:]
        self.turns.clear()
        for t in recent:
            self.turns.append(t)

class MemoryStore:
    def __init__(self):
        self._sessions: dict[str, SessionMemory] = {}

    def get(self, session_id: str) -> SessionMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionMemory()
        return self._sessions[session_id]

    def add_turn(self, session_id: str, role: str, text: str) -> None:
        self.get(session_id).add_turn(role, text)

    def get_compact_memory(self, session_id: str, max_chars: int = 800) -> str:
        return self.get(session_id).compact_text(max_chars=max_chars)

    # convenience wrapper
    def maybe_summarize(self, session_id: str, summarize_fn, keep_last: int = 6, threshold: int = 10) -> None:
        s = self.get(session_id)
        if len(s.turns) >= threshold:
            s.update_summary(summarize_fn, keep_last=keep_last)



client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_fn(old_turns: List[Turn], existing_summary: str) -> str:
    """
    Returns a single short line that captures the essence of old_turns + existing_summary.
    """
    # Build a tiny text payload (keep it cheap)
    lines = []
    if existing_summary:
        lines.append(f"Existing summary: {existing_summary}")
    for t in old_turns[-12:]:  # cap to avoid long prompts
        who = "User" if t.role == "user" else "Assistant"
        lines.append(f"{who}: {t.text.strip()}")
    text = "\n".join(lines)

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": (
                    "Summarize the following conversation history into ONE concise sentence "
                    "(<= 30 words), capturing task/topic, decisions, and any specific goals. "
                    "Avoid fluff.\n\n" + text
                )
            }]
        }],
        temperature=0.2,
    )
    return (resp.output_text or "").strip()

