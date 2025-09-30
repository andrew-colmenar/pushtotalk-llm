from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Dict
import os
from openai import OpenAI

# Config: last N verbatim, previous M summarized
LAST_N = 4
WINDOW_M = 6
MAX_TURNS_BUFFER = 200  # local ring buffer; not all of this is sent to the model
_MAX_CHARS = 8000

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class Turn:
    role: str   # "user" | "assistant"
    text: str

@dataclass
class SessionMemory:
    turns: Deque[Turn] = field(default_factory=lambda: deque(maxlen=MAX_TURNS_BUFFER))
    # Single plain-text block we pass straight into the final model:
    # 
    # MEMORIES:
    # - bullet...
    # - bullet...
    #
    # SUMMARY:
    # one concise sentence...
    memory_block: str = ""  # produced by recompute_memory_block()

    def add_turn(self, role: str, text: str) -> None:
        self.turns.append(Turn(role, text))

    def _window_slice(self) -> List[Turn]:
        """The M turns immediately before the last N: [-(N+M) : -N]."""
        total = len(self.turns) 
        if total < LAST_N + 1:
            return []
        start = max(0, total - (LAST_N + WINDOW_M))
        end = max(0, total - LAST_N)
        return list(self.turns)[start:end]

    def recompute_memory_block(self) -> None:
        """
        Refine the existing memory block (MEMORIES + SUMMARY) using the
        new window (M turns before last N). The model receives the current
        block and updates it in-place: prune/add MEMORIES and revise SUMMARY.
        """
        window = self._window_slice()
        if not window:
            # nothing to update; keep whatever we had
            return

        # Build the window text
        lines = []
        for t in window:
            who = "User" if t.role == "user" else "Assistant"
            lines.append(f"{who}: {t.text.strip()}")
        block = "\n".join(lines)

        existing = (self.memory_block or "").strip()

        prompt = f"""You are maintaining conversation memory.

    You will receive:
    1) CURRENT MEMORY STORE + SUMMARY (May be empty)
    2) NEW CONVERSATION TO INCORPORATE

    Your job is to update the CURRENT MEMORY STORE + SUMMARY in the following format:
    
    Format:

    SUMMARY:
    Summary that contextualizes the block so we know the recent general direction/idea, key steps, and current assistant thinking. 

    MEMORIES:
    - Only add if clearly long-term memory: explicit rules, hard preferences, identities, targets, vital constraints
    - Do NOT include status updates, recent steps, stack traces, one-offs, or summaries
    - Concise and precise bullets.

    EXISTING MEMORY STORE + SUMMARY:
    {existing if existing else "(none yet)"}

    WINDOW TO INCORPORATE:
    {block}
    """.strip()

        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            temperature=0.2,
        )

        text = (resp.output_text or "").strip()
        # Tolerate accidental code fences
        if text.startswith("```"):
            text = text.strip("`").strip()
            if "\n" in text:
                first, rest = text.split("\n", 1)
                if first.lower() in ("json", "txt", "text"):
                    text = rest.strip()

        # Store the whole refined block as-is (we don't parse)
        self.memory_block = text


#     def recompute_memory_block_old(self) -> None:
#         """
#         Ask the model to produce a single plain-text block:

#         MEMORIES:
#         - <0–5 durable bullets for long-term use, strictly curated>

#         SUMMARY:
#         <ONE concise sentence (<= 30 words) contextualizing ONLY the window (M turns before last N)>

#         We store that whole block (no parsing) in self.memory_block.
#         """
#         window = self._window_slice()
#         if not window:
#             self.memory_block = ""
#             return

#         # Build the window text
#         lines = []
#         for t in window:
#             who = "User" if t.role == "user" else "Assistant"
#             lines.append(f"{who}: {t.text.strip()}")
#         block = "\n".join(lines)

#         prompt = f"""You are maintaining conversation memory.

# Produce a plain text response in this structure:

# MEMORIES:
# - (durable facts/rules/preferences explicitly stated or vital long term items)
# - (each bullet concise and precise)

# SUMMARY:
# <summary that contextualizes the block so we know the general direction/idea AND key steps/decisions>
        
# WINDOW TO SUMMARIZE:
# {block}
# """.strip()

#         resp = client.responses.create(
#             model="gpt-4.1-mini",
#             input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
#             temperature=0.2,
#         )

#         text = (resp.output_text or "").strip()
#         # Tolerate accidental code fences
#         if text.startswith("```"):
#             text = text.strip("`").strip()
#             # if it starts with a language tag, drop the first line
#             if "\n" in text:
#                 first, rest = text.split("\n", 1)
#                 if first.lower() in ("json", "txt", "text"):
#                     text = rest.strip()

#         self.memory_block = text

    def compact_text(self, max_chars: int = _MAX_CHARS) -> str:
        """
        What we send to the final model as [MEMORY]:
        - memory_block (MEMORIES + SUMMARY)
        - last N turns verbatim
        """
        parts: List[str] = []
        if self.memory_block:
            parts.append(self.memory_block.strip())

        recent = list(self.turns)[-LAST_N:]
        if recent:
            parts.append("")  # blank line before verbatim
            for t in recent:
                prefix = "User:" if t.role == "user" else "Assistant:"
                parts.append(f"{prefix} {t.text.strip()}")

        out = "\n".join(parts).strip()
        return (out[: max_chars - 3] + "...") if len(out) > max_chars else out


class MemoryStore:
    def __init__(self):
        self._sessions: Dict[str, SessionMemory] = {}

    def get(self, session_id: str) -> SessionMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionMemory()
        return self._sessions[session_id]

    def add_turn(self, session_id: str, role: str, text: str) -> None:
        self.get(session_id).add_turn(role, text)

    def recompute_summary(self, session_id: str) -> None:
        """Recompute the plain-text memory block (MEMORIES + SUMMARY)."""
        self.get(session_id).recompute_memory_block()

    def get_compact_memory(self, session_id: str, max_chars: int = _MAX_CHARS) -> str:
        return self.get(session_id).compact_text(max_chars=max_chars)
