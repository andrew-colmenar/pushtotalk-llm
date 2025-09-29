from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple

@dataclass
class Turn:
    role: str   # user or assistant
    text: str

@dataclass
class SessionMemory:
    turns: Deque[Turn] = field(default_factory=lambda: deque(maxlen=12))  # last 12 turns
    summary: str = ""  # 1–2 lines you can update later

    def add_turn(self, role: str, text: str) -> None:
        self.turns.append(Turn(role, text))

    def compact_text(self, max_chars: int = 800) -> str:
        """
        Return a short memory string to pass to the model.
        Includes summary (if any) + last few turns, clipped to max_chars.
        """
        parts: List[str] = []
        if self.summary:
            parts.append(f"[SUMMARY] {self.summary}")

        # include most recent turns (user/assistant)
        for t in list(self.turns)[-8:]:  # last 8 items (≈ 4 exchanges)
            prefix = "U:" if t.role == "user" else "A:"
            parts.append(f"{prefix} {t.text.strip()}")

        out = "\n".join(parts)
        if len(out) > max_chars:
            out = out[: max_chars - 3].rstrip() + "..."
        return out

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
