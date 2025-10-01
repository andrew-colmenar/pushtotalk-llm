import os
import sys
import json
from typing import Dict, Any

"""
How to run
Backend:
Set env: export OPENAI_API_KEY=sk-...
Start API: python routing-ptt/test_ui.py
Frontend:
cd routing-ptt/web
npm install
npm run dev
Visit: http://localhost:5173
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

# Ensure this file's directory is importable as a module root (directory name has a hyphen)
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from memory import MemoryStore
from response_no_route import respond_always_enhanced


app = FastAPI(title="PTT Chat (Minimal)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


memory = MemoryStore()


def _get_session_id(data: Dict[str, Any]) -> str:
    sid = (data.get("session_id") or "web").strip()
    return sid or "web"


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse(
        """
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>PTT Backend</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }
      code { background:#f3f4f6; padding:2px 4px; border-radius:4px; }
    </style>
  </head>
  <body>
    <h1>PTT Backend is running</h1>
    <p>Open the Vite UI at <a href=\"http://localhost:5173\">http://localhost:5173</a>.</p>
    <p>API endpoints:</p>
    <ul>
      <li><code>GET /history</code></li>
      <li><code>POST /chat</code></li>
    </ul>
  </body>
  </html>
        """
    )


@app.get("/history")
async def history(session_id: str = "web") -> JSONResponse:
    s = memory.get(session_id)
    turns = [{"role": t.role, "text": t.text} for t in list(s.turns)]
    return JSONResponse({"session_id": session_id, "turns": turns})


@app.post("/chat")
async def chat(req: Request) -> JSONResponse:
    if not os.getenv("OPENAI_API_KEY"):
        return JSONResponse({"error": "OPENAI_API_KEY is not set"}, status_code=400)

    try:
        data = await req.json()
    except Exception:
        data = {}

    message = (data.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)

    session_id = _get_session_id(data)

    # Recompute memory and build the same payload style we send to the LLM
    memory.recompute_summary(session_id)
    mem_text = memory.get_compact_memory(session_id, max_chars=4000)

    # These flags keep the demo simple and deterministic
    include_web_search = False
    include_screenshot = False
    include_memory = True

    system_prompt = (
        "You are a voice-based assistant with resources to answer the user's question to the best of your ability.\n"
        "Resources: Conversational history is stored under [CONVERSATION HISTORY] section and a screenshot of the user's screen at time of question is provided. You may also use web search to find relevant information.\n"
        "Be concise, avoid long responses unless a longer narrative is needed.\n"
        "IMPORTANT: Respond in a TTS-friendly way: Avoid filler, emojis, bold text, markdown. Use minimal formatting and no links, if referncing a source just name the site.\n"
        "IMPORTANT: When using web search results: Provide only the direct answer concisely as possible. Do not dump raw articles or long summaries. Reformat all outputs for TTS.\n"
    )

    input_user_content = [
        {"type": "input_text", "text": "USER MESSAGE NEEDING RESPONSE: " + message}
    ]
    if include_memory and mem_text:
        input_user_content.append({
            "type": "input_text",
            "text": "[CONVERSATION HISTORY]\n" + mem_text
        })

    payload = [
        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
        {"role": "user", "content": input_user_content},
    ]

    # Clean text view for the UI
    payload_text_parts = [
        "[SYSTEM]",
        system_prompt.strip(),
        "",
        "[USER]",
        ("[CONVERSATION HISTORY]\n" + mem_text) if (include_memory and mem_text) else "",
        "QUESTION:",
        message,
    ]
    payload_text = "\n".join([p for p in payload_text_parts if p]).strip()

    # Generate the assistant answer using the same underlying function
    answer = respond_always_enhanced(
        session_id=session_id,
        question=message,
        include_web_search=include_web_search,
        include_screenshot=include_screenshot,
        include_memory=include_memory,
        memory_text=mem_text,
    )

    # Record turns and refresh memory
    memory.add_turn(session_id, "user", message)
    memory.add_turn(session_id, "assistant", answer)
    memory.recompute_summary(session_id)

    return JSONResponse({
        "session_id": session_id,
        "answer": answer,
        "payload": payload,
        "payload_text": payload_text,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


