import os
import sys
import json
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# Ensure this file's directory is importable as a module root (directory name has a hyphen)
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from plan_router import plan_tools
from response_generator import execute_plan
from memory import MemoryStore


app = FastAPI(title="PTT Chat UI")

# Allow local dev origins (including future npm dev server if wanted)
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
async def index() -> HTMLResponse:
    # Minimal single-file UI
    html = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>PTT Chat UI</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; background: #0b1120; color: #e5e7eb; }
    header { padding: 12px 16px; background: #111827; border-bottom: 1px solid #1f2937; }
    header h1 { margin: 0; font-size: 16px; font-weight: 600; }
    #app { display: flex; flex-direction: column; height: 100vh; }
    #messages { flex: 1; overflow: auto; padding: 16px; }
    .msg { max-width: 800px; margin: 0 auto 12px auto; padding: 12px 14px; border-radius: 10px; white-space: pre-wrap; line-height: 1.35; }
    .user { background: #1f2937; }
    .assistant { background: #0ea5e9; color: #0b1120; }
    .sysline { opacity: 0.7; font-size: 12px; text-align: center; margin: 8px 0; }
    #composer { display: flex; gap: 8px; padding: 12px; background: #111827; border-top: 1px solid #1f2937; }
    #input { flex: 1; background: #0f172a; color: #e5e7eb; border: 1px solid #1f2937; border-radius: 8px; padding: 10px 12px; }
    button { background: #22c55e; color: #052e16; border: none; border-radius: 8px; padding: 10px 14px; font-weight: 600; cursor: pointer; }
    button[disabled] { opacity: 0.6; cursor: not-allowed; }
  </style>
</head>
<body>
  <div id=\"app\">
    <header><h1>PTT Chat UI</h1></header>
    <div id=\"messages\"></div>
    <form id=\"composer\">
      <input id=\"input\" name=\"message\" placeholder=\"Type a message...\" autocomplete=\"off\" />
      <button id=\"send\" type=\"submit\">Send</button>
    </form>
  </div>
  <script>
    const elMessages = document.getElementById('messages');
    const elForm = document.getElementById('composer');
    const elInput = document.getElementById('input');
    const elSend = document.getElementById('send');

    function addMsg(role, text) {
      const div = document.createElement('div');
      div.className = 'msg ' + (role === 'user' ? 'user' : 'assistant');
      div.textContent = text;
      elMessages.appendChild(div);
      elMessages.scrollTop = elMessages.scrollHeight;
    }

    function addSys(text) {
      const div = document.createElement('div');
      div.className = 'sysline';
      div.textContent = text;
      elMessages.appendChild(div);
      elMessages.scrollTop = elMessages.scrollHeight;
    }

    async function loadHistory() {
      try {
        const res = await fetch('/history');
        if (!res.ok) return;
        const data = await res.json();
        elMessages.innerHTML = '';
        (data.turns || []).forEach(t => addMsg(t.role, t.text));
      } catch (e) {
        console.error(e);
      }
    }

    elForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const message = elInput.value.trim();
      if (!message) return;
      addMsg('user', message);
      elInput.value = '';
      elInput.disabled = true; elSend.disabled = true;
      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message })
        });
        const data = await res.json();
        if (res.ok) {
          addMsg('assistant', data.answer || '');
        } else {
          addSys('Error: ' + (data.error || res.status));
        }
      } catch (err) {
        addSys('Network error');
        console.error(err);
      } finally {
        elInput.disabled = false; elSend.disabled = false; elInput.focus();
      }
    });

    loadHistory();
  </script>
</body>
</html>
    """
    return HTMLResponse(content=html)


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

    try:
        # 1) Plan tools/context
        plan = plan_tools(message)

        # 2) Recompute memory block and get compact memory if needed
        memory.recompute_summary(session_id)
        mem_text = memory.get_compact_memory(session_id, max_chars=4000) if plan.get("needs_memory") else ""

        # 3) Generate answer
        answer = execute_plan(question=message, plan=plan, memory_text=mem_text)

        # 4) Record turn pair and refresh memory
        memory.add_turn(session_id, "user", message)
        memory.add_turn(session_id, "assistant", answer)
        memory.recompute_summary(session_id)

        return JSONResponse({
            "session_id": session_id,
            "answer": answer,
            "plan": plan,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


