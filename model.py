"""
Super‑simple background push‑to‑talk for LLM replies.

Hold F9 to record; release to send audio to Whisper for STT,
pipe transcript to an LLM, and speak the reply locally.

Tested on macOS/Windows/Linux (needs microphone permissions on macOS).

Dependencies (install once):
    pip install openai pynput sounddevice soundfile numpy pyttsx3 python-dotenv

Environment:
    OPENAI_API_KEY=...         # required
    OPENAI_MODEL=gpt-5         # optional; defaults to gpt-4o-mini if unset

Run:
    python talk_hotkey.py

Notes:
- Uses local TTS via pyttsx3 (offline). Swap to OpenAI TTS if you prefer.
- Uses OpenAI Whisper API for transcription. You can replace with faster-whisper
  to run STT fully local if desired.
"""
from __future__ import annotations
import os
import io
import time
import queue
import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import soundfile as sf
from pynput import keyboard
import pyttsx3

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"
HOTKEY = keyboard.Key.f9  # hold to talk
SILENCE_PAD_SEC = 0.2  # add a short tail so words aren’t clipped
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change to "gpt-5" if you have access
SYSTEM_PROMPT = (
    "You are a concise voice assistant. Be brief, direct, and helpful. "
    "If a question is ambiguous, pick the most likely meaning and answer."
)

_openai_client = None

def get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
    return _openai_client

@dataclass
class Recording:
    frames: list[np.ndarray]
    samplerate: int = SAMPLE_RATE
    channels: int = CHANNELS

    def to_wav_bytes(self) -> bytes:
        audio = np.concatenate(self.frames, axis=0) if self.frames else np.zeros((0, CHANNELS), dtype=DTYPE)
        # add a small silence pad to avoid clipping the last word
        pad = np.zeros((int(self.samplerate * SILENCE_PAD_SEC), self.channels), dtype=DTYPE)
        audio = np.vstack([audio, pad])
        buf = io.BytesIO()
        with sf.SoundFile(buf, mode="w", samplerate=self.samplerate, channels=self.channels, subtype="PCM_16", format="WAV") as f:
            f.write(audio)
        return buf.getvalue()

class PushToTalk:
    def __init__(self):
        self._recording = False
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):  # sd callback signature
        if status:
            # You could log status, but don't print to avoid console noise.
            pass
        # Ensure shape: (N, CHANNELS) int16
        chunk = np.array(indata, dtype=DTYPE)
        self._q.put(chunk)

    def start(self):
        with self._lock:
            if self._recording:
                return
            self._frames.clear()
            self._q = queue.Queue()
            self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=self._callback)
            self._stream.start()
            self._recording = True
            threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        # Pull chunks from queue while recording
        while self._recording or not self._q.empty():
            try:
                chunk = self._q.get(timeout=0.1)
                self._frames.append(chunk)
            except queue.Empty:
                pass

    def stop(self) -> Recording:
        with self._lock:
            if not self._recording:
                return Recording(frames=[])
            self._recording = False
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
        # give the drain loop a moment to flush
        time.sleep(0.15)
        return Recording(frames=self._frames[:])

# ------------------------------ TTS -----------------------------------
class Speaker:
    def __init__(self):
        self.engine = pyttsx3.init()
        # Slightly faster speech for snappiness
        rate = self.engine.getProperty('rate')
        self.engine.setProperty('rate', int(rate * 1.05))
        # Serialize TTS calls to avoid "run loop already started" on macOS
        self._say_q: queue.Queue[str | None] = queue.Queue()
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()

    def say(self, text: str):
        # Queue text to be spoken by the single TTS thread
        self._say_q.put(text)

    def _tts_worker(self):
        while True:
            text = self._say_q.get()
            if text is None:
                break
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception:
                # Swallow TTS errors to avoid crashing the app loop
                pass

    def stop(self):
        # Optional clean shutdown
        self._say_q.put(None)

# ---------------------------- LLM Logic -------------------------------

def transcribe_wav_bytes(wav_bytes: bytes) -> str:
    client = get_openai()
    # Use in-memory bytes with a named file-like object
    file_like = io.BytesIO(wav_bytes)
    file_like.name = "speech.wav"
    tr = client.audio.transcriptions.create(
        model="whisper-1",
        file=file_like,
        response_format="json",
        temperature=0.0,
    )
    return tr.text.strip()


def chat_llm(user_text: str) -> str:
    client = get_openai()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=0.4,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()

# --------------------------- App wiring -------------------------------
class App:
    def __init__(self):
        self.ptt = PushToTalk()
        self.speaker = Speaker()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self._is_down = False
        print("\nPush‑to‑Talk ready. Hold F9 to speak, release to send. Ctrl+C to exit.\n")

    def run(self):
        with self.listener:
            self.listener.join()

    def on_press(self, key):
        if key == HOTKEY and not self._is_down:
            self._is_down = True
            self.ptt.start()
            print("[Recording…]")

    def on_release(self, key):
        if key == HOTKEY and self._is_down:
            self._is_down = False
            rec = self.ptt.stop()
            print("[Transcribing…]")
            try:
                wav_bytes = rec.to_wav_bytes()
                text = transcribe_wav_bytes(wav_bytes)
                print(f"You: {text}")
                if not text:
                    self.speaker.say("I didn't catch that.")
                    return
                print("[Thinking…]")
                reply = chat_llm(text)
                print(f"Assistant: {reply}")
                self.speaker.say(reply)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Error: {e}")
                self.speaker.say("There was an error.")

if __name__ == "__main__":
    # Sanity check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in your environment.")
        raise SystemExit(1)
    try:
        App().run()
    except KeyboardInterrupt:
        print("\nExiting… Bye!\n")
