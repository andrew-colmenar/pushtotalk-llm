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
    python model.py

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
from typing import Callable, Optional
import sys
import subprocess
import shutil
import tempfile
import re
import datetime

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
HOTKEY = None  # using Option+Shift combo
SILENCE_PAD_SEC = 0.2  # add a short tail so words aren’t clipped
# Auto-stop and max duration settings
AUTO_SILENCE_MS = 800  # used only if USE_AUTO_SILENCE=1
USE_AUTO_SILENCE = os.getenv("USE_AUTO_SILENCE", "0") == "1"
MAX_UTTERANCE_SEC = int(os.getenv("MAX_UTTERANCE_SEC", "120"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change to "gpt-5" if you have access
# Session memory settings
MEMORY_DIR = os.getenv("MEMORY_DIR", "memories")
MEMORY_MODEL = os.getenv("OPENAI_MEMORY_MODEL", "gpt-4o")  # summarizer model
MEMORY_MAX_CHARS = int(os.getenv("MEMORY_MAX_CHARS", "2000"))
DEBUG = os.getenv("DEBUG", "0") == "1"
WAKE_ENABLED = os.getenv("ENABLE_WAKE_WORD", "0") == "1"
WAKE_WORD = os.getenv("WAKE_WORD", "alfred").lower()
WAKE_COOLDOWN_SEC = float(os.getenv("WAKE_COOLDOWN_SEC", "3.0"))
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH")  # required if wake word enabled
NO_SPEECH_TIMEOUT_SEC = float(os.getenv("NO_SPEECH_TIMEOUT_SEC", "2.5"))
RMS_MIN = float(os.getenv("RMS_MIN", "200"))

def dbg(msg: str):
    if DEBUG:
        ts = time.strftime("%H:%M:%S")
        print(f"[DBG {ts}] {msg}")
SYSTEM_PROMPT = (
    "You are a concise voice assistant. Be direct, and helpful. "
    "Your name is Alfred and refer to me as Bruce."
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
    def __init__(self, on_auto_stop: Optional[Callable[["Recording", str], None]] = None):
        self._recording = False
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._on_stop_cb = on_auto_stop
        # VAD / silence detection state
        self._start_time = 0.0
        self._last_voice_time = 0.0
        self._spoken_once = False
        self._calibrate_until = 0.0
        self._noise_rms = 0.0
        self._calib_count = 0
        # Optional WebRTC VAD
        try:
            import webrtcvad  # type: ignore
            self._webrtcvad = webrtcvad.Vad(2)
            self._vad_frame_len_samples = int(0.02 * SAMPLE_RATE)  # 20 ms
        except Exception:
            self._webrtcvad = None
            self._vad_frame_len_samples = 0

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
            dbg("Recording start: muting TTS")
            # Mute TTS while recording
            try:
                # Speaker instance lives in App; we'll unmute in _process_recording
                pass
            except Exception:
                pass
            self._frames.clear()
            self._q = queue.Queue()
            self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=self._callback)
            self._stream.start()
            self._recording = True
            threading.Thread(target=self._drain, daemon=True).start()
            # Initialize auto-stop state
            now = time.time()
            self._start_time = now
            self._last_voice_time = now
            self._spoken_once = False
            self._calibrate_until = now + 0.5  # 500 ms calibration window for noise floor
            self._noise_rms = 0.0
            self._calib_count = 0
            dbg("Recording started")

    def _drain(self):
        # Pull chunks from queue while recording
        while self._recording or not self._q.empty():
            try:
                chunk = self._q.get(timeout=0.1)
                self._frames.append(chunk)
                # Auto-stop checks (silence detection optional)
                now = time.time()
                # Detect speech
                is_voice = False
                if self._webrtcvad is not None:
                    frame_bytes = (self._vad_frame_len_samples * CHANNELS * 2) if self._vad_frame_len_samples else 0
                    data = chunk.tobytes()
                    if frame_bytes > 0:
                        for i in range(0, len(data) - frame_bytes + 1, frame_bytes):
                            if self._webrtcvad.is_speech(data[i:i+frame_bytes], SAMPLE_RATE):
                                is_voice = True
                                break
                else:
                    # RMS fallback (simple, robust)
                    rms = float(np.sqrt(np.mean((chunk.astype(np.float32)) ** 2)))
                    dbg(f"RMS={rms:.1f} thr={RMS_MIN:.1f} spoken_once={self._spoken_once}")
                    is_voice = rms > RMS_MIN

                if is_voice:
                    self._last_voice_time = now
                    self._spoken_once = True

                # Stop on sustained silence after we've detected speech at least once
                if USE_AUTO_SILENCE:
                    if self._spoken_once and (now - self._last_voice_time) * 1000.0 >= AUTO_SILENCE_MS:
                        dbg("Auto-stop by silence")
                        self._auto_stop("silence")
                        break

                # Stop on max duration
                if (now - self._start_time) >= MAX_UTTERANCE_SEC:
                    dbg("Auto-stop by max duration")
                    self._auto_stop("max_duration")
                    break

                # Extra guard: if we never detected speech and it's been too long, stop
                if USE_AUTO_SILENCE:
                    if not self._spoken_once and (now - self._start_time) >= NO_SPEECH_TIMEOUT_SEC:
                        dbg("Auto-stop by no speech")
                        self._auto_stop("no_speech")
                        break
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

    def is_recording(self) -> bool:
        return self._recording

    def _auto_stop(self, reason: str):
        rec = self.stop()
        if self._on_stop_cb is not None:
            try:
                self._on_stop_cb(rec, reason)
            except Exception:
                pass

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
        # Prefer macOS 'say' for robustness unless disabled
        self._use_say = (sys.platform == "darwin" and os.getenv("USE_MAC_SAY", "1") != "0")
        self._say_voice = os.getenv("TTS_VOICE")  # e.g., Samantha, Alex
        # pyttsx3 uses words per minute; macOS say uses -r wpm
        self._say_rate = os.getenv("TTS_RATE")  # e.g., 190
        # Piper configuration (optional local neural TTS)
        self._piper_exec = os.getenv("PIPER_EXEC") or shutil.which("piper")
        self._piper_model = os.getenv("PIPER_MODEL")  # path to .onnx model file
        self._piper_speaker = os.getenv("PIPER_SPEAKER")  # optional speaker id for multispeaker models
        self._piper_len = os.getenv("PIPER_LENGTH_SCALE") or "0.9"  # slightly faster by default
        self._piper_noise = os.getenv("PIPER_NOISE_SCALE") or "0.667"
        # Use Piper if both binary and model are present
        self._use_piper = bool(self._piper_exec and self._piper_model and os.path.exists(self._piper_model))
        # Current external playback process (for barge-in)
        self._current_proc: subprocess.Popen | None = None
        self._is_speaking = False
        self._muted = False

    def say(self, text: str):
        # Queue text to be spoken by the single TTS thread
        if self._muted:
            dbg("TTS muted; dropping utterance")
            return
        self._say_q.put(text)

    def _tts_worker(self):
        while True:
            text = self._say_q.get()
            if text is None:
                break
            try:
                text = self._normalize_tts_text(text)
                if self._use_piper:
                    self._piper_say(text)
                elif self._use_say:
                    args = ["say"]
                    if self._say_voice:
                        args += ["-v", self._say_voice]
                    if self._say_rate:
                        args += ["-r", self._say_rate]
                    args += [text]
                    self._is_speaking = True
                    proc = subprocess.Popen(args)
                    self._current_proc = proc
                    proc.wait()
                    self._current_proc = None
                    self._is_speaking = False
                else:
                    try:
                        self._is_speaking = True
                        self.engine.say(text)
                        self.engine.runAndWait()
                    finally:
                        self._is_speaking = False
            except Exception:
                # Swallow TTS errors to avoid crashing the app loop
                pass

    def _normalize_tts_text(self, text: str) -> str:
        # Flatten newlines and excessive whitespace to avoid engine quirks
        try:
            return re.sub(r"\s+", " ", text).strip()
        except Exception:
            return text

    def _piper_say(self, text: str):
        # Generate one WAV via Piper and play atomically so cancel stops all speech
        if not self._piper_exec or not self._piper_model:
            return
        args = [
            self._piper_exec,
            "--model", self._piper_model,
            "--length_scale", self._piper_len,
            "--noise_scale", self._piper_noise,
            "--output_file", "/dev/stdout",
        ]
        json_path = self._piper_model + ".json"
        if os.path.exists(json_path):
            args += ["--config", json_path]
        if self._piper_speaker:
            args += ["--speaker", self._piper_speaker]
        proc = subprocess.run(
            args,
            input=(text.strip() + "\n").encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        data = proc.stdout
        if not data:
            return
        if sys.platform == "darwin":
            tmp = None
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(data)
                tmp.flush()
                tmp.close()
                self._is_speaking = True
                proc = subprocess.Popen(["afplay", tmp.name])
                self._current_proc = proc
                proc.wait()
                self._current_proc = None
                self._is_speaking = False
            finally:
                if tmp is not None:
                    try:
                        os.unlink(tmp.name)
                    except Exception:
                        pass
        else:
            try:
                import soundfile as _sf
                import io as _io
                wav_io = _io.BytesIO(data)
                audio, sr = _sf.read(wav_io, dtype="int16")
                if audio.ndim == 1:
                    audio = audio.reshape(-1, 1)
                self._is_speaking = True
                sd.play(audio, sr, blocking=True)
                self._is_speaking = False
            except Exception:
                pass

    def _split_into_sentences(self, text: str) -> list[str]:
        # Split on newlines first, then sentence boundaries.
        parts = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Split at punctuation followed by space/newline
            segments = re.split(r"(?<=[.!?…])\s+", line)
            parts.extend(segments)
        return parts if parts else [text]

    def stop(self):
        # Optional clean shutdown
        self._say_q.put(None)

    def cancel(self):
        """Stop current speech immediately and clear the queue."""
        # Clear pending items
        try:
            while True:
                self._say_q.get_nowait()
        except queue.Empty:
            pass
        # Stop external process if any
        try:
            if self._current_proc is not None:
                try:
                    self._current_proc.terminate()
                except Exception:
                    pass
                try:
                    self._current_proc.wait(timeout=0.5)
                except Exception:
                    pass
                self._current_proc = None
        except Exception:
            pass
        # Stop pyttsx3
        try:
            self.engine.stop()
        except Exception:
            pass
        self._is_speaking = False

    def is_speaking(self) -> bool:
        return self._is_speaking

    def set_muted(self, muted: bool):
        self._muted = muted
        if muted:
            # Ensure we stop any current speech
            self.cancel()

# ---------------------------- LLM Logic -------------------------------

class WakeWordDetectorVosk:
    """Lightweight always-on wake word detector using Vosk offline ASR.

    Runs a background input stream and triggers callback when WAKE_WORD is heard.
    Automatically pauses/resumes around active recording to avoid device conflicts.
    """

    def __init__(self, model_path: str, wake_word: str, on_detect: Callable[[], None]):
        self.model_path = model_path
        self.wake_word = wake_word.lower()
        self.on_detect = on_detect
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._paused = True
        self._last_trigger = 0.0
        self._rec = None
        self._started = False

        try:
            import vosk  # type: ignore
            self._vosk = vosk
        except Exception as e:
            self._vosk = None
            dbg(f"Vosk import failed: {e}")

    def start(self):
        if self._vosk is None:
            dbg("Wake word disabled: vosk not available")
            return
        if not self.model_path or not os.path.exists(self.model_path):
            dbg("Wake word disabled: VOSK_MODEL_PATH missing or invalid")
            return
        with self._lock:
            if self._started:
                return
            try:
                model = self._vosk.Model(self.model_path)
                self._rec = self._vosk.KaldiRecognizer(model, SAMPLE_RATE)
            except Exception as e:
                dbg(f"Vosk model init failed: {e}")
                return
            self._paused = False
            self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=self._callback)
            self._stream.start()
            self._started = True
            dbg("Wake detector started")

    def pause(self):
        with self._lock:
            self._paused = True
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            dbg("Wake detector paused")

    def resume(self):
        with self._lock:
            if not self._started:
                return
            if self._stream is not None:
                return
            self._paused = False
            try:
                self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=self._callback)
                self._stream.start()
                dbg("Wake detector resumed")
            except Exception as e:
                dbg(f"Wake resume failed: {e}")

    def stop(self):
        with self._lock:
            self._paused = True
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            self._started = False
            dbg("Wake detector stopped")

    def _callback(self, indata, frames, time_info, status):
        if self._paused or self._rec is None:
            return
        try:
            data = bytes(indata)
            trig = False
            # Use partial results for responsiveness
            if self._rec.AcceptWaveform(data):
                res = self._rec.Result()
            else:
                res = self._rec.PartialResult()
            text = ""
            if res:
                # JSON like {"text": "..."} or {"partial": "..."}
                if "\"text\":" in res or "\"partial\":" in res:
                    try:
                        import json as _json
                        j = _json.loads(res)
                        text = (j.get("partial") or j.get("text") or "").lower()
                    except Exception:
                        pass
            if text:
                if self.wake_word in text:
                    now = time.time()
                    if (now - self._last_trigger) >= WAKE_COOLDOWN_SEC:
                        self._last_trigger = now
                        dbg(f"Wake word detected in: '{text}'")
                        # Call outside of audio thread
                        threading.Thread(target=self.on_detect, daemon=True).start()
        except Exception:
            pass

class MemoryManager:
    """Maintains a concise per-session memory file and updates it via a summarizer model.

    The memory is a short, structured text blob intended to guide future replies.
    We keep it small to control token costs.
    """

    def __init__(self, memory_path: str, max_chars: int = MEMORY_MAX_CHARS):
        self.memory_path = memory_path
        self.max_chars = max_chars
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        if not os.path.exists(self.memory_path):
            with open(self.memory_path, "w", encoding="utf-8") as f:
                f.write("- Recent topics:\n- Preferences:\n- Open tasks:\n")

    def read(self) -> str:
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def write(self, text: str) -> None:
        # Trim to max_chars (keep tail since it reflects most recent items)
        trimmed = text.strip()
        if len(trimmed) > self.max_chars:
            trimmed = trimmed[-self.max_chars :]
        with open(self.memory_path, "w", encoding="utf-8") as f:
            f.write(trimmed)

    def summarize_and_update(self, user_text: str, assistant_text: str) -> None:
        current = self.read().strip()
        client = get_openai()
        # Instruction keeps the memory concise and structured
        sys_instr = (
            "You curate a short session memory to improve future answers. "
            "Keep it under the requested character budget, structured with concise bullets. "
            "Include only stable facts, user preferences, context, and open tasks. "
            "Do not restate generic chit-chat. Prefer rewriting/merging over appending."
        )
        user_prompt = (
            f"Existing memory (may be empty):\n{current}\n\n"
            f"New exchange to incorporate:\nUser: {user_text}\nAssistant: {assistant_text}\n\n"
            f"Update the memory. Maximum {self.max_chars} characters."
        )
        try:
            resp = client.chat.completions.create(
                model=MEMORY_MODEL,
                messages=[
                    {"role": "system", "content": sys_instr},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=500,
            )
            updated = resp.choices[0].message.content or ""
            self.write(updated)
            try:
                print("[Memory summary]\n" + updated + "\n")
            except Exception:
                pass
        except Exception:
            # Best-effort: on failure, keep previous memory
            pass

def transcribe_wav_bytes(wav_bytes: bytes) -> str:
    client = get_openai()
    # Use in-memory bytes with a named file-like object
    file_like = io.BytesIO(wav_bytes)
    file_like.name = "speech.wav"
    dbg(f"Transcribing {len(wav_bytes)} bytes")
    tr = client.audio.transcriptions.create(
        model="whisper-1",
        file=file_like,
        response_format="json",
        temperature=0.0,
    )
    dbg("Transcription received")
    return tr.text.strip()


def chat_llm(
    user_text: str,
    memory_text: str | None = None,
    last_user_text: str | None = None,
    last_assistant_text: str | None = None,
) -> str:
    client = get_openai()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if memory_text:
        snippet = memory_text.strip()
        if len(snippet) > MEMORY_MAX_CHARS:
            snippet = snippet[-MEMORY_MAX_CHARS :]
        dbg(f"Injecting memory ({len(snippet)} chars)")
        messages.append({
            "role": "system",
            "content": (
                "Session memory (use only if relevant to the question):\n" + snippet
            ),
        })
    if last_user_text and last_assistant_text:
        # Provide the immediately previous exchange as optional context.
        messages.append({
            "role": "system",
            "content": (
                "Previous exchange (use only if relevant; ignore if off-topic):\n"
                f"User: {last_user_text}\n"
                f"Assistant: {last_assistant_text}"
            ),
        })
    dbg(f"LLM user text len={len(user_text)}")
    messages.append({"role": "user", "content": user_text})
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=300,
    )
    dbg("LLM reply received")
    return resp.choices[0].message.content.strip()

# --------------------------- App wiring -------------------------------
class App:
    def __init__(self):
        self.ptt = PushToTalk(on_auto_stop=self._on_auto_stop)
        self.speaker = Speaker()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        # Track pressed keys for modifiers
        self._pressed: set = set()
        self._combo_active = False
        print("\nPush‑to‑Talk ready. Press Option+Shift to toggle. Ctrl+C to exit.\n")
        # Set up per-session memory file
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        mem_dir = MEMORY_DIR
        os.makedirs(mem_dir, exist_ok=True)
        self.memory_path = os.path.join(mem_dir, f"session-{ts}.md")
        self.memory = MemoryManager(self.memory_path, max_chars=MEMORY_MAX_CHARS)
        # Track the most recent single-turn exchange to optionally include next time
        self._last_user_text: str | None = None
        self._last_assistant_text: str | None = None
        # Wake word detector (optional)
        self.wake: WakeWordDetectorVosk | None = None
        if WAKE_ENABLED:
            self.wake = WakeWordDetectorVosk(
                model_path=VOSK_MODEL_PATH or "",
                wake_word=WAKE_WORD,
                on_detect=self._on_wake_detect,
            )

    def run(self):
        if self.wake is not None:
            self.wake.start()
        with self.listener:
            self.listener.join()

    def on_press(self, key):
        # Record key press
        self._pressed.add(key)
        alt_keys = {keyboard.Key.alt, getattr(keyboard.Key, 'alt_l', keyboard.Key.alt), getattr(keyboard.Key, 'alt_r', keyboard.Key.alt)}
        shift_keys = {keyboard.Key.shift, getattr(keyboard.Key, 'shift_l', keyboard.Key.shift), getattr(keyboard.Key, 'shift_r', keyboard.Key.shift)}
        alt_down = any(k in self._pressed for k in alt_keys)
        shift_down = any(k in self._pressed for k in shift_keys)
        if alt_down and shift_down and not self._combo_active:
            self._combo_active = True
            # If currently speaking, cancel speech and start recording immediately
            if self.speaker.is_speaking():
                self.speaker.cancel()
                dbg("Hotkey cancel speech -> start recording")
                if self.wake is not None:
                    self.wake.pause()
                # Mute TTS during recording
                self.speaker.set_muted(True)
                self.ptt.start()
                print("[Recording…]")
                return
            # Otherwise toggle record/stop as usual
            if not self.ptt.is_recording():
                if self.wake is not None:
                    self.wake.pause()
                self.speaker.set_muted(True)
                self.ptt.start()
                print("[Recording…]")
            else:
                rec = self.ptt.stop()
                self._process_recording(rec, reason="manual")

    def on_release(self, key):
        # Track key release
        try:
            self._pressed.remove(key)
        except KeyError:
            pass
        # Reset combo when either modifier is released
        if key in (keyboard.Key.alt, getattr(keyboard.Key, 'alt_l', keyboard.Key.alt), getattr(keyboard.Key, 'alt_r', keyboard.Key.alt),
                   keyboard.Key.shift, getattr(keyboard.Key, 'shift_l', keyboard.Key.shift), getattr(keyboard.Key, 'shift_r', keyboard.Key.shift)):
            self._combo_active = False

    def _on_auto_stop(self, rec: Recording, reason: str):
        self._process_recording(rec, reason=reason)

    def _process_recording(self, rec: Recording, reason: str):
        print("[Transcribing…]")
        dbg(f"Process recording reason={reason} frames={len(rec.frames)}")
        try:
            wav_bytes = rec.to_wav_bytes()
            text = transcribe_wav_bytes(wav_bytes)
            print(f"You: {text}")
            dbg(f"Transcript len={len(text)}")
            if not text:
                self.speaker.say("I didn't catch that.")
                return
            print("[Thinking…]")
            mem_text = self.memory.read()
            dbg(f"Memory read len={len(mem_text)} from {self.memory_path}")
            reply = chat_llm(
                text,
                memory_text=mem_text,
                last_user_text=self._last_user_text,
                last_assistant_text=self._last_assistant_text,
            )
            print(f"Assistant: {reply}")
            dbg(f"Reply len={len(reply)}")
            # Unmute before queuing speech to avoid dropping
            self.speaker.set_muted(False)
            self.speaker.say(reply)
            # Store last exchange for next turn context (optional usage by the model)
            self._last_user_text = text
            self._last_assistant_text = reply
            # Update memory (best-effort; do not block UI)
            threading.Thread(target=self.memory.summarize_and_update, args=(text, reply), daemon=True).start()
            # Resume wake detector after speaking queues
            if self.wake is not None:
                # Give a brief delay to avoid self-trigger on TTS tail
                def _resume():
                    time.sleep(0.4)
                    self.wake.resume()
                threading.Thread(target=_resume, daemon=True).start()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error: {e}")
            dbg(f"Error detail: {e}")
            self.speaker.say("There was an error.")
            self.speaker.set_muted(False)

    def _on_wake_detect(self):
        dbg("Wake callback fired")
        # If already recording, ignore
        if self.ptt.is_recording():
            return
        # Stop any speech and start recording
        self.speaker.cancel()
        if self.wake is not None:
            self.wake.pause()
        self.speaker.set_muted(True)
        self.ptt.start()
        print("[Recording…]")

if __name__ == "__main__":
    # Sanity check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in your environment.")
        raise SystemExit(1)
    try:
        App().run()
    except KeyboardInterrupt:
        print("\nExiting… Bye!\n")
