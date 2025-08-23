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
HOTKEY = keyboard.Key.f9  # press to toggle start/stop
SILENCE_PAD_SEC = 0.2  # add a short tail so words aren’t clipped
# Auto-stop and max duration settings
AUTO_SILENCE_MS = 800  # stop after this long of silence following speech
MAX_UTTERANCE_SEC = int(os.getenv("MAX_UTTERANCE_SEC", "120"))
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

    def _drain(self):
        # Pull chunks from queue while recording
        while self._recording or not self._q.empty():
            try:
                chunk = self._q.get(timeout=0.1)
                self._frames.append(chunk)
                # Auto-stop checks
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
                    # RMS fallback
                    rms = float(np.sqrt(np.mean((chunk.astype(np.float32)) ** 2)))
                    if now < self._calibrate_until:
                        # running average for noise floor
                        self._noise_rms = (self._noise_rms * self._calib_count + rms) / (self._calib_count + 1)
                        self._calib_count += 1
                    # Threshold: 3x noise or absolute minimum
                    threshold = max(self._noise_rms * 3.0 if self._calib_count > 0 else 0.0, 500.0)
                    is_voice = rms > threshold

                if is_voice:
                    self._last_voice_time = now
                    self._spoken_once = True

                # Stop on sustained silence after we've detected speech at least once
                if self._spoken_once and (now - self._last_voice_time) * 1000.0 >= AUTO_SILENCE_MS:
                    self._auto_stop("silence")
                    break

                # Stop on max duration
                if (now - self._start_time) >= MAX_UTTERANCE_SEC:
                    self._auto_stop("max_duration")
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
        self._piper_len = os.getenv("PIPER_LENGTH_SCALE") or "1.0"
        self._piper_noise = os.getenv("PIPER_NOISE_SCALE") or "0.667"
        # Use Piper if both binary and model are present
        self._use_piper = bool(self._piper_exec and self._piper_model and os.path.exists(self._piper_model))

    def say(self, text: str):
        # Queue text to be spoken by the single TTS thread
        self._say_q.put(text)

    def _tts_worker(self):
        while True:
            text = self._say_q.get()
            if text is None:
                break
            try:
                if self._use_piper:
                    self._piper_say(text)
                elif self._use_say:
                    args = ["say"]
                    if self._say_voice:
                        args += ["-v", self._say_voice]
                    if self._say_rate:
                        args += ["-r", self._say_rate]
                    args += [text]
                    subprocess.run(args, check=False)
                else:
                    self.engine.say(text)
                    self.engine.runAndWait()
            except Exception:
                # Swallow TTS errors to avoid crashing the app loop
                pass

    def _piper_say(self, text: str):
        # Generate WAV via Piper and play for each sentence to ensure complete playback
        if not self._piper_exec or not self._piper_model:
            return
        sentences = self._split_into_sentences(text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            args = [
                self._piper_exec,
                "--model", self._piper_model,
                "--length_scale", self._piper_len,
                "--noise_scale", self._piper_noise,
                "--output_file", "/dev/stdout",
                "--sentence-silence", "0.30",
            ]
            # Include model config JSON if present for better prosody/params
            json_path = self._piper_model + ".json"
            if os.path.exists(json_path):
                args += ["--config", json_path]
            if self._piper_speaker:
                args += ["--speaker", self._piper_speaker]
            proc = subprocess.run(
                args,
                input=(sentence + "\n").encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            data = proc.stdout
            if not data:
                continue
            if sys.platform == "darwin":
                tmp = None
                try:
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp.write(data)
                    tmp.flush()
                    tmp.close()
                    subprocess.run(["afplay", tmp.name], check=False)
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
                    sd.play(audio, sr, blocking=True)
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
        self.ptt = PushToTalk(on_auto_stop=self._on_auto_stop)
        self.speaker = Speaker()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        print("\nPush‑to‑Talk ready. Press F9 to start/stop. Ctrl+C to exit.\n")

    def run(self):
        with self.listener:
            self.listener.join()

    def on_press(self, key):
        if key == HOTKEY:
            if not self.ptt.is_recording():
                self.ptt.start()
                print("[Recording…]")
            else:
                rec = self.ptt.stop()
                self._process_recording(rec, reason="manual")

    def on_release(self, key):
        # No-op for toggle behavior
        pass

    def _on_auto_stop(self, rec: Recording, reason: str):
        self._process_recording(rec, reason=reason)

    def _process_recording(self, rec: Recording, reason: str):
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
