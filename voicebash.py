#!/usr/bin/env python3
"""
Voice → Bash  •  speak a request, review the command, run it on approval
"""

import io
import os
import shlex
import subprocess
import sys
import tempfile
import wave

import pyaudio
import torch
import whisper
from dotenv import load_dotenv
import openai
import threading

# ─── Config ──────────────────────────────────────────────────────────────────
WHISPER_MODEL_NAME = "medium"   # tiny | base | small | medium | large-v3
SAMPLE_RATE = 16_000
SYSTEM_PROMPT = (
    "You are an expert in POSIX shell. "
    "Return exactly ONE bash command line, with NO explanations, NO markdown, NO formatting, and NOTHING else. "
    "Only output the raw command, on a single line."
)
load_dotenv()                                     # picks up .env
openai.api_key = os.environ["OPENAI_API_KEY"]     # raises KeyError if missing
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
# ─── Audio capture ───────────────────────────────────────────────────────────
def record(rate: int = SAMPLE_RATE) -> io.BytesIO:
    """Record mono audio from the default mic until the user presses Enter, then return as a BytesIO."""
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=rate,
                     input=True,
                     frames_per_buffer=4096)
    print("🎙️  Speak... (press Enter to stop recording)")
    frames = []
    stop_recording = threading.Event()

    def wait_for_enter():
        input()
        stop_recording.set()

    t = threading.Thread(target=wait_for_enter)
    t.start()

    while not stop_recording.is_set():
        frames.append(stream.read(4096))

    stream.stop_stream()
    stream.close()
    pa.terminate()

    # wrap into a WAV container in‑memory
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)        # int16 = 2 bytes
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))
    buf.seek(0)
    return buf


# ─── Speech‑to‑text (local Whisper) ──────────────────────────────────────────
def transcribe(buf: io.BytesIO) -> str:
    """
    Convert BytesIO WAV -> temp WAV file (for ffmpeg) -> text via Whisper.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(buf.getbuffer())
        tmp_path = tmp.name

    try:
        result = whisper_model.transcribe(
            tmp_path,
            fp16=torch.cuda.is_available()
        )
        return result["text"].strip()
    finally:
        os.remove(tmp_path)       # tidy up the temp file


# ─── Natural‑language request → Bash via ChatGPT ─────────────────────────────
def bash_from_llm(prompt: str) -> str:
    rsp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ]
    )
    return rsp.choices[0].message.content.strip()


# ─── Main flow ───────────────────────────────────────────────────────────────
def main():
    audio = record()
    spoken = transcribe(audio)
    print("\n📝 You said:", spoken)

    cmd = bash_from_llm(spoken)
    print("\n💻 Proposed command:\n", cmd)

    if input("\nRun it? [y/N] ").lower() != "y":
        sys.exit("Aborted.")

    try:
        print("\n⏱  Running…\n")
        subprocess.run(shlex.split(cmd), check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌  Exit code {e.returncode}")


if __name__ == "__main__":
    main()
