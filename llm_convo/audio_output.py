from abc import ABC, abstractmethod
from typing import Optional
import os
import tempfile
import subprocess
import logging

from gtts import gTTS
import pyaudio
import wave


class TTSClient(ABC):
    @abstractmethod
    def text_to_mp3(self, text: str, output_fn: Optional[str] = None) -> str:
        pass

    def play_text(self, text: str) -> str:
        tmp_mp3 = self.text_to_mp3(text)
        tmp_wav = tmp_mp3.replace(".mp3", ".wav")
        subprocess.call(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", tmp_mp3, tmp_wav])

        wf = wave.open(tmp_wav, "rb")
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=audio.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )

        data = wf.readframes(1024)
        while data != b"":
            stream.write(data)
            data = wf.readframes(1024)

        stream.close()
        audio.terminate()

    def get_duration(self, audio_fn: str) -> float:
        popen = subprocess.Popen(
            ["ffprobe", "-hide_banner", "-loglevel", "error", "-show_entries", "format=duration", "-i", audio_fn],
            stdout=subprocess.PIPE,
        )
        popen.wait()
        output = popen.stdout.read().decode("utf-8")
        duration = float(output.split("=")[1].replace("\r\n", "\n").split("\n")[0])
        return duration


def _tmp_mp3_path(output_fn: Optional[str]) -> str:
    return output_fn or os.path.join(tempfile.mkdtemp(), "tts.mp3")


class GoogleTTS(TTSClient):
    def __init__(self, lang: str = "en"):
        self.lang = lang

    def text_to_mp3(self, text: str, output_fn: Optional[str] = None) -> str:
        tmp_fn = _tmp_mp3_path(output_fn)
        gTTS(text, lang=self.lang).save(tmp_fn)
        return tmp_fn


class OpenAITTS(TTSClient):
    """Higher-quality TTS using OpenAI's `tts-1` / `tts-1-hd` models.

    Voices: alloy, echo, fable, onyx, nova, shimmer.
    """

    def __init__(
        self,
        voice: str = "alloy",
        model: str = "tts-1",
        api_key: Optional[str] = None,
    ):
        from openai import OpenAI

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set; cannot use OpenAITTS.")
        self.client = OpenAI(api_key=api_key)
        self.voice = voice
        self.model = model

    def text_to_mp3(self, text: str, output_fn: Optional[str] = None) -> str:
        tmp_fn = _tmp_mp3_path(output_fn)
        try:
            response = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="mp3",
            )
            response.stream_to_file(tmp_fn)
        except Exception:
            logging.exception("OpenAI TTS failed; falling back to gTTS.")
            gTTS(text, lang="en").save(tmp_fn)
        return tmp_fn


class ElevenLabsTTS(TTSClient):
    """High-quality TTS via ElevenLabs.

    Requires `pip install elevenlabs` and ELEVENLABS_API_KEY in the environment.
    """

    def __init__(
        self,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # "Rachel" default voice
        model_id: str = "eleven_turbo_v2_5",
        api_key: Optional[str] = None,
    ):
        try:
            from elevenlabs.client import ElevenLabs
        except ImportError as e:
            raise RuntimeError(
                "elevenlabs package not installed. Run: pip install elevenlabs"
            ) from e

        api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not set; cannot use ElevenLabsTTS.")
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.model_id = model_id

    def text_to_mp3(self, text: str, output_fn: Optional[str] = None) -> str:
        tmp_fn = _tmp_mp3_path(output_fn)
        try:
            audio_iter = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                model_id=self.model_id,
                text=text,
                output_format="mp3_44100_128",
            )
            with open(tmp_fn, "wb") as f:
                for chunk in audio_iter:
                    if chunk:
                        f.write(chunk)
        except Exception:
            logging.exception("ElevenLabs TTS failed; falling back to gTTS.")
            gTTS(text, lang="en").save(tmp_fn)
        return tmp_fn
