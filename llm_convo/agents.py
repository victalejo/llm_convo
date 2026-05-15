from typing import List, Optional, Dict, Any, Callable
from abc import ABC, abstractmethod
import logging

from llm_convo.audio_input import WhisperMicrophone
from llm_convo.audio_output import TTSClient, GoogleTTS
from llm_convo.openai_io import OpenAIChatCompletion
from llm_convo.twilio_io import TwilioCallSession


FALLBACK_ERROR_PHRASE = "Sorry, I'm having trouble responding right now. Could you repeat that?"


class ChatAgent(ABC):
    @abstractmethod
    def get_response(self, transcript: List[str]) -> str:
        pass

    def start(self):
        pass


class MicrophoneInSpeakerTTSOut(ChatAgent):
    def __init__(self, tts: Optional[TTSClient] = None):
        self.mic = WhisperMicrophone()
        self.speaker = tts or GoogleTTS()

    def get_response(self, transcript: List[str]) -> str:
        if len(transcript) > 0:
            self.speaker.play_text(transcript[-1])
        return self.mic.get_transcription()


class TerminalInPrintOut(ChatAgent):
    def get_response(self, transcript: List[str]) -> str:
        if len(transcript) > 0:
            print(transcript[-1])
        return input(" response > ")


class OpenAIChat(ChatAgent):
    def __init__(
        self,
        system_prompt: str,
        init_phrase: Optional[str] = None,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_handlers: Optional[Dict[str, Callable[..., str]]] = None,
        max_history: int = 20,
    ):
        self.openai_chat = OpenAIChatCompletion(
            system_prompt=system_prompt,
            model=model,
            tools=tools,
            tool_handlers=tool_handlers,
            max_history=max_history,
        )
        self.init_phrase = init_phrase

    def get_response(self, transcript: List[str]) -> str:
        if len(transcript) == 0:
            return self.init_phrase or ""
        try:
            return self.openai_chat.get_response(transcript)
        except Exception:
            logging.exception("OpenAIChat.get_response failed; returning fallback phrase.")
            return FALLBACK_ERROR_PHRASE


class TwilioCaller(ChatAgent):
    def __init__(self, session: TwilioCallSession, tts: Optional[TTSClient] = None, thinking_phrase: str = "OK"):
        self.session = session
        self.speaker = tts or GoogleTTS()
        self.thinking_phrase = thinking_phrase

    def _say(self, text: str):
        if not text:
            return
        try:
            key, tts_fn = self.session.get_audio_fn_and_key(text)
            self.speaker.text_to_mp3(text, output_fn=tts_fn)
            duration = self.speaker.get_duration(tts_fn)
            self.session.play(key, duration)
        except Exception:
            logging.exception("Failed to synthesize/play text on Twilio call.")

    def get_response(self, transcript: List[str]) -> str:
        if len(transcript) > 0:
            self._say(transcript[-1])
        resp = self.session.sst_stream.get_transcription()
        self._say(self.thinking_phrase)
        return resp
