from typing import List, Optional, Callable, Dict, Any
import os
import logging
import time

from openai import OpenAI, APIError, APITimeoutError, RateLimitError


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_HISTORY = 20
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3


class OpenAIChatCompletion:
    """Wrapper around the OpenAI chat completion API.

    Supports function calling (tools), conversation history truncation, and
    automatic retries on transient errors.
    """

    def __init__(
        self,
        system_prompt: str,
        model: Optional[str] = None,
        max_history: int = DEFAULT_MAX_HISTORY,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_handlers: Optional[Dict[str, Callable[..., str]]] = None,
        temperature: float = 0.7,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        api_key: Optional[str] = None,
    ):
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Export it or pass api_key= to OpenAIChatCompletion."
            )

        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)
        self.system_prompt = system_prompt
        self.model = model or DEFAULT_MODEL
        self.max_history = max_history
        self.tools = tools
        self.tool_handlers = tool_handlers or {}
        self.temperature = temperature
        self.max_retries = max_retries

    def _build_messages(self, transcript: List[str]) -> List[Dict[str, str]]:
        # Keep only the last `max_history` turns to stay within context limits.
        recent = transcript[-self.max_history :] if self.max_history > 0 else transcript
        # The agent that called us speaks on odd indices of the full transcript.
        # We need to preserve the role of each message relative to the full transcript.
        offset = len(transcript) - len(recent)
        messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        for i, text in enumerate(recent):
            # Even global index = user (the other party), odd = assistant (this agent).
            role = "user" if (offset + i) % 2 == 0 else "assistant"
            messages.append({"role": role, "content": text})
        return messages

    def _call_with_retry(self, **kwargs) -> Any:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self.client.chat.completions.create(**kwargs)
            except (APITimeoutError, RateLimitError) as e:
                last_err = e
                wait = 2**attempt
                logging.warning(
                    "OpenAI transient error (attempt %d/%d): %s. Retrying in %ds.",
                    attempt + 1,
                    self.max_retries,
                    e,
                    wait,
                )
                time.sleep(wait)
            except APIError as e:
                last_err = e
                logging.error("OpenAI API error: %s", e)
                break
        raise RuntimeError(f"OpenAI request failed after {self.max_retries} attempts") from last_err

    def get_response(self, transcript: List[str]) -> str:
        messages = self._build_messages(transcript)

        for _ in range(5):  # cap tool-calling loops to avoid infinite recursion
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            if self.tools:
                kwargs["tools"] = self.tools
                kwargs["tool_choice"] = "auto"

            response = self._call_with_retry(**kwargs)
            message = response.choices[0].message

            if not getattr(message, "tool_calls", None):
                return (message.content or "").strip()

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [tc.model_dump() for tc in message.tool_calls],
                }
            )
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                handler = self.tool_handlers.get(name)
                if handler is None:
                    result = f"Error: no handler registered for tool '{name}'."
                else:
                    try:
                        import json

                        args = json.loads(tool_call.function.arguments or "{}")
                        result = str(handler(**args))
                    except Exception as e:
                        logging.exception("Tool '%s' raised an error", name)
                        result = f"Error executing {name}: {e}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

        logging.warning("Tool-calling loop exhausted; returning last assistant content.")
        return (message.content or "").strip()
