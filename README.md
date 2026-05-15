# llm_convo

Build an AI phone agent powered by ChatGPT and Twilio. Supports incoming and outgoing calls, function calling (tools), multiple TTS providers, and signed webhook validation.

### How it works

```
Twilio Webhook → Flask app → Twilio Media Stream (websocket)
   → Whisper (speech-to-text) → ChatGPT (LLM + optional tools)
   → TTS (gTTS / OpenAI / ElevenLabs) → Twilio <Play>
```

### Features

- **OpenAI SDK v1+** with automatic retries on transient errors (rate limits, timeouts).
- **Function calling (tools)** — let the LLM look up customer data, schedule appointments, etc.
- **Conversation history truncation** to avoid blowing the context window on long calls.
- **Multiple TTS providers**: Google TTS (default), OpenAI TTS, ElevenLabs.
- **Twilio request signature validation** — webhooks are rejected if the signature is missing or invalid.
- **Robust error handling** — a failed LLM call or transcription does not crash the bridge; the agent says a fallback phrase and continues.
- **Configurable transcription language** instead of hard-coded English.
- **UUID-based audio cache keys** to avoid collisions across utterances.

### Setup

```bash
pip install git+https://github.com/sshh12/llm_convo
```

Environment variables:

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | yes | OpenAI chat completions + TTS. |
| `TWILIO_ACCOUNT_SID` | yes (phone) | Twilio account ID. |
| `TWILIO_AUTH_TOKEN` | yes (phone) | Used for API calls and webhook signature validation. |
| `TWILIO_PHONE_NUMBER` | yes (phone) | The "from" number for outgoing calls. |
| `ELEVENLABS_API_KEY` | optional | Only if using `ElevenLabsTTS`. |

### Demo

#### Basic Text Chat

```bash
python examples/keyboard_chat_with_gpt.py
```

#### Twilio Helpline (incoming calls)

```bash
python examples/twilio_ngrok_ml_rhyme_hotline.py --preload_whisper --start_ngrok
```

This creates an ngrok tunnel and prints a webhook URL to point to in Twilio settings for a purchased phone number.

#### Twilio Pizza Order (outgoing call)

```bash
python examples/twilio_ngrok_pizza_order.py --preload_whisper --start_ngrok --phone_number "+1XXXXXXXXXX"
```

### Code Snippets

#### Basic Haiku hotline

```python
from gevent import monkey

monkey.patch_all()

from llm_convo.agents import OpenAIChat, TwilioCaller
from llm_convo.twilio_io import TwilioServer
from llm_convo.conversation import run_conversation
import logging
import time

logging.getLogger().setLevel(logging.INFO)

tws = TwilioServer(remote_host="abcdef.ngrok.app", port=8080, static_dir="/path/to/static")
# Point twilio voice webhook to https://abcdef.ngrok.app/incoming-voice
tws.start()
agent_a = OpenAIChat(
    system_prompt="You are a Haiku Assistant. Answer whatever the user wants but always in a rhyming Haiku.",
    init_phrase="This is Haiku Bot, how can I help you.",
    model="gpt-4o-mini",
)

def run_chat(sess):
    agent_b = TwilioCaller(sess)
    while not agent_b.session.media_stream_connected():
        time.sleep(0.1)
    run_conversation(agent_a, agent_b)

tws.on_session = run_chat
```

#### Function calling (tools)

Let the LLM look up order status by calling Python functions:

```python
from llm_convo.agents import OpenAIChat

def lookup_order(order_id: str) -> str:
    # Replace with your DB lookup, API call, etc.
    return f"Order {order_id} is out for delivery, arriving 6:30pm."

tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up the current status of an order by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order ID."}
                },
                "required": ["order_id"],
            },
        },
    }
]

agent = OpenAIChat(
    system_prompt="You are a customer support agent for Pizza Palace.",
    init_phrase="Hi, this is Pizza Palace — how can I help?",
    tools=tools,
    tool_handlers={"lookup_order": lookup_order},
)
```

#### Higher-quality voices

```python
from llm_convo.audio_output import OpenAITTS, ElevenLabsTTS
from llm_convo.agents import TwilioCaller

# OpenAI TTS (alloy, echo, fable, onyx, nova, shimmer)
caller = TwilioCaller(session, tts=OpenAITTS(voice="nova", model="tts-1-hd"))

# ElevenLabs (requires `pip install elevenlabs`)
caller = TwilioCaller(session, tts=ElevenLabsTTS(voice_id="21m00Tcm4TlvDq8ikWAM"))
```

#### Disable signature validation (local dev only)

```python
tws = TwilioServer(..., validate_signature=False)
```

> **Do not** disable signature validation in production — without it, anyone who knows your webhook URL can trigger calls on your account.

### Architecture

| Module | Purpose |
|---|---|
| [`agents.py`](llm_convo/agents.py) | Abstract `ChatAgent` plus implementations for OpenAI, microphone, terminal and Twilio. |
| [`openai_io.py`](llm_convo/openai_io.py) | OpenAI SDK v1+ wrapper with retries, history truncation and tool calling. |
| [`audio_input.py`](llm_convo/audio_input.py) | Whisper transcription from microphone and Twilio media streams. |
| [`audio_output.py`](llm_convo/audio_output.py) | TTS providers: Google, OpenAI, ElevenLabs. |
| [`twilio_io.py`](llm_convo/twilio_io.py) | Flask + websocket bridge to Twilio Media Streams. |
| [`conversation.py`](llm_convo/conversation.py) | Drives turn-taking between two `ChatAgent`s. |

### Notes on costs

Rough order-of-magnitude per minute of call:

- Twilio voice: ~$0.013/min (US)
- OpenAI `gpt-4o-mini`: a few tenths of a cent per turn
- OpenAI TTS `tts-1`: ~$0.015 / 1k chars
- ElevenLabs: varies by plan
- Whisper (local): free, but needs a GPU for low latency
