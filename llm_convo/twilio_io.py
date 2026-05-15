import threading
import logging
import os
import base64
import json
import time
import uuid

from gevent.pywsgi import WSGIServer
from twilio.rest import Client
from twilio.request_validator import RequestValidator
from flask import Flask, send_from_directory, request, abort
from flask_sock import Sock
import simple_websocket
import audioop

from llm_convo.audio_input import WhisperTwilioStream


XML_MEDIA_STREAM = """
<Response>
    <Start>
        <Stream name="Audio Stream" url="wss://{host}/audiostream" />
    </Start>
    <Pause length="60"/>
</Response>
"""


class TwilioServer:
    def __init__(
        self,
        remote_host: str,
        port: int,
        static_dir: str,
        validate_signature: bool = True,
    ):
        """Twilio bridge server.

        Args:
            remote_host: Public hostname (without scheme) Twilio uses to reach us.
            port: Local port to bind.
            static_dir: Directory to serve generated TTS audio from.
            validate_signature: When True (default), reject incoming webhooks
                that do not carry a valid X-Twilio-Signature. Set to False only
                in local development environments without internet exposure.
        """
        self.app = Flask(__name__)
        self.sock = Sock(self.app)
        self.remote_host = remote_host
        self.port = port
        self.static_dir = static_dir
        self.validate_signature = validate_signature
        self.server_thread = threading.Thread(target=self._start, daemon=True)
        self.on_session = None

        try:
            account_sid = os.environ["TWILIO_ACCOUNT_SID"]
            auth_token = os.environ["TWILIO_AUTH_TOKEN"]
            self.from_phone = os.environ["TWILIO_PHONE_NUMBER"]
        except KeyError as e:
            raise RuntimeError(
                f"Missing required Twilio environment variable: {e.args[0]}"
            ) from e

        self.client = Client(account_sid, auth_token)
        self._validator = RequestValidator(auth_token)

        @self.app.route("/audio/<key>")
        def audio(key):
            # `key` is a filename stem we generated ourselves — sanitize anyway
            # to prevent path traversal.
            safe = os.path.basename(key)
            return send_from_directory(self.static_dir, safe + ".mp3")

        @self.app.route("/incoming-voice", methods=["POST"])
        def incoming_voice():
            if self.validate_signature and not self._is_valid_twilio_request():
                logging.warning("Rejected /incoming-voice request: invalid Twilio signature.")
                abort(403)
            return XML_MEDIA_STREAM.format(host=self.remote_host)

        @self.sock.route("/audiostream", websocket=True)
        def on_media_stream(ws):
            session = TwilioCallSession(
                ws, self.client, remote_host=self.remote_host, static_dir=self.static_dir
            )
            if self.on_session is not None:
                thread = threading.Thread(target=self._run_session_safe, args=(session,), daemon=True)
                thread.start()
            try:
                session.start_session()
            except Exception:
                logging.exception("Twilio media stream session crashed.")

    def _run_session_safe(self, session):
        try:
            self.on_session(session)
        except Exception:
            logging.exception("on_session handler raised an exception.")

    def _is_valid_twilio_request(self) -> bool:
        signature = request.headers.get("X-Twilio-Signature", "")
        url = request.url
        params = request.form.to_dict()
        return self._validator.validate(url, params, signature)

    def start_call(self, to_phone: str):
        try:
            return self.client.calls.create(
                twiml=XML_MEDIA_STREAM.format(host=self.remote_host),
                to=to_phone,
                from_=self.from_phone,
            )
        except Exception:
            logging.exception("Failed to start outgoing Twilio call to %s", to_phone)
            raise

    def _start(self):
        logging.info("Starting Twilio Server on port %d", self.port)
        WSGIServer(("0.0.0.0", self.port), self.app, log=None).serve_forever()

    def start(self):
        self.server_thread.start()


class TwilioCallSession:
    def __init__(self, ws, client: Client, remote_host: str, static_dir: str):
        self.ws = ws
        self.client = client
        self.sst_stream = WhisperTwilioStream()
        self.remote_host = remote_host
        self.static_dir = static_dir
        self._call = None
        self._audio_keys: dict = {}

    def media_stream_connected(self):
        return self._call is not None

    def _read_ws(self):
        while True:
            try:
                message = self.ws.receive()
            except simple_websocket.ws.ConnectionClosed:
                logging.warning("Call media stream connection lost.")
                break
            except Exception:
                logging.exception("Unexpected error reading from media stream.")
                break
            if message is None:
                logging.warning("Call media stream closed.")
                break

            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logging.warning("Received non-JSON media stream message; skipping.")
                continue

            event = data.get("event")
            if event == "start":
                logging.info("Call connected, %s", data.get("start"))
                self._call = self.client.calls(data["start"]["callSid"])
            elif event == "media":
                media = data.get("media", {})
                payload = media.get("payload")
                if not payload:
                    continue
                try:
                    chunk = base64.b64decode(payload)
                    if self.sst_stream.stream is not None:
                        self.sst_stream.stream.write(audioop.ulaw2lin(chunk, 2))
                except Exception:
                    logging.exception("Failed to process media chunk.")
            elif event == "stop":
                logging.info("Call media stream ended.")
                break

    def get_audio_fn_and_key(self, text: str):
        # Use UUID-based keys to avoid hash collisions between distinct utterances.
        if text in self._audio_keys:
            key = self._audio_keys[text]
        else:
            key = uuid.uuid4().hex
            self._audio_keys[text] = key
        path = os.path.join(self.static_dir, key + ".mp3")
        return key, path

    def play(self, audio_key: str, duration: float):
        if self._call is None:
            logging.warning("Tried to play audio before call connected; skipping.")
            return
        try:
            self._call.update(
                twiml=f'<Response><Play>https://{self.remote_host}/audio/{audio_key}</Play><Pause length="60"/></Response>'
            )
            time.sleep(duration + 0.2)
        except Exception:
            logging.exception("Failed to play audio on Twilio call.")

    def start_session(self):
        self._read_ws()
