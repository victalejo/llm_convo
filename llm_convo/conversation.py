import logging
from typing import Optional

from llm_convo.agents import ChatAgent


def run_conversation(agent_a: ChatAgent, agent_b: ChatAgent, max_turns: Optional[int] = None):
    """Alternate turns between two agents until one returns an empty response
    or `max_turns` total responses have been produced.
    """
    transcript = []
    turns = 0
    while max_turns is None or turns < max_turns:
        for label, agent in (("A", agent_a), ("B", agent_b)):
            try:
                text = agent.get_response(transcript)
            except Exception:
                logging.exception("Agent %s.get_response failed; ending conversation.", label)
                return transcript
            if not text:
                logging.info("Agent %s returned an empty response; ending conversation.", label)
                return transcript
            transcript.append(text)
            turns += 1
            logging.info("-> [%s] %s", label, text)
    return transcript
