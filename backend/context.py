"""
R7 – Conversation context management

Keep the in-memory transcript bounded so multi-turn sessions do not grow
latency or token cost without limit.
"""

from __future__ import annotations

from typing import List
from llm_protocol import ChatMessage


# Max non-system messages retained (user + assistant + tool)
DEFAULT_MAX_MESSAGES = 24
# When trimming, keep at least this many most recent messages
DEFAULT_KEEP_RECENT = 12


def cap_messages(
    messages: List[ChatMessage],
    max_messages: int = DEFAULT_MAX_MESSAGES,
    keep_recent: int = DEFAULT_KEEP_RECENT,
) -> List[ChatMessage]:
    """
    Preserve the leading system message(s), drop oldest dialogue turns,
    and never break a tool result away from its following assistant reply
    more than necessary (simple tail keep).
    """
    if len(messages) <= max_messages:
        return messages

    systems: List[ChatMessage] = []
    rest: List[ChatMessage] = []
    for m in messages:
        if m.role == "system" and not rest:
            systems.append(m)
        else:
            rest.append(m)

    if len(rest) <= keep_recent:
        return systems + rest

    trimmed = rest[-keep_recent:]

    # Avoid starting mid-tool: if first kept message is tool, drop it
    while trimmed and trimmed[0].role == "tool":
        trimmed = trimmed[1:]

    # Optional marker so the model knows history was truncated
    note = ChatMessage(
        role="system",
        content="[Earlier conversation turns were dropped to stay within context limits.]",
    )
    return systems + [note] + trimmed
