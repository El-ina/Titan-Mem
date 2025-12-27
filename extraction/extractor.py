import json
import re
from typing import List

from .prompts import EXTRACT_PROMPT
from chat.adapters import ChatAdapter


def sanitize_memories(memories: List[str]) -> List[str]:
    cleaned = []
    seen = set()
    for mem in memories:
        text = re.sub(r"^[\-\d\.\s]+", "", str(mem)).strip()
        if not text or len(text) < 6:
            continue
        if text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
    return cleaned


def fallback_memories(user_text: str, max_sentences: int = 3) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", user_text.strip())
    return sanitize_memories(parts[:max_sentences])


def extract_atomic_memories(
    user_text: str,
    assistant_text: str,
    adapter: ChatAdapter
) -> List[str]:
    prompt = EXTRACT_PROMPT.format(
        user=user_text.strip(),
        assistant=assistant_text.strip()
    )

    content = adapter.chat(
        [
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": prompt}
        ],
        format_hint="json"
    )

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return fallback_memories(user_text)

    if isinstance(data, list):
        return sanitize_memories([str(x) for x in data])

    memories = data.get("memories") if isinstance(data, dict) else None
    if not isinstance(memories, list):
        return fallback_memories(user_text)

    return sanitize_memories([str(x) for x in memories])
