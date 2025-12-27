from chat.adapters import ChatAdapter

from extraction.extractor import extract_atomic_memories

from chat.conversation import get_chat_adapter as get_adapter


def get_extraction_adapter() -> ChatAdapter:
    return get_adapter()
