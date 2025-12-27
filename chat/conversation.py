from typing import List, Dict
import yaml

from .adapters import ChatAdapter, OllamaChatAdapter, OpenRouterChatAdapter, OpenAIChatAdapter
from storage.models import Session
from storage.sessions import load_session, get_next_turn, add_message


SYSTEM_PROMPT = (
    "You are a calm, precise assistant. Keep answers concise but useful. "
    "Ask brief clarifying questions when needed."
)


def load_chat_config() -> dict:
    config_path = "config/chat_models.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_chat_adapter() -> ChatAdapter:
    config = load_chat_config()
    current = config.get("current", "ollama")

    if current == "ollama":
        ollama_cfg = config["ollama"]
        return OllamaChatAdapter(
            model=ollama_cfg["model"],
            base_url=ollama_cfg["base_url"]
        )
    elif current == "openrouter":
        openrouter_cfg = config["openrouter"]
        return OpenRouterChatAdapter(
            model=openrouter_cfg["model"],
            api_key=openrouter_cfg["api_key"],
            base_url=openrouter_cfg.get("base_url", "https://openrouter.ai/api/v1")
        )
    elif current == "openai":
        openai_cfg = config["openai"]
        return OpenAIChatAdapter(
            model=openai_cfg["model"],
            api_key=openai_cfg["api_key"],
            base_url=openai_cfg.get("base_url", "https://api.openai.com/v1")
        )
    else:
        raise ValueError(f"Unsupported chat backend: {current}")


def send_chat_message(session_id: str, user_message: str) -> tuple[Session, str]:
    session = load_session(session_id)
    user_text = user_message.strip()

    if not user_text:
        raise ValueError("Message cannot be empty.")

    turn = get_next_turn(session)

    chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    chat_messages.extend([{"role": msg.role, "content": msg.content} for msg in session.messages])
    chat_messages.append({"role": "user", "content": user_text})

    adapter = get_chat_adapter()
    assistant_text = adapter.chat(chat_messages)

    add_message(session, "user", user_text, turn)
    add_message(session, "assistant", assistant_text, turn)

    return session, assistant_text
