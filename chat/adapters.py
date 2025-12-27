from typing import Any, Dict, List, Optional
import requests


class ChatAdapter:
    def chat(self, messages: List[Dict[str, str]], format_hint: Optional[str] = None) -> str:
        raise NotImplementedError


class OllamaChatAdapter(ChatAdapter):
    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: List[Dict[str, str]], format_hint: Optional[str] = None) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if format_hint:
            payload["format"] = format_hint

        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
            r.raise_for_status()
        except requests.HTTPError:
            if format_hint:
                payload.pop("format", None)
                r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
                r.raise_for_status()
            else:
                raise

        data = r.json()
        return data["message"]["content"]


class OpenRouterChatAdapter(ChatAdapter):
    def __init__(self, model: str, api_key: str, base_url: str = "https://openrouter.ai/api/v1") -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: List[Dict[str, str]], format_hint: Optional[str] = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        r = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=120)
        r.raise_for_status()

        data = r.json()
        return data["choices"][0]["message"]["content"]


class OpenAIChatAdapter(ChatAdapter):
    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1") -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: List[Dict[str, str]], format_hint: Optional[str] = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        r = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=120)
        r.raise_for_status()

        data = r.json()
        return data["choices"][0]["message"]["content"]
