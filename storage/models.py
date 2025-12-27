from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    ts: Optional[str] = Field(None, description="ISO timestamp")
    turn: Optional[int] = Field(None, description="Turn number in conversation")


class Memory(BaseModel):
    id: str = Field(..., description="Unique memory ID")
    text: str = Field(..., description="Memory text content")
    ts: str = Field(..., description="ISO timestamp")
    session_id: str = Field(..., description="Source session ID")
    turn: int = Field(..., description="Turn number")
    provenance: Dict[str, str] = Field(..., description="Original user/assistant messages")


class Session(BaseModel):
    id: str = Field(..., description="Unique session ID")
    created_at: str = Field(..., description="ISO timestamp of creation")
    messages: List[Message] = Field(default_factory=list, description="Conversation messages")


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Session ID")
    message: str = Field(..., description="User message")


class ChatResponse(BaseModel):
    session_id: str = Field(..., description="Session ID")
    turn: int = Field(..., description="Turn number")
    assistant: str = Field(..., description="Assistant response")
    memory_status: str = Field(..., description="Memory pipeline status")


class SessionResponse(BaseModel):
    session_id: str = Field(..., description="New session ID")


class HistoryResponse(BaseModel):
    messages: List[Message] = Field(..., description="Session messages")


class MemoriesResponse(BaseModel):
    memories: List[Memory] = Field(..., description="Memory records")
    count: int = Field(..., description="Total memory count")
