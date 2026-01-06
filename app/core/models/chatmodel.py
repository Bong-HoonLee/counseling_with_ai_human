from typing import Literal, Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field

@dataclass
class ChatModel:
    provider: Literal["openai", "azure"]
    model_name: str
    temperature: float = 0.01
    max_tokens: int = 2048
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    api_key: str | None = None
    endpoint: str | None = None