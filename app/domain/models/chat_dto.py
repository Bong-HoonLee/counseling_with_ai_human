from typing import Literal, Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field

@dataclass
class ChatRES:
    chatbotmessage: str
    input_token: Optional[int] = None
    ouput_token: Optional[int] = None
    total_token: Optional[int] = None
    reasoning_token: Optional[int] = None

    retriever_used: bool = False
    retriever_calls: int = 0
    retrieved_docs: list[dict[str, Any]] | None = None