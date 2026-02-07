from typing import Iterable, List, Protocol

from app.domain.models import ChatRES, SearchQuery

class ChatbotAgentPort(Protocol):
    def generate(self, query: SearchQuery) -> ChatRES: ...