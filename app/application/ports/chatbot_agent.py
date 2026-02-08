from typing import Iterable, List, Protocol

from app.domain.models import ChatRES, AgentSearchQuery

class ChatbotAgentPort(Protocol):
    def generate(self, query: AgentSearchQuery) -> ChatRES: ...