from typing import Iterable, List, Protocol

from app.core.models import ChatRES, SearchQuery

class ChatmodelPort(Protocol):
    def response(self, query: SearchQuery) -> ChatRES: ...