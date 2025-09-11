from typing import Protocol, List
from app.core.models.embedding_dto import EmbeddingRequest, EmbeddingResponse

class EmbeddingPort(Protocol):
    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        ...

class SparseEmbeddingPort(Protocol):
    def embed(self, text: str) -> List[float]: ...