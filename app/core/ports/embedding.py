from typing import Protocol
from app.models.embedding_dto import EmbeddingRequest, EmbeddingResponse

class EmbeddingPort(Protocol):
    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        ...