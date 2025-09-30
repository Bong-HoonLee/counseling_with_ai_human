from typing import Protocol
from app.core.models import EmbeddingRequest, EmbeddingResponse, SparseVectorTypes

class EmbeddingPort(Protocol):
    # def embed_query(self, request: EmbeddingRequest) -> EmbeddingResponse:
    #     ...

    def embed_documents(self, request: EmbeddingRequest) -> EmbeddingResponse:
        ...

class SparseEmbeddingPort:
    '''
    아직 구현 x, 나중에 시그니처로 확장 가능
    '''
    # def encode(self, text: str) -> SparseVectorTypes: ...
    # def batch_encode(self, texts: list[str]) -> list[SparseVectorTypes]: ...
    pass