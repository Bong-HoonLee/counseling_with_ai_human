from dataclasses import dataclass
from typing import Optional

from qdrant_client import QdrantClient
from langchain_qdrant.sparse_embeddings import SparseEmbeddings
from langchain_qdrant.qdrant import RetrievalMode
from langchain_core.embeddings import Embeddings

@dataclass
class QdrantConfig:
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None
    prefer_grpc: bool = True
    timeout: float = 10.0
    in_memory: bool = False

@dataclass
class QdrantVSConfig:
    client: Optional[QdrantClient] = None
    collection_name: str
    embedding: Optional[Embeddings] = None
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    sparse_embedding: Optional[SparseEmbeddings] = None