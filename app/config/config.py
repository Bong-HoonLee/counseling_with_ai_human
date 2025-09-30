from typing import Literal, Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant.qdrant import RetrievalMode
from langchain_qdrant.sparse_embeddings import SparseEmbeddings

@dataclass
class QdrantVSClint:
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None
    prefer_grpc: bool = True
    timeout: float = 10.0
    in_memory: bool = False
    collection_name: str
    embedding: OpenAIEmbeddings = field(default_factory=OpenAIEmbeddings)
    retrieval_mode: Optional[RetrievalMode] = None
    sparse_embedding: Optional[SparseEmbeddings] = None