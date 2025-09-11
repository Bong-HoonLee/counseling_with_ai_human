from typing import Literal, Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field

from app.core.ports import VectorStorePort, EmbeddingPort



from langchain_qdrant.sparse_embeddings import SparseEmbeddings
from langchain_qdrant.qdrant import RetrievalMode

@dataclass
class VSClintConfig:
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None
    prefer_grpc: bool = True
    timeout: float = 10.0
    in_memory: bool = False

@dataclass
class VSConfig:
    client: Optional[VectorStorePort] = None
    collection_name: str
    embedding: Optional[EmbeddingPort] = None
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    sparse_embedding: Optional[SparseEmbeddings] = None

@dataclass
class PointUpsert:
    id: Optional[str] = None
    dense: Dict[str, List[float]]           # {"q_vec": [...], "a_vec": [...]}
    sparse: Optional[Tuple[List[int], List[float]]] = None
    payload: Optional[Dict[str, Any]] = None


@dataclass
class SearchQuery:
    query: Optional[str] = None
    top_k: int = 10
    dense: Dict[str, List[float]] = field(default_factory=dict)  # {"q_vec": [...], "a_vec": [...]}
    sparse: Optional[Tuple[List[int], List[float]]] = None
    weights: Dict[str, float] = field(default_factory=dict)        # {"q_vec": 0.7, "sparse": 0.3}
    options: Dict[str, Any] = field(default_factory=dict)