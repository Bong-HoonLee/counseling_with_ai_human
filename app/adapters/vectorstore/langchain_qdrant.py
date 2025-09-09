from typing import Iterable, List, Protocol, Optional

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from uuid import uuid4

from app.models import PointUpsert, SearchQuery
from app.infrastructure.vectorstore.qdrant import qdrant_client, health_check
from app.models import QdrantConfig, QdrantVSConfig

class LangchainQdrant:
    def __init__(
            self,
            client_config: QdrantConfig,
            vs_config: QdrantVSConfig,
            ):
        self.client_config = client_config
        self.vs_config = vs_config
        self._client: Optional[QdrantClient] = None
        self._vs: Optional[QdrantVectorStore] = None

    def _ensure_client(self) -> QdrantClient:
        if self._client is None:
            if self.client_config.in_memory:
                self._client = QdrantClient(":memory:")
            else:
                self._client = QdrantClient(
                    host=self.client_config.host,
                    port=self.client_config.port,
                    api_key=self.client_config.api_key,
                    grpc=self.client_config.prefer_grpc,
                    timeout=self.client_config.timeout,
                )
        return self._client
    
    def _ensure_vs(self) -> QdrantVectorStore:
        if self._vs is None:
            self._vs = QdrantVectorStore(
                client=self._ensure_client(),
                collection_name=self.vs_config.collection_name,
                embedding=self.vs_config.embedding,
                retrieval_mode=self.vs_config.retrieval_mode,
                sparse_embedding=self.vs_config.sparse_embedding
            )
        return self._vs
    

    def upsert(self, points: Iterable[PointUpsert]) -> None:
        vector_store = self._ensure_vs()

        documents = []
        ids = []

        for point in points:
            id = str(uuid4()) if point.id is None else point.id
            content = point.payload.pop('content', None)

            document = Document(page_content=content, metadata=point.payload)
            documents.append(document)
            ids.append(id)
        
        vector_store.add_documents(documents=documents, ids=ids)

    def search(self, query: SearchQuery) -> list[tuple[Document, float]]:
        '''
          query: str,                                   # 검색할 텍스트 쿼리
            k: int = 4,                                   # 반환할 결과 개수 (Top-k)
            filter: Optional[models.Filter] = None,       # Qdrant payload 조건 필터
            search_params: Optional[models.SearchParams] = None, # HNSW 등 검색 파라미터
            offset: int = 0,                              # 검색 결과 건너뛸 개수 (페이징)
            score_threshold: Optional[float] = None,      # 유사도 점수 임계값 (걸러내기)
            consistency: Optional[models.ReadConsistency] = None, # 분산환경 읽기 일관성
            hybrid_fusion: Optional[models.FusionQuery] = None,   # dense+sparse 융합 설정 (RRF/DBSF 등)
            **kwargs: Any,                                # 추가 확장 인자
        '''
        vector_store = self._ensure_vs()
        return vector_store.similarity_search_with_score(
            query= query.query,
            k= query.top_k,
            **query.options
        )