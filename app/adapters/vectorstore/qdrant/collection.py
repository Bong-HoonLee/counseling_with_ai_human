from typing import Iterable, List, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from .client import MakeQdrantClient

from app.core.models import PointUpsert
from app.config.config import QdrantClintConfig, QdrantVsConfig, QdrantSchema

class Qdrant:
    def __init__(
            self,
            client_cfg: QdrantClintConfig,
            vs_cfg: QdrantVsConfig,
            schema: QdrantSchema = None
            ):
        self.client_cfg = client_cfg
        self.vs_cfg = vs_cfg
        self.schema = schema
        self._client: Optional[QdrantClient] = None
        self._vs: Optional[QdrantVectorStore] = None

    def _ensure_client(self) -> QdrantClient:
        if self._client is None:
            self._client = MakeQdrantClient.make_qdrant_client(self.client_cfg)
        return self._client
    
    def _ensure_vs(self) -> QdrantVectorStore:
        """Initialize a new instance of `QdrantVectorStore`.

        Example:
            .. code-block:: python
            qdrant = Qdrant(
                client=client,
                collection_name="my-collection",
                embedding=OpenAIEmbeddings(),
                retrieval_mode=RetrievalMode.HYBRID,
                sparse_embedding=FastEmbedSparse(),
            )

        """
        if self._vs is None:
            self._vs = QdrantVectorStore(
                client=self._ensure_client(),
                collection_name=self.vs_cfg.collection_name,
                embedding=self.vs_cfg.embedding,
                retrieval_mode=self.vs_cfg.retrieval_mode,
                sparse_embedding=self.vs_cfg.sparse_embedding,
                vector_name=self.vs_cfg.vector_name,
                sparse_vector_name= self.vs_cfg.sparse_vector_name,
            )
        return self._vs
    

    def upsert(self, points: Iterable[PointUpsert]) -> None:
        '''
        시그니처 등록하지 않음
        '''
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
    
    def create_index(self) -> None:
        '''
        qdrant collection 생성 메서드
        시그니처 등록하지 않음
        '''
        client = self._ensure_client()
        schema = self.schema
        collection_name = schema.collection_name
        vectors_config = schema.vectors_config
        sparse_vectors_config = schema.sparse_vectors_config

        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config = sparse_vectors_config
        )

    