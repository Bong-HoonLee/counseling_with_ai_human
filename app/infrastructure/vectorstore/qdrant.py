from typing import Literal, Optional, List, Dict, Any, Tuple
from __future__ import annotations
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, HnswConfigDiff,
    SparseVectorParams, SparseIndexParams,
)
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
import yaml
from dotenv import load_dotenv

from app.config import yaml_cfg

Provider = Literal["openai", "azure"]

@dataclass
class DualEmbeddingConfig:
    """
    A/B 임베딩 분리
    """
    provider: Provider = "openai"
    a_kwargs: Optional[dict] = None   
    b_kwargs: Optional[dict] = None   


load_dotenv()

class QdrantCustom():
    def __init__(
            self,
            host: str = 'localhost',
            port: int = 6333,
            collection_name: str = 'aihuman',
            emb_cfg: Optional[DualEmbeddingConfig] = None,
            use_sparse: bool = False,
        )-> None:
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.use_sparse = use_sparse

        # lazy singleton
        self._client: Optional[QdrantClient] = None
        self._embA = None
        self._embB = None
        self._vs_q: Optional[QdrantVectorStore] = None
        self._vs_a: Optional[QdrantVectorStore] = None
        self._vs_cache = {} # key = (vector_name, retrieval_mode)

        self._emb_cfg = emb_cfg or DualEmbeddingConfig(provider="openai")

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(host=self.host, port=self.port)
        return self._client

    @property
    def embA(self):
        if self._embA is None:
            if self._emb_cfg.provider == "azure":
                if not self._emb_cfg.a_kwargs:
                    raise ValueError("Azure A-embedding kwargs가 필요합니다.")
                self._embA = AzureOpenAIEmbeddings(**self._emb_cfg.a_kwargs)
            else:
                if not self._emb_cfg.a_kwargs:
                    raise ValueError("OpenAI A-embedding kwargs가 필요합니다.")
                self._embA = OpenAIEmbeddings(**self._emb_cfg.a_kwargs)
        return self._embA

    @property
    def embB(self):
        if self._embB is None:
            kwargs = self._emb_cfg.b_kwargs or self._emb_cfg.a_kwargs
            if self._emb_cfg.provider == "azure":
                if not kwargs:
                    raise ValueError("Azure B-embedding kwargs가 필요합니다.")
                self._embB = AzureOpenAIEmbeddings(**kwargs)
            else:
                if not kwargs:
                    raise ValueError("OpenAI B-embedding kwargs가 필요합니다.")
                self._embB = OpenAIEmbeddings(**kwargs)
        return self._embB
    
    def create_if_absent(
        self,
        vector_size: Optional[int] = None,
        m: int = 16,
        ef_construct: int = 100,
        distance: Distance = Distance.COSINE,
        full_scan_threshold: Optional[int] = None,
        sparse_on_disk: bool = False,
    ) -> None:
        """
        컬렉션이 없을 때만 생성 
        recreate 하나만 사용하는 위험 부담 때문에 만든 함수
        """
        if self.client.collection_exists(self.collection_name):
            return

        vs = vector_size
        q_params = HnswConfigDiff(m=m, ef_construct=ef_construct)
        a_params = HnswConfigDiff(m=m, ef_construct=ef_construct)
        if full_scan_threshold is not None:
            q_params.full_scan_threshold = full_scan_threshold
            a_params.full_scan_threshold = full_scan_threshold

        vectors_cfg = {
            "q_vec": VectorParams(size=vs, distance=distance, hnsw_config=q_params),
            "a_vec": VectorParams(size=vs, distance=distance, hnsw_config=a_params),
        }
        sparse_cfg = {"q_sparse": SparseVectorParams(index=SparseIndexParams(on_disk=sparse_on_disk))} if self.use_sparse else None

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_cfg,
            sparse_vectors_config=sparse_cfg,
        )

    def recreate(
        self,
        vector_size: Optional[int] = None,
        m: int = 16,
        ef_construct: int = 100,
        distance: Distance = Distance.COSINE,
        full_scan_threshold: Optional[int] = None,
        sparse_on_disk: bool = False,
    ) -> None:
        """ 
        컬렉션을 드롭 후 재생성 (데이터 삭제)
        """
        vs = vector_size
        q_params = HnswConfigDiff(m=m, ef_construct=ef_construct)
        a_params = HnswConfigDiff(m=m, ef_construct=ef_construct)
        if full_scan_threshold is not None:
            q_params.full_scan_threshold = full_scan_threshold
            a_params.full_scan_threshold = full_scan_threshold

        vectors_cfg = {
            "q_vec": VectorParams(size=vs, distance=distance, hnsw_config=q_params),
            "a_vec": VectorParams(size=vs, distance=distance, hnsw_config=a_params),
        }
        sparse_cfg = {"q_sparse": SparseVectorParams(index=SparseIndexParams(on_disk=sparse_on_disk))} if self.use_sparse else None

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_cfg,
            sparse_vectors_config=sparse_cfg,
        )
        
    def create_payload_index(self, field_name, schema):
        return self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name=field_name,
            field_schema=schema
        )
    
    def get_vs(
            self, 
            vector_name: str, 
            use_A: bool = True,
            mode: str = "DENSE",
            sparse_embedding=None
        ) -> QdrantVectorStore:
        """
        lazy sigletone이며,
        init의 _vs_cache에 key값으로 저장을 합니다.
        """
        key = (vector_name, mode)
        if key not in self._vs_cache:
            emb = self.embA if use_A else self.embB
            rm = RetrievalMode[mode.upper()]  # "DENSE"|"HYBRID"
            self._vs_cache[key] = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=emb,
                vector_name=vector_name,
                retrieval_mode=rm,
                sparse_embedding=sparse_embedding,  # HYBRID일 때만 작동
            )
        return self._vs_cache[key]
    


    # def embedding_text(self, 
    #                    texts: list,
    #                    metadatas: list
    #                 ):
    #     '''
    #     texts=[
    #     "Azure Blob Storage 인증 문제 해결 방법",
    #     "Python S3 파일 업로드 방법",
    #     "FastAPI로 REST API 서버 만들기"
    #     ],
    #     metadatas=[
    #         {"source": "azure_docs"},
    #         {"source": "aws_docs"},
    #         {"source": "fastapi_docs"}
    #     ]
    #     '''

    #     self.vector_store.add_texts(
    #         texts = texts,
    #         metadatas = metadatas
    #     )
    
    # def similarity_search(self, 
    #                       search_cfg
    #                       ):
    #     '''
    #      cfg : "API 서버 만들기",
    #             k=2,
    #             filter={"source": "fastapi_docs"}
    #     '''
    #     docs = self.vector_store.similarity_search(search_cfg)

    #     return docs
    
    # def similarity_search_with_score(self, 
    #                       search_cfg
    #                       ):
    #     '''
    #      cfg : "API 서버 만들기",
    #             k=2,
    #             filter={"source": "fastapi_docs"},
    #             score_threshold=0.4
        
    #     return : [(Document(metadata={'source': 'azure_docs', '_id': '00be5158-4a9e-4cc0-b9c8-5d7190a2691c', 
    #                 '_collection_name': 'chatbot'}, page_content='Azure Blob Storage 인증 문제 해결 방법'),
    #                 0.83140427)]
    #     '''
    #     docs = self.vector_store.similarity_search_with_score(search_cfg)
        
    #     return docs
        
        