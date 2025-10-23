import os
from typing import Literal, Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant.qdrant import RetrievalMode
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_qdrant.sparse_embeddings import SparseEmbeddings
from qdrant_client import models
from qdrant_client.http.models import HnswConfigDiff, Distance, VectorParams

from app.core.models import VSClientConn

@dataclass
class QdrantSchema:
    collection_name: str = "default"
    vectors_config={
        "dense": VectorParams(
        size=3072,
        distance=Distance.COSINE,
        hnsw_config=HnswConfigDiff(
            m=16,                    # 그래프 degree (기본 16)
            ef_construct=100,        # 인덱스 구축시 탐색 범위
            full_scan_threshold=1000 # 소량 데이터 일때 hnsw보다 faster scan
            )
        )
        }
    sparse_vectors_config = {
        "sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=True))
    }

@dataclass
class QdrantClintConfig(VSClientConn):
    in_memory: bool = False
    prefer_grpc: bool = True

    @classmethod
    def default(cls) -> "QdrantClintConfig":
        return cls(
            host="localhost",
            port=6334,
            timeout=10,
        )


@dataclass
class QdrantVsConfig:
    '''
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        ### With the `text-embedding-3` class
        ### of models, you can specify the size
        ### of the embeddings you want returned.
        ### dimensions=1024
        )
    '''
    collection_name: str = "default"
    embedding: OpenAIEmbeddings = field(default_factory=OpenAIEmbeddings)
    retrieval_mode: Optional[RetrievalMode] = None
    sparse_embedding: Optional[SparseEmbeddings] = None
    vector_name = "dense"
    sparse_vector_name= "sparse"
    @classmethod
    def default(cls) -> "QdrantVsConfig":
        return cls(
            embedding= OpenAIEmbeddings(
                    model="text-embedding-3-large"),
            retrieval_mode= RetrievalMode.HYBRID,
            sparse_embedding=FastEmbedSparse(),
        )


@dataclass
class OpenAIConfig:
    provider: Literal["openai"]
    api_key: str
    model: str
    params: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        return cls(
            provider = "openai",
            api_key=os.environ["OPENAI_API_KEY"],   
            model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o")
        )

@dataclass
class AzureConfig:
    provider: Literal["azure"]
    api_key: str
    model: str
    endpoint: str
    api_version: str
    params: Optional[Dict[str, Any]] = None

    @classmethod
    def from_env(cls) -> "AzureConfig":
        return cls(
            provider = "azure",
            api_key=os.environ["AZURE_API_KEY"],
            model=os.getenv("AZURE_LLM_MODEL", "gpt-4o"),
            endpoint=os.getenv("AZURE_ENDPOINT"),
            api_version=os.getenv("AZURE_API_VERSION")
        )