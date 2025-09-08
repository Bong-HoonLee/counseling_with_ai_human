from typing import Any, Dict, Iterable, List, Optional, Tuple

from qdrant_client.models import PointStruct, NamedSparseVector

from app.core.ports import VectorStorePort, SparseVector
from app.models import SparseVectorTypes


class QdrantRepository:
    def __init__(self, client, collection: str):
        self.client = client
        self.collection = collection

    def upsert(self, ids, dense_vectors=None, sparse_vectors=None, payloads=None, vector_name="q_vec") -> None:
        ids = list(ids)
        dense = list(dense_vectors) if dense_vectors is not None else [None]*len(ids)
        sparse = list(sparse_vectors) if sparse_vectors is not None else [None]*len(ids)
        payloads = list(payloads) if payloads is not None else [None]*len(ids)

        points = []
        for i, pid in enumerate(ids):
            vectors = {}
            if dense[i] is not None:
                vectors[vector_name] = dense[i]
            if sparse[i] is not None:
                ind, val = sparse[i]
                vectors["sparse"] = NamedSparseVector(indices=ind, values=val)
            points.append(PointStruct(id=pid, vector=vectors, payload=payloads[i] or {}))

        # infra raw 호출 사용
        from app.infrastructure.vectorstore.qdrant.ops import upsert_points_raw
        upsert_points_raw(self.client, self.collection, points)

    def search(
        self,
        query_dense: Optional[List[float]] = None,
        query_sparse: Optional[SparseVectorTypes] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        vector_name: str = "q_vec",
    ) -> List[Dict[str, Any]]:
        # (정책) 하이브리드: dense + sparse를 Qdrant “query”로 결합하거나, rerank 방식 선택
        if query_dense is not None and query_sparse is not None:
            # 예: query 구조체로 결합(실제 SDK의 hybrid/query API에 맞춰 구현)
            # 여기서는 간단히 dense 우선 검색 후 스파스 재랭크 등 정책 구현 지점
            pass

        from app.infrastructure.vectorstore.qdrant.ops import search_raw
        res = search_raw(self.client, self.collection, vector_name, query_vector=query_dense, query_filter=filters, top=top_k)
        return [hit for hit in res]
