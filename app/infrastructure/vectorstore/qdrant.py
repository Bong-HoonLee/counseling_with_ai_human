from __future__ import annotations

import hashlib, uuid
from typing import Literal, Optional, Iterable, Callable, List, Dict, Any, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, 
    VectorParams, 
    HnswConfigDiff,
    SparseVectorParams, 
    SparseIndexParams,
    PointStruct,
    SparseVector
)
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.qdrant import RetrievalMode
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
import yaml
from dotenv import load_dotenv
import pandas as pd

from app.config import yaml_cfg
from app.models import DualEmbeddingConfig

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
        self._embeddings: Dict[str, Any] = {}
        self._vs_cache: Dict[Tuple[str, str], QdrantVectorStore] = {} # key = (vector_name, retrieval_mode)

        self._emb_cfg = emb_cfg or DualEmbeddingConfig(provider="openai")

    def get_embedding(self, key: str = "A"):
        if key not in self._embeddings:
            if key == "A":
                kwargs = self._emb_cfg.a_kwargs
            elif key == "B":
                kwargs = self._emb_cfg.b_kwargs or self._emb_cfg.a_kwargs
            else:
                raise ValueError(f"Unknown embedding key: {key}")

            if self._emb_cfg.provider == "azure":
                if not kwargs:
                    raise ValueError(f"Azure {key}-embedding kwargs가 필요합니다.")
                self._embeddings[key] = AzureOpenAIEmbeddings(**kwargs)
            else:
                if not kwargs:
                    raise ValueError(f"OpenAI {key}-embedding kwargs가 필요합니다.")
                self._embeddings[key] = OpenAIEmbeddings(**kwargs)

        return self._embeddings[key]

    @property
    def embA(self):
        return self.get_embedding("A")

    @property
    def embB(self):
        return self.get_embedding("B")
    
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
        lazy sigletone,
        init의 _vs_cache에 key값으로 저장을 합니다.
        """

        if vector_name == "q_vec" and not use_A:
            raise ValueError("q_vec에는 embA를 사용해야 합니다(use_A=True).")
        if vector_name == "a_vec" and use_A:
            raise ValueError("a_vec에는 embB를 사용해야 합니다(use_A=False).")
        
        key = (vector_name, mode)

        if mode.upper() == "HYBRID" and sparse_embedding is None:
            sparse_embedding = True

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
    
    def _normalize_text(
        self,
        s: Optional[str]
    ) -> str:
        if s is None:
            return ""
        return " ".join(str(s).strip().split())
    
    def _split_category_path(
        self,
        path: str
    ) -> Tuple[str, str, str]:
        parts = (path or "").split("/")
        parts = [p.strip() for p in parts if p is not None]
        lvl1 = parts[0] if len(parts) > 0 else ""
        lvl2 = parts[1] if len(parts) > 1 else ""
        lvl3 = parts[2] if len(parts) > 2 else ""
        return lvl1, lvl2, lvl3
    
    def _sha1(self, s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()
    
    def _content_hash(
        self,
        q: str,
        a: str
    ) -> str:
        return self._sha1(q + "|||A|||" + a)
    
    def _make_point_id(
            self,
            pair_id: str,
            version: int
        ) -> str:
    # Qdrant에 저장할 최종 키(결정적)
        return self._sha1(f"{pair_id}:{version}")

    def build_enriched_df(
        self,
        df_raw: pd.DataFrame,
        *,
        source_label: str = "severance",
        lang: str = "ko",
        pair_id_prefix: str = "severance",
        start_version: int = 1,
    ) -> pd.DataFrame:
        """
        입력: df_raw[ q_test, a_test, 구분 ]
        출력: df_enriched[ q_text, a_text, category_path, cat_lvl1/2/3, domain,
                        lang, source, status, error, retry_count,
                        pair_id, version, content_hash, point_id ]
        """
        df = df_raw.copy()

        # 1) 텍스트 정리
        df["q_text"] = df["유저"].map(self._normalize_text)
        df["a_text"] = df["챗봇"].map(self._normalize_text)

        # 둘 중에 하나라도 비었으면 제거
        df = df[~((df["q_text"] == "") | (df["a_text"] == ""))].reset_index(drop=True)

        # 2) 카테고리 분해해서 레벨별로 나누기
        df["category_path"] = df["구분"].fillna("").astype(str).str.strip()
        split_cols = df["category_path"].apply(self._split_category_path)
        df["cat_lvl1"] = split_cols.apply(lambda x: x[0])
        df["cat_lvl2"] = split_cols.apply(lambda x: x[1])
        df["cat_lvl3"] = split_cols.apply(lambda x: x[2])

        # 3) 메타
        df["domain"] = df["cat_lvl1"]       # 최상위 레벨을 도메인으로
        df["lang"] = lang
        df["source"] = source_label

        # 4) 운영 컬럼 초기화
        df["status"] = "NEW"
        df["error"] = ""
        df["retry_count"] = 0

        # 5) ID / 버전 / 해시 / point_id
        # pair_id: 초기는 행번호 기반으로 간단히(원하면 외부키로 대체 가능)
        df["pair_id"] = [
            f"{pair_id_prefix}-{i:08d}" for i in range(len(df))
        ]
        df["version"] = int(start_version)

        df["content_hash"] = [
            self._content_hash(q, a) for q, a in zip(df["q_text"], df["a_text"])
        ]
        df["point_id"] = [
            self._make_point_id(pid, ver) for pid, ver in zip(df["pair_id"], df["version"])
        ]

        # 품질 지표
        df["q_len"] = df["q_text"].str.len()
        df["a_len"] = df["a_text"].str.len()

        return df
    
    def _normalize_sparse(
        self, 
        q_sparse: Optional[dict]
    ) -> Optional[SparseVector]:
        """
        허용 입력:
          1) {"indices":[int,...], "values":[float,...]}  # Qdrant 표준
          2) {"token": weight, ...}  # 토큰->가중치 맵 (정렬 후 인덱스로 변환 필요 시 사용)
        반환: qdrant_client.http.models.SparseVector | None
        """
        if not q_sparse:
            return None

        # 표준 형태
        if "indices" in q_sparse and "values" in q_sparse:
            return SparseVector(indices=q_sparse["indices"], values=q_sparse["values"])

        # 토큰→가중치 맵(dict)을 받았다면, 정렬하여 인덱스/값 배열로 변환
        # (여기서는 간단히 토큰을 해시/사전순 인덱스로 매핑하지 않고,
        #  사전순 정렬 후 0..N-1의 가짜 인덱스 부여. 실제 운영에서는
        #  SPLADE 등에서 나온 정수 인덱스를 사용하세요.)
        if isinstance(q_sparse, dict):
            items = sorted(q_sparse.items(), key=lambda x: x[0])
            indices = list(range(len(items)))
            values = [float(v) for _, v in items]
            return SparseVector(indices=indices, values=values)

        raise ValueError("지원되지 않는 sparse 벡터 형식입니다.")
    
    def _ensure_dense_vec(
        self, 
        text: Optional[str],
        vec: Optional[List[float]],
        use_A: bool
    ) -> List[float]:
        """
        vec이 없으면 text로 dense 벡터를 생성합니다
        use_A=True => embA(embed_query), False => embB(embed_query)
        """
        if vec is not None:
            return vec
        if text is None:
            raise ValueError("dense 벡터를 만들 수 없습니다. text 또는 vec 중 하나는 제공되어야 합니다.")
        emb = self.embA if use_A else self.embB
        return emb.embed_query(text)
    
    def upsert_point(
        self,
        point_id: str | int,
        text_q: Optional[str] = None,
        text_a: Optional[str] = None,
        q_vec: Optional[List[float]] = None,
        a_vec: Optional[List[float]] = None,
        q_sparse: Optional[dict] = None,
        payload: Optional[Dict[str, Any]] = None,
        wait: bool = True,
    ):
        """
        하나의 포인트에 대해 named vectors(q_vec, a_vec)와 q_sparse를 업서트합니다.
        - text_q만 주면 embA로 q_vec 생성
        - text_a만 주면 embB로 a_vec 생성
        - q_sparse는 스파스 사용 시에만 전달 (self.use_sparse=True)
        """
        # dense vec 확보
        qv = self._ensure_dense_vec(text_q, q_vec, use_A=True) if (text_q is not None or q_vec is not None) else None
        av = self._ensure_dense_vec(text_a, a_vec, use_A=False) if (text_a is not None or a_vec is not None) else None

        if qv is None and av is None:
            raise ValueError("최소 하나의 dense 벡터(q_vec 또는 a_vec)가 필요합니다.")

        dense_vectors: Dict[str, List[float]] = {}
        if qv is not None:
            dense_vectors["q_vec"] = qv
        if av is not None:
            dense_vectors["a_vec"] = av

        sparse_vectors = None
        if self.use_sparse and q_sparse is not None:
            sparse_vectors = {"q_sparse": self._normalize_sparse(q_sparse)}

        point = PointStruct(
            id=point_id,
            vector=dense_vectors,                # dense vectors : dict
            payload=payload or {},
            sparse_vectors=sparse_vectors  # {"q_sparse": SparseVector(...)} | None
        )

        # 업서트
        return self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
            wait=wait
        )
    
    def upsert_batch(
        self,
        items: List[Dict[str, Any]],
        wait: bool = True,
        batch_size: int = 256,
    ):
        """
        items:
        [{
          "id": "doc-1",
          "text_q": "질문 텍스트",        
          "text_a": "응답 텍스트",        
          "q_vec": [...],                 
          "a_vec": [...],               
          "q_sparse": {"indices":[...], "values":[...] }  # 선택(self.use_sparse=True일 때)
          "payload": {"title":"...", "lang":"ko", ...}    # 선택
        },
        ...
        ]
        """
        buf: List[PointStruct] = []

        def _flush():
            if not buf:
                return None
            res = self.client.upsert(
                collection_name=self.collection_name,
                points=buf,
                wait=wait
            )
            buf.clear()
            return res

        last_res = None
        for it in items:
            pid = it.get("id")
            if pid is None:
                raise ValueError("각 item에는 고유 'id'가 필요합니다.")

            # dense
            text_q = it.get("text_q")
            text_a = it.get("text_a")
            q_vec = it.get("q_vec")
            a_vec = it.get("a_vec")

            qv = self._ensure_dense_vec(text_q, q_vec, use_A=True) if (text_q is not None or q_vec is not None) else None
            av = self._ensure_dense_vec(text_a, a_vec, use_A=False) if (text_a is not None or a_vec is not None) else None

            if qv is None and av is None:
                raise ValueError(f"[{pid}] 최소 하나의 dense 벡터(q_vec 또는 a_vec)가 필요합니다.")

            vectors: Dict[str, List[float]] = {}
            if qv is not None: vectors["q_vec"] = qv
            if av is not None: vectors["a_vec"] = av

            sparse_vectors = None
            if self.use_sparse and ("q_sparse" in it) and it["q_sparse"] is not None:
                sparse_vectors = {"q_sparse": self._normalize_sparse(it["q_sparse"])}

            p = PointStruct(
                id=pid,
                vector=vectors,
                payload=it.get("payload") or {},
                sparse_vectors=sparse_vectors
            )
            buf.append(p)

            if len(buf) >= batch_size:
                last_res = _flush()

        # 남은 것 처리
        tail = _flush()
        return tail or last_res
    
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
        
        