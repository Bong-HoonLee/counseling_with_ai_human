from typing import Optional
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseIndexParams, OptimizersConfigDiff, HnswConfigDiff
)

def build_collection_config(
    dim_q: int,
    dim_a: Optional[int] = None,
    distance: Distance = Distance.COSINE,
    m: int = 16,
    ef_construct: int = 100,
    full_scan_threshold: Optional[int] = None,
    use_sparse: bool = True,
    sparse_on_disk: bool = False,
):
    # HNSW 세부 파라미터
    q_params = HnswConfigDiff(m=m, ef_construct=ef_construct)
    a_params = HnswConfigDiff(m=m, ef_construct=ef_construct)

    if full_scan_threshold is not None:
        q_params.full_scan_threshold = full_scan_threshold
        a_params.full_scan_threshold = full_scan_threshold

    # 벡터 설정
    vectors_cfg = {
        "q_vec": VectorParams(size=dim_q, distance=distance, hnsw_config=q_params),
    }
    if dim_a:
        vectors_cfg["a_vec"] = VectorParams(size=dim_a, distance=distance, hnsw_config=a_params)

    # 희소 벡터 설정
    sparse_cfg = None
    if use_sparse:
        sparse_cfg = {
            "q_sparse": SparseVectorParams(index=SparseIndexParams(on_disk=sparse_on_disk))
        }

    optim = OptimizersConfigDiff(default_segment_number=2)

    return vectors_cfg, sparse_cfg, optim
