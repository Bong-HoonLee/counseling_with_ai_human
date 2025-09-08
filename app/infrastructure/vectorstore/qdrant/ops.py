from qdrant_client import QdrantClient
from qdrant_client.models import CollectionInfo, CreateCollection, NamedVector, PointStruct

def ensure_collection(client: QdrantClient, name: str, vectors, sparse, optim) -> None:
    try:
        info: CollectionInfo = client.get_collection(name)
        # (필요 시 파라미터 비교 후 업데이트 로직)
    except Exception:
        client.create_collection(
            collection_name=name,
            vectors_config=vectors,
            sparse_vectors_config=sparse,
            optimizers_config=optim,
        )

def upsert_points_raw(client: QdrantClient, name: str, points: list[PointStruct]) -> None:
    client.upsert(collection_name=name, points=points)

def search_raw(client: QdrantClient, name: str, vector_name: str, query_vector=None, query_filter=None, top=10):
    return client.search(collection_name=name, query_vector=NamedVector(name=vector_name, vector=query_vector),
                         query_filter=query_filter, limit=top)
