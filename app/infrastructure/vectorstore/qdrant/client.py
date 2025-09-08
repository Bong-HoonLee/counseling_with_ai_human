from qdrant_client import QdrantClient

def make_qdrant_client(
    host: str, port: int, api_key: str | None = None, prefer_grpc: bool = True, timeout: float = 10.0
) -> QdrantClient:
    return QdrantClient(host=host, port=port, api_key=api_key, grpc=prefer_grpc, timeout=timeout)

def health_check(client: QdrantClient) -> bool:
    try:
        client.get_collections()
        return True
    except Exception:
        return False
