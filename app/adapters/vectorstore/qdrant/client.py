from qdrant_client import QdrantClient

from app.config import QdrantClintConfig

class MakeQdrantClient:
    def make_qdrant_client(
        config: QdrantClintConfig
    ) -> QdrantClient:
        return QdrantClient(":memory:") if config.in_memory else QdrantClient(host=config.host, port=config.port, api_key=config.api_key, prefer_grpc=config.prefer_grpc, timeout=config.timeout)

    def health_check(client: QdrantClient) -> bool:
        try:
            client.get_collections()
            return True
        except Exception:
            return False