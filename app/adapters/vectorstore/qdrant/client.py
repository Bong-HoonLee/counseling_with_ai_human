from qdrant_client import QdrantClient

from config.config import QdrantVSClint

class MakeQdrantClient:
    def make_qdrant_client(
        config: QdrantVSClint
    ) -> QdrantClient:
        return QdrantClient(":memory:") if config.in_memory else QdrantClient(host=config.host, port=config.port, api_key=config.api_key, grpc=config.prefer_grpc, timeout=config.timeout)

    def health_check(client: QdrantClient) -> bool:
        try:
            client.get_collections()
            return True
        except Exception:
            return False