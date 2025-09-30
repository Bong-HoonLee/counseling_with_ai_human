from typing import Iterable, List, Protocol


class VectotStoreClient(Protocol):
    def make_qdrant_client():...

    def health_check() -> bool:...