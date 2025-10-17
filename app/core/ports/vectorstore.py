from typing import Iterable, List, Protocol
from app.core.models import PointUpsert, SearchQuery, VSIndexSchema

class VectorStorePort(Protocol):
    def search(self, query: SearchQuery): ...


class VectorStoreCIndexPort(Protocol):
    def create_index(self, schema: VSIndexSchema) -> None: ...
    def upsert(self, points: Iterable[PointUpsert]) -> None: ...