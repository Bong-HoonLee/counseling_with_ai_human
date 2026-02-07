from typing import Iterable, List, Protocol
from app.domain.models import PointUpsert, SearchQuery, VSClientConn

class VectorStorePort(Protocol):
    def search(self, query: SearchQuery): ...


class VectorStoreCIndexPort(Protocol):
    def create_index(self, schema: VSClientConn) -> None: ...
    def upsert(self, points: Iterable[PointUpsert]) -> None: ...