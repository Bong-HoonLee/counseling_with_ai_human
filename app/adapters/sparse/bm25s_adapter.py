from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass

from bm25s import BM25

from app.core.ports.tokenizer import Tokenizer
from app.models.sparse_dto import SparseVectorTypes

@dataclass(frozen=True)
class BM25Params:
    k1: float = 1.5
    b: float = 0.75


class BM25ADAPTER:
    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        vocab: Optional[Dict[str, int]] = None,
        params: Optional[BM25Params] = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._params = params or BM25Params()
        self._bm25 = None
        self._vocab: Optional[Dict[str, int]] = vocab  # 토큰 -> 인덱스
        self._idf: Optional[Dict[str, float]] = None   # 토큰 -> idf
        self._avgdl: Optional[float] = None

    # def encode(self, text: str) -> SparseVectorTypes:
    #     toks = self._tokenizer(text)
    #     indices, values = self._terms_to_bm25_vector(toks)
    #     return SparseVectorTypes(indices=indices, values=values)

    # def batch_encode(self, texts: List[str]) -> List[SparseVectorTypes]:
    #     return [self.encode(t) for t in texts]