from __future__ import annotations
from typing import List, Tuple

try:
    from konlpy.tag import Okt
except Exception:
    Okt = None

class OktTokenizer():
    """
    OKT 토크나이저.
    - 표제어화(stem), 정규화(norm), POS 필터링, 불용어 제거
    - 어댑터 형식으로 관련 코드들과 독립적 설계
    """
    name = "okt"

    def __init__(
        self,
        *,
        stem: bool = True,
        norm: bool = True,
    ):
        if Okt is None:
            raise RuntimeError("Okt를 사용할 수 없습니다.")
        self.okt = Okt()
        self.stem = stem
        self.norm = norm

    def tokenize(self, text: str) -> List[Tuple[str, str]]:
        if not text:
            return []
        
        return self.okt.pos(text, stem=self.stem, norm=self.norm)
