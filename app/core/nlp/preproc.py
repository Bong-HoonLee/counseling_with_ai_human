# app/core/text/preproc.py
from __future__ import annotations
from typing import Iterable, List, Optional, Set
import re

try:
    from konlpy.tag import Okt
except Exception:
    Okt = None

# 불용어
from app.core.nlp.pos import resolve_allowed_pos
from app.core.nlp.stopwords import load_stopwords


class KoreanPreproc:
    def __init__(
        self,
        stopwords: Optional[Iterable[str]] = None,
        okt_stem: bool = True,
        okt_norm: bool = True,
        lower: bool = True,
        allowed_pos: Optional[Set[str]] = None,
        allowed_pos_categories: Optional[Set[str]] = None,
        pos_backend: str = "okt",   # "okt" | "mecab" 
        min_token_len: int = 1,
        use_okt: bool = True,
    ):
        # 기본 STOPWORDS_KO 사용, 필요하면 추가 리스트로 덮어쓰기 가능
        self.stop = set(stopwords or load_stopwords())
        self.okt_stem = okt_stem
        self.okt_norm = okt_norm
        self.lower = lower
        self.min_token_len = min_token_len
        self.allowed_pos = set(allowed_pos) if allowed_pos else resolve_allowed_pos(
            backend=pos_backend,
            categories=set(allowed_pos_categories) if allowed_pos_categories else None,
        )

        self.ws = re.compile(r"\s+")
        self.okt = None
        if use_okt:
            if Okt is None:
                raise RuntimeError("konlpy의 Okt를 사용할 수 없습니다.")
            self.okt = Okt()

    def normalize(self, text: Optional[str]) -> str:
        if not text:
            return ""
        s = text.strip()
        s = self.ws.sub(" ", s)
        if self.lower:
            s = s.lower()
        return s

    def tokenize(self, text: Optional[str]) -> List[str]:
        s = self.normalize(text)
        if not s:
            return []

        if self.okt is None:
            toks = s.split(" ")
        else:
            toks = []
            for token, pos in self.okt.pos(s, stem=self.okt_stem, norm=self.okt_norm):
                if self.allowed_pos is None or pos in self.allowed_pos:
                    toks.append(token)

        # stopwords + 길이 필터
        toks = [t for t in toks if (len(t) >= self.min_token_len) and (t not in self.stop)]
        return toks
