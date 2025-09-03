from __future__ import annotations
from typing import Iterable, List, Optional, Set, Tuple
import re

try:
    from konlpy.tag import Okt
except Exception:
    Okt = None

from app.core.ports.tokenizer import Tokenizer
from app.core.nlp.pos import resolve_allowed_pos
from app.core.nlp.stopwords import load_stopwords


class KoreanPreproc:
    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        stopwords: Optional[Iterable[str]] = None,
        lower: bool = True,
        min_token_len: int = 1,
        allowed_pos: Optional[Set[str]] = None,
        allowed_pos_categories: Optional[Set[str]] = None,
        backend: Optional[str] = None,   # "okt" | "mecab" 
    ):
        self.tokenizer = tokenizer
        self.stop = set(stopwords or load_stopwords())
        self.lower = lower
        self.min_token_len = min_token_len

        # pos settins
        self.allowed_pos = set(allowed_pos) if allowed_pos else resolve_allowed_pos(
            backend=backend,
            categories=set(allowed_pos_categories) if allowed_pos_categories else None,
        )

        self.ws = re.compile(r"\s+")

    def _normalize(self, text: Optional[str]) -> str:
        if not text:
            return ""
        s = text.strip()
        s = self.ws.sub(" ", s)
        if self.lower:
            s = s.lower()
        return s

    def tokenize(self, text: Optional[str]) -> List[str]:
        s = self._normalize(text)
        if not s:
            return []

        pairs: List[Tuple[str, str]] = self.tokenizer.tokenize(s)

    
        if self.allowed_pos is not None:
            toks = [tok for tok, tag in pairs if tag in self.allowed_pos]
        else:
            toks = [tok for tok, _ in pairs]

        # stopwords + 길이 필터
        toks = [t for t in toks if (len(t) >= self.min_token_len) and (t not in self.stop)]
        return toks
