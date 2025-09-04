from __future__ import annotations

import os, json, time, hashlib, re
from dataclasses import dataclass, asdict
from typing import Dict, List, Iterable, Optional, Tuple
from collections import Counter

from app.core.ports.tokenizer import Tokenizer
from app.core.utils import HashStrategy, point_id, content_hash

# 스냅샷 메타 정보
@dataclass(frozen=True)
class VocabMeta:
    version: str
    tokenizer: str                # "Okt" 등
    stem: bool                     # 표제어화 사용 여부(옵션)
    drop_pos: List[str]            # 제거 POS 목록(옵션)
    stopwords_hash: str            # 불용어 집합 해시(옵션)
    min_df: int
    max_vocab: int
    num_docs: int                  # N
    avgdl: float                   # 평균 토큰 길이
    created_at: str                # ISO UTC


class VocabManager:
    """
    전역 vocab/통계 스냅샷을 관리하고, 텍스트/토큰을 ID 시퀀스로 변환하는 클래스.
    - vocab: token -> id
    - df:    token -> document frequency (선택적으로 필터링된 항목만 저장)
    - N:     문서 수
    - avgdl: 평균 문서 길이(토큰 수)
    """
    def __init__(
        self,
        vocab: Dict[str, int],
        df: Dict[str, int],
        N: int,
        avgdl: float,
        meta: Optional[VocabMeta] = None,
    ) -> None:
        if N <= 0:
            raise ValueError("N (num_docs)는 0보다 커야 합니다.")
        self.vocab = vocab
        self.df = df
        self.N = int(N)
        self.avgdl = float(avgdl)
        self.meta = meta

    
    @classmethod
    def build_from_texts(
        cls,
        texts: Iterable[str],
        tokenizer: Tokenizer,
        hasher: HashStrategy,
        *,
        version: str = "v1",
        tokenizer_name: str = "Okt",
        stem: bool = True,                 # 메타 기록용(실제 정책은 tokenizer가 가짐)
        drop_pos: Iterable[str] = (),
        stopwords: Iterable[str] = (),
        min_df: int = 2,
        max_vocab: int = 200_000,
    ) -> "VocabManager":
        """
        VocabManager 인스턴스 생성 메서드
        """
        df_counter = Counter()
        lengths = []
        num_docs = 0

        for text in texts:
            num_docs += 1
            tokens = tokenizer.tokenize(text or "")
            lengths.append(len(tokens))
            if tokens:
                df_counter.update(set(tokens))  # DF: 문서 내 중복 제외

        N = num_docs
        avgdl = (sum(lengths) / N) if N > 0 else 0.0

        # DF 필터링 + 빈도순 정렬
        items: List[Tuple[str, int]] = [(tok, df) for tok, df in df_counter.items() if df >= min_df]
        items.sort(key=lambda x: x[1], reverse=True)
        if max_vocab is not None:
            items = items[:max_vocab]

        vocab: Dict[str, int] = {tok: i for i, (tok, _) in enumerate(items)}
        df_out: Dict[str, int] = {tok: df for tok, df in items}

        stopwords_hash = hasher.hexdigest("\n".join(sorted(set(stopwords))))

        meta = VocabMeta(
            version=version,
            tokenizer=tokenizer_name,
            stem=bool(stem),
            drop_pos=list(drop_pos),
            stopwords_hash=stopwords_hash,
            min_df=min_df,
            max_vocab=max_vocab if max_vocab is not None else -1,
            num_docs=N,
            avgdl=avgdl,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        return cls(vocab=vocab, df=df_out, N=N, avgdl=avgdl, meta=meta)

    # 저장/불러오기
    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)
        with open(os.path.join(out_dir, "df_counter.json"), "w", encoding="utf-8") as f:
            json.dump(self.df, f, ensure_ascii=False)
        meta_obj = self.meta or VocabMeta(
            version="unknown",
            tokenizer="unknown",
            stem=False,
            drop_pos=[],
            stopwords_hash="",
            min_df=-1,
            max_vocab=-1,
            num_docs=self.N,
            avgdl=self.avgdl,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(meta_obj), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, 
            snapshot_dir: str
            ) -> "VocabManager":
        """
        저장된 스냅샷 json파일로 VocabManager 인스턴스 생성
        """
        with open(os.path.join(snapshot_dir, "vocab.json"), "r", encoding="utf-8") as f:
            vocab = json.load(f)
        with open(os.path.join(snapshot_dir, "df_counter.json"), "r", encoding="utf-8") as f:
            df = json.load(f)
        with open(os.path.join(snapshot_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = VocabMeta(**json.load(f))
        return cls(vocab=vocab, df=df, N=meta.num_docs, avgdl=meta.avgdl, meta=meta)

    # -------------------- 조회/인코딩 헬퍼 --------------------
    @property
    def size(self) -> int:
        return len(self.vocab)

    @property
    def version(self) -> str:
        return self.meta.version if self.meta else "unknown"

    def token_id(self, token: str) -> Optional[int]:
        return self.vocab.get(token)

    def tokens_to_ids(self, tokens: Iterable[str]) -> List[int]:
        return [self.vocab[t] for t in tokens if t in self.vocab]

    def encode_text(self, text: str, tokenizer: Tokenizer) -> List[int]:
        """텍스트를 토큰화한 뒤 vocab ID 시퀀스로 변환(존재하는 토큰만)."""
        return self.tokens_to_ids(tokenizer.tokenize(text or ""))

    # 필요 시 유효성 검증(예: df에 있으나 vocab에 없는 항목 체크 등)
    def validate(self) -> None:
        missing = [tok for tok in self.df.keys() if tok not in self.vocab]
        if missing:
            raise ValueError(f"DF에 있는데 vocab엔 없는 토큰이 있습니다: {missing[:10]} ... (총 {len(missing)}개)")
