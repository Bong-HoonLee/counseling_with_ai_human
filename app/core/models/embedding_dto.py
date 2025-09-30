from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

from app.core.models.types import model_provider

@dataclass
class EmbeddingConfig:
    """
    임베딩 config
    """
    provider: Optional[model_provider] = None
    kwargs: Optional[dict] = None

@dataclass
class EmbeddingItemInput:
    id: str
    text: str
    # space: str = "default"               # "query"/"passage" 등 채널 구분
    # metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetryPolicy:
    max_retries: int = 2
    backoff_initial_ms: int = 300
    backoff_multiplier: float = 2.0
    retry_on: List[str] = field(default_factory=lambda: ["429", "5xx", "timeout"])

@dataclass
class BatchingHints:
    batch_size_hint: Optional[int] = None
    rate_limit_tier: Optional[Literal["low","standard","high"]] = None
    timeout_sec: Optional[float] = None
    retry: RetryPolicy = field(default_factory=RetryPolicy)

@dataclass
class EmbeddingRequest:
    items: List[EmbeddingItemInput]
    emb_config: EmbeddingConfig
    # batching: BatchingHints = field(default_factory=BatchingHints)

@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    total_tokens: int = 0

@dataclass
class EmbeddingResultItem:
    id: str
    vector: List[float]
    dim: int
    space: str = "default"
    model_version: Optional[str] = None
    token_usage: Optional[TokenUsage] = None

@dataclass
class FailedItem:
    id: str
    error_code: str
    message: str
    is_transient: bool = True
    retry_after_ms: Optional[int] = None

@dataclass
class EmbeddingResponse:
    results: List[EmbeddingResultItem] = field(default_factory=list)
    failed: List[FailedItem] = field(default_factory=list)
    emb_config: Optional[EmbeddingConfig] = None
    usage_total: Optional[TokenUsage] = None
    warnings: List[str] = field(default_factory=list)
    elapsed_ms: Optional[int] = None
