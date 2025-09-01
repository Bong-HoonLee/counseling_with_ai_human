from typing import Literal, Optional
from dataclasses import dataclass
from app.models.types import Provider


@dataclass
class DualEmbeddingConfig:
    """
    A/B 임베딩 분리
    """
    provider: Provider = "openai"
    a_kwargs: Optional[dict] = None   
    b_kwargs: Optional[dict] = None  