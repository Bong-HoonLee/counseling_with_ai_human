from typing import Literal, Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field

@dataclass
class PointUpsert:
    id: str
    vectors: Dict[str, List[float]]           # {"q_vec": [...], "a_vec": [...]}
    sparse: Optional[Tuple[List[int], List[float]]] = None
    payload: Optional[Dict[str, Any]] = None

@dataclass
class SearchQuery:
    vectors: Dict[str, List[float]] = field(default_factory=dict)  # {"q_vec": [...], "a_vec": [...]}
    sparse: Optional[Tuple[List[int], List[float]]] = None
    weights: Dict[str, float] = field(default_factory=dict)        # {"q_vec": 0.7, "sparse": 0.3}
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 10