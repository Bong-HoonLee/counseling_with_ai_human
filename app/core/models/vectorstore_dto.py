from typing import Literal, Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field

@dataclass
class VSClientConn:
    host: Optional[str] = None
    port: Optional[int] = None
    api_key: Optional[str] = None
    timeout: Optional[float] = None
    

@dataclass
class PointUpsert:
    id: Optional[str] = None
    # dense: Dict[str, List[float]]           # {"q_vec": [...], "a_vec": [...]}
    # sparse: Optional[Tuple[List[int], List[float]]] = None
    payload: Optional[Dict[str, Any]] = None
    '''
    payload must contains content
    '''

@dataclass
class SearchQuery:
    query: Optional[str] = None
    top_k: int = 10
    # dense: Dict[str, List[float]] = field(default_factory=dict)  # {"q_vec": [...], "a_vec": [...]}
    # sparse: Optional[Tuple[List[int], List[float]]] = None
    # weights: Dict[str, float] = field(default_factory=dict)        # {"q_vec": 0.7, "sparse": 0.3}
    options: Dict[str, Any] = field(default_factory=dict) # score_threshold: float = 0.4