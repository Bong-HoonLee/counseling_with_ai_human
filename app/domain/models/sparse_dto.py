from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class SparseVectorTypes:
    indices: List[int]
    values: List[float]