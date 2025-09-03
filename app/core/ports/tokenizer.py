from __future__ import annotations
from typing import Protocol, List, Tuple

class Tokenizer(Protocol):
    def tokenize(self, text: str) -> List[Tuple[str, str]]:
        ...
