import hashlib
from typing import Protocol

class HashStrategy(Protocol):
    name: str
    def hexdigest(self, s: str) -> str: ...


class SHA1Strategy:
    name = "sha1"
    def hexdigest(self, s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

class SHA256Strategy:
    name = "sha256"
    def hexdigest(self, s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()