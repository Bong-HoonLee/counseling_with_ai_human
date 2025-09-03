from app.core.utils.hash import HashStrategy

def content_hash(strategy: HashStrategy, q: str, a: str) -> str:
    return strategy.hexdigest(f"{q}|||A|||{a}")

def point_id(strategy: HashStrategy, pair_id: str, version: int) -> str:
    return strategy.hexdigest(f"{pair_id}:{version}")
