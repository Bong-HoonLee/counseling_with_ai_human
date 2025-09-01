# POS 카테고리 (대문자 권장)
UNIVERSAL_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM", "PRON"}

# 백엔드별 매핑
OKT_MAP = {
    "NOUN": {"Noun"},
    "VERB": {"Verb"},
    "ADJ":  {"Adjective"},
    # Okt는 PROPN 구분이 약해서 NOUN로 흡수
    "PROPN": {"Noun"},
    "ADV": set(), "NUM": set(), "PRON": set(),
}

MECAB_MAP = {
    "NOUN": {"NNG","NNB"},        # 일반/의존 명사
    "PROPN": {"NNP"},             # 고유명사
    "VERB": {"VV"},               # 동사
    "ADJ":  {"VA"},               # 형용사
    "ADV":  {"MAG","MAJ"},
    "NUM":  {"SN"},
    "PRON": {"NP"},
}

BACKEND2MAP = {
    "okt": OKT_MAP,
    "mecab": MECAB_MAP,
}

def resolve_allowed_pos(
    backend: str = "okt",
    categories: set[str] | None = None,
    explicit_tags: set[str] | None = None,
) -> set[str]:
    """
    categories: POS 세트(예: {"NOUN","VERB","ADJ"})
    explicit_tags: 백엔드 태그를 직접 지정하고 싶을 때 사용
    """
    if explicit_tags:
        return set(explicit_tags)
    cats = (categories or {"NOUN","VERB","ADJ"})
    mp = BACKEND2MAP.get(backend.lower())
    if mp is None:
        raise ValueError(f"Unknown POS backend: {backend}")
    tags: set[str] = set()
    for c in cats:
        tags |= mp.get(c, set())
    return tags