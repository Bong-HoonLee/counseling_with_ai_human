STOPWORDS_KO: set[str] = {
    "은","는","이","가","을","를","에","와","과","도","만","까지","보다","으로","로","에서","한테",
    "처럼","그리고","그러나","하지만","그래서","또는","혹은",
    "하다","되다","있다","없다","주다","보다",
    "이","그","저","것","수","등",
}

def load_stopwords(extra_path: str | None = None) -> set[str]:
    """
    위의 정의된 stopwords 반환, 또는 외부 txt 파일 반환
    """
    stopwords = set(STOPWORDS_KO)

    if extra_path:
        try:
            with open(extra_path, encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word:
                        stopwords.add(word)
        except FileNotFoundError:
            raise RuntimeError(f"Stopwords 파일을 찾을 수 없습니다: {extra_path}")

    return stopwords