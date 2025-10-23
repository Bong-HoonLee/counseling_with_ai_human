from typing import Optional

from langchain_core.tools import tool

from app.core.ports import VectorStorePort


# application의 adapter 의존성 문제를 해결하기 위해 전역 변수로 설계
VS: Optional[VectorStorePort] = None
_LLM = None


def set_llm(llm) -> None:
    global _LLM
    _LLM = llm


def bind_vs(vs: VectorStorePort):
    '''
    composition단에서 설정
    '''
    global VS; VS = vs

def vs_or_raise() -> VectorStorePort:
    if VS is None: raise RuntimeError("VS not bound")
    return VS

@tool
def retrieve_tool(query: str, k: int = 6) -> dict:
    vs = vs_or_raise()
    docs = vs.search(query, k=k)
    return {"docs": docs}