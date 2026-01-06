from typing import Any
from langchain_core.tools import tool

from app.core.ports.vectorstore import VectorStorePort
from app.core.models import SearchQuery

def make_retrieve_tool(vs: VectorStorePort):
    @tool
    def retrieve_tool(query: SearchQuery) -> dict:
        '''
        Perform document retrieval for Retrieval-Augmented Generation (RAG) pipelines.

        This tool takes a natural language query and retrieves the top-k most relevant 
        documents from the configured vector store (e.g., Qdrant). 
        It supports both dense and sparse retrieval modes if hybrid retrieval is enabled.

        The retrieved documents are then typically passed to a generator model (LLM) 
        to ground its response on factual, context-rich information.

        Args:
            query (str): 
                The natural language search query from the user or rewritten form.
            k (int, optional): 
                Number of top documents to retrieve. Defaults to 6.
            **options (Any):
                Additional keyword arguments forwarded to the vector store search method, 
                such as filters, offsets, score thresholds, or hybrid fusion parameters.

        '''
        results = vs.search(query)
        docs = [
            {"page_content": d.page_content, "metadata": dict(d.metadata or {}), "score": float(s)}
            for d, s in results
        ]
        return {"docs": docs}
    return retrieve_tool