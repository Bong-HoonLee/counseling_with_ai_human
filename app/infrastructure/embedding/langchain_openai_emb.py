# from typing import Literal, Optional

# from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# from app.core.models import EmbeddingConfig, EmbeddingRequest, EmbeddingResponse
# from app.core.ports import EmbeddingPort

# class OpneaiEmbedding:
#     def __init__(
#             self,
#             config: EmbeddingConfig
#                  ):
#         self.cfg = config
#         self._emb = Optional[EmbeddingPort] = None

#     def _get_model(self) -> EmbeddingPort:
#         if self._emb is None:
#             _emb = OpenAIEmbeddings(**self.cfg.kwargs)

#         return _emb
    

#     # def embed_query(self, request: EmbeddingRequest) -> EmbeddingResponse:
#     #     emb_model = self._get_model()
        
    
#     def embed_documents(self, request: EmbeddingRequest) -> EmbeddingResponse:
#         request