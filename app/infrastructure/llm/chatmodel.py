from typing import Any, Dict, Union, Optional
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.language_models.base import LanguageModelLike
from langchain.chat_models import init_chat_model

from app.config import OpenAIConfig, AzureConfig

LlmConfig = Union[OpenAIConfig, AzureConfig]

class LangchainChat:
    def __init__(self, cfg: LlmConfig):
        self.cfg = cfg
        self._model: Optional[LanguageModelLike] = None

    def build_llm(self):
        if self._model is None:
            if self.cfg.provider == "azure":
                self._model =  AzureChatOpenAI(
                    azure_endpoint=self.cfg.endpoint,
                    api_version=self.cfg.api_version,
                    openai_api_key=self.cfg.api_key,
                    model=self.cfg.model,
                    # **self.cfg.params,
                )
            else:
                self._model =  ChatOpenAI(
                    api_key=self.cfg.api_key,
                    model=self.cfg.model,
                    # **self.cfg.params,
                    )
        return self._model

    # def response(self, query: str):
    #     return self._model.invoke(
    #         query
    #     )