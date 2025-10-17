from typing import Any, Dict, Union
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.language_models.base import LanguageModelLike
from langchain.chat_models import init_chat_model

from app.config import OpenAIConfig, AzureConfig

LlmConfig = Union[OpenAIConfig, AzureConfig]

def build_llm(cfg: LlmConfig) -> LanguageModelLike:
    params: Dict[str, Any] = cfg.params or {}
    if cfg.provider == "azure":
        return AzureChatOpenAI(
            azure_endpoint=cfg.endpoint,
            api_version=cfg.api_version,
            openai_api_key=cfg.api_key,
            model=cfg.model,
            **params,
        )
    return ChatOpenAI(
        api_key=cfg.api_key,
        model=cfg.model,
        **params,
    )