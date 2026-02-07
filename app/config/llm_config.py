from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Union
import os

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Union


# 공통 옵션(모델 파라미터)
@dataclass(frozen=True)
class ChatModelOptions:
    temperature: float = 0.01
    max_tokens: int = 2048
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    extra_params: Optional[Dict[str, Any]] = None  # provider별 추가 옵션


# Provider별 연결 정보(필수값을 타입으로 강제)
@dataclass(frozen=True)
class OpenAIConfig:
    provider: Literal["openai"] = "openai"
    api_key: str = ""
    model_name: str = "gpt-4o"


@dataclass(frozen=True)
class AzureConfig:
    provider: Literal["azure"] = "azure"
    api_key: str = ""
    model_name: str = "gpt-4o"
    endpoint: str = ""
    api_version: str = ""


ProviderConfig = Union[OpenAIConfig, AzureConfig]


# 최종 로드하는 config
@dataclass(frozen=True)
class LLMConfig:
    provider: ProviderConfig
    options: ChatModelOptions

