import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

from dotenv import load_dotenv
load_dotenv()

from app.core.models.types import model_provider

@dataclass
class OpenAIConfig:
    provider: Literal["openai"]
    api_key: str
    model: str
    params: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        return cls(
            provider = "openai",
            api_key=os.environ["OPENAI_API_KEY"],   
            model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o")
        )
    
@dataclass
class AzureConfig:
    provider: Literal["azure"]
    api_key: str
    model: str
    endpoint: str
    api_version: str
    params: Optional[Dict[str, Any]] = None

    @classmethod
    def from_env(cls) -> "AzureConfig":
        return cls(
            provider = "azure",
            api_key=os.environ["AZURE_API_KEY"],
            model=os.getenv("AZURE_LLM_MODEL", "gpt-4o"),
            endpoint=os.getenv("AZURE_ENDPOINT"),
            api_version=os.getenv("AZURE_API_VERSION")
        )