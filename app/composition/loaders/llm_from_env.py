import os
from app.config import (
    ChatModelOptions,
    OpenAIConfig,
    AzureConfig,
    LLMConfig
)


def _require_env(name: str) -> str:
    value = os.getenv(name, "")
    if not value:
        raise ValueError(f"{name} is required")
    return value


def load_llm_config_from_env(provider: str | None = None) -> LLMConfig:
    """
    - 기본은 ENV의 LLM_PROVIDER를 따름
    - 인자로 provider를 주면 ENV를 override (테스트/로컬에서 편함)
    """
    resolved_provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()

    opts = ChatModelOptions(
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.01")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        top_p=float(os.getenv("LLM_TOP_P", "1.0")),
        frequency_penalty=float(os.getenv("LLM_FREQUENCY_PENALTY", "0.0")),
        presence_penalty=float(os.getenv("LLM_PRESENCE_PENALTY", "0.0")),
    )

    if resolved_provider == "openai":
        cfg = OpenAIConfig(
            api_key=_require_env("OPENAI_API_KEY"),
            model_name=os.getenv("OPENAI_LLM_MODEL", "gpt-4o"),
        )
        return LLMConfig(provider=cfg, options=opts)

    if resolved_provider == "azure":
        cfg = AzureConfig(
            api_key=_require_env("AZURE_API_KEY"),
            model_name=os.getenv("AZURE_LLM_MODEL", "gpt-4o"),
            endpoint=_require_env("AZURE_ENDPOINT"),
            api_version=_require_env("AZURE_API_VERSION"),
        )
        return LLMConfig(provider=cfg, options=opts)

    raise ValueError(f"Unsupported provider: {resolved_provider}")
