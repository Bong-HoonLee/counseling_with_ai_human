# app/agents/specs.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Sequence, TYPE_CHECKING, Any
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
)
from langgraph.typing import ContextT
from typing_extensions import TypeVar
from langchain.agents.structured_output import (
    ResponseFormat,
)

if TYPE_CHECKING:
    from langgraph.types import Checkpointer
    from langgraph.cache.base import BaseCache

ResponseT = TypeVar("ResponseT")

@dataclass
class AgentSpec:
    model: str | BaseChatModel
    name: str | None = None
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None
    system_prompt: str | None = None
    middleware: Sequence[AgentMiddleware[AgentState[ResponseT], ContextT]] = ()
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | None = None
    state_schema: type[AgentState[ResponseT]] | None = None
    context_schema: type[ContextT] | None = None
    checkpointer: Checkpointer | None = None
    cache: BaseCache | None = None