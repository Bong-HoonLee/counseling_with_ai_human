# app/agents/factory.py
from __future__ import annotations
from typing import Dict, Any, Optional
from functools import lru_cache
from langchain.agents import create_agent
from app.application.agent.langchain_agent.model import AgentSpec

def build_agent(spec: AgentSpec, agent_prompt: str):
    return create_agent(
    model = spec.model,
    tools=spec.tools,
    system_prompt=agent_prompt,
    response_format= spec.response_format,
    state_schema= spec.state_schema,
    context_schema= spec.context_schema,
    checkpointer= spec.checkpointer
)