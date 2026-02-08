from __future__ import annotations
from typing import Dict, Any, Optional
import json
from functools import lru_cache

from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentState,
    _InputAgentState,
    _OutputAgentState,
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.typing import ContextT
from typing_extensions import TypeVar

from .model.agent_spec import AgentSpec
from app.domain.models import ChatRES, AgentSearchQuery
        
ResponseT = TypeVar("ResponseT")

class AgentAdapter:
    def __init__(self, spec: AgentSpec, agent_prompt: str):
        self._agent: Optional[
            CompiledStateGraph[
                AgentState[ResponseT],
                ContextT,
                _InputAgentState,
                _OutputAgentState[ResponseT]
                ]
            ] = None
        
        self.spec = spec
        self.agent_prompt = agent_prompt

    def _build_agent(self
        ) -> CompiledStateGraph[
                AgentState[ResponseT],
                ContextT,
                _InputAgentState,
                _OutputAgentState[ResponseT]]:
        if self._agent is None:
            self._agent =  create_agent(
            model = self.spec.model,
            tools=self.spec.tools,
            system_prompt=self.agent_prompt,
            response_format= self.spec.response_format,
            state_schema= self.spec.state_schema,
            context_schema= self.spec.context_schema,
            checkpointer= self.spec.checkpointer,
            middleware=self.spec.middleware or (),
        )
        
        return self._agent
    
    def _extract_ai_message_info(self, messages):
        RETRIEVER_TOOL_NAME = "retrieve_tool"

        def _tool_name(call: dict) -> str | None:
            # LC/모델 버전에 따라 tool_calls 구조가 달라서 둘 다 방어
            if not isinstance(call, dict):
                return None
            return call.get("name") or (call.get("function") or {}).get("name")

        # 0) retriever 호출 여부/횟수 + (가능하면) docs까지 추출
        retriever_calls = 0
        retrieved_docs = None

        for m in messages:
            # (A) AIMessage 안의 tool_calls에 retrieve_tool이 있는지
            if isinstance(m, AIMessage):
                calls = getattr(m, "tool_calls", None) or []
                for c in calls:
                    if _tool_name(c) == RETRIEVER_TOOL_NAME:
                        retriever_calls += 1

            # (B) ToolMessage로 반환된 결과에서 docs 꺼내기
            if isinstance(m, ToolMessage) and getattr(m, "name", None) == RETRIEVER_TOOL_NAME:
                # content가 dict일 수도, JSON string일 수도 있어서 방어
                payload = m.content
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except Exception:
                        payload = None

                if isinstance(payload, dict) and "docs" in payload:
                    retrieved_docs = payload.get("docs")

        retriever_used = retriever_calls > 0

        target_msg = None

        # 1) 먼저 AIMessage만 필터링
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]

        # 2) 뒤에서부터 보면서 "실제 답변" 후보 찾기
        for msg in reversed(ai_messages):
            # LangChain AIMessage에는 tool_calls 속성이 있을 수 있음
            has_tool_calls = getattr(msg, "tool_calls", None)

            # 툴콜용 메시지는 스킵
            if has_tool_calls:
                continue

            # 내용이 비어있는 것도 스킵
            if not msg.content:
                continue

            target_msg = msg
            break

        # 3) 그래도 못 찾았으면(툴콜만 있었던 경우 등), 그냥 마지막 AIMessage라도 사용
        if target_msg is None:
            if not ai_messages:
                return None  # AIMessage 자체가 없는 경우
            target_msg = ai_messages[-1]

        content = target_msg.content

        # 필요하다면 여기서 \n\n 같은 것 정리 가능
        # content = content.replace("\\n", "\n")  # 만약 실제로 문자열에 \n이 그대로 박혀있다면

        usage = target_msg.usage_metadata or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")

        output_details = usage.get("output_token_details", {}) or {}
        reasoning = output_details.get("reasoning")

        return ChatRES(
            chatbotmessage=content or "",
            input_token=input_tokens,
            ouput_token=output_tokens,
            total_token=total_tokens,
            reasoning_token=reasoning,
            retriever_used=retriever_used,
            retriever_calls=retriever_calls,
            retrieved_docs=retrieved_docs,
        )

    def generate(self, query_payload: AgentSearchQuery) -> ChatRES:
        query = query_payload.query
        
        agent = self._build_agent()
        config = {"configurable": {"thread_id": "1"}}
        payload = {"messages": [{"role": "user", "content": query}]}

        res = agent.invoke(
            payload,
            config=config,
            # context=Context(user_id="1")
        )
        
        msg = res.get('messages')

        return msg
        # return self._extract_ai_message_info(msg)