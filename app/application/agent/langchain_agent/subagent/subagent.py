from typing import Optional

from langchain.agents import create_agent

from app.application.agent.langchain_agent.subagent.tools import *

SEARCH_AGENT_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. "
    "Extract recipient information and craft appropriate subject lines and body text. "
    "Use send_email to send the message. "
    "Always confirm what was sent in your final response."
)

email_agent = create_agent(
    model = ,
    tools=[],
    system_prompt=SEARCH_AGENT_PROMPT,
)