from pydantic import BaseModel
from typing import Optional

# 요청 바디 모델 정의
class ChatRequest(BaseModel):
    message: str

# 응답 모델 정의 (선택사항)
class ChatResponse(BaseModel):
    reply: dict