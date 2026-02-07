from pydantic import BaseModel

# 요청 바디 모델 정의
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: dict