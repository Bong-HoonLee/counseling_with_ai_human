from fastapi import FastAPI
from graph_builder import build_graph
from graph_node import GraphState
from langgraph.checkpoint.memory import MemorySaver

from fastapi.middleware.cors import CORSMiddleware

from app.models import ChatRequest, ChatResponse

app = FastAPI()

origins = [
    "http://localhost:5173",   # React 개발 서버
    # "https://your-domain.com"  # 배포 시 프론트엔드 도메인
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



memory = MemorySaver()
workflow = build_graph()

# compiled_graph = workflow.compile(checkpointer=memory)
compiled_graph = workflow.compile()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    
    user_message = chat_request.message
    input = GraphState(question=[("user",  user_message)])
    
    
    output = compiled_graph.invoke(input)
                
    return ChatResponse(reply=output)
