from app.core.ports import ChatmodelPort

class ChatbotPipe:
    def __init__(self, chat_model: ChatmodelPort) -> None:
        self.chat_model = chat_model
    
    def run(self, query: str):
        res = self.chat_model.response(query)

        return res