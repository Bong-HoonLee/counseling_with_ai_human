from ..ports import ChatbotAgentPort

class ChatbotPipe:
    def __init__(self, chat_model: ChatbotAgentPort) -> None:
        self.chat_model = chat_model
    
    def run(self, query: str):
        res = self.chat_model.generate(query)

        return res