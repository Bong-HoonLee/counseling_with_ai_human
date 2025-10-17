from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, PydanticOutputParser
from langchain.docstore import InMemoryDocstore
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from langchain_core.language_models.base import LanguageModelLike

from .parser import RagOutput, ConvOutput
from .state import GraphState

from app.core.ports import VectorStorePort
from app.core.models import SearchQuery


class ChatbotNode:
    def __init__(self,
                 llm: LanguageModelLike,
                 retriever: VectorStorePort,
                 search_cfg: SearchQuery,
                 ):
        self.llm_model = llm
        self.retriever = retriever
        self.search_cfg = search_cfg

    def _load_prompt(self, prompt_path):
        from langchain_core.prompts import load_prompt    
        prompt = load_prompt(prompt_path)
        
        return prompt

    def retrieve(self, state: GraphState) -> GraphState:
        self.search_cfg.query = state['question'][-1][-1]
        # retrieve: 검색
        docs = self.retriever.search(self.search_cfg)

        # 임시코드
        if docs:
            '''
            고려할 사항 :
            1. 리랭커등을 해서 상위 3개정도를 집어낼건데 전부 통합할 때 컨텍스트 길이 모델에 넣을 수 있는지 만약에 넘치면 요약하는 분기를 타야되는지 고려
            2. 테스트 모드에서는 어떤 출처가 뽑혔는지 나타내야 되기 때문에 state에 기록 남기기(문서 이름, 스코어)
            3. 벡터DB 고도화하기 필드단으로 디테일한 세팅값, 검색 알고리즘, 리랭커, 벡터 양자화 등
            '''
            documents = [docs[0][0].page_content]
        else:
            documents = []

        return {'context': documents}


    def rewrite_query(self, state: GraphState) -> GraphState:
        prompt_path = self.cfg.get('prompt').get('query_rewrite')

        prompt = self._load_prompt(prompt_path)

        chain = (
            prompt
            | self.llm_model
        )

        # question에 rewrite된 query 쌓기
        query_rewrite = chain.invoke(
            {
                "question" : state['question'][-1],
            }
        )

        return {'question': query_rewrite}


    def rag_response(self, state: GraphState) -> GraphState:
        
        prompt_path = self.cfg.get('prompt').get('rag_prompt_path')

        prompt = self._load_prompt(prompt_path)

        parser = JsonOutputParser(pydantic_object=RagOutput)
        prompt = prompt.partial(format_instructions=parser.get_format_instructions())

        chain = (
            prompt
            | self.llm_model
            | JsonOutputParser()
        )
        
        contexts = state.get('context')[-1]
        question = state.get('question')[-1]
            
        response = chain.invoke({
        # 'chat_history': state['answer'],
        'question': question,
        'contexts' : contexts
        })
        
        # answer = response['answer'][0]['Answer']
        answer = response['Answer']

        return {'answer': [answer]}

    def conversation_response(self, state: GraphState) -> GraphState:
        
        prompt_path = self.cfg.get('prompt').get('conv_prompt_path')


        prompt = self._load_prompt(prompt_path)
        parser = JsonOutputParser(pydantic_object=ConvOutput)
        prompt = prompt.partial(format_instructions=parser.get_format_instructions())

        chain = (
            prompt
            | self.llm_model
            | JsonOutputParser()
        )
            
        question = state.get('question', [])
 
        response = chain.invoke({
        'question': question,
        })
        
        answer = response['Answer']

        return {'answer': [answer]}
    
    def summerize(self, state: GraphState) -> GraphState:
        answer_summerize = ''

        return {'answer': [answer_summerize]}


    def relevance_check(self, state: GraphState) -> GraphState:
        # Relevance Check: 관련성 확인
        binary_score = "Relevance Score"

        return {'binary_score': binary_score}



    def search_on_web(self, state: GraphState) -> GraphState:
        # Search on Web: 웹 검색
        documents = state["context"] = "기존 문서"
        searched_documents = "검색된 문서"
        documents += searched_documents

        return {'context': documents}


    # tools
    def is_retrieve(state: GraphState):
        context = state.get("context", [])
        
        if context:
            return "rag"
        else:
            return "conversation"