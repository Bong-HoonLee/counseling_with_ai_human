from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, PydanticOutputParser
from langchain.docstore import InMemoryDocstore
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
import yaml
import os
from dotenv import load_dotenv

from .state import GraphState

load_dotenv()


class Chatbot_node():
    def __init__(self, 
                 model_type: str,
                 cfg_path: str = 'app/core/config/config.yml',
                 search_type: str = 'qdrnat',
                 index: str = 'chatbot',
                 ):
        # self.model, self.emb_model = self._get_model(model_type, llm_cfg, emb_cfg)
        self.cfg = self._get_cfg(cfg_path)
        self.llm_model, self.emb_model = self._get_model(model_type)
        self.search_type = search_type
        self.index = index
        
    def _get_cfg(self, path):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _get_model(self, model_type):
        if model_type == 'Azure_gpt':
            llm_cfg = self.cfg['aoai']['llm_model']
            emb_cfg = self.cfg['aoai']['emb_model']
            llm_model = AzureChatOpenAI(**llm_cfg)
            emb_model = AzureOpenAIEmbeddings(**emb_cfg)
            
        elif model_type == 'openai_gpt':
            llm_cfg = self.cfg['open_ai']['llm_model']
            emb_cfg = self.cfg['open_ai']['emb_model']
            llm_model = ChatOpenAI(
                **llm_cfg
            )
            emb_model = OpenAIEmbeddings(**emb_cfg)

        return llm_model, emb_model

    def _load_prompt(self, prompt_path, prompt_type):
        from langchain_core.prompts import load_prompt
        from parser import RagOutput, ConvOutput
        prompt = load_prompt(prompt_path)
        
        if prompt_type == 'rag':
            parser = JsonOutputParser(pydantic_object=RagOutput)
            prompt = prompt.partial(format_instructions=parser.get_format_instructions())

        elif prompt_type == 'conversation':
            parser = JsonOutputParser(pydantic_object=ConvOutput)
            prompt = prompt.partial(format_instructions=parser.get_format_instructions())

        elif prompt_type == 'rewrite':
            pass
        
        return prompt

    def retrieve(self, state: GraphState) -> GraphState:
        if self.search_type == 'qdrnat':
            retrieve_cfg = self.cfg['retriever']['qdrant']
            host, port = retrieve_cfg.get('host'), retrieve_cfg.get('port')
            search_cfg = retrieve_cfg.get('search_cfg')
            user_query = state['question'][-1][-1]
            # retrieve: 검색
            client = QdrantClient(host=host, port=port)
            vector_store = QdrantVectorStore(client=client, collection_name=self.index, embedding=self.emb_model)
            docs = vector_store.similarity_search_with_score(user_query, **search_cfg)

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
            
        return GraphState(context=documents)


    def rewrite_query(self, state: GraphState) -> GraphState:
        prompt_path = self.cfg.get('prompt').get('query_rewrite')
        prompt_type = 'rewrite'

        prompt = self._load_prompt(prompt_path, prompt_type)

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

        return GraphState(question=query_rewrite)


    def rag_response(self, state: GraphState) -> GraphState:
        
        prompt_path = self.cfg.get('prompt').get('rag_prompt_path')

        prompt_type = 'rag'
        prompt = self._load_prompt(prompt_path, prompt_type)
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

        return GraphState(answer=[answer])

    def conversation_response(self, state: GraphState) -> GraphState:
        
        prompt_path = self.cfg.get('prompt').get('conv_prompt_path')

        prompt_type = 'conversation'
        prompt = self._load_prompt(prompt_path, prompt_type)
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

        return GraphState(answer=[answer])
    
    def summerize(self, state: GraphState) -> GraphState:
        answer_summerize = ''
        return GraphState(answer=answer_summerize)

    def relevance_check(self, state: GraphState) -> GraphState:
        # Relevance Check: 관련성 확인
        binary_score = "Relevance Score"
        return GraphState(binary_score=binary_score)


    def search_on_web(self, state: GraphState) -> GraphState:
        # Search on Web: 웹 검색
        documents = state["context"] = "기존 문서"
        searched_documents = "검색된 문서"
        documents += searched_documents
        return GraphState(context=documents)


    # tools
    def is_retrieve(state: GraphState):
        context = state.get("context", [])
        
        if context:
            return "rag"
        else:
            return "conversation"