from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import HnswConfigDiff
from langchain_qdrant import QdrantVectorStore
import yaml
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

class Qdrnat_custom():
    def __init__(self,
                 host: str = 'localhost',
                 port: int = 6333,
                 collection_name: str = 'chatbot',
                 cfg_path: str = "app/config/config.yml",
                 model_type: str = 'openai_gpt',
                 ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = self._client()
        self.vector_store = self._vector_store()
        self.cfg = self._get_cfg(cfg_path)
        self.emb_model = self._get_model(model_type)

    def _get_cfg(self, path):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        
        return config

    def _client(self):
        client = QdrantClient(host=self.host, port=self.port)
        return client

    def _get_model(self, model_type):
        if model_type == 'Azure_gpt':
            emb_cfg = self.cfg['aoai']['emb_model']
            emb_model = AzureOpenAIEmbeddings(**emb_cfg)
            
        elif model_type == 'openai_gpt':
            emb_cfg = self.cfg['open_ai']['emb_model']
            emb_model = OpenAIEmbeddings(**emb_cfg)

        return emb_model
    
    def _vector_store(self):
        vector_store = QdrantVectorStore(
        client=self.client,
        collection_name= self.collection_name,
        embedding= self.emb_model
        )

        return vector_store

    
    def create_index(self, vector_size:int = 3072):
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(
                    m=16,                    # 그래프 degree (기본 16)
                    ef_construct=100,        # 인덱스 구축시 탐색 범위
                    full_scan_threshold=1000 # 소량 데이터 일때 hnsw보다 faster scan
                )
            )
        )

    def embedding_text(self, texts: list, metadatas: list):
        '''
        texts=[
        "Azure Blob Storage 인증 문제 해결 방법",
        "Python S3 파일 업로드 방법",
        "FastAPI로 REST API 서버 만들기"
        ],
        metadatas=[
            {"source": "azure_docs"},
            {"source": "aws_docs"},
            {"source": "fastapi_docs"}
        ]
        '''

        self.vector_store.add_texts(
            texts = texts,
            metadatas = metadatas
        )
    
    def similarity_search(self, 
                          search_cfg
                          ):
        '''
         cfg : "API 서버 만들기",
                k=2,
                filter={"source": "fastapi_docs"}
        '''
        docs = self.vector_store.similarity_search(search_cfg)

        return docs
    
    def similarity_search_with_score(self, 
                          search_cfg
                          ):
        '''
         cfg : "API 서버 만들기",
                k=2,
                filter={"source": "fastapi_docs"},
                score_threshold=0.4
        
        return : [(Document(metadata={'source': 'azure_docs', '_id': '00be5158-4a9e-4cc0-b9c8-5d7190a2691c', 
                    '_collection_name': 'chatbot'}, page_content='Azure Blob Storage 인증 문제 해결 방법'),
                    0.83140427)]
        '''
        docs = self.vector_store.similarity_search_with_score(search_cfg)
        
        return docs
        
        