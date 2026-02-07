from app.common.models.vectorstore_dto import SearchQuery
from app.adapters.vectorstore.qdrant.langchain_qdrant import Qdrant
from app.config import QdrantClintConfig, QdrantVsConfig

def main(qdrant_vs: Qdrant, query_dto: SearchQuery):
    
    result = qdrant_vs.search(query_dto)
    print(result)

if __name__ == "__main__":
    client_cfg = QdrantClintConfig.default()
    vs_cfg = QdrantVsConfig.default()

    qdrant_vs = Qdrant(
        client_cfg,
        vs_cfg
    )

    query_dto = SearchQuery()
    query_dto.top_k = 3
    query_dto.query = '지금 상태가 너무 안 좋아서 학교 안 나가고 있어요.'

    main(qdrant_vs, query_dto)