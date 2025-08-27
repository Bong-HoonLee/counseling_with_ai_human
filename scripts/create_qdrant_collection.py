from app.infrastructure.vectorstore.qdrant import QdrantCustom
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Create Qdrant collection (dense).")
    parser.add_argument("--host", default=os.getenv("QDRANT_HOST", "localhost"))
    parser.add_argument("--port", default=os.getenv("QDRANT_PORT", 6333))
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "aihuman"))
    parser.add_argument("--dim", type=int, default=3072)
    parser.add_argument("--drop-if-exists", action="store_true",
                        help="컬렉션이 이미 있으면 삭제 후 재생성")
    parser.add_argument("--emb", default=os.getenv("OPENAI_EMBED_MODEL", "openai_gpt"),
                        help="예: text-embedding-3-small, 지정 시 차원 자동 감지 시도")

    args = parser.parse_args()

    host = args.host
    port = args.port
    collection_name = args.collection
    dim = args.dim
    model_type = args.emb

    qdrant_inst = QdrantCustom(
        host,
        port,
        collection_name,
        model_type
    )
    qdrant_inst.create_index(
        dim
    )


    # 필드 인덱스 생성
    for f, schema in [
    ("lang", "keyword"),
    ("doc_type", "keyword"),
    ("product", "keyword"),
    ("region", "keyword"),
    ("created_at", "integer"),
    ("tags", "keyword"),
    ("doc_id", "keyword"),
    ]:
        client.create_payload_index(COL, field_name=f, field_schema=schema)

if __name__ == "__main__":
    main()