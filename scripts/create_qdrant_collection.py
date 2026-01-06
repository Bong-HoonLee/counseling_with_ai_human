import argparse
import os

from app.adapters.vectorstore.qdrant.collection import Qdrant
from app.config import QdrantClintConfig, QdrantSchema, QdrantVsConfig

def main(
    client_cfg: QdrantClintConfig,
    vs_cfg: QdrantVsConfig,
    schema: QdrantSchema,
    ) -> None:

    # get qdrant client
    qdrant_inst = Qdrant(
        client_cfg=client_cfg,
        vs_cfg=vs_cfg,
    )
    qdrant_inst.create_index(schema=schema)

    return None


parser = argparse.ArgumentParser(description="Create Qdrant collection")
parser.add_argument("--host", type=str, default=None)
parser.add_argument("--port", type=int, default=None)
parser.add_argument("--timeout", type=int, default=None)
parser.add_argument("--collection", type=str, default=None)

args = parser.parse_args()

if __name__ == "__main__":

    client_cfg = QdrantClintConfig.default()
    schema = QdrantSchema()
    vs_cfg = QdrantVsConfig.default()

    client_cfg.host = args.host if args.host is not None else client_cfg.host
    client_cfg.port = int(args.port) if args.port is not None else client_cfg.port
    timeout = int(args.timeout) if args.timeout is not None else client_cfg.timeout
    schema.collection_name = args.collection if args.collection is not None else schema.collection_name
    schema.vectors_config = schema.vectors_config

    main(client_cfg, vs_cfg, schema)