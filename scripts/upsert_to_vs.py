import pandas as pd

from app.adapters.vectorstore.qdrant.collection import Qdrant
from app.config import QdrantClintConfig, QdrantVsConfig
from app.common.models.vectorstore_dto import PointUpsert

def main(qdrant_vs: Qdrant, data_path: str):
    data_og_df = pd.read_csv(data_path)

    fixed_df = data_og_df.where(pd.notnull(data_og_df), None)

    points = []
    for idx, row in fixed_df.iterrows():
        payload = {
            'content' : row['contents'],
            'symp1' : row['level1'],
            'symp2' : row['level2'],
            'symp3' : row['level3'],
            'symp4' : row['level4'],
        }
        point = PointUpsert()
        point.payload = payload
        points.append(point)
    

    qdrant_vs.upsert(points)


if __name__ == "__main__":
    # qdrant inst
    
    client_cfg = QdrantClintConfig.default()
    vs_cfg = QdrantVsConfig.default()

    qdrant_vs = Qdrant(
        client_cfg,
        vs_cfg
    )

    data_path = 'app/data/filtered_wellness_dataset_v1.1_20251020.csv'
    main(qdrant_vs, data_path)