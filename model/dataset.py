import torch
import pickle
from datasets import load_dataset
import pandas as pd
import pathlib
import os


class MSMARCO(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
    ):
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, f"../data/triplets_{split}.tsv")
        # file_path = f"C:\\Users\\Joshua\\Desktop\\Computer-Science-Courses\\Machine-Learning-Institute\\two-tower-retrieval\\data\\triplets_{split}.tsv"
        # print(pathlib.Path(file_path).is_file())
        self.triplets_df = pd.read_csv(file_path, sep="\t")
        # self.triplets_df = self.triplets_df.to_pandas()
        self.encoded_queries = pickle.load(open(f"../data/encoded_queries_{split}.pkl", "rb"))
        self.encoded_documents = pickle.load(
            open(f"../data/encoded_documents_{split}.pkl", "rb")
        )

    def __len__(self):
        return len(self.triplets_df)

    def __getitem__(self, idx):
        triplet = self.triplets_df.iloc[idx]
        query_id = triplet["query_id"]
        pos_doc_id = triplet["pos_doc_id"]
        neg_doc_id = triplet["neg_doc_id"]

        query = self.encoded_queries[query_id]
        pos_doc = self.encoded_documents[pos_doc_id]
        neg_doc = self.encoded_documents[neg_doc_id]

        return {
            "query": query,
            "pos_doc": pos_doc,
            "neg_doc": neg_doc,
        }