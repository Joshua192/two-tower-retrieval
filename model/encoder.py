import pickle
from gensim.utils import simple_preprocess
import numpy as np
import pandas as pd
import argparse
import os
from gensim.models import KeyedVectors

base_dir = os.path.dirname(__file__)
embedding_path = os.path.join(base_dir, "embeddings/GoogleNews-vectors-negative300.bin")

print("Loading embeddings from:", embedding_path)

model = KeyedVectors.load_word2vec_format(embedding_path, binary=True)


def sentence_to_w2v(sentence):
    tokens = simple_preprocess(sentence)
    vectors = [model[w] for w in tokens if w in model]
    if not vectors:
        return np.zeros(300)
    return np.mean(vectors, axis=0)


def encode_and_save_queries(split: str):
    queries_df = pd.read_csv(f"../data/queries_{split}.tsv", sep="\t")
    encoded_queries = {
        row["query_id"]: sentence_to_w2v(row["query_text"])
        for _, row in queries_df.iterrows()
    }
    with open(f"../data/encoded_queries_{split}.pkl", "wb") as f:
        pickle.dump(encoded_queries, f)


def encode_and_save_documents(split: str):
    documents_df = pd.read_csv(f"../data/documents_{split}.tsv", sep="\t")
    encoded_documents = {
        row["doc_id"]: sentence_to_w2v(row["doc_text"])
        for _, row in documents_df.iterrows()
    }
    with open(f"../data/encoded_documents_{split}.pkl", "wb") as f:
        pickle.dump(encoded_documents, f)


def main(split: str):
    encode_and_save_queries(split)
    encode_and_save_documents(split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode and save queries and documents."
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "validation", "test"], required=True
    )
    args = parser.parse_args()

    main(args.split)