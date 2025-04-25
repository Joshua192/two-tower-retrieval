import torch
import torch.nn.functional as F
import numpy as np
from model import QueryTower, DocumentTower
from process_data import doc_id_to_text
from encoder import sentence_to_w2v

# Load trained towers
query_tower = QueryTower()
doc_tower = DocumentTower()
query_tower.load_state_dict(torch.load("./saved_models/query_tower.pt"))
doc_tower.load_state_dict(torch.load("./saved_models/doc_tower.pt"))
query_tower.eval()
doc_tower.eval()

# Step 1: Encode all documents (you only need to do this once)
# Assume `doc_text_to_id` and `sentence_to_w2v` from training
doc_embeddings = []
doc_ids = []

# print(f"doc id -> txt: {doc_id_to_text}") # id to text is empty.

for doc_id, doc_text in doc_id_to_text.items():
    doc_vector = sentence_to_w2v(doc_text)  # shape (300,)
    doc_vector = torch.tensor(doc_vector).unsqueeze(0).float()  # shape (1, 300)
    with torch.no_grad():
        emb = doc_tower(doc_vector)  # shape (1, 300)
        # print(f"embedding: {emb}")
    doc_embeddings.append(emb)
    doc_ids.append(doc_id)
    break

doc_embeddings = torch.cat(doc_embeddings, dim=0)  # shape (num_docs, 300)

def get_top_k(query_text, k=5):
    query_vector = sentence_to_w2v(query_text)
    query_vector = torch.tensor(query_vector).unsqueeze(0).float()
    with torch.no_grad():
        query_embedding = query_tower(query_vector)

    # Compute cosine similarities
    similarities = F.cosine_similarity(query_embedding, doc_embeddings)

    # Get top-k indices
    top_k_indices = torch.topk(similarities, k).indices
    top_k_doc_ids = [doc_ids[i] for i in top_k_indices.tolist()]
    top_k_texts = [doc_id_to_text[doc_id] for doc_id in top_k_doc_ids]
    
    return list(zip(top_k_doc_ids, top_k_texts))

# Example usage
query = "What is the capital of France?"
top_docs = get_top_k(query, k=3)
for doc_id, text in top_docs:
    print(f"Doc ID: {doc_id} | Text: {text}")
