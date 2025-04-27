import chromadb
from model import DocumentTower
from encoder import sentence_to_w2v
from process_data import doc_id_to_text
import torch
import os

base_dir = os.path.dirname(__file__)

doc_tower = DocumentTower()
doc_path = os.path.join(base_dir, "./saved_models/doc_tower.pt")

doc_tower.load_state_dict(torch.load(doc_path))
doc_tower.eval()

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="doc_embed")

# Encode and add docs
for doc_id, doc_text in doc_id_to_text.items():
    vec = sentence_to_w2v(doc_text)
    vec = torch.tensor(vec).unsqueeze(0).float()
    with torch.no_grad():
        emb = doc_tower(vec).squeeze().tolist()

    collection.add(
        documents=[doc_text],
        embeddings=[emb],
        ids=[str(doc_id)],
        metadatas=[{"source": "msmarco"}]
    )

client = chromadb.PersistentClient(path="./chroma_stuff/")