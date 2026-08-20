from database import conn, get_detection
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import chromadb




client = chromadb.PersistentClient(path='chroma_db')
collection = client.get_or_create_collection('detections')
model = SentenceTransformer('all-MiniLM-L6-v2')

rows = get_detection(conn)
grouped = defaultdict(list)
for timestamp, label in rows:
    grouped[timestamp].append(label)

for timestamp, labels  in grouped.items():
    text =' '.join(labels)
    vector = model.encode(text)
    collection.add(
        ids=[str(timestamp)],
        embeddings=[vector.tolist()],
        documents=[text],
        metadatas=[{'timestamp':timestamp, 'Labels':text}]
    )





