import pickle

import chromadb
from sentence_transformers import SentenceTransformer
import spacy
from ultralytics import YOLO
from database import conn, count_label , lookup



clf = pickle.load(open('intent_model.pk1','rb'))
model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load('en_core_web_sm')


query = str(input('Ask me the Question'))
vector = model.encode([query])
intent = clf.predict(vector)[0]

class_list = list(YOLO('yolo11n.pt').names.values())

#code to get the details.
doc = nlp(query)

for i in doc:
    if i.lemma_ in class_list:
        target = i.lemma_
        if intent == 'count':
            print(count_label(conn, target))
        elif intent == 'lookup':
            print(lookup(conn, target))

        elif intent =='fuzzy':
            client = chromadb.PersistentClient(path='chroma_db')
            collection = client.get_collection('detections')
            results = collection.query(
                query_embeddings=vector.tolist(),
                n_results=5
            )
            print(results['documents'])
            print(results['metadatas'])




