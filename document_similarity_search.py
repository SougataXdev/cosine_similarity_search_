from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Artificial intelligence is transforming the world.",
    "Machine learning is a subset of artificial intelligence.",
    "Natural language processing enables computers to understand human language.",
    "The weather is sunny today."
]


queary = "hows the weather today ?"


document_embeddings = embedding.embed_documents(documents)

queary_embedding = embedding.embed_query(queary)


scores = cosine_similarity([queary_embedding] , document_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(index,score)


print(documents[index])

print(f"similarity score is {score}")
