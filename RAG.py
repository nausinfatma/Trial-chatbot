import os
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()
api_key = os.getenv("GEMINI_KEY")
gemini_client = genai.Client(api_key = api_key)
chroma_client = chromadb.Client()

documents = [
    "Python is a high level programming language.",
    "It is widely used in web development and data science.",
    "The sky is blue.",
    "Python is also popular for machine learning.",
    "The weather is sunny today.",
    "It has a simple and readable syntax."
]

query = "What is python?"

collection = chroma_client.create_collection(
    name = "my_collection"
    )

collection.add(
    documents = documents,
    ids = [str(i) for i in range(len(documents))]
    )

result = collection.query(
    query_texts= [query],
    n_results= 4
)

docs = []
threshold = 0.8

for dis, doc in zip(result["distances"][0], result["documents"][0]):
    if dis <= threshold:
        docs.append(doc)
        
context = '\n'.join(docs)

response = gemini_client.interactions.create(
    model = "gemini-3.6-flash",
    input = f"""
    Use the following context for info: {context}
    question = {query}
    """
)

print(response.output_text)