import os
from dotenv import load_dotenv
from google import genai
import numpy as np

load_dotenv()
api_key = os.getenv("GEMINI_KEY")
client = genai.Client(api_key = api_key)

document = """
    Python is a high level programming language.
    It is widely used in web development and data science.
    The sky is blue.
    Python is also popular for machine learning.
    The weather is sunny today.
    It has a simple and readable syntax.
"""

query = client.models.embed_content(
    model = "gemini-embedding-2",
    contents = "What is python?"
)
words = document.split(" ")
chunk_size = 10
overlap = 3

start = 0
chunks = []

while start < len(words):
    end = start + chunk_size
    chunk = " ".join(words[start:end])
    chunks.append(chunk)
    start = end - overlap

q_em = query.embeddings[0].values

em = []

for x in chunks:
    result = client.models.embed_content(
        model = "gemini-embedding-2",
        contents = x
        )
    
    em.append(result.embeddings[0].values)
    
similarity = []

for x in em:
    similarity.append(np.dot(q_em, x) / (np.linalg.norm(q_em) * np.linalg.norm(x)))
    
similar = []
threshold = 0.7

for x in similarity:
    if x > threshold:
        similar.append(x)

content =""

for i in range(len(similar)):
    content += chunks[i] + "\n"


response = client.interactions.create(
    model = "gemini-3.6-flash",
    input = f"""
You are a helpful assistant. Use the following context to answer the question.
Context: {content}
Question: What is python?
"""
)

print("Assistant:", response.output_text)
    
    