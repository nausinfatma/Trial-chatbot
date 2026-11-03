import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

load_dotenv()

api_key = os.getenv("GEMINI_KEY")

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash",
    google_api_key = api_key
)

query = "What is python?"


documents = [
    "Python is a high level programming language.",
    "It is widely used in web development and data science.",
    "The sky is blue.",
    "Python is also popular for machine learning.",
    "The weather is sunny today.",
    "It has a simple and readable syntax."
]

vectorspace = Chroma(
    collection_name= "docs"
)

vectorspace.add_texts(documents)

retriever = vectorspace.as_retriever(
    search_kwargs = {"k" : 3}
)

info = retriever.invoke(query)
information = ''

for x in info:
    information += x.page_content + '\n'
    
    
prompt = ChatPromptTemplate.from_template(
    """
    Answer only from the information given
    
    info: {information}
    question: {query}
    """
)

chain = prompt | llm 

response = chain.invoke({
    "information" : information,
    "query" : query
})

print(response.text)