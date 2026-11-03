import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

api_key = os.getenv("GEMINI_KEY")

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash",
    google_api_key = api_key
)

history = []
    
while True:
    query = input("User: ")
    
    if query == 'quit':
        break
    
    prompt = ChatPromptTemplate.from_template(
    """
    Answer the question in 1 to 4 lines. Answer like you're a software developer, considering the previous messages too:
    
    previous : {history}
    
    question: {query}
    """
)
    
    chain = prompt | llm 

    response = chain.invoke({
        "history" : history,
        "query" : query
    })

    history.append(f"""
                    User: {query}
                    System : {response.content}
                    """)
    
    print(response.text)


