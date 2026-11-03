import os
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
api_key = os.getenv("GEMINI_KEY")
client = genai.Client(api_key = api_key)

class programming(BaseModel):
    name: str
    difficulty: str

prev_id = None

while True:
    user_input = input("User: ")
    
    if user_input.lower() == "quit":
        break
    
    if prev_id is None:
        response = client.interactions.create(
            model = "gemini-3.6-flash",
            system_instruction="You're a python developer and professor. Add examples to the parts you'll teach.",
            input = user_input,
            response_format = {
                "type": "text",
                "mime_type": "application/json",
                "schema": programming.model_json_schema()
            }
        )
        
    else:
        response = client.interactions.create(
                    model = "gemini-3.6-flash",
                    input = user_input,
                    previous_interaction_id = prev_id,
                    response_format = {
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": programming.model_json_schema()
                    }
                )
        
    prev_id = response.id
        
    print("Assistant:", response.output_text)
    
     