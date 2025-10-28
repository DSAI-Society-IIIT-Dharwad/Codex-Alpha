import os
import uuid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Define your graph state
class GraphState(BaseModel):
    query: str
    agent_route: str = ""
    response: str = ""
    routing_reasoning: str = ""
    session_id: str = ""
    memory_updated: bool = False

    def __init__(self, **data):
        if not data.get('session_id'):
            data['session_id'] = str(uuid.uuid4())
        super().__init__(**data)

# Initialize LLM (Chat model)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")  # type: ignore
)