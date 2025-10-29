import pymongo
from pydantic import BaseModel, Field
import datetime
from datetime import datetime

# myclient = pymongo.MongoClient("mongodb://localhost:27017/")

# mydb = myclient["legal-nav"]
# user_collection = mydb["user_collection"]
# chat_collection = mydb["chat_collection"]

"""{
  "_id": ObjectId("66d99b56a1"),   // unique chat id
  "chatname": "Project Discussion 1",
  "participants": ["saksham", "assistant"],
  "created_at": ISODate("2025-08-31T12:00:00Z"),
  "last_message": "Sure, I’ll send the file.",  // optional, for sidebar preview
  "last_updated": ISODate("2025-08-31T12:05:00Z")
}
{
  "_id": ObjectId("66d99c01b3"),
  "chat_id": ObjectId("66d99b56a1"),   // points to chats._id
  "sender": "saksham",
  "role": "user",                       // could also be "assistant"
  "text": "Hello, how are you?",
  "timestamp": ISODate("2025-08-31T12:01:00Z")
}"""

class User(BaseModel):
    username: str
    # created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat()) #type: ignore

class ChatMessage(BaseModel):
    user_id: str
    sender: str
    chatname: str
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat()) #type: ignore

