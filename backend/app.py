from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from pydantic import BaseModel
from config.database import User, ChatMessage # user_collection,  chat_collection,
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os, sys, uuid
from bson import ObjectId
from datetime import datetime
from models.router import process_query


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

origins = [
    '*'
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # your React dev server (Vite) or http://localhost:3000 if CRA
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# @app.post("/api/user")
# def create_user(user: User):
#     user_dict = user.dict() #type: ignore
#     result = user_collection.insert_one(user_dict)
#     return {"user_id": str(result.inserted_id), "status": "User created"}

# @app.post("/api/chat/{chat_id}")
# def create_chat_message(chat_id: str, message: ChatMessage):
#     message_dict = message.dict()   #type: ignore
#     message_dict["chat_id"] = ObjectId(chat_id)

#     message_dict["timestamp"] = datetime.utcnow().isoformat() #type: ignore

#     result = chat_collection.insert_one(message_dict)

#     return {
#         "message_id": str(result.inserted_id),
#         "chat_id": chat_id,
#         "status": "Message added to chat"
#     }

@app.get("/")
def health_check():
    return {"status": "ok"}
    
@app.post("/api/chat")
async def chat(
    message: str = Form(...),
    sender: str = Form(...),
    file: UploadFile = File(None),
    session_id: str = Form(None)
):
    print("Message:", message)
    print("Sender:", sender)
    if file:
        print("File received:", file.filename)

    try:
        result = process_query(message, session_id)
        print("process_query result:", result)
    except Exception as e:
        print("Error in process_query:", e)
        return {
            "reply": "⚠️ Sorry, something went wrong while processing your query.",
            "agent": "system",
            "reasoning": str(e),
            "session_id": session_id or "unknown"
        }
    final_session_id = result.get("session_id") or session_id or str(uuid.uuid4())
    # Ensure response format
    return {
        "reply": result.get("response", "No response generated."),
        "agent": result.get("agent", "default"),
        "reasoning": result.get("reasoning", ""),
        "session_id": final_session_id
    }
