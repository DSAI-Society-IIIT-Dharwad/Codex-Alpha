from typing import Dict, Any
from .shared import GraphState, llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage

# Initialize memory storage for sessions
session_memories = {}

def get_or_create_memory(session_id: str) -> ConversationBufferMemory:
    """Get existing memory for session or create new one"""
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
    return session_memories[session_id]

def constitutional_ai_agent(state: GraphState) -> Dict[str, Any]:
    """Constitutional AI Agent - handles constitutional law queries with memory"""
    
    # Get or create memory for this session
    memory = get_or_create_memory(state.session_id)
    
    # Retrieve conversation history
    chat_history = memory.chat_memory.messages
    
    # Format chat history for the prompt
    history_text = ""
    if chat_history:
        history_text = "\n\nPrevious Conversation:\n"
        for message in chat_history[-6:]:  # Keep last 6 messages for context
            if isinstance(message, HumanMessage):
                history_text += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                history_text += f"Assistant: {message.content}\n"
    
    constitutional_prompt = ChatPromptTemplate.from_template("""
    You are a Constitutional AI Agent, an expert in constitutional law, fundamental rights, and constitutional interpretation.

    Your expertise includes:
    - Indian Constitution and its articles
    - Fundamental Rights (Articles 12-35)
    - Directive Principles of State Policy
    - Constitutional amendments
    - Judicial review and constitutional interpretation
    - Landmark constitutional cases
    - Civil liberties and constitutional protections

    {chat_history}

    Current User Query: {query}

    Please provide a comprehensive, accurate, and well-structured answer. Include relevant constitutional articles, legal principles, and examples where appropriate. Use clear language while maintaining legal accuracy.
    
    If this query relates to previous questions in our conversation, reference that context appropriately and build upon previous discussions.

    Answer:""")
    
    try:
        constitutional_chain = constitutional_prompt | llm | StrOutputParser()
        response = constitutional_chain.invoke({
            "query": state.query,
            "chat_history": history_text
        })
        
        # Save the current interaction to memory
        memory.chat_memory.add_user_message(state.query)
        memory.chat_memory.add_ai_message(response)
        
        formatted_response = f"""🏛️ **Constitutional AI Agent Response:**
        
{response}

---
**Agent Specialization:** Constitutional Law, Fundamental Rights, Constitutional Interpretation
**Session ID:** {state.session_id}"""
        
        return {
            "response": formatted_response,
            "agent_route": state.agent_route,
            "session_id": state.session_id,
            "memory_updated": True
        }
    
    except Exception as e:
        return {
            "response": f"🏛️ **Constitutional AI Agent** encountered an error: {str(e)}\n\nPlease try rephrasing your constitutional law question.",
            "agent_route": state.agent_route,
            "session_id": state.session_id,
            "memory_updated": False
        }

def clear_session_memory(session_id: str) -> bool:
    """Clear memory for a specific session"""
    if session_id in session_memories:
        del session_memories[session_id]
        return True
    return False

def get_session_history(session_id: str) -> list:
    """Get conversation history for a session"""
    if session_id in session_memories:
        return session_memories[session_id].chat_memory.messages
    return []