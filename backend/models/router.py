import os
import json
from datetime import datetime
import uuid
from typing import Dict, Any, Literal, Optional
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import re

# Importing the agents
from .legal_court_agent_main import legal_court_agent
from .constitutional_agent_main import constitutional_ai_agent

# Load environment variables
load_dotenv()
# Global memory storage for sessions
session_memories = {}

# Define the routing decision model
class RouterDecision(BaseModel):
    agent: Literal["constitutional_ai", "legal_court"]
    confidence: float
    reasoning: str

# Updated Graph state with memory support
class GraphState(BaseModel):
    query: str
    agent_route: str = ""
    response: str = ""
    routing_reasoning: str = ""
    session_id: str = ""
    memory_context: str = ""
    
    def __init__(self, **data):
        if not data.get('session_id'):
            data['session_id'] = str(uuid.uuid4())
        super().__init__(**data)

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")  # type: ignore
)

# Memory management functions
def get_or_create_memory(session_id: str) -> ConversationBufferMemory:
    """Get existing memory for session or create new one"""
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
    return session_memories[session_id]

def get_memory_context(session_id: str, limit: int = 6) -> str:
    """Get formatted conversation history for context"""
    if session_id not in session_memories:
        return ""
    
    memory = session_memories[session_id]
    messages = memory.chat_memory.messages
    
    if not messages:
        return ""
    
    # Get last 'limit' messages
    recent_messages = messages[-limit:]
    context = "Previous conversation context:\n"
    
    for message in recent_messages:
        if isinstance(message, HumanMessage):
            context += f"User: {message.content[:200]}...\n"
        elif isinstance(message, AIMessage):
            # Extract just the main response, not formatting
            content = message.content
            if "**Agent Response:**" in content:
                content = content.split("**Agent Response:**")[1].split("---")[0] #type: ignore
            context += f"Assistant: {content[:200]}...\n"
    
    return context

def update_memory(session_id: str, user_query: str, agent_response: str):
    """Update session memory with new interaction"""
    memory = get_or_create_memory(session_id)
    memory.chat_memory.add_user_message(user_query)
    memory.chat_memory.add_ai_message(agent_response)

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

def create_router_node():
    """Creates the router node that analyzes queries and routes to appropriate agents"""
    routing_prompt = ChatPromptTemplate.from_template("""
    You are an intelligent query router that analyzes user queries and determines which specialized agent should handle them.

    Available Agents:
    1. **Constitutional AI Agent**: Handles queries about constitutional law, fundamental rights, constitutional amendments, judicial review, constitutional interpretation, and civil liberties.
    2. **Legal Court Agent**: Handles queries about court procedures, case law, legal precedents, litigation processes, court systems, legal documentation, and general legal practice.

    Previous conversation context (if any):
    {memory_context}

    Current Query: {query}

    Based on the current query and previous context, determine which agent should handle this query.
    Respond with only the agent name: constitutional_ai or legal_court

    Agent:""")

    def router_node(state: GraphState) -> Dict[str, Any]:
        """Router node function that analyzes the query and determines routing"""
        query = state.query
        session_id = state.session_id
        
        # Get memory context for routing decision
        memory_context = get_memory_context(session_id, limit=4)
        
        # Create chain for routing decision
        routing_chain = routing_prompt | llm | StrOutputParser()

        try:
            agent_decision = routing_chain.invoke({
                "query": query,
                "memory_context": memory_context
            }).strip().lower()

            # Validate the decision
            valid_agents = ["constitutional_ai", "legal_court"]
            if agent_decision not in valid_agents:
                agent_decision = keyword_based_routing(query)

            reasoning = generate_routing_reasoning(query, agent_decision, bool(memory_context))

            return {
                "agent_route": agent_decision,
                "routing_reasoning": reasoning,
                "memory_context": memory_context
            }

        except Exception as e:
            agent_decision = keyword_based_routing(query)
            reasoning = f"Fallback routing based on keywords due to error: {str(e)}"

            return {
                "agent_route": agent_decision,
                "routing_reasoning": reasoning,
                "memory_context": memory_context
            }

    return router_node

def keyword_based_routing(query: str) -> str:
    """Fallback keyword-based routing logic"""
    query_lower = query.lower()

    # Constitutional AI keywords
    constitutional_keywords = [
        "constitution", "constitutional", "fundamental rights", "amendment",
        "judicial review", "civil liberties", "article", "preamble",
        "directive principles", "constitutional interpretation", "supreme court constitution"
    ]

    # Legal Court keywords
    legal_court_keywords = [
        "court", "case law", "precedent", "litigation", "lawsuit", "trial",
        "judge", "jurisdiction", "appeal", "legal procedure", "evidence",
        "contract law", "criminal law", "civil law", "legal document"
    ]

    # Count keyword matches
    constitutional_score = sum(1 for keyword in constitutional_keywords if keyword in query_lower)
    legal_court_score = sum(1 for keyword in legal_court_keywords if keyword in query_lower)

    # Determine best match
    if constitutional_score > legal_court_score:
        return "constitutional_ai"
    else:
        return "legal_court"

def generate_routing_reasoning(query: str, agent: str, has_memory: bool) -> str:
    """Generate reasoning for why a particular agent was chosen"""
    reasoning_map = {
        "constitutional_ai": "Query involves constitutional law, fundamental rights, or constitutional interpretation",
        "legal_court": "Query involves court procedures, case law, or general legal practice"
    }
    
    base_reasoning = reasoning_map.get(agent, "Query routed based on content analysis")
    
    if has_memory:
        base_reasoning += " (considered conversation context)"
    
    return base_reasoning

def route_to_agent(state: GraphState) -> str:
    """Conditional routing function for the graph"""
    return state.agent_route

def create_routing_graph():
    """Creates the complete LangGraph with router and agents"""
    workflow = StateGraph(GraphState)

    # Add nodes
    router_node = create_router_node()
    workflow.add_node("router", router_node)
    workflow.add_node("constitutional_ai", constitutional_ai_agent)
    workflow.add_node("legal_court", legal_court_agent)

    # Set entry point
    workflow.set_entry_point("router")

    # Add conditional routing edges
    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "constitutional_ai": "constitutional_ai",
            "legal_court": "legal_court"
        }
    )

    # Add edges to END
    workflow.add_edge("constitutional_ai", END)
    workflow.add_edge("legal_court", END)

    return workflow.compile()

def get_user_input(session_id=None):
    """Get user input with session management and memory commands"""
    if not session_id:
        session_id = str(uuid.uuid4())
        print(f"🆔 New session started: {session_id[:8]}...")
    
    print("=" * 80)
    print("🤖 **INTELLIGENT LEGAL AI ROUTER SYSTEM WITH MEMORY** 🤖")
    print("=" * 80)
    print(f"📱 Session ID: {session_id[:8]}...")
    
    # Show memory status
    memory_count = len(get_session_history(session_id)) // 2  # Divide by 2 for user/assistant pairs
    print(f"🧠 Conversation turns: {memory_count}")
    
    print("\n📋 **Available Commands:**")
    print(" 💬 Ask any legal question")
    print(" 🧹 Type 'clear' to clear conversation memory")
    print(" 📜 Type 'history' to view conversation history")
    print(" 🔄 Type 'new' to start a new session")
    print(" 🚪 Type 'quit' to exit")
    
    while True:
        user_input = input("\n💬 **Enter your question or command**: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            return None, session_id
        elif user_input.lower() == 'clear':
            if clear_session_memory(session_id):
                print("✅ Conversation memory cleared!")
            else:
                print("ℹ️ No memory to clear for this session.")
            continue
        elif user_input.lower() == 'new':
            new_session_id = str(uuid.uuid4())
            print(f"🆕 New session started: {new_session_id[:8]}...")
            return get_user_input(new_session_id)
        elif user_input.lower() == 'history':
            history = get_session_history(session_id)
            if history:
                print("\n📜 **Conversation History:**")
                for i in range(0, len(history), 2):
                    turn_num = (i // 2) + 1
                    if i < len(history):
                        user_msg = history[i].content[:150] + "..." if len(history[i].content) > 150 else history[i].content
                        print(f"{turn_num}. User: {user_msg}")
                    if i + 1 < len(history):
                        ai_msg = history[i + 1].content[:150] + "..." if len(history[i + 1].content) > 150 else history[i + 1].content
                        print(f"   Assistant: {ai_msg}")
            else:
                print("ℹ️ No conversation history for this session.")
            continue
        elif user_input:
            return user_input, session_id
        else:
            print("❌ Please enter a valid question or command.")

def extract_clean_response(formatted_response: str) -> str:
    """Extract clean LLM response from formatted agent output"""
    
    # Split by lines to process
    lines = formatted_response.split('\n')
    clean_lines = []
    found_content = False
    
    # Skip agent headers and stop at disclaimers/metadata
    for line in lines:
        # Skip headers
        if any(header in line for header in [
            "🏛️ **Constitutional AI Agent Response:**",
            "⚖️ **Legal Educational Assistant Response:**",
            "**Constitutional AI Agent Response:**",
            "**Legal Educational Assistant Response:**",
            "**Constitutional AI Agent** encountered an error:",
            "**Legal Educational Assistant** encountered an error:"
        ]):
            found_content = True
            continue
            
        # Stop at metadata/disclaimers
        if any(stopper in line for stopper in [
            "**📚 Sources Referenced",
            "**🧠 Memory Status",
            "**⚠️ Important Disclaimer",
            "**📊 Vector Store",
            "---",
            "**Agent Specialization:"
        ]):
            break
        
        # If we found content start, collect non-empty lines that aren't metadata
        if found_content and line.strip():
            # Skip lines that are clearly formatting/metadata
            if not any(skip in line for skip in [
                "**", "📚", "🧠", "⚠️", "📊", "---"
            ]):
                clean_lines.append(line)
        # If no header found, assume it's already clean content
        elif not found_content and line.strip():
            # Check if this line contains actual content (not just formatting)
            if not any(skip in line for skip in [
                "**📚", "**🧠", "**⚠️", "**📊", "---", "**Agent Specialization"
            ]):
                clean_lines.append(line)
    
    # If no clean lines found, try to extract everything before first metadata marker
    if not clean_lines:
        for line in lines:
            if any(stopper in line for stopper in [
                "**📚", "**🧠", "**⚠️", "**📊", "---", "**Agent Specialization"
            ]):
                break
            if line.strip() and not line.strip().startswith("🏛️") and not line.strip().startswith("⚖️"):
                clean_lines.append(line)
    
    # Join and clean up
    clean_response = '\n'.join(clean_lines).strip()
    
    # Remove any remaining formatting artifacts
    clean_response = clean_response.replace("🏛️", "").replace("⚖️", "").strip()
    
    # If still empty, return the original response
    if not clean_response:
        clean_response = formatted_response
    
    return clean_response

def get_results_as_json(result):
    """Return only the clean LLM response"""
    
    # Extract clean response
    clean_response = extract_clean_response(result['response'])
    
    output_data = {
        "message": clean_response,
        "agent": result.get("agent_route", ""),
        "reasoning": result.get("routing_reasoning", ""),
        "session_id": result.get("session_id", ""),
    }
    
    return output_data

def display_results(result):
    """Display only the clean LLM response in JSON format"""
    json_data = get_results_as_json(result)
    print(json.dumps(json_data, indent=2, ensure_ascii=False))

def main():
    """Main interactive function with memory support"""
    print("🚀 Initializing Intelligent Legal AI Router System with Memory...")

    try:
        # Create the graph
        app = create_routing_graph()
        print("✅ System initialized successfully!")
        
        session_id = None

        while True:
            # Get user input
            user_query, session_id = get_user_input(session_id)

            if user_query is None:
                print(f"\n👋 Session {session_id[:8]} ended. Thank you for using the system!")
                break

            print(f"\n🔄 Analyzing and processing your query...")
            print(f"📝 Query: '{user_query}'")

            try:
                # Create initial state with session ID
                initial_state = GraphState(
                    query=user_query,
                    session_id=session_id
                )

                # Run the graph
                print("⚡ Routing to appropriate agent...")
                result = app.invoke(initial_state)

                # Extract clean response for memory storage
                clean_response = extract_clean_response(result['response'])

                # Update memory with the clean interaction
                update_memory(session_id, user_query, clean_response)

                # Display results (only clean response)
                display_results(result)

            except Exception as e:
                print(f"\n❌ **Error processing query:** {str(e)}")
                print("Please try again with a different question.")

            # Ask if user wants to continue
            continue_choice = input("\n🔄 Would you like to ask another question? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                print(f"\n👋 Session {session_id[:8]} ended. Thank you!")
                break

    except Exception as e:
        print(f"\n❌ **System initialization error:** {str(e)}")
        print("Please check your API key and internet connection.")

# Test function with memory
def test_agents_with_memory():
    """Test function to demonstrate memory capabilities"""
    test_queries = [
        {
            "query": "What are the fundamental rights guaranteed by the Indian Constitution?",
            "expected_agent": "constitutional_ai"
        },
        {
            "query": "Can you tell me more about the right to equality mentioned earlier?",
            "expected_agent": "constitutional_ai"  # Should remember previous context
        },
        {
            "query": "How do I file an appeal in the High Court?",
            "expected_agent": "legal_court"
        }
    ]

    app = create_routing_graph()
    session_id = str(uuid.uuid4())

    print("\n🧪 **TESTING MEMORY-ENABLED AGENT RESPONSES**")
    print("=" * 60)

    for i, test in enumerate(test_queries, 1):
        print(f"\n--- Test {i} ---")
        print(f"Query: {test['query']}")
        
        initial_state = GraphState(
            query=test['query'],
            session_id=session_id
        )
        
        result = app.invoke(initial_state)
        
        # Extract clean response and update memory
        clean_response = extract_clean_response(result['response'])
        update_memory(session_id, test['query'], clean_response)
        
        print(f"Routed to: {result['agent_route']}")
        print(f"Expected: {test['expected_agent']}")
        print(f"✅ Correct routing: {result['agent_route'] == test['expected_agent']}")
        print(f"Memory context used: {bool(result.get('memory_context'))}")
        print(f"Clean response preview: {clean_response[:200]}...")
        print("-" * 60)

# Create routing app instance
routing_app = create_routing_graph()

def process_query(user_query: str, session_id: str): 
    """Process a query using the routing graph + memory and return clean response"""
    print(f"process_query called with query='{user_query}' and session='{session_id}'")
    if not session_id:
        session_id = str(uuid.uuid4())

    initial_state = GraphState(
        query=user_query,
        session_id=session_id
    )

    result = routing_app.invoke(initial_state)

    # Extract clean response
    clean_response = extract_clean_response(result['response'])

    # Update memory with clean response
    try:
        update_memory(session_id, user_query, clean_response)
    except Exception as e:
        print("Error updating memory:", e)

    return {
        "response": clean_response,  # Only the clean LLM response
        "agent": result['agent_route'],
        "reasoning": result.get("routing_reasoning", ""),
        "session_id": session_id,
    }

if __name__ == "__main__":
    # Run the main interactive system
    main()
    
    # Uncomment to run memory tests instead
    # test_agents_with_memory()