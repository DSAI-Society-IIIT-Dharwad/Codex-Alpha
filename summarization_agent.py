import os
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import operator
from dotenv import load_dotenv
load_dotenv()
# Set your Groq API key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Define the state structure
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    knowledge_base_path: str
    query: str
    documents: List[str]
    retrieved_chunks: List[str]
    summaries: Annotated[List[str], operator.add]
    final_summary: str
    chunk_size: int
    top_k: int


# Initialize the LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=2000
)

# ----------------------------
# 1. Load and chunk the knowledge base
# ----------------------------
def load_and_chunk(state: GraphState) -> GraphState:
    print("📘 Loading and chunking knowledge base...")
    with open(state["knowledge_base_path"], "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=state["chunk_size"],
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)
    print(f"✅ Created {len(chunks)} text chunks.")
    state["documents"] = chunks
    return state


# ----------------------------
# 2. Build FAISS Vector Store and retrieve
# ----------------------------
def retrieve_chunks(state: GraphState) -> GraphState:
    print("🔍 Creating embeddings and retrieving relevant chunks...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_texts(state["documents"], embedding=embeddings)
    docs = db.similarity_search(state["query"], k=state["top_k"])

    retrieved_chunks = [doc.page_content for doc in docs]
    print(f"✅ Retrieved top {len(retrieved_chunks)} relevant chunks.")
    state["retrieved_chunks"] = retrieved_chunks
    return state


# ----------------------------
# 3. Summarize each retrieved chunk
# ----------------------------
def summarize_chunks(state: GraphState) -> GraphState:
    print("🧠 Summarizing each retrieved chunk...")
    prompt = PromptTemplate(
        input_variables=["chunk"],
        template=(
            "Summarize the following legal text in concise, factual language:\n\n"
            "{chunk}\n\nSummary:"
        ),
    )
    summaries = []
    for idx, chunk in enumerate(state["retrieved_chunks"], start=1):
        chain_input = prompt.format(chunk=chunk)
        response = llm.invoke(chain_input)
        summary_text = response.content if hasattr(response, "content") else str(response)
        print(f"   ➤ Chunk {idx} summarized.")
        summaries.append(summary_text)
    state["summaries"] = summaries
    return state


# ----------------------------
# 4. Combine individual summaries into a final answer
# ----------------------------
def combine_summaries(state: GraphState) -> GraphState:
    print("🧩 Combining all summaries into a single concise output...")
    combined_text = "\n\n".join(state["summaries"])
    prompt = PromptTemplate(
        input_variables=["summaries", "query"],
        template=(
            "You are a legal summarization expert.\n"
            "Based on the following partial summaries, produce a single concise, coherent summary "
            "that answers the user's question accurately and references legal concepts clearly.\n\n"
            "User Question:\n{query}\n\nPartial Summaries:\n{summaries}\n\nFinal Answer:"
        ),
    )

    chain_input = prompt.format(query=state["query"], summaries=combined_text)
    response = llm.invoke(chain_input)
    final_summary = response.content if hasattr(response, "content") else str(response)
    print("✅ Final summary generated.")
    state["final_summary"] = final_summary
    return state


# ----------------------------
# 5. Build LangGraph pipeline
# ----------------------------
def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("load_and_chunk", load_and_chunk)
    workflow.add_node("retrieve_chunks", retrieve_chunks)
    workflow.add_node("summarize_chunks", summarize_chunks)
    workflow.add_node("combine_summaries", combine_summaries)

    workflow.set_entry_point("load_and_chunk")
    workflow.add_edge("load_and_chunk", "retrieve_chunks")
    workflow.add_edge("retrieve_chunks", "summarize_chunks")
    workflow.add_edge("summarize_chunks", "combine_summaries")
    workflow.add_edge("combine_summaries", END)

    return workflow.compile()


# ----------------------------
# 6. Main Execution
# ----------------------------
if __name__ == "__main__":
    print("\n🚀 LangGraph RAG Summarization Agent (Groq LLM)\n")
    kb_path = "kaanoon_all_answers.txt"

    if not os.path.exists(kb_path):
        print(f"❌ Knowledge base file not found: {kb_path}")
        exit(1)

    user_query = input("Enter your legal query: ").strip()
    if not user_query:
        print("❌ No query entered. Exiting.")
        exit(1)

    initial_state: GraphState = {
        "knowledge_base_path": kb_path,
        "query": user_query,
        "documents": [],
        "retrieved_chunks": [],
        "summaries": [],
        "final_summary": "",
        "chunk_size": 1200,
        "top_k": 6,
    }

    app = build_graph()
    final_state = app.invoke(initial_state)

    print("\n==============================")
    print("📜 FINAL SUMMARY:")
    print("==============================\n")
    print(final_state["final_summary"])
