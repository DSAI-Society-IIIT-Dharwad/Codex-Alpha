from .shared import llm, GraphState
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import re
import hashlib
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Memory storage for RAG agent
rag_session_memories = {}

def get_rag_memory(session_id: str) -> ConversationBufferMemory:
    """Get or create memory for RAG agent sessions"""
    if session_id not in rag_session_memories:
        rag_session_memories[session_id] = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
    return rag_session_memories[session_id]

def get_rag_context(session_id: str, limit: int = 4) -> str:
    """Get conversation context for RAG agent"""
    if not session_id or session_id not in rag_session_memories:
        return ""
    
    memory = rag_session_memories[session_id]
    messages = memory.chat_memory.messages
    
    if not messages:
        return ""
    
    recent_messages = messages[-limit:]
    context = ""
    
    for message in recent_messages:
        if isinstance(message, HumanMessage):
            context += f"Previous User Question: {message.content}\n"
        elif isinstance(message, AIMessage):
            content = message.content
            if len(content) > 200:
                content = content[:200] + "..." #type: ignore
            context += f"Previous Response Summary: {content}\n"
    
    return context

def extract_section_headers(text: str) -> List[Tuple[str, int]]:
    """Extract section headers and their positions from legal document text"""
    patterns = [
        r'^([A-Z\s]{5,}):',
        r'^(\d+\.\s*[A-Z][^.]*):',
        r'^(SECTION\s+\d+[^:\n]*):',
        r'^(CHAPTER\s+[IVX]+[^:\n]*):',
        r'^(ARTICLE\s+\d+[^:\n]*):',
        r'^([A-Z][a-z\s]+\s*\([a-z]\)[^:\n]*):',
    ]
    
    headers = []
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
        for match in matches:
            headers.append((match.group(1).strip(), match.start()))
    
    headers.sort(key=lambda x: x[1])
    return headers

def redact_pii(text: str) -> str:
    """Basic PII redaction for legal documents"""
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REDACTED]', text)
    text = re.sub(r'\+91[-\s]?\d{10}\b', '[PHONE_REDACTED]', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]', text)
    text = re.sub(r'\b\d{4}\s?\d{4}\s?\d{4}\b', '[AADHAR_REDACTED]', text)
    text = re.sub(r'\b[A-Z]{5}\d{4}[A-Z]\b', '[PAN_REDACTED]', text)
    return text

def mmr_selection(query_embedding: List[float], doc_embeddings: List[List[float]], 
                  documents: List[str], k: int = 5, lambda_param: float = 0.7) -> List[int]:
    """Maximal Marginal Relevance selection for diverse document retrieval"""
    if not doc_embeddings or k <= 0:
        return []
    
    query_embedding = np.array(query_embedding).reshape(1, -1)
    doc_embeddings = np.array(doc_embeddings)
    
    query_similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    selected_indices = []
    remaining_indices = list(range(len(documents)))
    
    first_idx = remaining_indices[np.argmax(query_similarities)]
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    for _ in range(min(k - 1, len(remaining_indices))):
        mmr_scores = []
        
        for idx in remaining_indices:
            query_sim = query_similarities[idx]
            
            if selected_indices:
                selected_embeddings = doc_embeddings[selected_indices]
                doc_similarities = cosine_similarity(
                    doc_embeddings[idx:idx+1], selected_embeddings
                )[0]
                max_doc_sim = max(doc_similarities)
            else:
                max_doc_sim = 0
            
            mmr_score = lambda_param * query_sim - (1 - lambda_param) * max_doc_sim
            mmr_scores.append(mmr_score)
        
        if mmr_scores:
            best_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
    
    return selected_indices

class LegalCourtRAGAgent:
    def __init__(self, documents_directory: str = "./legal_documents", 
                 persist_directory: str = "./chroma_storage",
                 knowledge_base_file: str = "./rag_knowledge_base.txt"):
        """Initialize the Legal Court RAG Agent with enhanced vector store, embeddings and knowledge base"""
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize ChromaDB client with persistence
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection_name = "indian_legal_documents"
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection with {self.collection.count()} documents")
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Indian legal documents, case law, and knowledge base with enhanced metadata"}
            )
            print("Created new collection")
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""]
        )
        
        self.documents_directory = documents_directory
        self.knowledge_base_file = knowledge_base_file
        
        # Load knowledge base first
        self.load_knowledge_base()
        
        # Load documents if directory exists
        if os.path.exists(documents_directory) and self.collection.count() == 0:
            self.load_documents_from_directory()

    def load_knowledge_base(self) -> None:
        """Load and process the knowledge base text file"""
        if not os.path.exists(self.knowledge_base_file):
            print(f"Knowledge base file {self.knowledge_base_file} not found. Creating sample file...")
            self.create_sample_knowledge_base()
            return
        
        try:
            print(f"Loading knowledge base from {self.knowledge_base_file}...")
            
            with open(self.knowledge_base_file, 'r', encoding='utf-8') as f:
                knowledge_text = f.read()
            
            if not knowledge_text.strip():
                print("Knowledge base file is empty")
                return
            
            # Check if knowledge base is already loaded
            existing_kb = self.collection.get(
                where={"source_type": "knowledge_base"}
            )
            
            if existing_kb['documents']:
                print(f"Knowledge base already loaded with {len(existing_kb['documents'])} entries")
                return
            
            # Split knowledge base into chunks
            kb_chunks = self.parse_knowledge_base(knowledge_text)
            
            if kb_chunks:
                print(f"Processing {len(kb_chunks)} knowledge base entries...")
                self.add_knowledge_base_chunks(kb_chunks)
            else:
                print("No valid knowledge base entries found")
                
        except Exception as e:
            print(f"Error loading knowledge base: {str(e)}")

    def create_sample_knowledge_base(self) -> None:
        """Load knowledge base from existing legal_knowledge_base.txt file"""
        try:
            if os.path.exists("legal_knowledge_base.txt"):
                with open("legal_knowledge_base.txt", 'r', encoding='utf-8') as f:
                    content = f.read()
                if content.strip():
                    with open(self.knowledge_base_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.load_knowledge_base()
        except Exception as e:
            print(f"Error loading knowledge base: {str(e)}")

    def parse_knowledge_base(self, text: str) -> List[Dict[str, str]]:
        """Parse the knowledge base text file into structured Q&A pairs"""
        kb_entries = []
        
        # Split by major sections (##) and questions (###)
        sections = re.split(r'^##\s+(.+?)$', text, flags=re.MULTILINE)
        
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                section_title = sections[i].strip()
                section_content = sections[i + 1].strip()
                
                # Split questions within the section
                questions = re.split(r'^###\s+(.+?)$', section_content, flags=re.MULTILINE)
                
                for j in range(1, len(questions), 2):
                    if j + 1 < len(questions):
                        question = questions[j].strip()
                        answer = questions[j + 1].strip()
                        
                        if question and answer:
                            kb_entries.append({
                                'section': section_title,
                                'question': question,
                                'answer': answer,
                                'combined_text': f"Section: {section_title}\nQuestion: {question}\nAnswer: {answer}"
                            })
        
        return kb_entries

    def add_knowledge_base_chunks(self, kb_entries: List[Dict[str, str]]) -> None:
        """Add knowledge base entries to the vector store"""
        if not kb_entries:
            return
        
        print(f"Adding {len(kb_entries)} knowledge base entries to vector store...")
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for i, entry in enumerate(kb_entries):
            # Create document text for embedding
            documents.append(entry['combined_text'])
            
            # Create metadata
            metadata = {
                'source': self.knowledge_base_file,
                'source_type': 'knowledge_base',
                'document_title': f"Knowledge Base - {entry['section']}",
                'section': entry['section'],
                'question': entry['question'],
                'kb_entry_index': i,
                'chunk_length': len(entry['combined_text']),
                'created_at': datetime.now().isoformat(),
                'extraction_method': 'knowledge_base_parser'
            }
            
            metadatas.append(metadata)
            
            # Create unique ID
            entry_hash = hashlib.md5(entry['combined_text'].encode()).hexdigest()[:8]
            ids.append(f"kb_{entry_hash}_entry_{i}")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas, #type: ignore
            ids=ids
        )
        
        print(f"Successfully added {len(kb_entries)} knowledge base entries to vector store")

    def create_enhanced_metadata(self, chunk: str, source_file: str, chunk_index: int, 
                                page_info: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        """Create enhanced metadata for document chunks"""
        
        # Extract document title from filename
        document_title = os.path.splitext(source_file)[0].replace('_', ' ').title()
        
        # Try to determine page number from chunk content
        page_number = None
        page_match = re.search(r'--- Page (\d+) ---', chunk)
        if page_match:
            page_number = int(page_match.group(1))
        
        # Extract potential section/heading information
        section_headers = extract_section_headers(chunk)
        section = section_headers[0][0] if section_headers else None
        
        # Calculate chunk hash for deduplication
        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
        
        # Apply PII redaction
        redacted_chunk = redact_pii(chunk)
        is_redacted = redacted_chunk != chunk
        
        metadata = {
            "source": source_file,
            "source_type": "document",
            "document_title": document_title,
            "chunk_index": chunk_index,
            "page_number": page_number,
            "section": section,
            "chunk_length": len(chunk),
            "chunk_hash": chunk_hash,
            "extraction_method": page_info.get("extraction_method", "unknown"),
            "total_pages": page_info.get("total_pages", 0),
            "is_pii_redacted": is_redacted,
            "created_at": datetime.now().isoformat()
        }
        
        return metadata, redacted_chunk

    def load_documents_from_directory(self) -> None:
        """Enhanced document loading with better metadata and persistence"""
        if not os.path.exists(self.documents_directory):
            print(f"Documents directory {self.documents_directory} not found")
            return
        
        pdf_files = list(Path(self.documents_directory).glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.documents_directory}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process with enhanced extraction")
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
            
            # Check if document already exists in collection
            existing_docs = self.collection.get(
                where={"source": pdf_file.name}
            )
            
            if existing_docs['documents']:
                print(f"Document {pdf_file.name} already exists in collection, skipping...")
                continue
            
            # Extract text from PDF with metadata using OCR processor
            document_text, page_info = self.pdf_extractor.extract_text_from_pdf(str(pdf_file), use_ocr=True)
            
            if document_text.strip():
                # Chunk the document
                chunks = self.text_splitter.split_text(document_text)
                print(f"Created {len(chunks)} chunks from {pdf_file.name}")
                
                # Add chunks to vector store with enhanced metadata
                self.add_document_chunks_enhanced(chunks, str(pdf_file.name), page_info)
            else:
                print(f"Warning: No text extracted from {pdf_file.name}")
    
    def add_document_chunks_enhanced(self, chunks: List[str], source_file: str, 
                                   page_info: Dict[str, Any]) -> None:
        """Add document chunks with enhanced metadata to the vector store"""
        if not chunks:
            return
        
        # Filter out very short chunks that might be noise
        filtered_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
        
        if not filtered_chunks:
            print(f"No meaningful chunks found in {source_file}")
            return
        
        # Generate embeddings for chunks
        print(f"Generating embeddings for {len(filtered_chunks)} chunks...")
        embeddings = self.embedding_model.encode(filtered_chunks).tolist()
        
        # Create enhanced metadata for chunks
        chunk_ids = []
        metadatas = []
        processed_chunks = []
        
        for i, chunk in enumerate(filtered_chunks):
            metadata, redacted_chunk = self.create_enhanced_metadata(
                chunk, source_file, i, page_info
            )
            
            chunk_id = f"{source_file}_{metadata['chunk_hash'][:8]}_chunk_{i}"
            
            chunk_ids.append(chunk_id)
            metadatas.append(metadata)
            processed_chunks.append(redacted_chunk)
        
        # Add to ChromaDB collection
        self.collection.add(
            embeddings=embeddings,
            documents=filtered_chunks,
            metadatas=metadatas, #type: ignore
            ids=chunk_ids
        )
        
        print(f"Successfully added {len(filtered_chunks)} chunks from {source_file} to persistent vector store")
    
    def retrieve_relevant_documents_mmr(self, query: str, k: int = 5, 
                                      fetch_k: int = 15, lambda_param: float = 0.7,
                                      include_knowledge_base: bool = True) -> List[Dict]:
        """Enhanced retrieval using MMR for diversity and better relevance, including knowledge base"""
        try:
            if self.collection.count() == 0:
                print("Warning: Vector store is empty. Please add documents first.")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Determine search filters
            where_filter = None
            if not include_knowledge_base:
                where_filter = {"source_type": {"$ne": "knowledge_base"}}
            
            # Fetch more documents than needed for MMR selection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(fetch_k, self.collection.count()),
                include=['documents', 'metadatas', 'distances', 'embeddings'],
                where=where_filter
            )
            
            if not results['documents'] or not results['documents'][0]:
                return []
            
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            embeddings = results['embeddings'][0] if 'embeddings' in results else None
            
            # Apply MMR selection if we have embeddings
            if embeddings and len(documents) > k:
                selected_indices = mmr_selection(
                    query_embedding, embeddings, documents, k, lambda_param
                )
            else:
                selected_indices = list(range(min(k, len(documents))))
            
            # Format results with enhanced metadata
            relevant_docs = []
            for idx in selected_indices:
                doc_metadata = metadatas[idx]
                
                # Create citation information based on source type
                if doc_metadata.get('source_type') == 'knowledge_base':
                    citation = f"Knowledge Base - {doc_metadata.get('section', 'General')}"
                    if doc_metadata.get('question'):
                        citation += f": {doc_metadata['question']}"
                else:
                    citation = f"{doc_metadata.get('document_title', 'Unknown')}"
                    if doc_metadata.get('page_number'):
                        citation += f", Page {doc_metadata['page_number']}"
                    if doc_metadata.get('section'):
                        citation += f", Section: {doc_metadata['section']}"
                
                relevant_docs.append({
                    'content': documents[idx],
                    'source': doc_metadata.get('source', 'unknown'),
                    'source_type': doc_metadata.get('source_type', 'document'),
                    'document_title': doc_metadata.get('document_title', 'Unknown'),
                    'page_number': doc_metadata.get('page_number'),
                    'section': doc_metadata.get('section'),
                    'question': doc_metadata.get('question'),
                    'distance': distances[idx],
                    'citation': citation,
                    'extraction_method': doc_metadata.get('extraction_method', 'unknown'),
                    'is_pii_redacted': doc_metadata.get('is_pii_redacted', False)
                })
            
            return relevant_docs
        
        except Exception as e:
            print(f"Error retrieving documents with MMR: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the current collection including knowledge base"""
        try:
            count = self.collection.count()
            
            if count == 0:
                return {
                    "total_chunks": 0,
                    "unique_sources": 0,
                    "sources": [],
                    "knowledge_base_entries": 0,
                    "pdf_documents": 0,
                    "extraction_methods": {},
                    "total_pages": 0,
                    "redacted_chunks": 0
                }
            
            # Get all metadata for analysis
            all_results = self.collection.get(
                limit=count,
                include=['metadatas']
            )
            
            metadatas = all_results['metadatas']
            sources = set()
            extraction_methods = {}
            total_pages = 0
            redacted_chunks = 0
            knowledge_base_entries = 0
            pdf_documents = 0
            
            for meta in metadatas:
                sources.add(meta.get('source', 'unknown'))
                
                method = meta.get('extraction_method', 'unknown')
                extraction_methods[method] = extraction_methods.get(method, 0) + 1
                
                if meta.get('source_type') == 'knowledge_base':
                    knowledge_base_entries += 1
                else:
                    pdf_documents += 1
                
                if meta.get('total_pages'):
                    total_pages = max(total_pages, meta['total_pages'])
                
                if meta.get('is_pii_redacted', False):
                    redacted_chunks += 1
            
            return {
                "total_chunks": count,
                "unique_sources": len(sources),
                "sources": list(sources),
                "knowledge_base_entries": knowledge_base_entries,
                "pdf_documents": pdf_documents,
                "extraction_methods": extraction_methods,
                "total_pages_processed": total_pages,
                "redacted_chunks": redacted_chunks,
                "persist_directory": self.persist_directory,
                "knowledge_base_file": self.knowledge_base_file
            }
            
        except Exception as e:
            return {"error": str(e)}

    def legal_court_agent(self, state: GraphState) -> Dict[str, Any]:
        """Enhanced Legal Court Agent with knowledge base integration"""
        
        session_id = getattr(state, 'session_id', '')
        
        # Get conversation context from memory
        memory_context = get_rag_context(session_id, limit=3)
        
        # Get collection statistics
        stats = self.get_collection_stats()
        print(f"Enhanced vector store stats: {stats}")
        
        # Enhanced retrieval with MMR (including knowledge base)
        relevant_docs = self.retrieve_relevant_documents_mmr(
            state.query, k=6, fetch_k=20, lambda_param=0.7, include_knowledge_base=True
        )
        
        # Separate knowledge base and document results
        kb_docs = [doc for doc in relevant_docs if doc['source_type'] == 'knowledge_base']
        pdf_docs = [doc for doc in relevant_docs if doc['source_type'] != 'knowledge_base']
        
        # Prepare enhanced context with precise citations
        rag_context = ""
        citations = []
        
        # Add knowledge base content first
        if kb_docs:
            rag_context += "\n=== KNOWLEDGE BASE INFORMATION ===\n"
            for i, doc in enumerate(kb_docs, 1):
                rag_context += f"\n[KB-{i}] {doc['citation']}\n"
                rag_context += f"Content: {doc['content']}\n"
                rag_context += "---\n"
                
                citations.append({
                    'source': doc['source'],
                    'source_type': 'knowledge_base',
                    'document_title': doc['document_title'],
                    'question': doc.get('question'),
                    'section': doc['section'],
                    'citation': doc['citation']
                })
        
        # Add PDF document content
        if pdf_docs:
            rag_context += "\n=== DOCUMENT CONTENT ===\n"
            for i, doc in enumerate(pdf_docs, len(kb_docs) + 1):
                rag_context += f"\n[Document {i}] {doc['citation']}\n"
                rag_context += f"Content: {doc['content']}\n"
                rag_context += f"Extraction Method: {doc['extraction_method']}\n"
                if doc['is_pii_redacted']:
                    rag_context += "[Note: This document has been processed for PII redaction]\n"
                rag_context += "---\n"
                
                citations.append({
                    'source': doc['source'],
                    'source_type': 'document',
                    'document_title': doc['document_title'],
                    'page_number': doc['page_number'],
                    'section': doc['section'],
                    'citation': doc['citation']
                })
        
        # Enhanced faithfulness-focused prompt with knowledge base integration
        legal_court_prompt = ChatPromptTemplate.from_template("""
        You are an Educational Legal Assistant specializing in Indian law with strict faithfulness to source documents and knowledge base.

        **CRITICAL FAITHFULNESS REQUIREMENTS:**
        1. ONLY use information explicitly stated in the retrieved documents and knowledge base
        2. Prioritize knowledge base information for general legal questions
        3. Use document content for specific case law or detailed legal provisions
        4. Quote directly from sources when making specific claims
        5. If information is not clearly specified, state: "The retrieved sources do not clearly specify this information"
        6. Always cite the exact source (KB or Document) when referencing information
        7. Do not infer, paraphrase, or extrapolate beyond what is explicitly written

        **SAFETY INSTRUCTIONS:**
        - You are an EDUCATIONAL ASSISTANT ONLY - NEVER provide personalized legal advice
        - ALWAYS recommend consulting qualified legal professionals for specific situations
        - Focus on explaining what the sources say, not what someone should do

        **Previous Conversation Context:**
        {memory_context}

        **Retrieved Legal Information (Knowledge Base + Documents):**
        {rag_context}

        **Current User Query:** {query}

        **Response Instructions:**
        1. Check knowledge base information first for general legal concepts
        2. Use document content for specific cases or detailed legal provisions
        3. Provide educational explanation based ONLY on retrieved content
        4. Include precise citations for all claims (KB-X for knowledge base, Document-X for documents)
        5. Clearly state what information is missing or unclear
        6. End with appropriate legal disclaimer

        **Educational Response with Precise Citations:**""")
        
        try:
            legal_court_chain = legal_court_prompt | llm | StrOutputParser()
            response = legal_court_chain.invoke({
                "query": state.query,
                "rag_context": rag_context if rag_context else "No relevant legal information found for this specific query.",
                "memory_context": memory_context if memory_context else "No previous conversation context."
            })
            
            # Update memory with current interaction
            if session_id:
                memory = get_rag_memory(session_id)
                memory.chat_memory.add_user_message(state.query)
                memory.chat_memory.add_ai_message(response)
            
            # Format enhanced response with detailed citations and stats
            citation_list = ""
            if citations:
                kb_citations = [c for c in citations if c['source_type'] == 'knowledge_base']
                doc_citations = [c for c in citations if c['source_type'] == 'document']
                
                if kb_citations:
                    citation_list += "**Knowledge Base Sources:**\n"
                    for cite in kb_citations:
                        citation_list += f"• {cite['citation']}\n"
                
                if doc_citations:
                    citation_list += "\n**Document Sources:**\n"
                    for cite in doc_citations:
                        citation_list += f"• {cite['citation']}\n"
            
            formatted_response = f"""⚖️ **Enhanced Legal Educational Assistant Response:**

            {response}

            **📚 Precise Source Citations:**
            {citation_list if citations else "• No specific sources retrieved for this query"}

            **⚠️ Important Legal Disclaimer:**
            This response is for educational purposes only and does not constitute legal advice. The information is based solely on the knowledge base and retrieved documents and may not be comprehensive. For specific legal matters, please consult with a qualified legal professional."""

            return {
                "response": formatted_response,
                "agent_route": getattr(state, 'agent_route', 'legal_court'),
                "sources_used": [doc['source'] for doc in relevant_docs],
                "knowledge_base_used": len(kb_docs) > 0,
                "documents_used": len(pdf_docs),
                "precise_citations": citations,
                "session_id": session_id,
                "memory_used": bool(memory_context),
                "memory_context": memory_context,
                "retrieval_method": "MMR with Knowledge Base",
                "pii_redacted": any(doc['is_pii_redacted'] for doc in relevant_docs),
                "faithfulness_enforced": True
            }

        
        except Exception as e:
            return {
                "response": f"""⚖️ **Enhanced Legal Educational Assistant** encountered an error: {str(e)}

**⚠️ Disclaimer:** For legal assistance, please consult with a qualified legal professional.

Please try rephrasing your legal education question.""",
                "agent_route": getattr(state, 'agent_route', 'legal_court'),
                "sources_used": [],
                "knowledge_base_used": False,
                "precise_citations": [],
                "session_id": session_id,
                "memory_used": False,
                "retrieval_method": "Failed",
                "error": str(e)
            }

# Global instance for use in router
_legal_rag_agent = None

def get_legal_rag_agent():
    """Get or create the global Enhanced Legal RAG Agent instance"""
    global _legal_rag_agent
    if _legal_rag_agent is None:
        _legal_rag_agent = LegalCourtRAGAgent()
    return _legal_rag_agent

# Function to use in your existing router system
def legal_court_agent(state: GraphState) -> Dict[str, Any]:
    """Enhanced wrapper function for integration with existing router system"""
    legal_rag_agent = get_legal_rag_agent()
    return legal_rag_agent.legal_court_agent(state)

def clear_rag_memory(session_id: str) -> bool:
    """Clear RAG agent memory for a specific session"""
    if session_id in rag_session_memories:
        del rag_session_memories[session_id]
        return True
    return False

def add_single_document(pdf_path: str, use_ocr: bool = True) -> Dict[str, Any]:
    """Add a single document to the enhanced RAG system"""
    agent = get_legal_rag_agent()
    
    if not os.path.exists(pdf_path):
        return {"success": False, "error": f"File {pdf_path} not found"}
    
    try:
        # Check if document already exists
        filename = os.path.basename(pdf_path)
        existing_docs = agent.collection.get(where={"source": filename})
        
        if existing_docs['documents']:
            return {
                "success": False, 
                "error": f"Document {filename} already exists in collection",
                "existing_chunks": len(existing_docs['documents'])
            }
        
        # Extract text with enhanced metadata using OCR processor
        document_text, page_info = agent.pdf_extractor.extract_text_from_pdf(pdf_path, use_ocr=use_ocr)
        
        if document_text.strip():
            # Chunk the document
            chunks = agent.text_splitter.split_text(document_text)
            
            # Add chunks to vector store
            agent.add_document_chunks_enhanced(chunks, filename, page_info)
            
            return {
                "success": True,
                "filename": filename,
                "chunks_created": len(chunks),
                "extraction_method": page_info.get("extraction_method", "unknown"),
                "total_pages": page_info.get("total_pages", 0)
            }
        else:
            return {
                "success": False,
                "error": f"No text could be extracted from {filename}"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Error processing {pdf_path}: {str(e)}"
        }

def get_collection_info() -> Dict[str, Any]:
    """Get detailed information about the current collection"""
    agent = get_legal_rag_agent()
    return agent.get_collection_stats()

def search_documents(query: str, k: int = 5, use_mmr: bool = True) -> Dict[str, Any]:
    """Search documents in the collection with optional MMR"""
    agent = get_legal_rag_agent()
    
    try:
        if use_mmr:
            results = agent.retrieve_relevant_documents_mmr(query, k=k)
        else:
            # Fallback to simple retrieval
            query_embedding = agent.embedding_model.encode([query]).tolist()[0]
            chroma_results = agent.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            results = []
            if chroma_results['documents'] and chroma_results['documents'][0]:
                for i, doc in enumerate(chroma_results['documents'][0]):
                    metadata = chroma_results['metadatas'][0][i]
                    results.append({
                        'content': doc,
                        'source': metadata['source'],
                        'document_title': metadata.get('document_title', 'Unknown'),
                        'page_number': metadata.get('page_number'),
                        'section': metadata.get('section'),
                        'distance': chroma_results['distances'][0][i],
                        'citation': f"{metadata.get('document_title', metadata['source'])}"
                    })
        
        return {
            "success": True,
            "query": query,
            "results_count": len(results),
            "results": results,
            "retrieval_method": "MMR" if use_mmr else "Simple",
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Search failed: {str(e)}"
        }

def remove_document(source_filename: str) -> Dict[str, Any]:
    """Remove all chunks of a specific document from the collection"""
    agent = get_legal_rag_agent()
    
    try:
        # Get all chunks for this document
        existing_docs = agent.collection.get(
            where={"source": source_filename},
            include=['ids']
        )
        
        if not existing_docs['ids']:
            return {
                "success": False,
                "error": f"Document {source_filename} not found in collection"
            }
        
        # Delete all chunks
        agent.collection.delete(ids=existing_docs['ids'])
        
        return {
            "success": True,
            "filename": source_filename,
            "chunks_removed": len(existing_docs['ids'])
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error removing document: {str(e)}"
        }

def rebuild_collection(documents_directory: str = "./legal_documents") -> Dict[str, Any]:
    """Rebuild the entire collection from scratch"""
    global _legal_rag_agent
    
    try:
        # Delete existing collection
        if _legal_rag_agent:
            try:
                _legal_rag_agent.chroma_client.delete_collection(_legal_rag_agent.collection_name)
            except:
                pass
        
        # Create new agent instance
        _legal_rag_agent = LegalCourtRAGAgent(documents_directory=documents_directory)
        
        stats = _legal_rag_agent.get_collection_stats()
        
        return {
            "success": True,
            "message": "Collection rebuilt successfully",
            "stats": stats
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error rebuilding collection: {str(e)}"
        }

def update_knowledge_base(knowledge_base_file: str = "./rag_knowledge_base.txt", 
                         new_content: str = "") -> Dict[str, Any]:
    """Update the knowledge base file and reload it"""
    try:
        with open(knowledge_base_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n{new_content}")
        
        # Get the global agent and reload knowledge base
        global _legal_rag_agent
        if _legal_rag_agent:
            # Remove existing knowledge base entries
            existing_kb = _legal_rag_agent.collection.get(
                where={"source_type": "knowledge_base"},
                include=['ids']
            )
            if existing_kb['ids']:
                _legal_rag_agent.collection.delete(ids=existing_kb['ids'])
            
            # Reload knowledge base
            _legal_rag_agent.load_knowledge_base()
        
        return {
            "success": True,
            "message": "Knowledge base updated and reloaded successfully"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Error updating knowledge base: {str(e)}"
        }

def search_knowledge_base_only(query: str, k: int = 5) -> Dict[str, Any]:
    """Search only the knowledge base entries"""
    agent = get_legal_rag_agent()
    
    try:
        results = agent.retrieve_relevant_documents_mmr(
            query, k=k, include_knowledge_base=True
        )
        
        # Filter only knowledge base results
        kb_results = [doc for doc in results if doc['source_type'] == 'knowledge_base']
        
        return {
            "success": True,
            "query": query,
            "results_count": len(kb_results),
            "results": kb_results,
            "retrieval_method": "Knowledge Base Only"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Knowledge base search failed: {str(e)}"
        }