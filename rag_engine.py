"""
RAG Engine - Retrieval Augmented Generation pipeline using LangChain, FAISS, and Google GenAI
"""

import os
import json
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "data", "vectorstore")

# Embedding model (Google GenAI)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# Google GenAI configuration
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


class RAGEngine:
    """RAG Engine that handles document indexing and querying."""

    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.active_provider = None  # "gemini" or None
        self.active_model = None
        self._initialized = False
        self._llm_available = False
        self._gemini_available = False

    def _init_gemini(self) -> bool:
        """Initialize Google GenAI (Gemini) via langchain-google-genai."""
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("  [INFO] No Google/Gemini API key found in environment.")
            return False

        try:
            print(f"  [>] Initializing Google GenAI ({GEMINI_MODEL})...")
            self.llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                api_key=api_key,
                temperature=0.3,
                max_output_tokens=1024,
            )
            self.active_provider = "gemini"
            self.active_model = GEMINI_MODEL
            self._gemini_available = True
            self._llm_available = True
            print(f"  [OK] Google GenAI ready: {GEMINI_MODEL}")
            return True
        except Exception as e:
            print(f"  [WARN] Google GenAI initialization error: {e}")
            return False

    def initialize(self):
        """Initialize the RAG engine components."""
        print("[*] Initializing RAG Engine...")

        # Initialize Google GenAI embeddings
        print("  [>] Loading Google GenAI embedding model...")
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key,
        )
        print(f"  [OK] Embedding model loaded: {EMBEDDING_MODEL}")

        # Initialize Google GenAI
        if not self._init_gemini():
            print("  [INFO] Running in retrieval-only mode (configure GEMINI_API_KEY for AI generation)")

        # Load or create vectorstore
        self._load_or_create_vectorstore()
        self._initialized = True
        print("[OK] RAG Engine initialized successfully!\n")

    def _load_or_create_vectorstore(self):
        """Load existing vectorstore or create from documents."""
        index_file = os.path.join(VECTORSTORE_DIR, "index.faiss")
        if os.path.exists(index_file):
            print("  [>] Loading existing vectorstore...")
            try:
                self.vectorstore = FAISS.load_local(
                    VECTORSTORE_DIR, self.embeddings, allow_dangerous_deserialization=True
                )
                test_dim = len(self.embeddings.embed_query("test"))
                if self.vectorstore.index.d != test_dim:
                    print(f"  [WARN] Vectorstore dimension mismatch ({self.vectorstore.index.d} vs {test_dim}). Rebuilding index...")
                    self.rebuild_index()
                else:
                    print(f"  [OK] Vectorstore loaded")
            except Exception as e:
                print(f"  [WARN] Could not load vectorstore ({e}), rebuilding index...")
                self.rebuild_index()
        else:
            print("  [>] No vectorstore found, creating from documents...")
            self._create_vectorstore()

    def _create_vectorstore(self):
        """Create vectorstore from scraped documents."""
        docs_path = os.path.join(DATA_DIR, "rbi_documents.json")

        if not os.path.exists(docs_path):
            print("  [WARN] No documents found. Running scraper first...")
            from scraper import run_scraper

            run_scraper(use_sample=True, scrape_live=False)

        with open(docs_path, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)

        print(f"  [>] Processing {len(raw_docs)} documents...")

        # Convert to LangChain Documents
        documents = []
        for doc in raw_docs:
            content = f"Title: {doc['title']}\n\n{doc['content']}"
            metadata = {
                "id": doc["id"],
                "title": doc["title"],
                "date": doc.get("date", ""),
                "url": doc.get("url", ""),
                "type": doc.get("type", ""),
                "source": doc.get("source", "RBI"),
            }
            documents.append(Document(page_content=content, metadata=metadata))

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)
        print(f"  [>] Created {len(chunks)} text chunks")

        # Create vectorstore
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # Save vectorstore
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)
        self.vectorstore.save_local(VECTORSTORE_DIR)
        print(f"  [OK] Vectorstore created and saved to {VECTORSTORE_DIR}")

    def rebuild_index(self):
        """Rebuild the vectorstore from scratch."""
        import shutil

        if os.path.exists(VECTORSTORE_DIR):
            shutil.rmtree(VECTORSTORE_DIR)
        self._create_vectorstore()

    def query(self, question: str, top_k: int = 5) -> Dict:
        """Query the RAG system."""
        if not self._initialized:
            self.initialize()

        # Retrieve relevant documents
        retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}
        )
        relevant_docs = retriever.invoke(question)

        # Prepare sources
        sources = []
        seen_ids = set()
        for doc in relevant_docs:
            doc_id = doc.metadata.get("id", "")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                sources.append(
                    {
                        "title": doc.metadata.get("title", "Unknown"),
                        "date": doc.metadata.get("date", ""),
                        "url": doc.metadata.get("url", ""),
                        "type": doc.metadata.get("type", ""),
                        "snippet": doc.page_content[:300] + "..."
                        if len(doc.page_content) > 300
                        else doc.page_content,
                    }
                )

        # Generate answer using LLM or return context-based response
        if self._llm_available and self.llm:
            answer = self._generate_answer(question, relevant_docs)
        else:
            answer = self._generate_retrieval_answer(question, relevant_docs)

        return {
            "answer": answer,
            "sources": sources,
            "query": question,
            "provider": self.active_provider,
            "model": self.active_model,
        }

    def _generate_answer(self, question: str, docs: List[Document]) -> str:
        """Generate answer using Google GenAI."""
        context = "\n\n---\n\n".join(
            [
                f"Source: {doc.metadata.get('title', 'Unknown')}\n{doc.page_content}"
                for doc in docs
            ]
        )

        prompt = f"""You are an expert assistant specializing in Reserve Bank of India (RBI) regulations, circulars, notifications, and policies. 

Based on the following context from official RBI documents, provide a comprehensive and accurate answer to the question. If the context doesn't contain enough information to fully answer the question, acknowledge what you can answer and mention what additional information might be needed.

Always cite specific RBI documents, circulars, or notifications when possible.

Context:
{context}

Question: {question}

Answer:"""

        models_to_try = [self.active_model, "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"]
        models = [m for m in dict.fromkeys(models_to_try) if m]

        last_error = None
        for model_name in models:
            try:
                if model_name != self.active_model or not self.llm:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                    active_llm = ChatGoogleGenerativeAI(
                        model=model_name,
                        api_key=api_key,
                        max_output_tokens=1024,
                    )
                else:
                    active_llm = self.llm

                response = active_llm.invoke(prompt)
                self.active_model = model_name

                # Handle response from ChatGoogleGenerativeAI
                if isinstance(response, str):
                    return response
                elif hasattr(response, "content"):
                    content = response.content
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        return "".join(
                            part.get("text", "") if isinstance(part, dict) else str(part)
                            for part in content
                        )
                    return str(content)
                return str(response)
            except Exception as e:
                last_error = e
                continue

        provider_name = self.active_provider or "Google GenAI"
        return f"Error generating response from {provider_name}: {str(last_error)}"

    def _generate_retrieval_answer(
        self, question: str, docs: List[Document]
    ) -> str:
        """Generate a structured answer from retrieved documents when LLM is not available."""
        if not docs:
            return "I couldn't find any relevant information in the RBI documents database. Please try rephrasing your question."

        answer_parts = [
            "**Based on the retrieved RBI documents, here is the relevant information:**\n"
        ]

        for i, doc in enumerate(docs[:3], 1):
            title = doc.metadata.get("title", "Unknown Document")
            content = doc.page_content.strip()
            # Truncate very long content
            if len(content) > 500:
                content = content[:500] + "..."
            answer_parts.append(f"### {i}. {title}\n{content}\n")

        answer_parts.append(
            "\n---\n*Note: This response is generated from document retrieval only. "
            "Configure GEMINI_API_KEY for AI-generated summaries.*"
        )

        return "\n".join(answer_parts)

    def get_document_stats(self) -> Dict:
        """Get statistics about indexed documents."""
        docs_path = os.path.join(DATA_DIR, "rbi_documents.json")
        stats = {
            "total_documents": 0,
            "types": {},
            "vectorstore_exists": os.path.exists(
                os.path.join(VECTORSTORE_DIR, "index.faiss")
            ),
            "llm_available": self._llm_available,
            "active_provider": self.active_provider,
            "active_model": self.active_model,
            "gemini_available": self._gemini_available,
            "gemini_model": GEMINI_MODEL,
            "embedding_model": EMBEDDING_MODEL,
        }

        if os.path.exists(docs_path):
            with open(docs_path, "r", encoding="utf-8") as f:
                docs = json.load(f)
            stats["total_documents"] = len(docs)
            for doc in docs:
                doc_type = doc.get("type", "unknown")
                stats["types"][doc_type] = stats["types"].get(doc_type, 0) + 1

        return stats


# Singleton instance
rag_engine = RAGEngine()
