"""
LangChain RAG Example with RTX-STone

This example demonstrates how to build a Retrieval-Augmented Generation (RAG)
system using LangChain and RTX-STone optimized PyTorch.

Features:
- Document loading and processing
- Vector embeddings with local models
- Semantic search with FAISS
- LLM inference with RTX 50-series optimization
- Chat interface

Components:
- Embeddings: sentence-transformers (runs on GPU)
- Vector Store: FAISS (GPU-accelerated)
- LLM: Llama, Mistral, or Qwen (optimized with RTX-STone)
- Framework: LangChain

Installation:
    pip install langchain langchain-community sentence-transformers faiss-gpu

Usage:
    python integrations/langchain_rag_example.py --documents ./docs --query "What is RAG?"

Author: RTX-STone Contributors
License: BSD-3-Clause
"""

import argparse
import os
import torch
from pathlib import Path
from typing import List, Optional


def check_rtx_stone():
    """Check if RTX-STone is properly configured."""
    print("=" * 70)
    print("RTX-STone + LangChain RAG")
    print("=" * 70)

    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability(0)
        print(f"GPU: {gpu_name}")
        print(f"Compute Capability: {compute_cap[0]}.{compute_cap[1]}")

        if compute_cap == (12, 0):
            print("✓ RTX 50-series detected - optimizations active!")
            return True
        else:
            print("⚠ Not RTX 50-series - standard performance")

    return False


class RTXOptimizedRAG:
    """
    RAG system optimized for RTX 50-series GPUs.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
    ):
        """
        Initialize RAG system.

        Args:
            model_name: LLM model name
            embedding_model: Embedding model name
            device: Device to run on
        """
        self.device = device
        self.model_name = model_name
        self.embedding_model_name = embedding_model

        print(f"\nInitializing RAG system...")
        print(f"LLM: {model_name}")
        print(f"Embeddings: {embedding_model}")

        # Initialize components
        self._init_embeddings()
        self._init_vector_store()
        self._init_llm()

    def _init_embeddings(self):
        """Initialize embedding model."""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            print(f"Loading embedding model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={"device": self.device, "batch_size": 32},
            )
            print("✓ Embeddings loaded")
        except ImportError:
            print("✗ sentence-transformers not installed")
            print("Install: pip install sentence-transformers")
            raise

    def _init_vector_store(self):
        """Initialize vector store."""
        try:
            from langchain_community.vectorstores import FAISS

            self.vector_store_class = FAISS
            print("✓ FAISS available")
        except ImportError:
            print("✗ FAISS not installed")
            print("Install: pip install faiss-gpu  # or faiss-cpu")
            raise

    def _init_llm(self):
        """Initialize LLM with RTX-STone optimizations."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            from langchain_community.llms import HuggingFacePipeline

            print(f"Loading LLM: {self.model_name}...")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load model with optimizations
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

            # Apply RTX-STone optimizations if available
            try:
                from huggingface_rtx5080 import optimize_for_rtx5080

                model = optimize_for_rtx5080(model)
                print("✓ RTX-STone optimizations applied")
            except ImportError:
                print("⚠ RTX-STone optimizations not available")

            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
            )

            self.llm = HuggingFacePipeline(pipeline=pipe)
            print("✓ LLM loaded")

        except ImportError as e:
            print(f"✗ Error loading LLM: {e}")
            print("Install: pip install transformers accelerate")
            raise

    def load_documents(self, document_path: str):
        """
        Load documents from a directory or file.

        Args:
            document_path: Path to documents
        """
        try:
            from langchain_community.document_loaders import (
                DirectoryLoader,
                TextLoader,
                PyPDFLoader,
            )
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            print(f"\nLoading documents from: {document_path}")

            path = Path(document_path)

            # Load documents
            if path.is_file():
                if path.suffix == ".pdf":
                    loader = PyPDFLoader(str(path))
                else:
                    loader = TextLoader(str(path))
                documents = loader.load()
            else:
                # Load all text files from directory
                loader = DirectoryLoader(
                    str(path),
                    glob="**/*.txt",
                    loader_cls=TextLoader,
                )
                documents = loader.load()

            print(f"Loaded {len(documents)} documents")

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )
            splits = text_splitter.split_documents(documents)
            print(f"Split into {len(splits)} chunks")

            # Create vector store
            print("Creating vector store...")
            self.vector_store = self.vector_store_class.from_documents(
                splits,
                self.embeddings,
            )
            print("✓ Vector store created")

            return len(splits)

        except Exception as e:
            print(f"✗ Error loading documents: {e}")
            raise

    def query(self, question: str, k: int = 4) -> str:
        """
        Query the RAG system.

        Args:
            question: Question to ask
            k: Number of documents to retrieve

        Returns:
            Answer string
        """
        try:
            from langchain.chains import RetrievalQA

            print(f"\nQuery: {question}")

            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": k}),
                return_source_documents=True,
            )

            # Run query
            result = qa_chain({"query": question})

            answer = result["result"]
            sources = result["source_documents"]

            print("\n" + "=" * 70)
            print("Answer:")
            print("=" * 70)
            print(answer)

            print("\n" + "=" * 70)
            print(f"Sources ({len(sources)} documents):")
            print("=" * 70)
            for i, doc in enumerate(sources, 1):
                print(f"\n{i}. {doc.metadata.get('source', 'Unknown')}")
                print(f"   {doc.page_content[:200]}...")

            return answer

        except Exception as e:
            print(f"✗ Error querying: {e}")
            raise

    def chat(self):
        """Interactive chat interface."""
        print("\n" + "=" * 70)
        print("RAG Chat Interface")
        print("=" * 70)
        print("Type 'exit' to quit\n")

        while True:
            try:
                question = input("You: ").strip()

                if question.lower() in ["exit", "quit", "q"]:
                    print("Goodbye!")
                    break

                if not question:
                    continue

                self.query(question)
                print("\n" + "-" * 70 + "\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="RTX-STone + LangChain RAG Example")
    parser.add_argument(
        "--documents",
        type=str,
        help="Path to documents directory or file",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to run (if not provided, enters chat mode)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="LLM model to use",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model",
    )

    args = parser.parse_args()

    # Check RTX-STone
    check_rtx_stone()

    if not args.documents:
        print("\n✗ No documents provided!")
        print("Usage: python langchain_rag_example.py --documents ./docs")
        print("\nExample:")
        print("  1. Create a docs/ directory")
        print("  2. Add .txt files to docs/")
        print("  3. Run: python langchain_rag_example.py --documents ./docs")
        return

    # Create RAG system
    rag = RTXOptimizedRAG(
        model_name=args.model,
        embedding_model=args.embeddings,
    )

    # Load documents
    rag.load_documents(args.documents)

    # Query or chat
    if args.query:
        rag.query(args.query)
    else:
        rag.chat()


if __name__ == "__main__":
    main()
