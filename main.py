"""
RAG Document Ingestion and Retrieval Pipeline

This module orchestrates the complete Retrieval-Augmented Generation (RAG) workflow for document
analysis and question answering. It performs the following steps:

1. Document Ingestion: Loads and processes PDF documents from the /docs directory
2. Chunking & Summarization: Breaks documents into manageable chunks and summarizes them
3. Vector Store Creation: Embeds chunks into a vector database (ChromaDB)
4. RAG Retrieval: Retrieves relevant chunks based on user queries using semantic similarity
5. Answer Generation: Generates contextual final answers using the retrieved chunks

The pipeline uses ChromaDB as the vector database for efficient semantic search and retrieval.
Processed chunks are cached in JSON format to avoid redundant processing.

Usage:
    run_pipeline("query_text") - Execute the full RAG pipeline with the given query
"""

import json
from ingestion_pipeline import run_complete_ingestion_pipeline,create_vector_store, load_chunks
from retrieval_pipleline import rag_retrieval_pipeline, generate_final_answer
from pathlib import Path


def run_pipeline(query: str) -> str:
    try:
        """Main ingestion pipeline"""
        print("=== RAG Document Ingestion Pipeline ===\n")

        summarised_chunks = []
        
        # Step 1-3: Ingest, chunk, summarise, and export to JSON
        if Path("./chunks_raw.json").is_file():
            print("chunks already exist, skipping ingestion and loading from disk")
            summarised_chunks = load_chunks()
        else:
            print("the chunks does not exist, running the complete ingestion pipeline")
            summarised_chunks = run_complete_ingestion_pipeline("./docs/prudential-plc-ar-2022.pdf")

        db = create_vector_store(summarised_chunks)

        # Step 4: RAG Retrieval
        chunks = rag_retrieval_pipeline(query, db)

        # Step 5: Generate final answer using retrieved chunks
        final_answer = generate_final_answer(chunks, query)
        print("Final Answer:")
        print(final_answer)
        return final_answer
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        raise ValueError(f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()