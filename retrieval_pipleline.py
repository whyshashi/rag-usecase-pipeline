"""
Retrieval-Augmented Generation (RAG) Retrieval Pipeline

This module implements the retrieval and answer generation portion of a RAG pipeline.
It handles:
- Querying a vector database (Chroma) to retrieve relevant document chunks
- Processing multimodal content (text, tables, and images) from retrieved documents
- Generating comprehensive answers using Google's Gemini AI with vision capabilities

Key Functions:
    - rag_retrieval_pipeline(): Retrieves relevant chunks from the vector store based on a query
    - generate_final_answer(): Generates a final answer using the retrieved chunks and multimodal content

Dependencies:
    - unstructured: Document parsing and chunking
    - langchain: LLM integration and document management
    - Google Generative AI: Chat and embedding models
    - ChromaDB: Vector database for similarity search
"""

import json
from typing import List

# Unstructured for document parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# LangChain components
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from ingestion_pipeline import export_chunks_to_json


load_dotenv()

def rag_retrieval_pipeline(query: str, db: Chroma):
    """Main RAG retrieval and answer generation pipeline"""

    # Query the vector store

    retriever = db.as_retriever(search_kwargs={"k": 3})
    chunks = retriever.invoke(query)
    export_chunks_to_json(chunks, "rag_results.json")
    return chunks

def generate_final_answer(chunks, query) -> str:
    """Generate final answer using multimodal content"""
    
    try:
        # Initialize LLM (needs vision model for images)
        # llm = ChatOpenAI(model="gpt-4o", temperature=0)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
        
        # Build the text prompt
        prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""
        
        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i+1} ---\n"
            
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                
                # Add raw text
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"
                
                # Add tables as HTML
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"
            
            prompt_text += "\n"
        
        prompt_text += """
Please provide a clear, comprehensive answer using the text, tables, and images above. If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

ANSWER:"""

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add all images from all chunks
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                images_base64 = original_data.get("images_base64", [])
                
                for image_base64 in images_base64:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        print(f" Answer generation failed: {e}")
        raise ValueError(f"Answer generation failed: {e}")
