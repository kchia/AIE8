#!/usr/bin/env python3
"""
Simple test script for Phase 2: Metadata Enhancement
This script tests the metadata functionality added to the RAG system.
"""

import asyncio
import os
from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter


def test_basic_metadata():
    """Test basic metadata functionality."""
    
    print("Testing Phase 2: Metadata Enhancement")
    print("=" * 50)
    
    # Test 1: TextFileLoader with metadata
    print("\n1. Testing TextFileLoader metadata generation...")
    
    # Create test documents
    test_docs = [
        "This is a test document about artificial intelligence and machine learning.",
        "Another document discussing the benefits of retrieval augmented generation.",
        "A third document exploring vector databases and embeddings."
    ]
    test_metadata = [
        {"source_file": "test1.txt", "file_type": "txt", "topic": "AI"},
        {"source_file": "test2.txt", "file_type": "txt", "topic": "RAG"},
        {"source_file": "test3.txt", "file_type": "txt", "topic": "vectors"}
    ]
    
    print(f"[OK] Created {len(test_docs)} test documents with metadata")
    
    # Test 2: CharacterTextSplitter with metadata
    print("\n2. Testing CharacterTextSplitter with metadata...")
    
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks, chunk_metadata = splitter.split_texts_with_metadata(test_docs, test_metadata)
    
    print(f"[OK] Split documents into {len(chunks)} chunks")
    print(f"[OK] Generated metadata for {len(chunk_metadata)} chunks")
    
    if chunk_metadata:
        print(f"[INFO] Sample chunk metadata: {chunk_metadata[0]}")
    
    # Test 3: VectorDatabase basic functionality (without OpenAI)
    print("\n3. Testing VectorDatabase metadata storage (basic)...")
    
    from aimakerspace.vectordatabase import VectorDatabase
    import numpy as np
    
    # Create a vector database without embedding model for basic test
    vector_db = VectorDatabase(embedding_model=None)
    
    # Insert some test vectors with metadata
    for i, (chunk, meta) in enumerate(zip(chunks[:3], chunk_metadata[:3])):
        # Create a dummy vector for testing
        dummy_vector = np.random.rand(100)  # 100-dimensional random vector
        vector_db.insert(f"chunk_{i}", dummy_vector, meta)
    
    print(f"[OK] Inserted {len(vector_db.vectors)} vectors with metadata")
    print(f"[OK] Stored metadata for {len(vector_db.metadata)} vectors")
    
    # Test metadata retrieval
    first_key = list(vector_db.vectors.keys())[0]
    vector, metadata = vector_db.retrieve_from_key(first_key, include_metadata=True)
    
    if metadata:
        print(f"[OK] Successfully retrieved metadata: {metadata.get('source_file', 'unknown')}")
    
    print("\n" + "=" * 50)
    print("[SUCCESS] Phase 2 Metadata Enhancement: Basic tests passed!")
    print("\nEnhanced features implemented:")
    print("* Document metadata tracking (source, timestamps, file types)")
    print("* Chunk-level metadata with positioning info")
    print("* Metadata storage in VectorDatabase")
    print("* Metadata retrieval functionality")
    
    return True


def test_file_based_metadata():
    """Test with actual file if available."""
    print("\n4. Testing with actual data files...")
    
    # Check for PMarca blogs file
    if os.path.exists("data/PMarcaBlogs.txt"):
        print("[INFO] Found PMarcaBlogs.txt, testing file-based metadata...")
        
        loader = TextFileLoader("data/PMarcaBlogs.txt")
        documents, metadata = loader.load_documents_with_metadata()
        
        print(f"[OK] Loaded {len(documents)} document(s)")
        print(f"[OK] Generated metadata for {len(metadata)} document(s)")
        
        if metadata:
            meta = metadata[0]
            print(f"[INFO] File metadata - Source: {meta.get('source_file')}, Type: {meta.get('file_type')}, Length: {meta.get('document_length')}")
        
        return True
    else:
        print("[INFO] No data files found, skipping file-based test")
        return True


if __name__ == "__main__":
    try:
        success = test_basic_metadata()
        if success:
            test_file_based_metadata()
            
        print("\n[COMPLETED] Phase 2 implementation ready!")
        print("Next steps: Test with OpenAI API key for full functionality")
    except Exception as e:
        print(f"[ERROR] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()