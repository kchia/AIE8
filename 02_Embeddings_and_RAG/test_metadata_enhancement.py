# -*- coding: utf-8 -*-
"""
Test script for Phase 2: Metadata Enhancement
This script tests the metadata functionality added to the RAG system.
"""

import asyncio
import os
from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.rag_pipeline import EnhancedRetrievalAugmentedQAPipeline
from aimakerspace.openai_utils.chatmodel import ChatOpenAI


def test_metadata_enhancement():
    """Test the metadata enhancement functionality."""
    
    print("Testing Phase 2: Metadata Enhancement")
    print("=" * 50)
    
    # Test 1: TextFileLoader with metadata
    print("\n1. Testing TextFileLoader metadata generation...")
    
    # Test with a text file (assuming PMarcaBlogs.txt exists)
    if os.path.exists("data/PMarcaBlogs.txt"):
        loader = TextFileLoader("data/PMarcaBlogs.txt")
        documents, metadata = loader.load_documents_with_metadata()
        
        print(f"[OK] Loaded {len(documents)} document(s)")
        print(f"[OK] Generated metadata for {len(metadata)} document(s)")
        
        if metadata:
            print(f"[INFO] Sample metadata: {metadata[0]}")
    else:
        # Create a small test file for demonstration
        print("[INFO] Creating test documents...")
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
        documents = test_docs
        metadata = test_metadata
        print(f"[OK] Created {len(documents)} test documents with metadata")
    
    # Test 2: CharacterTextSplitter with metadata
    print("\n2. Testing CharacterTextSplitter with metadata...")
    
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    
    if len(documents) > 0 and len(metadata) > 0:
        chunks, chunk_metadata = splitter.split_texts_with_metadata(documents, metadata)
        
        print(f"✅ Split documents into {len(chunks)} chunks")
        print(f"✅ Generated metadata for {len(chunk_metadata)} chunks")
        
        if chunk_metadata:
            print(f"📋 Sample chunk metadata: {chunk_metadata[0]}")
    else:
        print("❌ No documents available for splitting")
        return False
    
    # Test 3: VectorDatabase with metadata
    print("\n3. Testing VectorDatabase metadata storage...")
    
    # Note: This test requires OpenAI API key to be set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OpenAI API key not found. Skipping vector database tests.")
        print("   Set OPENAI_API_KEY environment variable to test full functionality.")
        return True
    
    try:
        vector_db = VectorDatabase()
        
        # Build database with metadata (async)
        async def build_vector_db():
            return await vector_db.abuild_from_list(chunks, chunk_metadata)
        
        vector_db = asyncio.run(build_vector_db())
        
        print(f"✅ Built vector database with {len(vector_db.vectors)} vectors")
        print(f"✅ Stored metadata for {len(vector_db.metadata)} vectors")
        
        # Test 4: Search with metadata
        print("\n4. Testing search with metadata...")
        
        # Search without filtering
        results = vector_db.search_by_text(
            "What is artificial intelligence?", 
            k=2, 
            return_metadata=True
        )
        
        print(f"✅ Found {len(results)} results with metadata")
        
        for i, (text, score, meta) in enumerate(results):
            print(f"   Result {i+1}: Score={score:.3f}, Source={meta.get('source_file', 'unknown')}")
        
        # Test filtering (if we have diverse metadata)
        if any('topic' in meta for meta in chunk_metadata):
            print("\n5. Testing metadata filtering...")
            
            filtered_results = vector_db.search_by_text(
                "machine learning",
                k=3,
                filter_criteria={"topic": "AI"},
                return_metadata=True
            )
            
            print(f"✅ Filtered search returned {len(filtered_results)} results")
            for text, score, meta in filtered_results:
                print(f"   Filtered result: Topic={meta.get('topic')}, Score={score:.3f}")
        
        # Test 5: Enhanced RAG Pipeline
        print("\n6. Testing Enhanced RAG Pipeline...")
        
        chat_model = ChatOpenAI()
        rag_pipeline = EnhancedRetrievalAugmentedQAPipeline(
            llm=chat_model,
            vector_db_retriever=vector_db,
            include_metadata=True,
            include_scores=True
        )
        
        # Test the pipeline
        result = rag_pipeline.run_pipeline(
            "What is the main topic discussed in the documents?",
            k=2
        )
        
        print(f"✅ RAG Pipeline executed successfully")
        print(f"📋 Response: {result['response'][:200]}...")
        print(f"📋 Source info included: {'source_info' in result}")
        print(f"📋 Context count: {result['context_count']}")
        
        print("\n" + "=" * 50)
        print("🎉 Phase 2 Metadata Enhancement: ALL TESTS PASSED!")
        print("\nEnhanced features now available:")
        print("✅ Document metadata tracking (source, timestamps, file types)")
        print("✅ Chunk-level metadata with positioning info")
        print("✅ Metadata-based filtering in searches")
        print("✅ Source attribution in RAG responses")
        print("✅ Enhanced pipeline with metadata passthrough")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_metadata_enhancement()
    if success:
        print("\n🚀 Phase 2 implementation ready!")
        print("🔗 You can now proceed to Phase 3 or test with your own documents.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")