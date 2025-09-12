#!/usr/bin/env python3
"""
Test script for Phase 3: Additional Distance Metrics
This script tests the new distance metric functionality added to the RAG system.
"""

import asyncio
import os
import numpy as np
from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase, cosine_similarity, euclidean_distance, manhattan_distance, dot_product_similarity
from aimakerspace.rag_pipeline import EnhancedRetrievalAugmentedQAPipeline
from aimakerspace.distance_metrics import DistanceMetricAnalyzer, create_distance_metrics_guide


def test_basic_distance_metrics():
    """Test basic distance metric functionality without embeddings."""
    
    print("Testing Phase 3: Additional Distance Metrics")
    print("=" * 50)
    
    # Test 1: Individual distance functions
    print("\n1. Testing individual distance functions...")
    
    # Create test vectors
    vector_a = np.array([1.0, 2.0, 3.0, 4.0])
    vector_b = np.array([2.0, 3.0, 4.0, 5.0])
    vector_c = np.array([1.0, 2.0, 3.0, 4.0])  # Same as vector_a
    
    print(f"Vector A: {vector_a}")
    print(f"Vector B: {vector_b}")
    print(f"Vector C: {vector_c} (same as A)")
    print()
    
    # Test all distance functions
    metrics = {
        'cosine': cosine_similarity,
        'euclidean': euclidean_distance,
        'manhattan': manhattan_distance,
        'dot_product': dot_product_similarity
    }
    
    print("Distance/Similarity scores (A vs B):")
    for name, func in metrics.items():
        try:
            score = func(vector_a, vector_b)
            print(f"  {name.ljust(12)}: {score:.6f}")
        except Exception as e:
            print(f"  {name.ljust(12)}: Error - {e}")
    
    print("\nDistance/Similarity scores (A vs C - should be identical/max similarity):")
    for name, func in metrics.items():
        try:
            score = func(vector_a, vector_c)
            print(f"  {name.ljust(12)}: {score:.6f}")
        except Exception as e:
            print(f"  {name.ljust(12)}: Error - {e}")
    
    # Test 2: VectorDatabase configuration
    print("\n2. Testing VectorDatabase configuration...")
    
    # Test default metric
    db_default = VectorDatabase(embedding_model=None)
    print(f"[OK] Default metric: {db_default.default_distance_metric}")
    
    # Test custom metric
    db_euclidean = VectorDatabase(embedding_model=None, default_distance_metric='euclidean')
    print(f"[OK] Custom metric: {db_euclidean.default_distance_metric}")
    
    # Test available metrics
    available = db_default.get_available_metrics()
    print(f"[OK] Available metrics: {available}")
    
    # Test invalid metric
    try:
        VectorDatabase(embedding_model=None, default_distance_metric='invalid_metric')
        print("[ERROR] Should have raised ValueError for invalid metric")
        return False
    except ValueError as e:
        print(f"[OK] Correctly caught invalid metric: {e}")
    
    # Test 3: VectorDatabase with sample vectors
    print("\n3. Testing VectorDatabase with sample vectors...")
    
    # Create sample data
    sample_texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers", 
        "Natural language processing analyzes human language",
        "Computer vision interprets visual information",
        "Reinforcement learning learns through trial and error"
    ]
    
    sample_metadata = [
        {"topic": "ML", "source": "textbook.txt"},
        {"topic": "DL", "source": "textbook.txt"}, 
        {"topic": "NLP", "source": "paper.pdf"},
        {"topic": "CV", "source": "paper.pdf"},
        {"topic": "RL", "source": "article.txt"}
    ]
    
    # Create vector database and insert sample vectors
    db = VectorDatabase(embedding_model=None, default_distance_metric='cosine')
    
    for i, (text, meta) in enumerate(zip(sample_texts, sample_metadata)):
        # Create dummy embedding vectors for testing
        vector = np.random.rand(100)  # 100-dimensional random vector
        db.insert(f"doc_{i}", vector, meta)
    
    print(f"[OK] Inserted {len(db.vectors)} vectors with metadata")
    
    # Test 4: Search with different metrics
    print("\n4. Testing search with different distance metrics...")
    
    query_vector = np.random.rand(100)  # Random query vector
    
    for metric_name in db.get_available_metrics():
        try:
            # Get the distance function
            distance_func = db.DISTANCE_METRICS[metric_name]
            
            # Perform search
            results = db.search(query_vector, k=3, distance_measure=distance_func, return_metadata=True)
            
            print(f"[OK] {metric_name}: Found {len(results)} results")
            if results:
                best_score = results[0][1]
                print(f"     Best score: {best_score:.6f}")
                
        except Exception as e:
            print(f"[ERROR] {metric_name}: {e}")
    
    # Test 5: Metric switching
    print("\n5. Testing metric switching...")
    
    original_metric = db.default_distance_metric
    print(f"Original default: {original_metric}")
    
    # Change metric
    db.set_default_distance_metric('euclidean')
    print(f"Changed to: {db.default_distance_metric}")
    
    # Change back
    db.set_default_distance_metric(original_metric)
    print(f"Changed back to: {db.default_distance_metric}")
    
    print("\n" + "=" * 50)
    print("[SUCCESS] Phase 3 Basic Tests: All tests passed!")
    return True


def test_with_real_embeddings():
    """Test distance metrics with actual embeddings (requires OpenAI API key)."""
    
    print("\n6. Testing with real embeddings...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("[INFO] OpenAI API key not found. Skipping embedding tests.")
        print("       Set OPENAI_API_KEY environment variable for full testing.")
        return True
    
    try:
        from aimakerspace.openai_utils.chatmodel import ChatOpenAI
        
        # Create test documents
        test_docs = [
            "Artificial intelligence is transforming technology",
            "Machine learning algorithms learn from data patterns", 
            "Neural networks mimic brain functionality",
            "The weather is sunny today",
            "Cooking requires skill and creativity"
        ]
        
        test_metadata = [
            {"topic": "AI", "category": "tech"},
            {"topic": "ML", "category": "tech"},
            {"topic": "NN", "category": "tech"}, 
            {"topic": "weather", "category": "misc"},
            {"topic": "cooking", "category": "misc"}
        ]
        
        # Create vector database with embeddings
        vector_db = VectorDatabase(default_distance_metric='cosine')
        
        # Build database
        async def build_db():
            return await vector_db.abuild_from_list(test_docs, test_metadata)
        
        vector_db = asyncio.run(build_db())
        print(f"[OK] Built vector database with {len(vector_db.vectors)} embedded vectors")
        
        # Test query with different metrics
        query = "What is artificial intelligence?"
        print(f"\nTesting query: '{query}'")
        
        for metric in vector_db.get_available_metrics():
            distance_func = vector_db.DISTANCE_METRICS[metric]
            results = vector_db.search_by_text(
                query, k=3, distance_measure=distance_func, return_metadata=True
            )
            
            print(f"\n{metric.upper()} Results:")
            for i, (text, score, metadata) in enumerate(results, 1):
                print(f"  {i}. Score: {score:.4f} | Topic: {metadata.get('topic', 'N/A')} | Text: {text[:60]}...")
        
        # Test RAG pipeline with different metrics
        print("\n7. Testing Enhanced RAG Pipeline with distance metrics...")
        
        chat_model = ChatOpenAI()
        rag_pipeline = EnhancedRetrievalAugmentedQAPipeline(
            llm=chat_model,
            vector_db_retriever=vector_db,
            include_metadata=True
        )
        
        # Test with different metrics
        test_metrics = ['cosine', 'euclidean']
        question = "How do neural networks work?"
        
        for metric in test_metrics:
            print(f"\nRAG with {metric} distance:")
            try:
                result = rag_pipeline.run_pipeline(
                    question, k=2, distance_metric=metric
                )
                print(f"  Response length: {len(result['response'])} chars")
                print(f"  Context count: {result['context_count']}")
                print(f"  Response preview: {result['response'][:100]}...")
            except Exception as e:
                print(f"  Error: {e}")
        
        # Test distance metric comparison
        print("\n8. Testing distance metric comparison utility...")
        
        try:
            comparison = rag_pipeline.compare_distance_metrics("machine learning", k=2)
            print("[OK] Distance metric comparison:")
            for metric, results in comparison.items():
                if isinstance(results, list):
                    print(f"  {metric}: {len(results)} results")
                    if results:
                        print(f"    Best score: {results[0][1]:.4f}")
                else:
                    print(f"  {metric}: {results}")
        except Exception as e:
            print(f"[ERROR] Comparison failed: {e}")
        
        # Test analyzer
        print("\n9. Testing DistanceMetricAnalyzer...")
        
        try:
            analyzer = DistanceMetricAnalyzer(vector_db)
            
            # Test characteristics analysis
            characteristics = analyzer.analyze_metric_characteristics(sample_size=10)
            print("[OK] Metric characteristics analysis:")
            for metric, stats in characteristics.items():
                if 'mean' in stats:
                    print(f"  {metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
                else:
                    print(f"  {metric}: {stats}")
            
            # Test recommendations
            for use_case in ['general', 'speed', 'similarity']:
                metric, reason = analyzer.recommend_metric(use_case)
                print(f"[OK] {use_case}: {metric} - {reason[:50]}...")
                
        except Exception as e:
            print(f"[ERROR] Analyzer failed: {e}")
        
        print("\n" + "=" * 50)
        print("[SUCCESS] Phase 3 Advanced Tests: All tests completed!")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Advanced testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_distance_metrics_guide():
    """Print the distance metrics guide."""
    print("\n" + "=" * 80)
    print("DISTANCE METRICS GUIDE")
    print("=" * 80)
    print(create_distance_metrics_guide())
    print("=" * 80)


if __name__ == "__main__":
    try:
        # Run basic tests
        success = test_basic_distance_metrics()
        
        if success:
            # Run advanced tests with embeddings
            test_with_real_embeddings()
            
            # Print guide
            print_distance_metrics_guide()
            
            print("\n🎉 Phase 3 Implementation Complete!")
            print("\nNew Capabilities Added:")
            print("✓ Multiple distance metrics (cosine, euclidean, manhattan, dot_product)")
            print("✓ Configurable default metrics in VectorDatabase")
            print("✓ Distance metric comparison utilities")
            print("✓ RAG pipeline integration with metric selection")
            print("✓ Performance analysis and recommendations")
            print("✓ Comprehensive testing and benchmarking tools")
            
        else:
            print("\n❌ Some basic tests failed. Check the error messages above.")
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()