"""
Phase 3 Usage Examples: Distance Metrics in RAG System

This file demonstrates how to use the new distance metric capabilities
added in Phase 3 of the RAG enhancement project.
"""

import asyncio
import os
from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.rag_pipeline import EnhancedRetrievalAugmentedQAPipeline
from aimakerspace.distance_metrics import DistanceMetricAnalyzer


async def example_1_basic_usage():
    """Example 1: Basic distance metric configuration and usage."""
    
    print("=== Example 1: Basic Distance Metric Usage ===")
    
    # Create VectorDatabase with different default metrics
    db_cosine = VectorDatabase(default_distance_metric='cosine')
    db_euclidean = VectorDatabase(default_distance_metric='euclidean') 
    db_manhattan = VectorDatabase(default_distance_metric='manhattan')
    
    print(f"Cosine DB default: {db_cosine.default_distance_metric}")
    print(f"Euclidean DB default: {db_euclidean.default_distance_metric}")
    print(f"Manhattan DB default: {db_manhattan.default_distance_metric}")
    
    # Show available metrics
    print(f"Available metrics: {db_cosine.get_available_metrics()}")
    
    # Change default metric dynamically
    db_cosine.set_default_distance_metric('dot_product')
    print(f"Changed cosine DB to: {db_cosine.default_distance_metric}")


async def example_2_search_with_different_metrics():
    """Example 2: Searching with different distance metrics."""
    
    print("\n=== Example 2: Search with Different Metrics ===")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI API key required for this example")
        return
    
    # Create sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to learn representations.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret and analyze visual information.",
        "The weather forecast shows rain tomorrow with high humidity levels.",
        "Cooking pasta requires boiling water and timing the cooking process carefully."
    ]
    
    # Load with metadata
    metadata = [
        {"topic": "ML", "category": "AI", "difficulty": "intermediate"},
        {"topic": "DL", "category": "AI", "difficulty": "advanced"},
        {"topic": "NLP", "category": "AI", "difficulty": "intermediate"},
        {"topic": "CV", "category": "AI", "difficulty": "intermediate"},
        {"topic": "weather", "category": "misc", "difficulty": "easy"},
        {"topic": "cooking", "category": "misc", "difficulty": "easy"}
    ]
    
    # Create and populate vector database
    vector_db = VectorDatabase(default_distance_metric='cosine')
    await vector_db.abuild_from_list(documents, metadata)
    
    print(f"Built database with {len(vector_db.vectors)} documents")
    
    # Test query with different metrics
    query = "How does artificial intelligence work?"
    
    print(f"\nQuery: '{query}'\n")
    
    for metric_name in vector_db.get_available_metrics():
        distance_func = vector_db.DISTANCE_METRICS[metric_name]
        results = vector_db.search_by_text(
            query, k=2, distance_measure=distance_func, return_metadata=True
        )
        
        print(f"{metric_name.upper()} Results:")
        for i, (text, score, meta) in enumerate(results, 1):
            topic = meta.get('topic', 'N/A')
            category = meta.get('category', 'N/A')
            print(f"  {i}. Score: {score:.4f} | Topic: {topic} | Category: {category}")
            print(f"     Text: {text[:80]}...")
        print()


async def example_3_rag_pipeline_with_metrics():
    """Example 3: Using RAG pipeline with different distance metrics."""
    
    print("\n=== Example 3: RAG Pipeline with Distance Metrics ===")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI API key required for this example")
        return
    
    # Reuse the vector database from example 2
    documents = [
        "Machine learning algorithms learn patterns from training data to make predictions.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "Supervised learning uses labeled examples to train predictive models.",
        "Unsupervised learning finds hidden patterns in data without labels.",
        "Reinforcement learning learns optimal actions through trial and error.",
    ]
    
    metadata = [
        {"topic": "ML basics", "source": "textbook.pdf"},
        {"topic": "Neural networks", "source": "research.pdf"},  
        {"topic": "Supervised learning", "source": "textbook.pdf"},
        {"topic": "Unsupervised learning", "source": "textbook.pdf"},
        {"topic": "Reinforcement learning", "source": "paper.pdf"}
    ]
    
    vector_db = VectorDatabase(default_distance_metric='cosine')
    await vector_db.abuild_from_list(documents, metadata)
    
    # Create RAG pipeline
    from aimakerspace.openai_utils.chatmodel import ChatOpenAI
    chat_model = ChatOpenAI()
    
    rag_pipeline = EnhancedRetrievalAugmentedQAPipeline(
        llm=chat_model,
        vector_db_retriever=vector_db,
        include_metadata=True,
        include_scores=True
    )
    
    question = "What is the difference between supervised and unsupervised learning?"
    
    # Test with different distance metrics
    metrics_to_test = ['cosine', 'euclidean', 'dot_product']
    
    for metric in metrics_to_test:
        print(f"\n--- RAG with {metric.upper()} distance ---")
        try:
            result = rag_pipeline.run_pipeline(
                question, 
                k=3, 
                distance_metric=metric
            )
            
            print(f"Context count: {result['context_count']}")
            print(f"Similarity scores: {result['similarity_scores']}")
            print(f"Response preview: {result['response'][:200]}...")
            
        except Exception as e:
            print(f"Error with {metric}: {e}")


async def example_4_distance_metric_analysis():
    """Example 4: Analyzing and comparing distance metrics."""
    
    print("\n=== Example 4: Distance Metric Analysis ===")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI API key required for this example")
        return
    
    # Create a larger dataset for analysis
    documents = [
        "Python is a high-level programming language known for its simplicity.",
        "JavaScript is essential for web development and frontend applications.", 
        "Java is a robust object-oriented programming language for enterprise applications.",
        "C++ provides low-level control and is used for system programming.",
        "HTML structures web content and defines document markup.",
        "CSS styles web pages and controls visual presentation.",
        "SQL manages and queries relational databases efficiently.",
        "Machine learning algorithms analyze data to find patterns.",
        "Data science combines statistics, programming, and domain expertise.",
        "Artificial intelligence simulates human intelligence in machines."
    ]
    
    metadata = [
        {"language": "Python", "type": "programming", "level": "high"},
        {"language": "JavaScript", "type": "programming", "level": "high"},
        {"language": "Java", "type": "programming", "level": "high"},
        {"language": "C++", "type": "programming", "level": "low"},
        {"language": "HTML", "type": "markup", "level": "high"},
        {"language": "CSS", "type": "styling", "level": "high"},
        {"language": "SQL", "type": "query", "level": "high"},
        {"field": "ML", "type": "data science", "level": "advanced"},
        {"field": "DS", "type": "data science", "level": "intermediate"},
        {"field": "AI", "type": "data science", "level": "advanced"}
    ]
    
    # Build vector database
    vector_db = VectorDatabase(default_distance_metric='cosine')
    await vector_db.abuild_from_list(documents, metadata)
    
    print(f"Built database with {len(vector_db.vectors)} technical documents")
    
    # Create analyzer
    analyzer = DistanceMetricAnalyzer(vector_db)
    
    # Analyze metric characteristics
    print("\n1. Metric Characteristics Analysis:")
    characteristics = analyzer.analyze_metric_characteristics(sample_size=20)
    for metric, stats in characteristics.items():
        if 'mean' in stats:
            print(f"  {metric.ljust(12)}: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, Range=[{stats['min']:.4f}, {stats['max']:.4f}]")
        else:
            print(f"  {metric.ljust(12)}: {stats}")
    
    # Compare similarity scores for a query
    print("\n2. Similarity Score Comparison:")
    test_query = "programming languages for web development"
    comparison = analyzer.compare_similarity_scores(test_query, k=3)
    
    for metric, results in comparison.items():
        print(f"\n  {metric.upper()}:")
        if isinstance(results, list):
            for i, (text, score) in enumerate(results, 1):
                print(f"    {i}. Score: {score:.4f} - {text}")
        else:
            print(f"    {results}")
    
    # Get recommendations
    print("\n3. Metric Recommendations:")
    use_cases = ["general", "similarity", "clustering", "speed"]
    for case in use_cases:
        metric, reason = analyzer.recommend_metric(case)
        print(f"  {case.ljust(10)}: {metric.ljust(10)} - {reason}")


async def example_5_performance_comparison():
    """Example 5: Performance comparison between distance metrics."""
    
    print("\n=== Example 5: Performance Comparison ===")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI API key required for this example")
        return
    
    # Create larger dataset for performance testing
    import random
    
    # Generate sample tech documents
    topics = ["programming", "databases", "web development", "mobile apps", "data science", 
              "machine learning", "cybersecurity", "cloud computing", "DevOps", "AI"]
    
    documents = []
    metadata = []
    
    for i in range(30):  # Create 30 documents
        topic = random.choice(topics)
        doc = f"This document discusses {topic} and its applications in modern technology. " \
              f"It covers best practices, tools, and methodologies related to {topic}. " \
              f"The content is relevant for professionals working with {topic} technologies."
        
        documents.append(doc)
        metadata.append({
            "topic": topic,
            "doc_id": f"doc_{i:03d}",
            "length": len(doc)
        })
    
    # Build database
    vector_db = VectorDatabase(default_distance_metric='cosine')
    await vector_db.abuild_from_list(documents, metadata)
    
    print(f"Built performance test database with {len(vector_db.vectors)} documents")
    
    # Create analyzer and run benchmark
    analyzer = DistanceMetricAnalyzer(vector_db)
    
    test_queries = [
        "programming languages and frameworks",
        "database management systems",
        "web development technologies", 
        "artificial intelligence applications",
        "cybersecurity best practices"
    ]
    
    print(f"\nBenchmarking with {len(test_queries)} queries...")
    
    try:
        benchmark_results = analyzer.benchmark_metrics(test_queries, k=5)
        
        print("\nPerformance Results:")
        print("-" * 70)
        print(f"{'Metric':<12} {'Avg Time (ms)':<15} {'Unique Sources':<15} {'Total Results':<15}")
        print("-" * 70)
        
        for metric, stats in benchmark_results.items():
            avg_time_ms = stats['avg_time_per_query'] * 1000
            unique_sources = stats['unique_sources_found']
            total_results = stats['total_results']
            
            print(f"{metric:<12} {avg_time_ms:<15.2f} {unique_sources:<15} {total_results:<15}")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")


def usage_guide():
    """Print a comprehensive usage guide for Phase 3 features."""
    
    guide = """
PHASE 3 DISTANCE METRICS - USAGE GUIDE
=====================================

1. BASIC CONFIGURATION
----------------------
# Create VectorDatabase with specific default metric
db = VectorDatabase(default_distance_metric='euclidean')

# Change default metric
db.set_default_distance_metric('manhattan')

# Get available metrics
metrics = db.get_available_metrics()  # ['cosine', 'euclidean', 'manhattan', 'dot_product']

2. SEARCH WITH SPECIFIC METRICS
-------------------------------
# Search with explicit distance function
results = db.search_by_text(
    "query text", 
    k=5,
    distance_measure=db.DISTANCE_METRICS['euclidean']
)

# Search using default metric (set in constructor)
results = db.search_by_text("query text", k=5)

3. RAG PIPELINE WITH DISTANCE METRICS
-------------------------------------
# Create pipeline with default metric preference
rag = EnhancedRetrievalAugmentedQAPipeline(
    llm=chat_model,
    vector_db_retriever=vector_db,
    default_distance_metric='cosine'
)

# Use specific metric for a query
result = rag.run_pipeline(
    "What is machine learning?",
    distance_metric='euclidean',
    k=3
)

# Compare metrics for same query
comparison = rag.compare_distance_metrics("query text", k=3)

4. DISTANCE METRIC ANALYSIS
---------------------------
# Create analyzer
analyzer = DistanceMetricAnalyzer(vector_db)

# Analyze metric characteristics
characteristics = analyzer.analyze_metric_characteristics()

# Compare scores for specific query
comparison = analyzer.compare_similarity_scores("test query", k=5)

# Get recommendations
metric, reason = analyzer.recommend_metric("similarity")

# Run performance benchmark
results = analyzer.benchmark_metrics(["query1", "query2"], k=3)

5. CHOOSING THE RIGHT METRIC
----------------------------
- Cosine: General purpose, good for text embeddings (DEFAULT)
- Euclidean: Geometric relationships, clustering tasks
- Manhattan: Sparse data, less sensitive to outliers
- Dot Product: Fastest computation, requires normalized vectors

6. PERFORMANCE CONSIDERATIONS
----------------------------
Speed ranking (fastest to slowest):
1. Dot Product (fastest)
2. Manhattan 
3. Cosine
4. Euclidean (slowest due to square root)

Memory usage: All metrics have similar memory requirements
Accuracy: Depends on your data and use case - test with your specific dataset
    """
    
    print(guide)


async def main():
    """Run all examples."""
    
    print("Phase 3: Distance Metrics - Usage Examples")
    print("=" * 50)
    
    # Run examples
    await example_1_basic_usage()
    await example_2_search_with_different_metrics()
    await example_3_rag_pipeline_with_metrics()
    await example_4_distance_metric_analysis()
    await example_5_performance_comparison()
    
    # Print usage guide
    usage_guide()
    
    print("\n" + "=" * 50)
    print("Phase 3 Examples Complete!")
    print("Explore different distance metrics to optimize your RAG system")
    print("for your specific use case and data characteristics.")


if __name__ == "__main__":
    asyncio.run(main())