"""
Distance Metrics Utilities for RAG System

This module provides additional utilities for analyzing and comparing
distance metrics performance in the RAG system.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
import matplotlib.pyplot as plt
from collections import defaultdict


class DistanceMetricAnalyzer:
    """
    Utility class for analyzing and comparing distance metric performance.
    """
    
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.performance_data = defaultdict(list)
    
    def benchmark_metrics(self, queries: List[str], k: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Benchmark all available distance metrics with multiple queries.
        
        Args:
            queries: List of test queries
            k: Number of results to retrieve per query
            
        Returns:
            Dictionary with performance metrics for each distance measure
        """
        if self.vector_db.embedding_model is None:
            raise ValueError("Embedding model required for benchmarking")
        
        results = {}
        available_metrics = self.vector_db.get_available_metrics()
        
        for metric_name in available_metrics:
            metric_func = self.vector_db.DISTANCE_METRICS[metric_name]
            
            # Measure performance
            start_time = time.time()
            query_results = []
            
            for query in queries:
                try:
                    search_results = self.vector_db.search_by_text(
                        query, 
                        k=k, 
                        distance_measure=metric_func,
                        return_metadata=True
                    )
                    query_results.append(search_results)
                except Exception as e:
                    print(f"Error with {metric_name} for query '{query}': {e}")
                    query_results.append([])
            
            end_time = time.time()
            
            # Calculate statistics
            total_time = end_time - start_time
            avg_time_per_query = total_time / len(queries)
            
            # Calculate result diversity (unique sources retrieved)
            all_sources = set()
            for query_result in query_results:
                for result in query_result:
                    if len(result) >= 3:  # Has metadata
                        metadata = result[2]
                        source = metadata.get('source_file', 'unknown')
                        all_sources.add(source)
            
            results[metric_name] = {
                'total_time': total_time,
                'avg_time_per_query': avg_time_per_query,
                'unique_sources_found': len(all_sources),
                'total_results': sum(len(qr) for qr in query_results)
            }
        
        return results
    
    def compare_similarity_scores(self, query: str, k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compare similarity scores across different metrics for a single query.
        
        Args:
            query: Query to test
            k: Number of results to compare
            
        Returns:
            Dictionary with metric names and their scored results
        """
        if self.vector_db.embedding_model is None:
            raise ValueError("Embedding model required for comparison")
        
        comparison = {}
        available_metrics = self.vector_db.get_available_metrics()
        
        for metric_name in available_metrics:
            metric_func = self.vector_db.DISTANCE_METRICS[metric_name]
            
            try:
                results = self.vector_db.search_by_text(
                    query,
                    k=k,
                    distance_measure=metric_func
                )
                # Extract just text and score for comparison
                scored_results = [(result[0][:100] + "...", result[1]) for result in results]
                comparison[metric_name] = scored_results
                
            except Exception as e:
                comparison[metric_name] = [f"Error: {str(e)}"]
        
        return comparison
    
    def analyze_metric_characteristics(self, sample_size: int = 100) -> Dict[str, Dict[str, Any]]:
        """
        Analyze the mathematical characteristics of each distance metric.
        
        Args:
            sample_size: Number of vector pairs to sample for analysis
            
        Returns:
            Analysis results for each metric
        """
        if len(self.vector_db.vectors) < 2:
            raise ValueError("Need at least 2 vectors in database for analysis")
        
        # Get sample of vectors
        vector_items = list(self.vector_db.vectors.items())
        if len(vector_items) > sample_size:
            import random
            vector_items = random.sample(vector_items, sample_size)
        
        results = {}
        available_metrics = self.vector_db.get_available_metrics()
        
        for metric_name in available_metrics:
            metric_func = self.vector_db.DISTANCE_METRICS[metric_name]
            scores = []
            
            # Calculate scores for all pairs
            for i in range(len(vector_items)):
                for j in range(i + 1, len(vector_items)):
                    vector_a = vector_items[i][1]
                    vector_b = vector_items[j][1]
                    
                    try:
                        score = metric_func(vector_a, vector_b)
                        scores.append(score)
                    except Exception as e:
                        continue
            
            if scores:
                scores = np.array(scores)
                results[metric_name] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'median': float(np.median(scores)),
                    'range': float(np.max(scores) - np.min(scores))
                }
            else:
                results[metric_name] = {'error': 'Could not calculate scores'}
        
        return results
    
    def recommend_metric(self, use_case: str = "general") -> Tuple[str, str]:
        """
        Recommend the best distance metric based on use case.
        
        Args:
            use_case: Type of use case ("general", "similarity", "clustering", "speed")
            
        Returns:
            Tuple of (recommended_metric, reasoning)
        """
        recommendations = {
            "general": ("cosine", "Cosine similarity works well for most text embedding applications and is normalized."),
            "similarity": ("cosine", "Cosine similarity is ideal for measuring semantic similarity in high-dimensional spaces."),
            "clustering": ("euclidean", "Euclidean distance preserves geometric relationships useful for clustering."),
            "speed": ("dot_product", "Dot product is computationally fastest but requires normalized vectors."),
            "sparse": ("manhattan", "Manhattan distance can work better with sparse or high-dimensional vectors."),
            "normalized": ("dot_product", "For normalized vectors, dot product is equivalent to cosine but faster.")
        }
        
        return recommendations.get(use_case, ("cosine", "Default recommendation for unknown use cases."))
    
    def print_comparison_report(self, query: str = "test query", k: int = 5):
        """
        Print a comprehensive comparison report of all distance metrics.
        
        Args:
            query: Query to use for testing
            k: Number of results to analyze
        """
        print("=" * 80)
        print("DISTANCE METRICS COMPARISON REPORT")
        print("=" * 80)
        
        # Basic information
        available_metrics = self.vector_db.get_available_metrics()
        print(f"Available Metrics: {', '.join(available_metrics)}")
        print(f"Default Metric: {self.vector_db.default_distance_metric}")
        print(f"Vector Database Size: {len(self.vector_db.vectors)} vectors")
        print()
        
        # Metric characteristics
        print("METRIC CHARACTERISTICS:")
        print("-" * 40)
        try:
            characteristics = self.analyze_metric_characteristics(sample_size=50)
            for metric, stats in characteristics.items():
                print(f"{metric.upper()}:")
                if 'error' in stats:
                    print(f"  {stats['error']}")
                else:
                    print(f"  Mean Score: {stats['mean']:.4f}")
                    print(f"  Std Dev: {stats['std']:.4f}")
                    print(f"  Range: {stats['min']:.4f} - {stats['max']:.4f}")
                print()
        except Exception as e:
            print(f"Could not analyze characteristics: {e}")
        
        # Similarity comparison for sample query
        if self.vector_db.embedding_model:
            print("SIMILARITY SCORES COMPARISON:")
            print("-" * 40)
            print(f"Query: '{query}'")
            print()
            
            try:
                comparison = self.compare_similarity_scores(query, k=3)
                for metric, results in comparison.items():
                    print(f"{metric.upper()}:")
                    if isinstance(results, list) and len(results) > 0:
                        for i, (text, score) in enumerate(results[:3]):
                            print(f"  {i+1}. Score: {score:.4f} - {text}")
                    else:
                        print(f"  {results}")
                    print()
            except Exception as e:
                print(f"Could not compare scores: {e}")
        
        # Recommendations
        print("RECOMMENDATIONS:")
        print("-" * 40)
        use_cases = ["general", "similarity", "clustering", "speed"]
        for case in use_cases:
            metric, reason = self.recommend_metric(case)
            print(f"{case.title()}: {metric} - {reason}")
        
        print("=" * 80)


def create_distance_metrics_guide() -> str:
    """
    Create a comprehensive guide to distance metrics.
    
    Returns:
        Formatted guide as string
    """
    guide = """
DISTANCE METRICS GUIDE
======================

1. COSINE SIMILARITY
   - Formula: dot(A, B) / (||A|| * ||B||)
   - Range: -1 to 1 (higher = more similar)
   - Best for: Text embeddings, high-dimensional data
   - Pros: Normalized, angle-based, handles different magnitudes well
   - Cons: Ignores magnitude information

2. EUCLIDEAN DISTANCE
   - Formula: sqrt(sum((A - B)²))
   - Range: 0 to infinity (converted to 1/(1+distance) for similarity)
   - Best for: Continuous data, clustering, geometric relationships
   - Pros: Intuitive, preserves spatial relationships
   - Cons: Sensitive to dimensionality and magnitude

3. MANHATTAN DISTANCE
   - Formula: sum(|A - B|)
   - Range: 0 to infinity (converted to 1/(1+distance) for similarity)  
   - Best for: Sparse data, grid-like structures
   - Pros: Less sensitive to outliers than Euclidean
   - Cons: Less intuitive for continuous similarity

4. DOT PRODUCT SIMILARITY
   - Formula: sum(A * B)
   - Range: -infinity to infinity
   - Best for: Normalized vectors, fast computation
   - Pros: Computationally fastest
   - Cons: Sensitive to vector magnitudes, needs normalized input

CHOOSING THE RIGHT METRIC:
- Text/Semantic similarity: Cosine
- Fast computation with normalized vectors: Dot Product
- Geometric/spatial relationships: Euclidean
- Sparse or high-dimensional data: Manhattan
- General purpose: Cosine (safest choice)

PERFORMANCE CONSIDERATIONS:
- Dot Product: Fastest
- Cosine: Moderate (requires normalization)
- Manhattan: Moderate
- Euclidean: Slowest (square root calculation)
    """
    
    return guide


if __name__ == "__main__":
    print(create_distance_metrics_guide())