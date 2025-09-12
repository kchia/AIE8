import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Dict, Any, Optional
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio
from datetime import datetime


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)


def euclidean_distance(vector_a: np.array, vector_b: np.array) -> float:
    """
    Computes the Euclidean distance between two vectors.
    Returns the inverse (1 / (1 + distance)) to maintain higher-is-better semantics.
    """
    distance = np.linalg.norm(vector_a - vector_b)
    # Return inverse for similarity semantics (higher = more similar)
    return 1.0 / (1.0 + distance)


def manhattan_distance(vector_a: np.array, vector_b: np.array) -> float:
    """
    Computes the Manhattan (L1) distance between two vectors.
    Returns the inverse (1 / (1 + distance)) to maintain higher-is-better semantics.
    """
    distance = np.sum(np.abs(vector_a - vector_b))
    # Return inverse for similarity semantics (higher = more similar)
    return 1.0 / (1.0 + distance)


def dot_product_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """
    Computes the dot product similarity between two vectors.
    Note: This assumes vectors are normalized or you want raw dot product.
    """
    return float(np.dot(vector_a, vector_b))


class VectorDatabase:
    # Available distance metrics
    DISTANCE_METRICS = {
        'cosine': cosine_similarity,
        'euclidean': euclidean_distance,
        'manhattan': manhattan_distance,
        'dot_product': dot_product_similarity
    }
    
    def __init__(self, embedding_model: EmbeddingModel = None, default_distance_metric: str = 'cosine'):
        self.vectors = defaultdict(np.array)
        self.metadata = defaultdict(dict)  # Store metadata for each vector
        self.embedding_model = embedding_model  # Allow None for basic testing
        
        # Set default distance metric
        if default_distance_metric not in self.DISTANCE_METRICS:
            raise ValueError(f"Unknown distance metric: {default_distance_metric}. "
                           f"Available metrics: {list(self.DISTANCE_METRICS.keys())}")
        
        self.default_distance_metric = default_distance_metric

    def get_distance_function(self, distance_measure: Optional[Callable] = None) -> Callable:
        """Get the distance function to use for calculations."""
        if distance_measure is not None:
            return distance_measure
        return self.DISTANCE_METRICS[self.default_distance_metric]
    
    def set_default_distance_metric(self, metric_name: str) -> None:
        """Set the default distance metric for this database."""
        if metric_name not in self.DISTANCE_METRICS:
            raise ValueError(f"Unknown distance metric: {metric_name}. "
                           f"Available metrics: {list(self.DISTANCE_METRICS.keys())}")
        self.default_distance_metric = metric_name
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available distance metrics."""
        return list(self.DISTANCE_METRICS.keys())

    def insert(self, key: str, vector: np.array, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.vectors[key] = vector
        if metadata is None:
            metadata = {}
        # Add timestamp if not provided
        if 'timestamp' not in metadata:
            metadata['timestamp'] = datetime.now().isoformat()
        self.metadata[key] = metadata

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Optional[Callable] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        return_metadata: bool = False,
    ):
        # Get the appropriate distance function
        distance_func = self.get_distance_function(distance_measure)
        
        # Filter vectors based on metadata criteria
        filtered_items = []
        for key, vector in self.vectors.items():
            if filter_criteria:
                metadata = self.metadata.get(key, {})
                if not self._matches_criteria(metadata, filter_criteria):
                    continue
            filtered_items.append((key, vector))
        
        scores = [
            (key, distance_func(query_vector, vector))
            for key, vector in filtered_items
        ]
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
        
        if return_metadata:
            return [(key, score, self.metadata.get(key, {})) for key, score in sorted_scores]
        return sorted_scores

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Optional[Callable] = None,
        return_as_text: bool = False,
        filter_criteria: Optional[Dict[str, Any]] = None,
        return_metadata: bool = False,
    ):
        if self.embedding_model is None:
            raise ValueError("Embedding model not set. Cannot search by text without an embedding model.")
        
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure, filter_criteria, return_metadata)
        
        if return_as_text:
            return [result[0] for result in results]
        return results

    def retrieve_from_key(self, key: str, include_metadata: bool = False):
        vector = self.vectors.get(key, None)
        if include_metadata:
            metadata = self.metadata.get(key, {})
            return (vector, metadata) if vector is not None else (None, {})
        return vector

    def _matches_criteria(self, metadata: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter criteria."""
        for key, value in criteria.items():
            if key not in metadata:
                return False
            if isinstance(value, dict) and "$in" in value:
                if metadata[key] not in value["$in"]:
                    return False
            elif isinstance(value, dict) and "$eq" in value:
                if metadata[key] != value["$eq"]:
                    return False
            elif metadata[key] != value:
                return False
        return True
    
    async def abuild_from_list(self, list_of_text: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None) -> "VectorDatabase":
        if self.embedding_model is None:
            raise ValueError("Embedding model not set. Cannot build vector database without an embedding model.")
        
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for i, (text, embedding) in enumerate(zip(list_of_text, embeddings)):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
            self.insert(text, np.array(embedding), metadata)
        return self


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
