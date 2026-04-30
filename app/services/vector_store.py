from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.core.config import get_settings
from app.utils.logger import setup_logger
from typing import List, Dict, Any
import uuid

settings = get_settings()
logger = setup_logger(__name__)


class VectorStore:
    """Vector store service using Qdrant."""
    
    COLLECTION_NAME = "documents"
    
    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url)
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.COLLECTION_NAME not in collection_names:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.COLLECTION_NAME}")
    
    def add_vectors(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add vectors to the collection.
        
        Args:
            vectors: List of embedding vectors
            payloads: List of metadata dictionaries
        
        Returns:
            List of point IDs
        """
        points = []
        ids = []
        
        for vector, payload in zip(vectors, payloads):
            point_id = str(uuid.uuid4())
            ids.append(point_id)
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            )
        
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points
        )
        
        logger.info(f"Added {len(points)} vectors to collection")
        return ids
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding
            limit: Maximum number of results
            score_threshold: Minimum similarity score
        
        Returns:
            List of search results with scores and payloads
        """
        results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in results
        ]
    
    def delete_by_document_id(self, document_id: str):
        """Delete all vectors for a document."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        self.client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        )
        logger.info(f"Deleted vectors for document: {document_id}")


def get_vector_store() -> VectorStore:
    """Get vector store dependency."""
    return VectorStore()