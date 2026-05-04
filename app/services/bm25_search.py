from rank_bm25 import BM25Okapi
from sqlalchemy.orm import Session
from app.models.database import DocumentChunk
from app.utils.logger import setup_logger
from typing import List, Dict, Any
import nltk
from nltk.tokenize import word_tokenize
import os

logger = setup_logger(__name__)

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class BM25SearchService:
    """BM25 keyword-based search service."""
    
    def __init__(self, db: Session):
        self.db = db
        self.bm25 = None
        self.documents = []
        self._index_documents()
    
    def _index_documents(self):
        """Index all document chunks for BM25 search."""
        logger.info("Indexing documents for BM25...")
        
        # Get all chunks
        chunks = self.db.query(DocumentChunk).all()
        
        if not chunks:
            logger.warning("No documents to index for BM25")
            return
        
        # Store documents and tokenize
        self.documents = []
        tokenized_corpus = []
        
        for chunk in chunks:
            self.documents.append({
                'id': chunk.id,
                'document_id': chunk.document_id,
                'content': chunk.content,
                'chunk_index': chunk.chunk_index,
                'metadata': chunk.metadata
            })
            
            # Tokenize content
            tokens = self._tokenize(chunk.content)
            tokenized_corpus.append(tokens)
        
        # Create BM25 index
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"BM25 index created with {len(self.documents)} documents")
        else:
            logger.warning("No documents to index")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        try:
            # Lowercase and tokenize
            tokens = word_tokenize(text.lower())
            # Filter out very short tokens and punctuation
            tokens = [t for t in tokens if len(t) > 2 and t.isalnum()]
            return tokens
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            return text.lower().split()
    
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search documents using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of results with scores
        """
        if not self.bm25:
            logger.warning("BM25 index not initialized")
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                doc = self.documents[idx]
                results.append({
                    'id': doc['id'],
                    'document_id': doc['document_id'],
                    'content': doc['content'],
                    'score': float(scores[idx]),
                    'chunk_index': doc['chunk_index'],
                    'metadata': doc['metadata'],
                    'source_type': 'bm25'
                })
        
        logger.info(f"BM25 search returned {len(results)} results")
        return results
    
    def reindex(self):
        """Rebuild BM25 index (call after adding/removing documents)."""
        self._index_documents()


def get_bm25_service(db: Session) -> BM25SearchService:
    """Get BM25 search service dependency."""
    return BM25SearchService(db)
