from app.utils.logger import setup_logger
from typing import List, Dict, Any
import re

logger = setup_logger(__name__)


class CitationService:
    """Service for tracking and managing citations."""
    
    def extract_citations(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract citation references from answer.
        
        Args:
            answer: Generated answer text
            sources: List of source documents
        
        Returns:
            Dictionary with answer and citation mappings
        """
        # Find all [Source N] references
        citation_pattern = r'\[Source (\d+)\]'
        citations = re.findall(citation_pattern, answer)
        
        # Map citations to sources
        citation_map = {}
        for cite_num in set(citations):
            cite_idx = int(cite_num) - 1
            if 0 <= cite_idx < len(sources):
                source = sources[cite_idx]
                citation_map[cite_num] = {
                    'source_number': cite_num,
                    'content': source['content'][:200] + '...',
                    'source_type': source['source_type'],
                    'metadata': source['metadata']
                }
        
        return {
            'answer': answer,
            'citations': citation_map,
            'citation_count': len(citation_map)
        }
    
    def format_answer_with_citations(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> str:
        """
        Format answer with detailed citations.
        
        Args:
            answer: Generated answer
            sources: List of sources
        
        Returns:
            Formatted answer with citations
        """
        formatted = answer + "\n\n## Sources\n\n"
        
        for idx, source in enumerate(sources, 1):
            source_type = source['source_type']
            
            if source_type == 'document':
                filename = source['metadata'].get('filename', 'Unknown')
                formatted += f"[{idx}] Document: {filename}\n"
            elif source_type == 'web':
                title = source['metadata'].get('title', 'Unknown')
                url = source['metadata'].get('url', '')
                formatted += f"[{idx}] Web: {title}\n    URL: {url}\n"
            
            # Add snippet
            content = source['content'][:150] + '...'
            formatted += f"    \"{content}\"\n\n"
        
        return formatted


def get_citation_service() -> CitationService:
    """Get citation service dependency."""
    return CitationService()
