from anthropic import Anthropic
from app.core.config import get_settings
from app.utils.logger import setup_logger
from typing import Dict
import json
import re

settings = get_settings()
logger = setup_logger(__name__)


class QueryAnalyzer:
    """Analyze and decompose complex queries."""
    
    def __init__(self):
        self.client = Anthropic(api_key=settings.anthropic_api_key)
    
    def analyze_query(self, query: str) -> Dict:
        """
        Analyze query complexity and intent.
        
        Returns:
            Dictionary with query analysis
        """
        prompt = f"""Analyze this user query and determine:
            1. Is it a simple or complex question?
            2. Does it need to be broken down into sub-questions?
            3. What type of information is needed? (factual, comparative, explanatory, etc.)

            Query: {query}

            Respond in JSON format only and use text, not markdown:
            {{
                "complexity": "simple" or "complex",
                "needs_decomposition": true/false,
                "query_type": "factual/comparative/explanatory/procedural",
                "sub_questions": ["q1", "q2", ...] or []
            }}"""

        try:
            response = self.client.messages.create(
                model=settings.llm_model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            result = json.loads(response.content[0].text)
            logger.info(f"Query Analyer Result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Query analysis error: {e}")
            return {
                "complexity": "simple",
                "needs_decomposition": False,
                "query_type": "factual",
                "sub_questions": []
            }


def get_query_analyzer() -> QueryAnalyzer:
    """Get query analyzer dependency."""
    return QueryAnalyzer()
