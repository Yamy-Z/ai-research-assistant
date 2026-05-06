from anthropic import Anthropic
from app.core.config import get_settings
from app.agents.state import AgentState, TaskType
from app.utils.logger import setup_logger
import json

settings = get_settings()
logger = setup_logger(__name__)


class TaskAnalyzer:
    """Analyzes queries to determine task type and routing."""
    
    def __init__(self):
        self.client = Anthropic(api_key=settings.anthropic_api_key)
    
    def analyze(self, state: AgentState) -> AgentState:
        """
        Analyze the query to determine task type.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state with task analysis
        """
        query = state["query"]
        logger.info(f"Analyzing query: {query}")
        
        analysis_prompt = f"""Analyze this user query and determine what type of task it requires.

            Query: {query}

            Determine:
            1. Does it require RESEARCH (finding information, answering questions)?
            2. Does it require CODE EXECUTION (data analysis, calculations, plotting)?
            3. What is the complexity level?

            Respond ONLY with valid JSON in this exact format:
            {{
                "task_type": "research" or "code" or "mixed",
                "requires_research": true/false,
                "requires_code": true/false,
                "complexity": "simple" or "medium" or "complex",
                "reasoning": "brief explanation"
            }}

            Examples:
            - "What is machine learning?" → {{"task_type": "research", "requires_research": true, "requires_code": false, "complexity": "simple"}}
            - "Calculate the mean of [1,2,3,4,5]" → {{"task_type": "code", "requires_research": false, "requires_code": true, "complexity": "simple"}}
            - "Find data on GDP and plot the trend" → {{"task_type": "mixed", "requires_research": true, "requires_code": true, "complexity": "complex"}}

            Your response (JSON only) in text format:"""

        try:
            response = self.client.messages.create(
                model=settings.llm_model,
                max_tokens=500,
                temperature=0,
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            
            # Parse response
            result_text = response.content[0].text.strip()
            
            # Remove markdown code blocks if present
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            result = json.loads(result_text)
            
            # Update state
            state["task_type"] = result["task_type"]
            state["requires_research"] = result["requires_research"]
            state["requires_code"] = result["requires_code"]
            state["complexity"] = result["complexity"]
            
            logger.info(
                f"Analysis: type={result['task_type']}, "
                f"research={result['requires_research']}, "
                f"code={result['requires_code']}, "
                f"complexity={result['complexity']}"
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            # Default to research on error
            state["task_type"] = TaskType.RESEARCH
            state["requires_research"] = True
            state["requires_code"] = False
            state["complexity"] = "simple"
            state["errors"].append(f"Analysis failed: {e}")
            return state


def create_analyzer_node():
    """Create the analyzer node for LangGraph."""
    analyzer = TaskAnalyzer()
    
    def analyze_node(state: AgentState) -> AgentState:
        state["current_step"] = "analyze"
        state = analyzer.analyze(state)
        state["steps_completed"].append("analyze")
        return state
    
    return analyze_node
