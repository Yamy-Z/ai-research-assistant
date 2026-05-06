from typing import Optional
from anthropic import Anthropic
from app.core.config import get_settings
from app.services.code_executor import CodeExecutor
from app.agents.state import AgentState
from app.utils.logger import setup_logger
import json

settings = get_settings()
logger = setup_logger(__name__)


class CodeAgent:
    """Agent that generates and executes code to solve tasks."""
    
    def __init__(self):
        self.client = Anthropic(api_key=settings.anthropic_api_key)
        self.executor = CodeExecutor()
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Generate and execute code based on the query.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state with code execution results
        """
        query = state["query"]
        logger.info(f"Code agent processing: {query}")
        
        # Generate code
        code = self._generate_code(query)
        
        if not code:
            state["code_result"] = {
                "success": False,
                "output": "",
                "error": "Failed to generate code"
            }
            state["errors"].append("Code generation failed")
            return state
        
        logger.info(f"Generated code:\n{code}")
        
        # Execute code
        execution_result = self.executor.execute_python(code)
        
        # Update state
        state["code_result"] = execution_result
        
        if not execution_result["success"]:
            logger.error(f"Code execution failed: {execution_result.get('error')}")
            state["errors"].append(f"Code execution failed: {execution_result.get('error')}")
        else:
            logger.info(f"Code executed successfully: {execution_result['output'][:100]}")
        
        return state
    
    def _generate_code(self, query: str) -> Optional[str]:
        """
        Generate Python code to answer the query.
        
        Args:
            query: User query
        
        Returns:
            Python code string
        """
        prompt = f"""Generate the final Python program to solve this task. The code will run in an isolated environment with only standard library available.

            Task: {query}

            Requirements:
            - Use only Python standard library (no external packages)
            - Code must be complete and runnable
            - Print the final result
            - Keep it simple and focused
            - No file I/O or network access
            - Do not call LLM APIs
            - Do not generate code that generates more code
            - Solve the task directly in Python
            - Maximum 50 lines

            Generate ONLY the Python code, no explanations or markdown:"""

        try:
            response = self.client.messages.create(
                model=settings.llm_model,
                max_tokens=2000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            code = response.content[0].text.strip()
            
            # Remove markdown code blocks if present
            if code.startswith("```python"):
                code = code.split("```python")[1].split("```")[0].strip()
            elif code.startswith("```"):
                code = code.split("```")[1].split("```")[0].strip()
            
            return code
            
        except Exception as e:
            logger.error(f"Code generation error: {e}")
            return None


def create_code_agent_node():
    """Create code agent node for LangGraph."""
    agent = CodeAgent()
    
    def code_node(state: AgentState) -> AgentState:
        state["current_step"] = "code"
        state = agent.execute(state)
        state["steps_completed"].append("code")
        return state
    
    return code_node
