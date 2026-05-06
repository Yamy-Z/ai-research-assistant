from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents.analyzer import create_analyzer_node
from app.agents.code_agent import create_code_agent_node
from app.agents.research_agent import create_research_agent_node
from app.services.embedding import EmbeddingService, get_embedding_service
from app.services.vector_store import VectorStore, get_vector_store
from app.services.bm25_search import BM25SearchService, get_bm25_service
from app.services.hybrid_search import HybridSearchService, get_hybrid_search_service
from app.services.web_search import WebSearchService, get_web_search_service
from app.services.reranker import RerankerService, get_reranker_service
from app.core.database import SessionLocal
from app.utils.logger import setup_logger
from typing import Dict, Any, Optional
import uuid
from datetime import datetime
import time

logger = setup_logger(__name__)


class Orchestrator:
    """Multi-agent orchestrator using LangGraph."""
    
    def __init__(self):
        # Initialize all services
        self.db = SessionLocal()
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store()
        self.bm25_service = get_bm25_service(self.db)
        self.reranker_service = get_reranker_service()
        self.web_search_service = get_web_search_service()
        
        self.hybrid_search_service = get_hybrid_search_service(
            self.embedding_service,
            self.vector_store,
            self.bm25_service,
            self.reranker_service
        )
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze", create_analyzer_node())
        workflow.add_node(
            "research",
            create_research_agent_node(
                self.hybrid_search_service,
                self.web_search_service
            )
        )
        workflow.add_node("code", create_code_agent_node())
        workflow.add_node("aggregate", self._aggregate_node)
        
        # Define routing
        workflow.set_entry_point("analyze")
        
        workflow.add_conditional_edges(
            "analyze",
            self._route_after_analysis,
            {
                "research": "research",
                "code": "code",
                "mixed": "research",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "research",
            self._route_after_research,
            {
                "code": "code",
                "aggregate": "aggregate",
                "end": END
            }
        )
        
        workflow.add_edge("code", "aggregate")
        workflow.add_edge("aggregate", END)
        
        return workflow.compile()
    
    def _route_after_analysis(self, state: AgentState) -> str:
        """Route after task analysis."""
        task_type = state.get("task_type", "research")
        
        if task_type == "research":
            return "research"
        elif task_type == "code":
            return "code"
        elif task_type == "mixed":
            return "research"
        else:
            return "end"
    
    def _route_after_research(self, state: AgentState) -> str:
        """Route after research."""
        if state.get("task_type") == "mixed" and state.get("requires_code"):
            return "code"
        else:
            return "aggregate"
    
    def _aggregate_node(self, state: AgentState) -> AgentState:
        """Aggregate results from all agents."""
        state["current_step"] = "aggregate"
        logger.info("Aggregating results")
        
        parts = []
        sources = []
        
        # Add research results
        if state.get("research_result"):
            research = state["research_result"]
            parts.append(research["answer"])
            sources.extend(research.get("sources", []))
        
        # Add code results
        if state.get("code_result"):
            code = state["code_result"]
            if code["success"]:
                parts.append(f"\nCode Execution Result:\n{code['output']}")
            else:
                parts.append(f"\nCode Execution Failed:\n{code.get('error', 'Unknown error')}")
        
        # Create final answer
        state["answer"] = "\n\n".join(parts) if parts else "No results generated"
        state["sources"] = sources
        state["steps_completed"].append("aggregate")
        
        return state
    
    def execute(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute query through orchestrator.
        
        Args:
            query: User query
            user_id: Optional user ID
        
        Returns:
            Final result dictionary
        """
        start_time = time.time()
        
        initial_state: AgentState = {
            "query": query,
            "user_id": user_id,
            "workflow_id": str(uuid.uuid4()),
            "current_step": "start",
            "steps_completed": [],
            "task_type": None,
            "requires_code": False,
            "requires_research": False,
            "complexity": "simple",
            "research_result": None,
            "code_result": None,
            "answer": None,
            "sources": [],
            "errors": [],
            "execution_time_ms": 0,
            "tokens_used": 0,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        logger.info(f"Starting workflow: {initial_state['workflow_id']}")
        
        try:
            final_state = self.graph.invoke(initial_state)
            final_state["execution_time_ms"] = (time.time() - start_time) * 1000
            
            logger.info(
                f"Workflow completed: {final_state['workflow_id']}, "
                f"steps: {' → '.join(final_state['steps_completed'])}, "
                f"time: {final_state['execution_time_ms']:.0f}ms"
            )
            
            return {
                "workflow_id": final_state["workflow_id"],
                "answer": final_state["answer"],
                "sources": final_state["sources"],
                "task_type": final_state["task_type"],
                "steps_completed": final_state["steps_completed"],
                "execution_time_ms": final_state["execution_time_ms"],
                "errors": final_state["errors"]
            }
            
        except Exception as e:
            logger.error(f"Workflow error: {e}", exc_info=True)
            return {
                "workflow_id": initial_state["workflow_id"],
                "answer": f"Error executing workflow: {e}",
                "sources": [],
                "task_type": "error",
                "steps_completed": initial_state["steps_completed"],
                "execution_time_ms": (time.time() - start_time) * 1000,
                "errors": [str(e)]
            }


# Singleton instance
_orchestrator_instance = None

def get_orchestrator() -> Orchestrator:
    """Get orchestrator singleton."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = Orchestrator()
    return _orchestrator_instance
