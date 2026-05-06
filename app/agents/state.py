from typing import TypedDict, List, Dict, Any, Optional, Literal
from datetime import datetime


class AgentState(TypedDict):
    """
    State schema for multi-agent orchestration.
    
    This state is passed between agents and tracks the entire workflow.
    """
    # Input
    query: str                          # Original user query
    user_id: Optional[str]              # User identifier
    
    # Workflow tracking
    workflow_id: str                    # Unique workflow identifier
    current_step: str                   # Current execution step
    steps_completed: List[str]          # Steps completed so far
    
    # Analysis
    task_type: Optional[str]            # "research", "code", "mixed"
    requires_code: bool                 # Whether code execution needed
    requires_research: bool             # Whether research needed
    complexity: str                     # "simple", "medium", "complex"
    
    # Agent results
    research_result: Optional[Dict[str, Any]]   # Results from research agent
    code_result: Optional[Dict[str, Any]]       # Results from code agent
    
    # Final output
    answer: Optional[str]               # Final answer to user
    sources: List[Dict[str, Any]]       # All sources used
    
    # Metadata
    errors: List[str]                   # Any errors encountered
    execution_time_ms: float            # Total execution time
    tokens_used: int                    # Total tokens consumed
    created_at: datetime
    updated_at: datetime


class TaskType(str):
    """Task type enumeration."""
    RESEARCH = "research"
    CODE = "code"
    MIXED = "mixed"


class WorkflowStep(str):
    """Workflow step enumeration."""
    START = "start"
    ANALYZE = "analyze"
    RESEARCH = "research"
    CODE = "code"
    AGGREGATE = "aggregate"
    END = "end"