import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["DEBUG"] = "false"

from app.agents.orchestrator import Orchestrator

def test_orchestrator():
    """Test orchestrator with different query types."""
    
    orchestrator = Orchestrator()
    
    test_queries = [
        "What is machine learning?",  # Research only
        "Calculate the mean of [1,2,3,4,5]",  # Code only
        "Find information about Python and show me example code"  # Mixed
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = orchestrator.execute(query)
        
        print(f"\nTask Type: {result['task_type']}")
        print(f"Steps: {' → '.join(result['steps_completed'])}")
        print(f"Time: {result['execution_time_ms']:.0f}ms")
        print(f"\nAnswer:\n{result['answer']}")
        
        if result['errors']:
            print(f"\nErrors: {result['errors']}")

if __name__ == "__main__":
    test_orchestrator()
    
