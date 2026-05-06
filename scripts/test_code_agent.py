import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["DEBUG"] = "false"
from app.agents.orchestrator import Orchestrator

def test_code_agent():
    """Test code agent with various tasks."""
    
    orchestrator = Orchestrator()
    
    test_queries = [
        "Calculate the mean of [1, 2, 3, 4, 5]",
        "Generate fibonacci sequence up to 10 numbers",
        "Sort this list: [5, 2, 8, 1, 9] and print the result",
        "Calculate factorial of 10",
        "Find prime numbers between 1 and 20"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = orchestrator.execute(query)
        
        print(f"Task Type: {result['task_type']}")
        print(f"Steps: {' → '.join(result['steps_completed'])}")
        
        if result.get('code_result'):
            code_result = result['code_result']
            print(f"\nCode Execution:")
            print(f"  Success: {code_result['success']}")
            print(f"  Output: {code_result['output']}")
            if code_result.get('error'):
                print(f"  Error: {code_result['error']}")
            print(f"  Time: {code_result['execution_time_ms']:.0f}ms")
        
        print(f"\nFinal Answer:\n{result['answer']}")

if __name__ == "__main__":
    test_code_agent()
