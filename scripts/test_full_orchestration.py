import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["DEBUG"] = "false"
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql://ai_user:ai_password@localhost:5432/ai_research_db"
)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

from app.agents.orchestrator import get_orchestrator
import time

def test_full_orchestration():
    """Test complete multi-agent system."""
    
    orchestrator = get_orchestrator()
    
    test_cases = [
        {
            "query": "What is machine learning?",
            "expected_type": "research",
            "expected_steps": ["analyze", "research", "aggregate"]
        },
        {
            "query": "Calculate the factorial of 8",
            "expected_type": "code",
            "expected_steps": ["analyze", "code", "aggregate"]
        },
        {
            "query": "Explain Python list comprehensions and show me 3 examples with code",
            "expected_type": "mixed",
            "expected_steps": ["analyze", "research", "code", "aggregate"]
        },
        {
            "query": "What are the latest developments in quantum computing and calculate 2^10",
            "expected_type": "mixed",
            "expected_steps": ["analyze", "research", "code", "aggregate"]
        }
    ]
    
    results = []
    
    for test in test_cases:
        print(f"\n{'='*70}")
        print(f"Test: {test['query']}")
        print('='*70)
        
        result = orchestrator.execute(test['query'])
        
        # Check task type
        task_match = result['task_type'] == test['expected_type']
        
        # Check steps
        steps_match = result['steps_completed'] == test['expected_steps']
        
        # Display results
        print(f"\n✓ Task Type: {result['task_type']} "
              f"{'✓' if task_match else '✗ Expected: ' + test['expected_type']}")
        print(f"✓ Steps: {' → '.join(result['steps_completed'])} "
              f"{'✓' if steps_match else '✗ Expected: ' + ' → '.join(test['expected_steps'])}")
        print(f"✓ Time: {result['execution_time_ms']:.0f}ms")
        
        if result['errors']:
            print(f"✗ Errors: {result['errors']}")
        
        print(f"\nAnswer Preview:\n{result['answer'][:200]}...")
        
        results.append({
            'query': test['query'],
            'task_type_correct': task_match,
            'steps_correct': steps_match,
            'has_errors': len(result['errors']) > 0,
            'execution_time_ms': result['execution_time_ms']
        })
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print('='*70)
    
    task_correct = sum(r['task_type_correct'] for r in results)
    steps_correct = sum(r['steps_correct'] for r in results)
    no_errors = sum(not r['has_errors'] for r in results)
    avg_time = sum(r['execution_time_ms'] for r in results) / len(results)
    
    print(f"Task Type Detection: {task_correct}/{len(results)}")
    print(f"Correct Routing: {steps_correct}/{len(results)}")
    print(f"Error-Free Execution: {no_errors}/{len(results)}")
    print(f"Average Time: {avg_time:.0f}ms")
    
    success_rate = (task_correct + steps_correct + no_errors) / (len(results) * 3)
    print(f"\nOverall Success Rate: {success_rate:.1%}")

if __name__ == "__main__":
    test_full_orchestration()
