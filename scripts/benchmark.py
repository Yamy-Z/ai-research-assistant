import requests
import time
import statistics

API_URL = "http://localhost:8000/api/v1"

test_queries = [
    "What is artificial intelligence?",
    "Explain machine learning",
    "What is supervised learning?",
    "How does deep learning work?",
    "What is natural language processing?"
]

def benchmark_queries():
    """Benchmark query performance."""
    latencies = []
    
    print("Running benchmark...")
    for query in test_queries:
        start = time.time()
        
        response = requests.post(
            f"{API_URL}/query/",
            json={"query": query, "top_k": 5}
        )
        
        end = time.time()
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
        
        if response.status_code == 200:
            result = response.json()
            server_time = result['query_time_ms']
            print(f"✓ Query: {query[:40]}...")
            print(f"  Total: {latency_ms:.2f}ms | Server: {server_time:.2f}ms")
        else:
            print(f"✗ Error: {response.status_code}")
    
    print(f"\nBenchmark Results:")
    print(f"  Mean latency: {statistics.mean(latencies):.2f}ms")
    print(f"  Median latency: {statistics.median(latencies):.2f}ms")
    print(f"  Min latency: {min(latencies):.2f}ms")
    print(f"  Max latency: {max(latencies):.2f}ms")

if __name__ == "__main__":
    benchmark_queries()