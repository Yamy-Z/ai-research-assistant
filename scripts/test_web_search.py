import requests

API_URL = "http://localhost:8000/api/v1"

def test_web_search():
    """Test query with web search."""
    
    queries = [
        "What are the latest developments in AI in 2026?",
        "Explain quantum computing breakthroughs",
        "What is the current state of climate change?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        response = requests.post(
            f"{API_URL}/query/",
            json={
                "query": query,
                "top_k": 3,
                "include_sources": True
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\nAnswer:\n{result['answer']}\n")
            print(f"Sources: {len(result['sources'])}")
            print(f"Time: {result['query_time_ms']:.2f}ms\n")
            
            # Show source breakdown
            doc_sources = [s for s in result['sources'] if s['source_type'] == 'document']
            web_sources = [s for s in result['sources'] if s['source_type'] == 'web']
            
            print(f"Documents: {len(doc_sources)}, Web: {len(web_sources)}")
            
            # Show web URLs
            if web_sources:
                print("\nWeb sources:")
                for source in web_sources:
                    print(f"  - {source['metadata']['title']}")
                    print(f"    {source['metadata']['url']}")
        else:
            print(f"Error: {response.status_code}")

if __name__ == "__main__":
    test_web_search()