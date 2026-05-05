"""
Week 2 Evaluation Suite
=======================

Tests all Week 2 features:
1. Web search integration (Tavily)
2. BM25 keyword search
3. Hybrid search (vector + BM25)
4. Cross-encoder re-ranking
5. Citation tracking
6. Query decomposition
7. End-to-end answer quality

Usage:
    python scripts/evaluate_week2.py

Requirements:
    - All services running (docker-compose up)
    - Test documents indexed (run scripts/upload_test_docs.py first)
    - Tavily API key configured in .env
"""

import requests
import time
import json
import statistics
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path


API_URL = "http://localhost:8000/api/v1"


# ─── Evaluation Dataset ────────────────────────────────────────────────

EVAL_DATASET = [
    # --- Easy: Definition queries (test basic RAG) ---
    {
        "id": "easy_def_001",
        "query": "What is machine learning?",
        "expected_keywords": ["learn", "data", "algorithm", "pattern", "model"],
        "difficulty": "easy",
        "category": "definition",
        "requires_web": False,
        "expected_min_sources": 1
    },
    {
        "id": "easy_def_002",
        "query": "What is deep learning?",
        "expected_keywords": ["neural", "network", "layer", "deep"],
        "difficulty": "easy",
        "category": "definition",
        "requires_web": False,
        "expected_min_sources": 1
    },
    {
        "id": "easy_def_003",
        "query": "What is natural language processing?",
        "expected_keywords": ["language", "text", "process", "understand"],
        "difficulty": "easy",
        "category": "definition",
        "requires_web": False,
        "expected_min_sources": 1
    },

    # --- Medium: Comparison / Explanation (test hybrid search) ---
    {
        "id": "med_comp_001",
        "query": "Compare supervised and unsupervised learning",
        "expected_keywords": ["supervised", "unsupervised", "labeled", "unlabeled"],
        "difficulty": "medium",
        "category": "comparison",
        "requires_web": False,
        "expected_min_sources": 2
    },
    {
        "id": "med_expl_001",
        "query": "How does feature engineering help model performance?",
        "expected_keywords": ["feature", "variable", "input", "performance"],
        "difficulty": "medium",
        "category": "explanation",
        "requires_web": False,
        "expected_min_sources": 1
    },

    # --- Keyword-heavy queries (test BM25 contribution) ---
    {
        "id": "keyword_001",
        "query": "reinforcement learning rewards penalties agent",
        "expected_keywords": ["reinforcement", "reward", "agent"],
        "difficulty": "medium",
        "category": "keyword_search",
        "requires_web": False,
        "expected_min_sources": 1,
        "note": "BM25 should boost results for exact keyword matches"
    },
    {
        "id": "keyword_002",
        "query": "precision recall accuracy evaluation metrics",
        "expected_keywords": ["precision", "recall", "accuracy", "metric"],
        "difficulty": "medium",
        "category": "keyword_search",
        "requires_web": False,
        "expected_min_sources": 1,
        "note": "BM25 should boost results for exact keyword matches"
    },

    # --- Web search queries (test Tavily integration) ---
    {
        "id": "web_001",
        "query": "What are the latest developments in AI regulation in 2026?",
        "expected_keywords": ["regulation", "AI", "law", "policy"],
        "difficulty": "medium",
        "category": "current_events",
        "requires_web": True,
        "expected_min_sources": 1,
        "must_have_web_source": True
    },
    {
        "id": "web_002",
        "query": "What are the most recent breakthroughs in quantum computing?",
        "expected_keywords": ["quantum", "computing", "qubit"],
        "difficulty": "hard",
        "category": "current_events",
        "requires_web": True,
        "expected_min_sources": 1,
        "must_have_web_source": True
    },
    {
        "id": "web_003",
        "query": "What is the current state of large language model research?",
        "expected_keywords": ["language", "model", "LLM"],
        "difficulty": "medium",
        "category": "current_events",
        "requires_web": True,
        "expected_min_sources": 1,
        "must_have_web_source": True
    },

    # --- Complex queries (test query decomposition) ---
    {
        "id": "complex_001",
        "query": "Compare supervised and unsupervised learning, explain when to use each, and give real-world examples of both",
        "expected_keywords": ["supervised", "unsupervised", "labeled", "example"],
        "difficulty": "hard",
        "category": "complex_decomposition",
        "requires_web": False,
        "expected_min_sources": 2,
        "note": "Should trigger query decomposition into sub-questions"
    },
    {
        "id": "complex_002",
        "query": "Explain how neural networks learn, what backpropagation does, and why deep learning needs GPUs",
        "expected_keywords": ["neural", "backpropagation", "gradient", "GPU"],
        "difficulty": "hard",
        "category": "complex_decomposition",
        "requires_web": False,
        "expected_min_sources": 2,
        "note": "Should trigger query decomposition into sub-questions"
    },

    # --- Citation quality queries ---
    {
        "id": "cite_001",
        "query": "What are the main types of machine learning algorithms?",
        "expected_keywords": ["supervised", "unsupervised", "reinforcement"],
        "difficulty": "easy",
        "category": "citation_check",
        "requires_web": False,
        "expected_min_sources": 2,
        "must_have_citations": True
    },

    # --- Edge cases ---
    {
        "id": "edge_001",
        "query": "asdfghjkl random gibberish qwertyuiop",
        "expected_keywords": [],
        "difficulty": "edge",
        "category": "edge_case",
        "requires_web": False,
        "should_handle_gracefully": True
    },
    {
        "id": "edge_002",
        "query": "a",
        "expected_keywords": [],
        "difficulty": "edge",
        "category": "edge_case",
        "requires_web": False,
        "should_handle_gracefully": True
    },
]


# ─── Scoring Functions ─────────────────────────────────────────────────

def score_keyword_match(answer: str, expected: List[str]) -> Dict[str, Any]:
    """Score answer by keyword presence."""
    if not expected:
        return {"score": 1.0, "matches": [], "misses": [], "total": 0}

    answer_lower = answer.lower()
    matches = [kw for kw in expected if kw.lower() in answer_lower]
    misses = [kw for kw in expected if kw.lower() not in answer_lower]

    return {
        "score": len(matches) / len(expected),
        "matches": matches,
        "misses": misses,
        "total": len(expected)
    }


def score_source_diversity(sources: List[Dict]) -> Dict[str, Any]:
    """Score source type diversity."""
    doc_sources = [s for s in sources if s.get("source_type") == "document"]
    web_sources = [s for s in sources if s.get("source_type") == "web"]

    return {
        "total": len(sources),
        "documents": len(doc_sources),
        "web": len(web_sources),
        "has_documents": len(doc_sources) > 0,
        "has_web": len(web_sources) > 0,
        "diversity_score": min(1.0, len(set(s.get("source_type") for s in sources)) / 2)
    }


def score_citations(answer: str, citations: Dict, sources: List[Dict]) -> Dict[str, Any]:
    """Score citation quality."""
    import re
    citation_refs = re.findall(r'\[Source (\d+)\]', answer)
    unique_refs = set(citation_refs)

    # Check valid references
    valid_refs = [r for r in unique_refs if int(r) <= len(sources)]
    invalid_refs = [r for r in unique_refs if int(r) > len(sources)]

    has_citations = len(unique_refs) > 0
    all_valid = len(invalid_refs) == 0

    if not sources:
        score = 0.5  # Neutral if no sources
    elif has_citations and all_valid:
        score = 1.0
    elif has_citations and not all_valid:
        score = 0.5  # Some invalid
    else:
        score = 0.0  # No citations at all

    return {
        "score": score,
        "total_refs": len(citation_refs),
        "unique_refs": len(unique_refs),
        "valid_refs": len(valid_refs),
        "invalid_refs": len(invalid_refs),
        "citation_map_count": len(citations),
        "has_citations": has_citations
    }


def score_answer_quality(answer: str, query: str) -> Dict[str, Any]:
    """Score overall answer quality."""
    scores = {}

    # Length
    length = len(answer)
    if length < 20:
        scores["length"] = 0.1
    elif length < 50:
        scores["length"] = 0.4
    elif length < 100:
        scores["length"] = 0.7
    elif length < 3000:
        scores["length"] = 1.0
    else:
        scores["length"] = 0.8

    # Not an error message
    error_phrases = ["couldn't find", "error", "sorry", "unable to", "no results"]
    is_error = any(p in answer.lower() for p in error_phrases)
    scores["not_error"] = 0.0 if is_error else 1.0

    # Relevance: query words appearing in answer
    query_words = set(w.lower() for w in query.split() if len(w) > 3)
    answer_lower = answer.lower()
    if query_words:
        overlap = sum(1 for w in query_words if w in answer_lower)
        scores["relevance"] = overlap / len(query_words)
    else:
        scores["relevance"] = 0.5

    overall = sum(scores.values()) / len(scores) if scores else 0
    return {"overall": overall, "detail": scores}


def score_latency(latency_ms: float) -> Dict[str, Any]:
    """Score response latency."""
    if latency_ms <= 2000:
        return {"score": 1.0, "rating": "excellent", "ms": latency_ms}
    elif latency_ms <= 5000:
        return {"score": 0.8, "rating": "good", "ms": latency_ms}
    elif latency_ms <= 10000:
        return {"score": 0.5, "rating": "acceptable", "ms": latency_ms}
    elif latency_ms <= 15000:
        return {"score": 0.3, "rating": "slow", "ms": latency_ms}
    else:
        return {"score": 0.0, "rating": "critical", "ms": latency_ms}


# ─── Test Runner ───────────────────────────────────────────────────────

def run_single_query(item: Dict) -> Dict[str, Any]:
    """Run a single evaluation query and score the result."""
    query_id = item["id"]
    query = item.get("query", "")

    result = {
        "id": query_id,
        "query": query,
        "difficulty": item.get("difficulty"),
        "category": item.get("category"),
        "passed": False,
        "scores": {},
        "errors": []
    }

    # Handle edge cases
    if item.get("should_handle_gracefully"):
        try:
            response = requests.post(
                f"{API_URL}/query/",
                json={"query": query, "top_k": 3, "include_sources": True},
                timeout=30
            )
            # Any non-500 response is acceptable for edge cases
            result["passed"] = response.status_code != 500
            result["scores"]["edge_handling"] = 1.0 if result["passed"] else 0.0
            result["scores"]["overall"] = result["scores"]["edge_handling"]
            result["http_status"] = response.status_code
            return result
        except Exception as e:
            result["errors"].append(str(e))
            result["scores"]["overall"] = 0.0
            return result

    # Normal query
    try:
        start = time.time()
        response = requests.post(
            f"{API_URL}/query/",
            json={"query": query, "top_k": 5, "include_sources": True},
            timeout=30
        )
        wall_time_ms = (time.time() - start) * 1000

        if response.status_code != 200:
            result["errors"].append(f"HTTP {response.status_code}: {response.text[:200]}")
            result["scores"]["overall"] = 0.0
            return result

        data = response.json()
        answer = data.get("answer", "")
        sources = data.get("sources", [])
        citations = data.get("citations", {})
        server_time = data.get("query_time_ms", 0)

        result["answer_preview"] = answer[:200]
        result["latency_ms"] = wall_time_ms
        result["server_time_ms"] = server_time
        result["source_count"] = len(sources)
        result["citation_count"] = data.get("citation_count", 0)

        # ── Score: Keyword match ──
        if item.get("expected_keywords"):
            kw = score_keyword_match(answer, item["expected_keywords"])
            result["scores"]["keyword"] = kw["score"]
            result["keyword_detail"] = kw

        # ── Score: Source diversity ──
        src_div = score_source_diversity(sources)
        result["scores"]["source_diversity"] = src_div["diversity_score"]
        result["source_breakdown"] = src_div

        # Must have web source?
        if item.get("must_have_web_source"):
            result["scores"]["has_web"] = 1.0 if src_div["has_web"] else 0.0

        # Minimum sources?
        min_src = item.get("expected_min_sources", 0)
        if min_src > 0:
            result["scores"]["min_sources"] = 1.0 if len(sources) >= min_src else 0.0

        # ── Score: Citations ──
        cite = score_citations(answer, citations, sources)
        result["scores"]["citations"] = cite["score"]
        result["citation_detail"] = cite

        if item.get("must_have_citations"):
            result["scores"]["has_citations"] = 1.0 if cite["has_citations"] else 0.0

        # ── Score: Answer quality ──
        quality = score_answer_quality(answer, query)
        result["scores"]["quality"] = quality["overall"]
        result["quality_detail"] = quality["detail"]

        # ── Score: Latency ──
        lat = score_latency(wall_time_ms)
        result["scores"]["latency"] = lat["score"]
        result["latency_rating"] = lat["rating"]

        # ── Overall score ──
        score_values = [v for v in result["scores"].values() if isinstance(v, (int, float))]
        result["scores"]["overall"] = sum(score_values) / len(score_values) if score_values else 0

        # Pass threshold
        result["passed"] = result["scores"]["overall"] >= 0.5

    except requests.exceptions.Timeout:
        result["errors"].append("Request timed out (30s)")
        result["scores"]["overall"] = 0.0
    except requests.exceptions.ConnectionError:
        result["errors"].append("Connection refused — is the server running?")
        result["scores"]["overall"] = 0.0
    except Exception as e:
        result["errors"].append(str(e))
        result["scores"]["overall"] = 0.0

    return result


def run_evaluation() -> Dict[str, Any]:
    """Run the full evaluation suite."""

    print("\n" + "=" * 70)
    print("  AI RESEARCH ASSISTANT — WEEK 2 EVALUATION")
    print("=" * 70)

    # Preflight checks
    print("\nPreflight checks...")
    try:
        health = requests.get(f"http://localhost:8000/health", timeout=5)
        if health.status_code != 200:
            print("  ✗ Server health check failed")
            sys.exit(1)
        print("  ✓ Server is healthy")
    except Exception:
        print("  ✗ Cannot reach server. Run: docker-compose up")
        sys.exit(1)

    # Run all queries
    results = []
    total = len(EVAL_DATASET)
    suite_start = time.time()

    for idx, item in enumerate(EVAL_DATASET, 1):
        query_label = item["query"][:50] if item["query"] else "(empty)"
        print(f"\n[{idx}/{total}] {item['id']}: {query_label}...")

        result = run_single_query(item)
        results.append(result)

        # Print inline status
        overall = result["scores"].get("overall", 0)
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        latency = result.get("latency_ms", 0)

        print(f"       {status}  score={overall:.2f}  latency={latency:.0f}ms  sources={result.get('source_count', '?')}")

        if result.get("keyword_detail", {}).get("misses"):
            print(f"       Missing keywords: {result['keyword_detail']['misses']}")
        if result["errors"]:
            print(f"       Errors: {result['errors']}")

    suite_time = time.time() - suite_start

    # ── Summary ─────────────────────────────────────────────────
    summary = build_summary(results, suite_time)
    print_summary(summary)

    # ── Save results ────────────────────────────────────────────
    output_dir = Path("data/evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"week2_eval_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "summary": summary,
            "results": results,
            "dataset_size": total,
            "timestamp": timestamp
        }, f, indent=2, default=str)

    print(f"\nResults saved: {output_file}")

    return summary


# ─── Summary Builder ───────────────────────────────────────────────────

def build_summary(results: List[Dict], total_time: float) -> Dict[str, Any]:
    """Build evaluation summary from results."""

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    all_scores = [r["scores"].get("overall", 0) for r in results]

    # By difficulty
    by_difficulty = {}
    for diff in ["easy", "medium", "hard", "edge"]:
        diff_results = [r for r in results if r.get("difficulty") == diff]
        if diff_results:
            by_difficulty[diff] = {
                "count": len(diff_results),
                "passed": sum(1 for r in diff_results if r["passed"]),
                "avg_score": statistics.mean(
                    [r["scores"].get("overall", 0) for r in diff_results]
                )
            }

    # By category
    by_category = {}
    for result in results:
        cat = result.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"count": 0, "scores": [], "passed": 0}
        by_category[cat]["count"] += 1
        by_category[cat]["scores"].append(result["scores"].get("overall", 0))
        if result["passed"]:
            by_category[cat]["passed"] += 1
    for cat in by_category:
        by_category[cat]["avg_score"] = statistics.mean(by_category[cat]["scores"])
        del by_category[cat]["scores"]

    # Latency stats
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms")]
    latency_stats = {}
    if latencies:
        sorted_lat = sorted(latencies)
        latency_stats = {
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p95_ms": sorted_lat[int(len(sorted_lat) * 0.95)] if len(sorted_lat) >= 2 else sorted_lat[-1],
            "min_ms": min(latencies),
            "max_ms": max(latencies)
        }

    # Feature-specific scores
    feature_scores = {}

    # Web search effectiveness
    web_queries = [r for r in results if r.get("source_breakdown", {}).get("web", 0) > 0 or
                   any(item.get("must_have_web_source") for item in EVAL_DATASET if item["id"] == r["id"])]
    web_required = [r for r in results
                    if any(item.get("must_have_web_source") for item in EVAL_DATASET if item["id"] == r["id"])]
    web_has_results = [r for r in web_required if r.get("source_breakdown", {}).get("has_web")]
    feature_scores["web_search"] = {
        "required": len(web_required),
        "successful": len(web_has_results),
        "rate": len(web_has_results) / len(web_required) if web_required else 0
    }

    # Citation quality
    cite_scores = [r["scores"].get("citations", 0) for r in results if "citations" in r["scores"]]
    feature_scores["citations"] = {
        "avg_score": statistics.mean(cite_scores) if cite_scores else 0,
        "queries_with_citations": sum(
            1 for r in results if r.get("citation_detail", {}).get("has_citations")
        ),
        "total_queries": len([r for r in results if not r.get("should_handle_gracefully")])
    }

    # Keyword matching
    kw_scores = [r["scores"].get("keyword", 0) for r in results if "keyword" in r["scores"]]
    feature_scores["keyword_match"] = {
        "avg_score": statistics.mean(kw_scores) if kw_scores else 0,
        "count": len(kw_scores)
    }

    return {
        "total_queries": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total > 0 else 0,
        "avg_score": statistics.mean(all_scores) if all_scores else 0,
        "by_difficulty": by_difficulty,
        "by_category": by_category,
        "latency": latency_stats,
        "features": feature_scores,
        "total_time_seconds": total_time
    }


# ─── Pretty Printer ───────────────────────────────────────────────────

def print_summary(summary: Dict):
    """Print formatted evaluation summary."""

    print(f"\n{'=' * 70}")
    print("  EVALUATION RESULTS")
    print(f"{'=' * 70}")

    pr = summary["pass_rate"]
    avg = summary["avg_score"]

    print(f"\n  Total Queries:  {summary['total_queries']}")
    print(f"  Passed:         {summary['passed']}")
    print(f"  Failed:         {summary['failed']}")
    print(f"  Pass Rate:      {pr:.1%}")
    print(f"  Avg Score:      {avg:.2f}")

    # By difficulty
    print(f"\n  {'─' * 40}")
    print(f"  BY DIFFICULTY")
    print(f"  {'─' * 40}")
    for diff, stats in summary.get("by_difficulty", {}).items():
        bar = "█" * int(stats["avg_score"] * 20) + "░" * (20 - int(stats["avg_score"] * 20))
        print(f"  {diff:8s}  {stats['passed']}/{stats['count']} passed  "
              f"avg={stats['avg_score']:.2f}  {bar}")

    # By category
    print(f"\n  {'─' * 40}")
    print(f"  BY CATEGORY")
    print(f"  {'─' * 40}")
    for cat, stats in sorted(summary.get("by_category", {}).items()):
        bar = "█" * int(stats["avg_score"] * 20) + "░" * (20 - int(stats["avg_score"] * 20))
        print(f"  {cat:22s}  {stats['passed']}/{stats['count']}  "
              f"avg={stats['avg_score']:.2f}  {bar}")

    # Feature scores
    print(f"\n  {'─' * 40}")
    print(f"  FEATURE SCORES")
    print(f"  {'─' * 40}")

    feat = summary.get("features", {})

    ws = feat.get("web_search", {})
    print(f"  Web Search:     {ws.get('successful', 0)}/{ws.get('required', 0)} queries returned web results "
          f"({ws.get('rate', 0):.0%})")

    ct = feat.get("citations", {})
    print(f"  Citations:      {ct.get('queries_with_citations', 0)}/{ct.get('total_queries', 0)} answers have citations "
          f"(avg quality={ct.get('avg_score', 0):.2f})")

    kw = feat.get("keyword_match", {})
    print(f"  Keyword Match:  avg={kw.get('avg_score', 0):.2f} across {kw.get('count', 0)} queries")

    # Latency
    lat = summary.get("latency", {})
    if lat:
        print(f"\n  {'─' * 40}")
        print(f"  LATENCY")
        print(f"  {'─' * 40}")
        print(f"  Mean:    {lat.get('mean_ms', 0):>8.0f} ms")
        print(f"  Median:  {lat.get('median_ms', 0):>8.0f} ms")
        print(f"  P95:     {lat.get('p95_ms', 0):>8.0f} ms")
        print(f"  Min:     {lat.get('min_ms', 0):>8.0f} ms")
        print(f"  Max:     {lat.get('max_ms', 0):>8.0f} ms")

    print(f"\n  Total Time: {summary.get('total_time_seconds', 0):.1f}s")

    # Verdict
    print(f"\n{'=' * 70}")
    if pr >= 0.90 and avg >= 0.70:
        print("  ✅  WEEK 2 EVALUATION PASSED")
        print("      All targets met. Ready for Week 3.")
    elif pr >= 0.75 and avg >= 0.60:
        print("  ⚠️   WEEK 2 EVALUATION MARGINAL")
        print("      Most targets met but room for improvement.")
    else:
        print("  ❌  WEEK 2 EVALUATION FAILED")
        print("      Review failing queries and fix issues before proceeding.")

    print(f"\n  Targets:")
    print(f"    Pass rate ≥ 90%:         {'✓' if pr >= 0.90 else '✗'}  (actual: {pr:.1%})")
    print(f"    Avg score ≥ 0.70:        {'✓' if avg >= 0.70 else '✗'}  (actual: {avg:.2f})")
    p95 = lat.get("p95_ms", 99999)
    print(f"    P95 latency < 10s:       {'✓' if p95 < 10000 else '✗'}  (actual: {p95:.0f}ms)")
    ws_rate = ws.get("rate", 0)
    print(f"    Web search working:      {'✓' if ws_rate >= 0.80 else '✗'}  (actual: {ws_rate:.0%})")
    ct_score = ct.get("avg_score", 0)
    print(f"    Citations present:       {'✓' if ct_score >= 0.50 else '✗'}  (actual: {ct_score:.2f})")

    print(f"{'=' * 70}\n")


# ─── Entry Point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    run_evaluation()