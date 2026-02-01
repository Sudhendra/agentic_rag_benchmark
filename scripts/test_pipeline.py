#!/usr/bin/env python3
"""Quick test script to validate the pipeline works end-to-end."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


async def main():
    """Run a quick validation test."""
    from src.core.llm_client import OpenAIClient
    from src.core.types import Question, QuestionType
    from src.data.hotpotqa import load_hotpotqa
    from src.retrieval.bm25 import BM25Retriever
    from src.architectures.vanilla_rag import VanillaRAG
    from src.evaluation.metrics import exact_match, f1_score
    from src.utils.cache import SQLiteCache
    
    print("=" * 60)
    print("Agentic RAG Benchmark - Pipeline Validation Test")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY not set")
        print("Please set your API key in a .env file or environment variable")
        return
    
    print("\n1. Loading HotpotQA (5 questions for testing)...")
    questions, corpus = load_hotpotqa(
        setting="distractor",
        split="validation",
        subset_size=5,
    )
    print(f"   ✓ Loaded {len(questions)} questions and {len(corpus)} documents")
    
    print("\n2. Setting up components...")
    cache = SQLiteCache(".cache/test_cache.db")
    llm = OpenAIClient(model="gpt-4o-mini", cache=cache)
    retriever = BM25Retriever()
    rag = VanillaRAG(llm, retriever, {"top_k": 5})
    print("   ✓ LLM Client: gpt-4o-mini")
    print("   ✓ Retriever: BM25")
    print("   ✓ Architecture: Vanilla RAG")
    
    print("\n3. Running inference on first question...")
    q = questions[0]
    print(f"   Question: {q.text}")
    print(f"   Gold answer: {q.gold_answer}")
    
    # Get corpus for this question (HotpotQA distractor provides per-question corpus)
    q_corpus = [doc for doc in corpus if doc.id.startswith(q.id)]
    if not q_corpus:
        q_corpus = corpus[:10]  # Fallback to first 10 docs
    
    response = await rag.answer(q, q_corpus)
    
    print(f"\n   Predicted: {response.answer}")
    print(f"   Tokens: {response.total_tokens}")
    print(f"   Cost: ${response.total_cost_usd:.6f}")
    print(f"   Latency: {response.latency_ms:.2f}ms")
    
    print("\n4. Computing metrics...")
    em = exact_match(response.answer, q.gold_answer or "")
    f1 = f1_score(response.answer, q.gold_answer or "")
    print(f"   Exact Match: {em:.2f}")
    print(f"   F1 Score: {f1:.2f}")
    
    print("\n5. Cache stats...")
    print(f"   Cache entries: {cache.size()}")
    
    print("\n" + "=" * 60)
    print("✓ Pipeline validation PASSED!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Create .env file with OPENAI_API_KEY if not done")
    print("  2. Run: python -m pytest tests/ -v")
    print("  3. Implement additional architectures")
    
    cache.close()


if __name__ == "__main__":
    asyncio.run(main())
