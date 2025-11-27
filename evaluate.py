import json
import time
import numpy as np
from typing import List, Dict
from app.retrieval import HybridRetriever

class RAGEvaluator:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.retriever.initialize()
    
    def calculate_hit_rate(self, retrieved_chunks: List[str], relevant_chunks: List[str], k: int = 5) -> float:
        """Calculate Hit@K metric"""
        top_k = retrieved_chunks[:k]
        hits = len(set(top_k) & set(relevant_chunks))
        return hits / min(k, len(relevant_chunks))
    
    def calculate_mrr(self, retrieved_chunks: List[str], relevant_chunks: List[str]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for rank, chunk_id in enumerate(retrieved_chunks, 1):
            if chunk_id in relevant_chunks:
                return 1.0 / rank
        return 0.0
    
    def evaluate_retrieval(self, test_queries: List[Dict]) -> Dict:
        """Evaluate retrieval performance"""
        hit_rates = []
        mrrs = []
        latencies = []
        
        for test_case in test_queries:
            query = test_case["query"]
            relevant_chunks = test_case["relevant_chunks"]
            
            start_time = time.time()
            results = self.retriever.retrieve(query, top_k=10)
            latency = (time.time() - start_time) * 1000
            
            retrieved_ids = [r["chunk_id"] for r in results]
            
            hit_rate = self.calculate_hit_rate(retrieved_ids, relevant_chunks, k=5)
            mrr = self.calculate_mrr(retrieved_ids, relevant_chunks)
            
            hit_rates.append(hit_rate)
            mrrs.append(mrr)
            latencies.append(latency)
        
        return {
            "hit_rate@5": np.mean(hit_rates),
            "mrr": np.mean(mrrs),
            "avg_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies)
        }

# Example test queries (you would need to create these based on your PDF)
test_queries = [
    {
        "query": "What are the main principles discussed in the document?",
        "relevant_chunks": ["chunk_1", "chunk_5", "chunk_8"]
    }
]

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_retrieval(test_queries)
    print("Evaluation Results:")
    print(json.dumps(results, indent=2))