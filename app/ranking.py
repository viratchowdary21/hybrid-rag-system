import numpy as np
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class HybridRanker:
    def __init__(self):
        # Custom weights for hybrid scoring
        self.dense_weight = 0.6
        self.sparse_weight = 0.4
        self.normalize_scores = True
        
        # Optional cross-encoder for re-ranking
        self.cross_encoder = None
        self.use_cross_encoder = False
        
        # Initialize cross-encoder if needed
        if self.use_cross_encoder:
            try:
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            except:
                print("Cross-encoder not available, continuing without it...")
                self.use_cross_encoder = False
    
    def rank_results(self, query: str, results: List[Dict], top_k: int = 10) -> List[Dict]:
        """Apply hybrid ranking to results"""
        if not results:
            return []
        
        # Calculate final scores using custom formula
        scored_results = self._calculate_hybrid_scores(results)
        
        # Apply cross-encoder re-ranking if available
        if self.use_cross_encoder and self.cross_encoder:
            scored_results = self._cross_encoder_rerank(query, scored_results)
        
        # Sort by final score and return top-k
        scored_results.sort(key=lambda x: x["final_score"], reverse=True)
        return scored_results[:top_k]
    
    def _calculate_hybrid_scores(self, results: List[Dict]) -> List[Dict]:
        """Calculate hybrid scores using custom formula"""
        # Extract scores for normalization
        dense_scores = [r["dense_score"] for r in results if r["dense_score"] > 0]
        sparse_scores = [r["sparse_score"] for r in results if r["sparse_score"] > 0]
        
        # Normalize scores if required
        if self.normalize_scores and dense_scores and sparse_scores:
            max_dense = max(dense_scores) if dense_scores else 1
            max_sparse = max(sparse_scores) if sparse_scores else 1
            
            for result in results:
                # Normalize scores to [0, 1] range
                norm_dense = result["dense_score"] / max_dense if max_dense > 0 else 0
                norm_sparse = result["sparse_score"] / max_sparse if max_sparse > 0 else 0
                
                # Apply custom hybrid formula
                result["final_score"] = self._hybrid_formula(norm_dense, norm_sparse)
        else:
            # Use raw scores with weights
            for result in results:
                result["final_score"] = self._hybrid_formula(
                    result["dense_score"], 
                    result["sparse_score"]
                )
        
        return results
    
    def _hybrid_formula(self, dense_score: float, sparse_score: float) -> float:
        """
        Custom hybrid scoring formula
        
        This formula emphasizes:
        - Strong performance in either retrieval method
        - Balanced contribution from both methods
        - Slight preference for dense retrieval (configurable via weights)
        """
        # Base weighted combination
        base_score = (self.dense_weight * dense_score + 
                     self.sparse_weight * sparse_score)
        
        # Boost for agreement between methods (harmonic mean component)
        if dense_score > 0 and sparse_score > 0:
            agreement_boost = (2 * dense_score * sparse_score) / (dense_score + sparse_score + 1e-8)
            base_score += 0.1 * agreement_boost
        
        # Penalize extreme disparities
        score_diff = abs(dense_score - sparse_score)
        if score_diff > 0.5:  # Large disparity
            base_score *= 0.9
        
        return base_score
    
    def _cross_encoder_rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """Apply cross-encoder re-ranking to top results"""
        if len(results) <= 1:
            return results
        
        # Prepare pairs for cross-encoder
        pairs = [(query, result["content"]) for result in results]
        
        # Get cross-encoder scores
        cross_scores = self.cross_encoder.predict(pairs)
        
        # Update final scores with cross-encoder influence
        for i, result in enumerate(results):
            # Blend cross-encoder score with hybrid score
            cross_score = float(cross_scores[i])
            result["final_score"] = 0.7 * result["final_score"] + 0.3 * cross_score
        
        return results