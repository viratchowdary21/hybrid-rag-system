import os
import pickle
import numpy as np
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import faiss
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
from app.ranking import HybridRanker

class HybridRetriever:
    def __init__(self):
        self.pdf_path = "data/sample.pdf"
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.embedding_model = None
        self.vector_store = None
        self.bm25_index = None
        self.chunks = []
        self.chunk_metadata = []
        self.ranker = HybridRanker()
        
    def initialize(self):
        """Initialize the retriever by loading and processing PDF"""
        print("Initializing Hybrid Retriever...")
        
        if not os.path.exists("data/processed"):
            os.makedirs("data/processed")
            
        # Check if processed data exists
        if (os.path.exists("data/processed/chunks.pkl") and 
            os.path.exists("data/processed/embeddings.index")):
            self._load_processed_data()
        else:
            self._process_pdf()
            self._build_indices()
            self._save_processed_data()
            
        print(f"Initialized with {len(self.chunks)} chunks")
    
    def _process_pdf(self):
        """Load and chunk PDF document"""
        print("Processing PDF...")
        
        doc = fitz.open(self.pdf_path)
        chunks = []
        metadata = []
        
        chunk_id = 0
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Simple text chunking
            words = text.split()
            for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = " ".join(chunk_words)
                
                if len(chunk_text.strip()) > 50:  # Minimum chunk length
                    chunks.append(chunk_text)
                    metadata.append({
                        "chunk_id": f"chunk_{chunk_id}",
                        "page_number": page_num + 1,
                        "start_word": i,
                        "end_word": i + len(chunk_words)
                    })
                    chunk_id += 1
        
        self.chunks = chunks
        self.chunk_metadata = metadata
        doc.close()
    
    def _build_indices(self):
        """Build both sparse and dense indices"""
        print("Building indices...")
        
        # Build BM25 sparse index
        tokenized_corpus = [doc.split() for doc in self.chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # Build TF-IDF embeddings as a simple alternative
        self._build_tfidf_embeddings()
    
    def _build_tfidf_embeddings(self):
        """Build TF-IDF embeddings as a lightweight alternative"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(max_features=512, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.chunks)
        
        # Convert to dense array for FAISS
        embeddings = tfidf_matrix.toarray().astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        self.vector_store = faiss.IndexFlatIP(embeddings.shape[1])
        self.vector_store.add(embeddings)
        self.vectorizer = vectorizer
        
    def _load_processed_data(self):
        """Load pre-processed data"""
        with open("data/processed/chunks.pkl", "rb") as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.chunk_metadata = data['metadata']
            
        self.vector_store = faiss.read_index("data/processed/embeddings.index")
        
        # Rebuild BM25
        tokenized_corpus = [doc.split() for doc in self.chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # Rebuild vectorizer
        self._build_tfidf_embeddings()
    
    def _save_processed_data(self):
        """Save processed data for future use"""
        with open("data/processed/chunks.pkl", "wb") as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.chunk_metadata
            }, f)
            
        faiss.write_index(self.vector_store, "data/processed/embeddings.index")
    
    def dense_retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Dense retrieval using TF-IDF similarity"""
        # Transform query to TF-IDF
        query_vec = self.vectorizer.transform([query]).toarray().astype('float32')
        faiss.normalize_L2(query_vec)
        
        # Search in FAISS
        scores, indices = self.vector_store.search(query_vec, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "chunk_id": self.chunk_metadata[idx]["chunk_id"],
                    "content": self.chunks[idx],
                    "dense_score": float(score),
                    "sparse_score": 0.0,
                    "final_score": 0.0,
                    "metadata": self.chunk_metadata[idx]
                })
        
        return results
    
    def sparse_retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Sparse retrieval using BM25"""
        tokenized_query = query.split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:  # Only include relevant results
                results.append({
                    "chunk_id": self.chunk_metadata[idx]["chunk_id"],
                    "content": self.chunks[idx],
                    "dense_score": 0.0,
                    "sparse_score": float(bm25_scores[idx]),
                    "final_score": 0.0,
                    "metadata": self.chunk_metadata[idx]
                })
        
        return results
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Hybrid retrieval combining dense and sparse methods"""
        # Get results from both retrievers
        dense_results = self.dense_retrieve(query, top_k * 2)
        sparse_results = self.sparse_retrieve(query, top_k * 2)
        
        # Merge results and apply hybrid ranking
        all_results = self._merge_results(dense_results, sparse_results)
        ranked_results = self.ranker.rank_results(query, all_results, top_k)
        
        return ranked_results
    
    def _merge_results(self, dense_results: List[Dict], sparse_results: List[Dict]) -> List[Dict]:
        """Merge results from both retrievers"""
        merged = {}
        
        # Add dense results
        for result in dense_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in merged:
                merged[chunk_id] = result
            else:
                merged[chunk_id]["dense_score"] = result["dense_score"]
        
        # Add sparse results
        for result in sparse_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in merged:
                merged[chunk_id] = result
            else:
                merged[chunk_id]["sparse_score"] = result["sparse_score"]
        
        return list(merged.values())