#!/usr/bin/env python3
"""
Vector Search Benchmarking Script for Gift Cards Dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

class VectorSearchBenchmark:
    def __init__(self, dataset_path: str):
        """Initialize the benchmark with the dataset"""
        print(f"Loading dataset from {dataset_path}")
        self.df = pd.read_parquet(dataset_path)
        self.embeddings = np.array(self.df['embedding'].tolist())
        print(f"Loaded {len(self.df)} records with {self.embeddings.shape[1]}-dimensional embeddings")
        
        # Load the same model used for dataset creation
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def similarity_search(self, query_text: str, top_k: int = 10, 
                         filters: Dict[str, Any] = None) -> Tuple[List[int], List[float], float]:
        """
        Perform similarity search with optional metadata filters
        Returns: (indices, similarities, search_time)
        """
        start_time = time.time()
        
        # Encode query
        query_embedding = self.model.encode([query_text])
        
        # Apply filters first to reduce search space
        if filters:
            mask = self._apply_filters(filters)
            filtered_df = self.df[mask]
            filtered_embeddings = self.embeddings[mask]
            filtered_indices = filtered_df.index.tolist()
        else:
            filtered_df = self.df
            filtered_embeddings = self.embeddings
            filtered_indices = list(range(len(self.df)))
        
        if len(filtered_embeddings) == 0:
            return [], [], time.time() - start_time
        
        # Compute similarities
        similarities = cosine_similarity(query_embedding, filtered_embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        # Map back to original indices
        original_indices = [filtered_indices[i] for i in top_indices]
        
        search_time = time.time() - start_time
        return original_indices, top_similarities.tolist(), search_time
    
    def _apply_filters(self, filters: Dict[str, Any]) -> np.ndarray:
        """Apply metadata filters and return boolean mask"""
        mask = np.ones(len(self.df), dtype=bool)
        
        for key, value in filters.items():
            if key == 'min_rating':
                mask &= (self.df['average_rating'] >= value)
            elif key == 'max_rating':
                mask &= (self.df['average_rating'] <= value)
            elif key == 'rating_tier':
                mask &= (self.df['rating_tier'] == value)
            elif key == 'min_reviews':
                mask &= (self.df['rating_number'] >= value)
            elif key == 'max_reviews':
                mask &= (self.df['rating_number'] <= value)
            elif key == 'review_volume':
                mask &= (self.df['review_volume'] == value)
            elif key == 'has_price':
                mask &= (self.df['has_price'] == value)
            elif key == 'store':
                mask &= (self.df['store'] == value)
            elif key == 'contains_category':
                mask &= self.df['categories'].apply(lambda x: value in x if isinstance(x, list) else False)
        
        return mask
    
    def run_benchmark_suite(self):
        """Run a comprehensive benchmark with different query types"""
        print("\n" + "="*80)
        print("VECTOR SEARCH BENCHMARK SUITE - Amazon Gift Cards Dataset")
        print("="*80)
        
        # Query sets for different scenarios
        queries = {
            "Product Search": [
                "amazon gift card for birthday",
                "digital gift card instant delivery",
                "physical gift card with greeting",
                "holiday themed gift card",
                "corporate gift cards bulk"
            ],
            "Feature Search": [
                "no expiration date gift card",
                "customizable gift card design",
                "gift card with fast shipping",
                "electronic gift card email",
                "gift card for any occasion"
            ]
        }
        
        # Filter scenarios with different selectivity
        filter_scenarios = {
            "No Filters (100% selectivity)": {},
            
            "High Rating Only (70% selectivity)": {
                "min_rating": 4.0
            },
            
            "Excellent Rating (70% selectivity)": {
                "rating_tier": "excellent"
            },
            
            "Popular Items (35% selectivity)": {
                "min_reviews": 100
            },
            
            "High Rating + Popular (25% selectivity)": {
                "min_rating": 4.0,
                "min_reviews": 100
            },
            
            "Premium Items (33% selectivity)": {
                "has_price": True
            },
            
            "Amazon Store Only (varies)": {
                "store": "Amazon"
            },
            
            "Highly Selective (5% selectivity)": {
                "min_rating": 4.5,
                "min_reviews": 500,
                "has_price": True
            }
        }
        
        # Run benchmarks
        results = []
        
        for query_type, query_list in queries.items():
            print(f"\n📊 Query Type: {query_type}")
            print("-" * 60)
            
            for scenario_name, filters in filter_scenarios.items():
                print(f"\n🔍 Scenario: {scenario_name}")
                
                # Calculate selectivity
                if filters:
                    mask = self._apply_filters(filters)
                    selectivity = mask.sum() / len(self.df) * 100
                    print(f"   Filter selectivity: {selectivity:.1f}% ({mask.sum()}/{len(self.df)} items)")
                else:
                    selectivity = 100.0
                    print(f"   Filter selectivity: {selectivity:.1f}% ({len(self.df)}/{len(self.df)} items)")
                
                scenario_times = []
                
                for query in query_list:
                    indices, similarities, search_time = self.similarity_search(
                        query, top_k=10, filters=filters
                    )
                    scenario_times.append(search_time * 1000)  # Convert to ms
                    
                    print(f"   Query: '{query[:40]}{'...' if len(query) > 40 else ''}'")
                    print(f"   Results: {len(indices)} items, Time: {search_time*1000:.2f}ms")
                    
                    if len(indices) > 0:
                        print(f"   Top result: {self.df.iloc[indices[0]]['title'][:50]}...")
                        print(f"   Similarity: {similarities[0]:.3f}")
                
                avg_time = np.mean(scenario_times)
                print(f"   📈 Average search time: {avg_time:.2f}ms")
                
                results.append({
                    'query_type': query_type,
                    'scenario': scenario_name,
                    'selectivity_pct': selectivity,
                    'avg_time_ms': avg_time,
                    'num_queries': len(query_list)
                })
        
        # Summary
        print(f"\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        results_df = pd.DataFrame(results)
        
        print(f"\nDataset Stats:")
        print(f"  Total items: {len(self.df):,}")
        print(f"  Embedding dimension: {self.embeddings.shape[1]}")
        print(f"  Dataset size: {Path('gift_cards_with_embeddings.parquet').stat().st_size / (1024*1024):.1f} MB")
        
        print(f"\nPerformance Summary:")
        for _, row in results_df.iterrows():
            print(f"  {row['query_type']} | {row['scenario'][:30]:30} | "
                  f"{row['selectivity_pct']:5.1f}% | {row['avg_time_ms']:6.2f}ms")
        
        print(f"\nOverall average search time: {results_df['avg_time_ms'].mean():.2f}ms")
        
        return results_df

def main():
    # Check if dataset exists
    dataset_file = "gift_cards_with_embeddings.parquet"
    if not Path(dataset_file).exists():
        print(f"Error: {dataset_file} not found. Please run process_dataset.py first.")
        return
    
    # Initialize and run benchmark
    benchmark = VectorSearchBenchmark(dataset_file)
    results = benchmark.run_benchmark_suite()
    
    # Save results
    results.to_csv("benchmark_results.csv", index=False)
    print(f"\nBenchmark results saved to benchmark_results.csv")

if __name__ == "__main__":
    main()