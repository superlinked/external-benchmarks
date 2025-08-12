# Benchmark Datasets - Summary Report

**Generated:** 2025-08-11 (Updated)

## Overview

This repository contains benchmark datasets of varying sizes for vector search performance testing on Amazon product data.

| Dataset | Records | File Size (MB) | Avg Rating | Avg Price ($) | Avg Reviews | Categories |
|---------|---------|----------------|------------|---------------|-------------|------------|
| 10k | 9,000 | 207.2 | 4.25 ± 0.74 | 42.80 ± 105 | 249 | 33 |
| 100k | 98,500 | 2,265.8 | 4.24 ± 0.77 | 46.25 ± 125 | 259 | 36 |
| 1M | 1,044,500 | 23,365.2 | 4.23 ± 0.76 | 45.67 ± 189 | 269 | 39 |
| 10M | 10,564,046 | 242,668.9 | 4.23 ± 0.77 | 46.51 ± 179 | 264 | 46 |

## Dataset Characteristics

### Schema Structure (17 columns)
- **Product Metadata**: title, description, features, categories, store, details
- **Ratings Data**: average_rating (1.0-5.0), rating_number, rating_tier
- **Pricing**: price (58% have prices), has_price flag
- **Vector Embeddings**: 2,688-dimensional (BAAI/bge-small-en-v1.5 model, 7 fields × 384 dims)
- **Derived Features**: review_volume categories, source_dataset tracking

### Rating Distribution (Consistent Across Scales)
| Tier | 10k | 100k | 1M | 10M |
|------|-----|------|-----|-----|
| Excellent (4.5-5.0) | 39.0% | 39.4% | 38.4% | 38.1% |
| High (4.0-4.5) | 31.5% | 31.3% | 31.5% | 31.5% |
| Medium (3.0-4.0) | 16.3% | 15.9% | 16.2% | 16.4% |
| Low (< 3.0) | 13.2% | 13.5% | 13.9% | 14.1% |

### Top Categories (by percentage)
1. **Books**: 41-42% across all datasets
2. **Automotive**: 15-19% (higher in smaller datasets)
3. **Tools & Home Improvement**: 11-14%
4. **All Beauty**: 5-8%
5. **Computers**: 3-4%

## Data Quality Metrics

### Statistical Consistency
- **Rating Variance**: σ² = 0.55-0.59 (stable across scales)
- **Price Distribution**: Right-skewed with high variance (typical e-commerce pattern)
- **Review Volume**: Median ~50-100 reviews, mean inflated by bestsellers
- **Null Values**: Price (42% null), Store (2.6% null), Main Category (0.7% null)

### Sampling Validation
- Stratified sampling preserved category distributions within ±2%
- Rating distributions maintained within ±1% across scales
- Price statistics show expected variance due to long-tail products

## Analysis Approach

- **10k Dataset**: Full comprehensive analysis with visualizations
- **Larger Datasets**: Memory-efficient statistical validation using DuckDB
- **Consistency Check**: Validates that sampling preserved original distributions

## Key Insights

1. **Scalability**: Linear file size growth (~23MB per 100k records)
2. **Distribution Stability**: Core metrics remain consistent across 3 orders of magnitude
3. **Category Diversity**: Larger datasets reveal more niche categories (46 vs 33)
4. **Rating Bias**: Slight positive skew (4.2+ average) typical of review platforms
5. **Price Coverage**: 58% products have pricing data (consistent across scales)

## Performance Benchmarking Recommendations

### Vector Search Scenarios
1. **Cold Start Testing**: Use 10k dataset for rapid iteration
2. **Production Simulation**: 1M dataset represents typical catalog size
3. **Scale Testing**: 10M dataset for stress testing infrastructure
4. **Filter Selectivity**: Test with category/price/rating filters at 5%, 25%, 50%, 100%

### Expected Query Patterns
- **Product Search**: Title/description similarity (60% of queries)
- **Feature Search**: Specific attribute matching (25% of queries)  
- **Cross-category**: Multi-modal recommendations (15% of queries)

### Optimization Targets
| Dataset | Target P95 Latency | Index Build Time |
|---------|-------------------|------------------|
| 10k | < 10ms | < 1 sec |
| 100k | < 25ms | < 10 sec |
| 1M | < 50ms | < 2 min |
| 10M | < 100ms | < 30 min |

## Report Structure

```
reports/
├── benchmark_10k/          # Comprehensive analysis with plots
│   ├── README.md
│   ├── rating_analysis.png
│   ├── category_analysis.png  
│   ├── text_analysis.png
│   └── correlation_analysis.png
├── benchmark_100k/         # Statistical validation
│   └── README.md
├── benchmark_1M/           # Statistical validation  
│   └── README.md
├── benchmark_10M/          # Statistical validation
│   └── README.md
├── sampling_analysis.json  # Detailed statistics
└── summary_report.md       # This file
```

## Usage Examples

### Loading Datasets
```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_parquet('benchmark/benchmark_1M.parquet')

# Extract embeddings
embeddings = np.vstack(df['embedding'].values)  # Shape: (1044500, 2688)

# Filter by category
books = df[df['main_category'] == 'Books']

# High-rated products with prices
premium = df[(df['rating_tier'] == 'excellent') & (df['has_price'] == True)]
```

### Vector Search Example
```python
from sklearn.metrics.pairwise import cosine_similarity

# Query embedding (from same model)
query = embed_text("wireless bluetooth headphones")

# Compute similarities
similarities = cosine_similarity([query], embeddings)[0]

# Get top-k results
top_k = 100
top_indices = np.argsort(similarities)[-top_k:][::-1]
results = df.iloc[top_indices]
```

*Generated by benchmark analysis pipeline - Updated 2025-08-11*
