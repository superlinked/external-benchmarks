# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a vector search benchmarking repository focused on Amazon product data. It contains:
- Dataset processing scripts for Amazon Reviews 2023 data
- Vector embedding generation using sentence transformers
- Comprehensive benchmarking suite for testing vector search performance
- Sample dataset: Amazon Gift Cards (1,137 items with 384-dimensional embeddings)

## Development Setup

```bash
# Install Python dependencies
pip install pandas pyarrow torch transformers sentence-transformers numpy tqdm scikit-learn

# Or use the requirements file
pip install -r requirements.txt
```

## Common Commands

### Dataset Processing
```bash
# Download and process Gift Cards dataset with embeddings
python process_dataset.py
```

### Benchmarking
```bash
# Run comprehensive vector search benchmark
python benchmark.py
```

### Performance Testing
The benchmark tests various scenarios:
- Different filter selectivity (5% to 100%)
- Product search vs feature search queries
- Metadata filtering combinations
- Performance optimization on Apple Silicon (MPS acceleration)

## Architecture Overview

### Core Components
- `process_dataset.py`: Downloads Amazon Reviews 2023 data, processes metadata, generates embeddings using all-MiniLM-L6-v2
- `benchmark.py`: Vector search benchmarking suite with configurable filters and query types  
- `gift_cards_with_embeddings.parquet`: Final dataset (3.3MB, 1,137 records)

### Data Pipeline
1. Download JSONL data from Amazon Reviews 2023 API
2. Parse and clean product metadata (title, description, features, ratings)
3. Generate vector embeddings using SentenceTransformer
4. Save to parquet with both metadata and embeddings
5. Benchmark various search scenarios with performance metrics

### Embedding Strategy
- Model: `all-MiniLM-L6-v2` (384 dimensions)
- Text: Combined title + description + features
- Hardware acceleration: MPS on Apple Silicon, CUDA fallback
- Batch processing: 32 items per batch for optimal performance