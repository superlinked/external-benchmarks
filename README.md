# Vector Search Benchmarking Suite

This repository contains a high-performance vector search benchmarking suite optimized for Apple Silicon M3 Max and other high-performance hardware. The suite processes Amazon Reviews 2023 datasets with advanced embedding generation and comprehensive performance testing.

## Datasets Available

### Appliances Dataset (Primary)
- **Size**: 94,327 products 
- **Embeddings**: 2,688-dimensional vectors (7 fields concatenated)
- **Model**: BAAI/bge-small-en-v1.5
- **Processing time**: ~40 minutes on M3 Max (estimated, in progress)
- **Current configuration**: Default dataset processed by `process_dataset.py`

### Gift Cards Dataset (Reference)
- **Size**: 1,137 products
- **Embeddings**: 2,688-dimensional vectors (7 fields concatenated)
- **Model**: BAAI/bge-small-en-v1.5
- **Processing time**: TBD (not yet processed with 7-field strategy)
- **Usage**: Available for smaller-scale testing and comparisons

## Data Storage

All large data files are stored in Google Cloud Storage for easy access:

**Bucket**: `gs://superlinked-benchmarks-external/`

### Download Data Files

```bash
# Download raw datasets
gsutil cp gs://superlinked-benchmarks-external/meta_Gift_Cards.jsonl .
gsutil cp gs://superlinked-benchmarks-external/meta_Appliances.jsonl .

# Download processed datasets with embeddings
gsutil cp gs://superlinked-benchmarks-external/gift_cards_with_embeddings.parquet .
gsutil cp gs://superlinked-benchmarks-external/appliances_with_embeddings.parquet .
```

## Advanced Embedding Strategy

- **Model**: `BAAI/bge-small-en-v1.5` (384 dimensions per field)
- **Fields Embedded**: 
  1. `title` - Product title
  2. `description` - Product description
  3. `features` - Product features list
  4. `main_category` - Main product category
  5. `store` - Store name
  6. `categories` - Category hierarchy
  7. `details` - Additional product details
- **Total Dimensions**: 7 × 384 = 2,688 dimensions
- **Hardware Acceleration**: MPS on Apple Silicon, CUDA on NVIDIA GPUs
- **Batch Processing**: Up to 1024 items per batch for maximum throughput
- **Parallel Processing**: 12-worker ThreadPoolExecutor for optimal CPU utilization

## Performance Optimizations

### M3 Max Optimizations
- **Memory Utilization**: Achieves 99%+ unified memory usage (127GB/128GB)
- **GPU Utilization**: 100% compute utilization during processing
- **Batch Size**: Aggressive 1024-item batches for maximum bandwidth
- **Parallel Workers**: 12 concurrent threads matching CPU core count
- **Processing Rate**: ~2,300 records/minute for large datasets

### Hardware Requirements
- **Minimum RAM**: 16GB (recommended: 64GB+ for large datasets)
- **GPU**: Apple Silicon MPS or NVIDIA CUDA support
- **Storage**: SSD recommended for dataset I/O

## Setup and Usage

### Prerequisites

```bash
pip install pandas pyarrow torch transformers numpy tqdm scikit-learn
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Process Datasets

The script is configured to process the Appliances dataset by default:

```bash
python process_dataset.py
```

This will:
- Download `meta_Appliances.jsonl` from Amazon Reviews 2023 (if not present)
- Process and clean 94,327 product records across 7 text fields
- Generate 2,688-dimensional embeddings using BAAI/bge-small-en-v1.5
- Save result as `appliances_with_embeddings.parquet`

To process Gift Cards instead, modify the script:
```python
metadata_file = "meta_Gift_Cards.jsonl"  # Change from "meta_Appliances.jsonl"
output_file = "gift_cards_with_embeddings.parquet"  # Change output filename
```

### Run Benchmarks

```bash
python benchmark.py
```

## Dataset Schema

The processed parquet files contain:

### Core Product Data
- `parent_asin`: Unique product identifier
- `title`: Product title
- `description`: Product description text
- `features`: Product features list
- `main_category`: Primary category
- `store`: Store/seller name
- `categories`: Category hierarchy array
- `details`: Additional product details dictionary

### Rating & Review Data
- `average_rating`: Average user rating (1-5 scale)
- `rating_number`: Number of reviews
- `rating_tier`: Categorical rating (low/medium/high/excellent)
- `review_volume`: Categorical review count (few/moderate/many/popular)

### Metadata
- `price`: Product price (when available)
- `has_price`: Boolean indicating price availability

### Vector Data
- `embedding`: 2,688-dimensional vector (7 fields × 384 dims each)

## Benchmark Results

### M3 Max Performance (128GB Unified Memory)
- **Appliances (94K records)**: ~40 minutes total processing (in progress)
- **Throughput**: ~2,300 records/minute sustained
- **Memory Usage**: 99%+ utilization (127GB/128GB)
- **GPU Utilization**: 100% compute usage
- **Gift Cards (1K records)**: Processing time TBD with current 7-field configuration

### Processing Breakdown
- Field-level parallel processing with 12 workers
- Single-batch processing for smaller datasets
- Progressive batch processing for large datasets
- L2 normalization and memory optimization

## Use Cases

This benchmarking suite is ideal for:

1. **Vector Database Performance Testing**: Measure search latency and accuracy
2. **Embedding Model Evaluation**: Compare different models at scale
3. **Hardware Optimization**: Test GPU/CPU utilization patterns
4. **Scalability Testing**: Understand performance characteristics across dataset sizes
5. **Multi-field Search**: Benchmark concatenated vs separate field strategies

## Architecture

### Parallel Processing Design
- ThreadPoolExecutor with CPU-core-matched worker count
- Field-level parallelization for maximum throughput
- Memory-efficient batch processing with automatic cleanup
- MPS/CUDA acceleration with fallback handling

### Optimization Features
- Aggressive batch sizing (up to 1024 items)
- Single-batch processing for small datasets
- Progressive batching with memory management
- Hardware-specific optimizations (MPS float32, etc.)

## License

This dataset is derived from the Amazon Reviews 2023 dataset. Please refer to the [original dataset's license](https://amazon-reviews-2023.github.io/) for usage terms.