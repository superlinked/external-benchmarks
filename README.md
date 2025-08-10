# Vector Search Benchmarking Suite

High-performance vector search benchmarking suite optimized for Apple Silicon and NVIDIA hardware. Processes Amazon Reviews 2023 datasets with 2,688-dimensional embeddings.

## Datasets

### Appliances (Primary)
- **94,327 products** processed in **42 minutes on M3 Max**
- **2,688-dimensional embeddings** (7 fields × 384 dims)
- **Model**: BAAI/bge-small-en-v1.5
- **Output**: 1.4GB parquet file

### Gift Cards (Reference)  
- **1,137 products** available for testing
- Same embedding strategy as Appliances
- Modify script configuration to process

## Data Access

Large files stored in Google Cloud Storage: `gs://superlinked-benchmarks-external/`

```bash
# Download datasets
gsutil cp gs://superlinked-benchmarks-external/meta_Appliances.jsonl .
gsutil cp gs://superlinked-benchmarks-external/appliances_with_embeddings.parquet .
```

## Embedding Strategy

- **Model**: BAAI/bge-small-en-v1.5 (384 dims per field)
- **Fields**: title, description, features, main_category, store, categories, details  
- **Total**: 7 × 384 = 2,688 dimensions
- **Hardware**: MPS/CUDA acceleration with 1024-item batches
- **Parallel**: 12-worker processing for maximum throughput

## Usage

```bash
pip install -r requirements.txt
python process_dataset.py  # Processes Appliances by default
python benchmark.py        # Run performance benchmarks
```

## Performance

**M3 Max (128GB)**: 94K records in 42 minutes, 99% memory usage, 100% GPU utilization

## Schema

Key columns in parquet files:
- `parent_asin`, `title`, `description`, `features`, `main_category`, `store`, `categories`, `details`
- `average_rating`, `rating_number`, `price` 
- `embedding`: 2,688-dimensional vector

## License

This dataset is derived from the Amazon Reviews 2023 dataset. Please refer to the [original dataset's license](https://amazon-reviews-2023.github.io/) for usage terms.