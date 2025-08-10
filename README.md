# Amazon Gift Cards Vector Search Benchmark Dataset

This repository contains a vector search benchmarking dataset based on Amazon Gift Cards from the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset. The dataset includes 1,137 gift card products with metadata and 384-dimensional vector embeddings generated from product descriptions.

## Dataset Overview

- **Source**: Amazon Reviews 2023 - Gift Cards category
- **Size**: 1,137 products
- **Embeddings**: 384-dimensional vectors using `all-MiniLM-L6-v2`
- **File Format**: Parquet (3.3 MB)
- **Text Fields**: Title, description, features (combined for embeddings)
- **Metadata**: Ratings, review counts, prices, categories, store info

## Dataset Statistics

```
Total records: 1,137
Records with descriptions: 875 (77%)
Records with features: 1,032 (91%)
Records with ratings: 1,137 (100%)
Records with prices: 380 (33%)
Average rating: 4.49/5.0

Rating Distribution:
- Excellent (4.5-5.0): 797 items (70%)
- High (4.0-4.5): 160 items (14%)
- Medium (3.5-4.0): 84 items (7%)
- Low (0-3.5): 96 items (8%)
```

## Setup and Usage

### Prerequisites

```bash
pip install pandas pyarrow torch transformers sentence-transformers numpy tqdm scikit-learn
```

### Generate the Dataset

1. Run the data processing script:
```bash
python process_dataset.py
```

This will:
- Download the Gift Cards metadata from Amazon Reviews 2023
- Process and clean the data
- Generate vector embeddings using `all-MiniLM-L6-v2`
- Save the result as `gift_cards_with_embeddings.parquet`

### Run Benchmarks

```bash
python benchmark.py
```

## Benchmark Query Types

The benchmark suite tests different types of vector search scenarios:

### 1. Product Search Queries
- "amazon gift card for birthday"
- "digital gift card instant delivery"  
- "physical gift card with greeting"
- "holiday themed gift card"
- "corporate gift cards bulk"

### 2. Feature-Based Queries
- "no expiration date gift card"
- "customizable gift card design"
- "gift card with fast shipping"
- "electronic gift card email"
- "gift card for any occasion"

## Filter Scenarios (Different Selectivity)

The benchmark tests various metadata filter combinations to simulate real-world search scenarios:

| Filter Scenario | Selectivity | Description |
|----------------|-------------|-------------|
| No Filters | 100% | Pure vector similarity search |
| High Rating Only | ~70% | Rating ≥ 4.0 |
| Excellent Rating | ~70% | Rating tier = "excellent" |
| Popular Items | ~35% | Review count ≥ 100 |
| High Rating + Popular | ~25% | Rating ≥ 4.0 AND reviews ≥ 100 |
| Premium Items | ~33% | Has price information |
| Amazon Store Only | Varies | Store = "Amazon" |
| Highly Selective | ~5% | Rating ≥ 4.5 AND reviews ≥ 500 AND has price |

## Dataset Schema

The parquet file contains the following columns:

### Core Product Data
- `parent_asin`: Unique product identifier
- `title`: Product title
- `description`: Product description text
- `features`: List of product features
- `combined_text`: Title + description + features (used for embeddings)

### Rating & Review Data
- `average_rating`: Average user rating (1-5 scale)
- `rating_number`: Number of reviews
- `rating_tier`: Categorical rating (low/medium/high/excellent)
- `review_volume`: Categorical review count (few/moderate/many/popular)

### Metadata
- `price`: Product price (when available)
- `has_price`: Boolean indicating price availability
- `main_category`: "Gift Cards"
- `categories`: List of product categories
- `store`: Store name (mostly "Amazon")
- `details`: Additional product details

### Vector Data
- `embedding`: 384-dimensional vector embedding

## Performance Characteristics

On Apple M3 Pro (using MPS acceleration):

### Typical Search Performance
- **No filters**: ~8-15ms per query
- **Simple filters** (1-2 conditions): ~10-20ms per query
- **Complex filters** (3+ conditions): ~15-30ms per query
- **Embedding generation**: ~4s for full dataset (1,137 items)

### Search Quality
- High-quality semantic search with `all-MiniLM-L6-v2` embeddings
- Good recall for product-specific queries
- Effective feature-based matching
- Handles synonyms and related concepts well

## Extending the Dataset

To create larger benchmarking datasets:

1. **Use more categories**: Download additional categories from Amazon Reviews 2023
2. **Include review text**: Add embeddings for actual user reviews
3. **Multi-modal**: Include image embeddings using vision models
4. **Temporal data**: Add time-based filtering scenarios

Example categories for expansion:
- Electronics (larger dataset ~2M items)
- Books (very large ~15M items)  
- Clothing (large ~5M items)

## Files

- `process_dataset.py`: Script to download and process the raw data
- `benchmark.py`: Comprehensive benchmarking suite
- `gift_cards_with_embeddings.parquet`: Final dataset with embeddings
- `requirements.txt`: Python dependencies
- `benchmark_results.csv`: Benchmark performance results

## Use Cases

This dataset is ideal for:

1. **Vector Database Benchmarking**: Test search performance and accuracy
2. **Hybrid Search Testing**: Combine vector similarity with metadata filters
3. **Embedding Model Evaluation**: Compare different embedding models
4. **Search Algorithm Development**: Develop and test new search approaches
5. **Performance Analysis**: Understand search latency under different conditions

## Hardware Optimization

The dataset generation and benchmarking code is optimized for Apple Silicon:

- **MPS acceleration**: Uses Metal Performance Shaders for embedding generation
- **Batch processing**: Efficient batch inference (32 items per batch)
- **Memory efficient**: Streaming processing for larger datasets
- **Fast storage**: Parquet format for quick loading

## License

This dataset is derived from the Amazon Reviews 2023 dataset. Please refer to the [original dataset's license](https://amazon-reviews-2023.github.io/) for usage terms.