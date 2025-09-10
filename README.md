# Vector Search Benchmarks

This repo contains datasets for benchmarking vector search performance, to help Superlinked prioritize integration partners.

## Overview

We reviewed a number of publicly available datasets and noted 3 core problems + here is how this dataset fixes them:

|Problems of other vector search benchmarks| How this dataset solves it                                         |
|-|--------------------------------------------------------------------|
|Not enough metadata of various types makes it hard to test filter performance| 3 number, 1 categorical, 3 text, 1 image column                    |
|Vectors too small, while SOTA models usually output 2k+ even 4k+ dims| 4154 dims                                                          |
|Dataset too small, especially if larger vectors are used| 100k, 1M and 10M item variants, all sampled from the large dataset |

## Available Datasets

### Product data

The folders contain `parquet` files with the metadata and vectors.

| Dataset        | Records    | # Files | Size    |
|----------------|------------|---------|---------|
| benchmark_10k  | 10,000     | 100     | ~230 MB |
| benchmark_100k | 100,000    | 100     | ~2.3 GB |
| benchmark_1M   | 1,000,000  | 100     | ~23 GB  |
| benchmark_10M  | 10,534,536 | 1000    | ~240 GB |

The structure of the files is the same throughout:

```
Schema([('parent_asin', String), # the id
        ('main_category', String),
        ('title', String),
        ('average_rating', Float64),
        ('rating_number', Float64),
        ('description', String),
        ('price', Float64),
        ('categories', String),
        ('image_url', String)])
        ('value', List(Float64)), # the vectors
```

### Queries

Some smaller dataset versions have a query set guaranteed to only contain parent_asins from the corresponding dataset version.
The smaller versions are created for testing purposes when only a smaller dataset was ingested. 
In the [query](superlinked_app/query.py) file the actual query structure can be seen.
The file structure is
```
{
    query_id: {
        product_id: str | None,       # parent_asin - get that value from the database and search with it
        rating_max: int | None,       # filter for product.average_rating <= rating_max
        rating_num_min: int | None,   # filter product.rating_number > rating_num_min
        main_category: str | None,    # filter for product.main_category == main_category
    },
    ...
}
```

| Dataset               | Queries |
|-----------------------|---------|
| query-params-100k     | 15      |
| query-params-1M       | 117     |
| query-params-10M      | 1,000   |

### Result set

Query results are stored in `ranked-results.json`. 
The structure is

```
{
    query_id: [ordered list of result parent_asins],
    ...
}
```

NOTE: The results expect all products ingested in the database!

## Data Access

Datasets are available via multiple ways:

1. You can use gsutil to download the dataset (as HTTPS download works best for individual files):
```bash
# Download benchmark datasets
gsutil cp -r "gs://superlinked-benchmarks-external/amazon-products-images/benchmark-10k/**" ./your/local/data/folder/
gsutil cp -r "gs://superlinked-benchmarks-external/amazon-products-images/benchmark-100k/**" ./your/local/data/folder/
gsutil cp -r "gs://superlinked-benchmarks-external/amazon-products-images/benchmark-1M/**" ./your/local/data/folder/
gsutil cp -r "gs://superlinked-benchmarks-external/amazon-products-images/benchmark-10M/**" ./your/local/data/folder/
```
As queries are individual files, even a simple https download works fine:
```bash
# Download queries
wget https://storage.googleapis.com/superlinked-benchmarks-external/amazon-products-images/query-params-100k.json
wget https://storage.googleapis.com/superlinked-benchmarks-external/amazon-products-images/query-params-1M.json
wget https://storage.googleapis.com/superlinked-benchmarks-external/amazon-products-images/query-params-10M.json
```
Same is true for results:
```bash
# Download the ground truth query results
wget https://storage.googleapis.com/superlinked-benchmarks-external/amazon-products-images/ranked-results.json
```
but gsutil works fine for these as well (you can infer the path from the URLs). For `ranked-results.json`:
```bash
gsutil cp "gs://superlinked-benchmarks-external/amazon-products-images/ranked-results.json" ./your/local/data/folder/
```
2. Using huggingface datasets

The product data is available using [HF Datasets](https://huggingface.co/docs/datasets/en/index).

```python
from datasets import load_dataset

benchmark_10k = load_dataset("superlinked/external-benchmarking", data_dir="benchmark-10k")
benchmark_100k = load_dataset("superlinked/external-benchmarking", data_dir="benchmark-100k")
benchmark_1M = load_dataset("superlinked/external-benchmarking", data_dir="benchmark-1M")
benchmark_10M = load_dataset("superlinked/external-benchmarking", data_dir="benchmark-10M")
```

For query and result data, please use one of the above methods (gsutil or direct download).

## Dataset Production

### Source Data
- **Origin**: [Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io/)
- **Categories**: `["Books", "Automotive", "Tools and Home Improvement", "All Beauty", "Electronics", "Software", "Health and Household"]`

### Embeddings

The embeddings are created via a [superlinked config](superlinked_app). The resulting 4154 dim vector contains:
- 1 categorical,
- 3 number,
- 3 text (`Qwen/Qwen3-Embedding-0.6B`),
- and 1 image (`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`)

embeddings concatenated.

## Running Benchmarks

For the `benchmark_10M` setup produce the following set of measurements - basically fill in the 'TBD' cells:

| # | Write | Target | Observed |Read | Target | Observed |
|-|-|-|-|-|-|-|
|1|Create Index from scratch | < 2hrs |TBD|-|-|-|
|2|- | - |-|20 QPS of 0.001% filter selectivity| 100ms @ p95 | TBD |
|3|- | - |-|20 QPS of 0.1% filter selectivity| 100ms @ p95 | TBD |
|4|- | - |-|20 QPS of 1% filter selectivity| 100ms @ p95 | TBD |
|5|- | - |-|20 QPS of 10% filter selectivity| 100ms @ p95 | TBD |
|6|20 QPS for single-object updates (incl. embedding)| 2s @ p95 | TBD |20 QPS of 1% filter selectivity| 100ms @ p95 | TBD |
|7|200 QPS for single-object updates (incl. embedding)| 2s @ p95 | TBD |20 QPS of 1% filter selectivity| 100ms @ p95 | TBD |

Formulate the queries like this:
1. **Vector Similarity**: Each query should contain `dot product` similarity scoring against a vector that you grab from the DB. 
The vector is specified in query_params under the `product_id` key.
2. **Filters**: To get the target filter selectivity, please use the filters specified in the `query_params` files.
3. **Results details**: Add `LIMIT 100` to all queries and only retrieve `parent_asin` for each record to minimize networking overhead.
4. **Vector Search Recall**: We expect that you can tune your system to produce >90% average hit rate for the ANN index and we expect that you run the above tests with such tuning.

|Selectivity| Predicate                                                                       |
|-|---------------------------------------------------------------------------------|
|0.001%| `average_rating <= 3.0 and rating_number > 130 and main_category == 'Computers'` |
|0.1%| `average_rating <= 3.5 and rating_number > 30 and main_category == 'Computers'` |
|1%| `rating_number > 45 and main_category == 'Computers'`                        |
|10%| `average_rating <= 3.5 and rating_number > 1`                                   |

## Query result quality evaluation

You are welcome to use the `calculate_hit_rates` function in [eval.py](eval.py).
It expects the prediction results in a similar format as the ground truth result set is provided.

## Pricing

To enable us to compare different vendors, we consider the above dataset size + performance to be a "unit" of vector search, for which we would like to know:
1. What are the vector search vendor parameters of the cloud instance that can support this "unit".
2. What is the price-per-GB-month for this instance, assuming a sustained average workload as described by the targets above.
3. How does the price scale with (a) 2x the size (b) 2x the read QPS (c) 2x the write QPS.

## License

This dataset is derived from the Amazon Reviews 2023 dataset. Please refer to the [original dataset's license](https://amazon-reviews-2023.github.io/) for usage terms.