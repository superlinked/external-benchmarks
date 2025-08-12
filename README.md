# Vector Search Benchmarks

This repo contains datasets for benchmarking vector search performance, to help Superlinked prioritize integration partners.

## Overview

We reviewed a number of publicly available datasets and noted 3 core problems + here is how this dataset fixes them:

|Problems of other vector search benchmarks|How this dataset solves it|
|-|-|
|Not enough metadata of various types makes it hard to test filter performance|17 metadata properties - numeric, categorical, relational|
|Vectors too small, while SOTA models usually output 2k+ even 4k+ dims|2688 dims|
|Dataset too small, especially if larger vectors are used|10k, 100k, 1M and 10M item variants, all sampled from the large dataset|

### Dataset Issues / Notes
1. In pre-processing we accidentally dropped `asin` which is the primary key of these datasets - to validate recall we will need to add it back in in the next version of this dataset. Right now, there is no PK.
2. The `details` column has a bunch of redundancy (null values for missing keys), which if prunned will reduce the dataset size by 20-30%.
3. The original dataset also contains images, but since we do not aim to test embedding model inference performance with vector search vendors, image URLs were not included.

## Available Datasets

The `benchmark_10M.parquet` dataset is the one to measure the vector search performance on. We have added smaller variants of this dataset (via uniform sampling) to make it easier to test your benchmarking setup.

| Dataset | Records | File Size |
|---------|---------|-----------|
| benchmark_10k | 9,000 | 207 MB |
| benchmark_100k | 98,500 | 2.3 GB |
| benchmark_1M | 1,044,500 | 23.4 GB |
| benchmark_10M | 10,564,046 | 243 GB |

To learn more about the datasets, see [`reports/summary_report.md`](reports/summary_report.md) and [`reports/benchmark_10k/README.md`](reports/benchmark_10k/README.md).

### Data Access

Datasets are available via HTTPS download:

```bash
# Download benchmark datasets
wget https://storage.googleapis.com/superlinked-benchmarks-external/amazon-products/benchmark_10k.parquet
wget https://storage.googleapis.com/superlinked-benchmarks-external/amazon-products/benchmark_100k.parquet
wget https://storage.googleapis.com/superlinked-benchmarks-external/amazon-products/benchmark_1M.parquet
wget https://storage.googleapis.com/superlinked-benchmarks-external/amazon-products/benchmark_10M.parquet
```

## Dataset Production

### Source Data
- **Origin**: [Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io/)
- **Categories**: Books, Automotive, Tools & Home Improvement, All Beauty, Computers

### Embeddings
Our goal was to mimic a SOTA model dimensionality (e.g. [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) at 2560 dims) but to save resources we built similar vectors by concatenating outputs of a smaller model applied to individual string fields of each item:
- **Model**: [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) (384 dims per field)
- **Fields Embedded**: 7 text fields (title, description, features, main_category, store, categories, details)
- **Final Embedding**: 7 × 384 = 2,688 dimensions per product


## Running Benchmarks

For `benchmark_10M.parquet` produce the following set of measurements - basically fill in the 'TBD' cells:

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
1. **Vector Similarity**: Each query should contain `dot product` similarity scoring against a vector that you grab at random from the dataset. Note - if your system caches the vector-specific computations, please rotate a large set of random vectors - otherwise you can use the same vector.
2. **Filters**: To get the target filter selectivity, please use one of the filter predicates below or similar.
3. **Results details**: Add `LIMIT 100` to all queries and only retrieve `parent_asin` for each record to minimize networking overhead (until we add `asin` back in, see *Dataset Issues* above).
4. **Vector Search Recall**: We expect that you can tune your system to produce >90% average recall for the ANN index and we expect that you run the above tests with such tunning.

|Selectivity|Predicate|
|-|-|
|0.001%|`average_rating <= 3.0 and rating_number > 130 and main_category == 'Computers'`|
|0.1%|`average_rating <= 3.5 and rating_number > 15 and main_category == 'Computers'`|
|1%|`average_rating >= 3.5 and rating_number > 10 and main_category == 'Computers'`|
|10%|`main_category in ['Computers', 'All Beauty', 'Buy a Kindle']`|

## Pricing

To enable us to compare different vendors, we consider the above dataset size + performance to be a "unit" of vector search, for which we would like to know:
1. What are the vector search vendor parameters of the cloud instance that can support this "unit".
2. What is the price-per-GB-month for this instance, assuming a sustained average workload as described by the targets above.
3. How does the price scale with (a) 2x the size (b) 2x the read QPS (c) 2x the write QPS.

## License

This dataset is derived from the Amazon Reviews 2023 dataset. Please refer to the [original dataset's license](https://amazon-reviews-2023.github.io/) for usage terms.