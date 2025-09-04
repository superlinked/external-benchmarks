# Vector Search Benchmarks

This repo contains datasets for benchmarking vector search performance, to help Superlinked prioritize integration partners.

## Overview

We reviewed a number of publicly available datasets and noted 3 core problems + here is how this dataset fixes them:

|Problems of other vector search benchmarks| How this dataset solves it                                              |
|-|-------------------------------------------------------------------------|
|Not enough metadata of various types makes it hard to test filter performance| 3 number, 1 categorical, 3 text, 1 image column                         |
|Vectors too small, while SOTA models usually output 2k+ even 4k+ dims| 4154 dims                                                               |
|Dataset too small, especially if larger vectors are used| 10k, 100k, 1M and 10M item variants, all sampled from the large dataset |

## Available Datasets

### Metadata

The individual `parquet` files contain the metadata and the encoder inputs.

```
#   Column          Dtype  
---  ------          -----  
 0   main_category   object 
 1   title           object 
 2   average_rating  float64
 3   rating_number   float64
 4   description     object 
 5   price           float64
 6   categories      object 
 7   parent_asin     object 
 8   image_url       object 
```

| Dataset                | Records    | File Size |
|------------------------|------------|-----------|
| benchmark_10k.parquet  | 10,000     | 9.5 MB    |
| benchmark_100k.parquet | 100,000    | 93.1 MB   |
| benchmark_1M.parquet   | 1,000,000  | 922.5 MB  |
| benchmark_10M.parquet  | 10,534,536 | 9.4 GB    |

### Vectors

The folders with `-vector` suffix contain the vectors. These folders have `parquet` files inside.
The structure is
```
 |-- parent_asin: string (nullable = true)
 |-- value: array (nullable = true)
 |    |-- element: double (containsNull = true)
```

| Dataset                | Files | File Size |
|------------------------|-------|-----------|
| benchmark_10k-vectors  | 1,000 | 221.92 MB |
| benchmark_100k-vectors | 1,000 | 1.28 GB   |
| benchmark_1M-vectors   | 1,000 | 20.36 GB  |
| benchmark_10M-vectors  | 5,000 | 214.44 GB |

### Queries

Some smaller dataset versions have a query set guaranteed to only contain parent_asins from the corresponding dataset version.
The smaller versions are created for testing purposes when only a smaller dataset was ingested.
The structure is
```
{
    query_id: {
        product_id: str | None,
        rating_max: int | None,
        rating_num_min: int | None,
        main_category: str | None,
    },
    ...
}
```

| Dataset               | Queries |
|-----------------------|---------|
| query-params-100k     | 15      |
| query-params-1M       | 117     |
| query-params-10M      | 1,000   |

#### Result set

Query results are stored in `ranked-results.json`. 
The structure is

```
{
    query_id: [ordered list of result ids],
    ...
}
```

NOTE: The results expect all products ingested in the database!

### Data Access

Datasets are available via HTTPS download:

```bash
# Download benchmark datasets
wget https://storage.googleapis.com/superlinked-benchmarks-external/amazon-products-images/benchmark-10k.parquet
wget https://storage.googleapis.com/superlinked-benchmarks-external/amazon-products-images/benchmark-100k.parquet
wget https://storage.googleapis.com/superlinked-benchmarks-external/amazon-products-images/benchmark-1M.parquet
wget https://storage.googleapis.com/superlinked-benchmarks-external/amazon-products-images/benchmark-10M.parquet
```

```bash
# Download vectors - WE NEED A SOLUTION HERE
gsutil -m cp -r gs://superlinked-benchmarks-external/amazon-products-images/benchmark-10k-vectors ./local_folder
gsutil -m cp -r gs://superlinked-benchmarks-external/amazon-products-images/benchmark-100k-vectors ./local_folder
gsutil -m cp -r gs://superlinked-benchmarks-external/amazon-products-images/benchmark-1M-vectors ./local_folder
gsutil -m cp -r gs://superlinked-benchmarks-external/amazon-products-images/benchmark-10M-vectors ./local_folder
```

```bash
# Download queries
wget https://storage.googleapis.com/superlinked-benchmarks-external/amazon-products-images/query-params-100k.json
wget https://storage.googleapis.com/superlinked-benchmarks-external/amazon-products-images/query-params-1M.json
wget https://storage.googleapis.com/superlinked-benchmarks-external/amazon-products-images/query-params-10M.json
```

## Dataset Production

### Source Data
- **Origin**: [Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io/)
- **Categories**: Books, Automotive, Tools & Home Improvement, All Beauty, Computers

### Embeddings

The embeddings are created via a [superlinked config](superlinked_app). The resulting 4154 dim vector contains:
- 1 categorical
- 3 number
- 3 text (Qwen/Qwen3-Embedding-0.6B)
- 1 image()
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