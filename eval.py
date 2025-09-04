def calculate_hit_rates(ground_truth: dict[str, list[str]],
                        predicted: dict[str, list[str]],
                        k_values: list[int]) -> dict[int, float]:
    """
    Calculate hit rates (precision@k) at different k values for information retrieval evaluation.

    Args:
        ground_truth: Dictionary mapping query_id to ordered list of relevant result ids
        predicted: Dictionary mapping query_id to ordered list of predicted result ids
        k_values: List of k values to calculate hit rates for (e.g., [1, 5, 10, 20])

    Returns:
        Dictionary mapping k values to hit rates (relevant docs retrieved / k, averaged across queries)

    Raises:
        ValueError: If query_ids don't match between ground_truth and predicted
    """
    # Validate input
    if set(ground_truth.keys()) != set(predicted.keys()):
        raise ValueError("Query IDs must match between ground_truth and predicted dictionaries")

    hit_rates = {}

    for k in k_values:
        total_precision = 0.0
        total_queries = len(ground_truth)

        for query_id in ground_truth:
            # Get relevant documents for this query
            relevant_docs = set(ground_truth[query_id])

            # Get top-k predictions for this query
            top_k_predictions = predicted[query_id][:k]

            # Count how many of the top-k predictions are relevant
            relevant_retrieved = sum(1 for doc_id in top_k_predictions if doc_id in relevant_docs)

            # Calculate precision@k for this query (relevant retrieved / k)
            precision_at_k = relevant_retrieved / k if k > 0 else 0.0
            total_precision += precision_at_k

        # Average precision@k across all queries
        hit_rates[k] = total_precision / total_queries if total_queries > 0 else 0.0

    return hit_rates


def example_usage():
    # Sample data
    ground_truth = {
        "q1": ["doc1", "doc3", "doc5"] + ["filler"] * 97,  # 100 items total
        "q2": ["doc2", "doc4"] + ["filler"] * 98,
        "q3": ["doc1", "doc6", "doc8"] + ["filler"] * 97
    }

    predicted = {
        "q1": ["doc1", "doc2", "doc7"] + ["other"] * 97,  # 1/1=1.0 at k=1, 1/3=0.33 at k=3
        "q2": ["doc1", "doc3", "doc2"] + ["other"] * 97,  # 0/1=0.0 at k=1, 1/3=0.33 at k=3
        "q3": ["doc9", "doc10", "doc1"] + ["other"] * 97  # 0/1=0.0 at k=1, 1/3=0.33 at k=3
    }

    k_values = [1, 3, 5, 10]
    results = calculate_hit_rates(ground_truth, predicted, k_values)

    print("Hit Rates (Precision@k):")
    for k, hit_rate in results.items():
        print(f"Hit@{k}: {hit_rate:.3f}")

    # Expected output:
    # Hit@1: 0.333 ((1.0 + 0.0 + 0.0) / 3)
    # Hit@3: 0.333 ((0.33 + 0.33 + 0.33) / 3)
    # Hit@5: 0.200 ((0.20 + 0.20 + 0.20) / 3)
    # Hit@10: 0.100 ((0.10 + 0.10 + 0.10) / 3)

# Uncomment to test:
example_usage()