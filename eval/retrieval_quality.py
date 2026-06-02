def precision_at_k(expected_doc_ids: set[str], retrieved_doc_ids: list[str], k: int = 5) -> float:
    if not expected_doc_ids or k <= 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved_doc_ids[:k] if doc_id in expected_doc_ids)
    return hits / min(k, len(expected_doc_ids))


def reciprocal_rank(expected_doc_ids: set[str], retrieved_doc_ids: list[str]) -> float:
    for index, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected_doc_ids:
            return 1 / index
    return 0.0


if __name__ == "__main__":
    print({"precision_at_5": precision_at_k({"a"}, ["b", "a"]), "mrr": reciprocal_rank({"a"}, ["b", "a"])})
