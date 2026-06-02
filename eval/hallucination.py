def score_faithfulness(answer: str, retrieved_chunks: list[str]) -> dict:
    context = " ".join(retrieved_chunks).lower()
    answer_terms = {term.strip(".,:;!?").lower() for term in answer.split() if len(term) > 4}
    if not answer_terms:
        return {"faithfulness": 0.0, "unsupported_terms": []}
    unsupported = sorted(term for term in answer_terms if term not in context)
    faithfulness = 1 - (len(unsupported) / len(answer_terms))
    return {"faithfulness": round(faithfulness, 3), "unsupported_terms": unsupported[:25]}


if __name__ == "__main__":
    sample = score_faithfulness("The policy requires local models.", ["Local models are required by policy."])
    print(sample)
