def routing_accuracy(expected: list[str], predicted: list[str]) -> dict:
    total = min(len(expected), len(predicted))
    if total == 0:
        return {"accuracy": 0.0, "total": 0}
    correct = sum(1 for left, right in zip(expected, predicted) if left == right)
    return {"accuracy": correct / total, "total": total}


if __name__ == "__main__":
    print(routing_accuracy(["RAG", "VISION"], ["RAG", "RAG"]))
