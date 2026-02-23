def weighted_average(metrics):

    total_examples = sum(num_examples for num_examples, _ in metrics)

    weighted_acc = sum(
        num_examples * m["accuracy"]
        for num_examples, m in metrics
    )

    return {"accuracy": weighted_acc / total_examples}
