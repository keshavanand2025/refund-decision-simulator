def calculate_total_refund_cost(decisions, order_values):
    total = 0
    for d, value in zip(decisions, order_values):
        if d == 1:
            total += value
    return total


def calculate_accuracy(predictions, actual):
    correct = sum(p == a for p, a in zip(predictions, actual))
    return correct / len(actual)
