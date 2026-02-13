def rule_based_decision(order_value, delay_minutes, past_refunds):
    if delay_minutes > 30:
        return 1  # refund
    if past_refunds > 3:
        return 0  # reject
    if order_value < 200:
        return 1
    return 0
