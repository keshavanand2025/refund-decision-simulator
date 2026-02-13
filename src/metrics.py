from metrics import calculate_total_refund_cost

# Rule-based cost (full dataset)
rule_cost = calculate_total_refund_cost(
    data["rule_prediction"],
    data["order_amount"]
)

# ML cost (test set only)
ml_cost = calculate_total_refund_cost(
    ml_predictions,
    data.loc[X_test.index, "order_amount"]
)

print("Rule-Based Total Cost:", rule_cost)
print("ML-Based Total Cost:", ml_cost)
