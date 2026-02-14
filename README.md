# Refund Decision Simulator

## Overview
This project compares rule-based and machine learning approaches 
for refund decision systems in online platforms.

Refund decisions must balance:
- Fraud risk
- Customer retention
- Operational cost

## Methodology
- Synthetic dataset (1000 simulated orders)
- Rule-based heuristic model
- Logistic Regression classifier
- Train-test split evaluation
- Economic cost simulation

## Evaluation Metrics
- Accuracy
- Confusion matrix
- Classification report
- Total economic cost comparison

## Objective
To evaluate whether ML-based decision systems reduce overall economic
cost compared to traditional rule-based heuristics.

## Core Concepts Demonstrated
- Cost-Sensitive Learning
- Decision Systems Engineering
- Economic Optimization vs Accuracy Optimization
- Simulation-Based Experimental Design
## Results Summary

On the test dataset:

- Rule-Based Test Cost: ₹128,401
- ML-Based Test Cost: ₹162,748

Although ML slightly improved prediction patterns, 
the rule-based system achieved a lower economic cost 
under the defined cost model.

This demonstrates that optimizing for accuracy alone 
does not guarantee optimal economic outcomes.
## Limitations

- Synthetic dataset (simulated environment)
- Simple logistic regression baseline
- Static cost assumptions


## This project demonstrates how machine learning systems should be evaluated not only on prediction accuracy but on real-world economic outcomes.
