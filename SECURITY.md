# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| < 2.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

1. **Do not** open a public GitHub issue.
2. Email: **keshav@gangsof18.com**
3. Include a description of the vulnerability, steps to reproduce, and potential impact.

You can expect an initial response within **48 hours**.

## Scope

This project is a research implementation. Security considerations include:
- Dependency vulnerabilities (scikit-learn, LightGBM, XGBoost, pandas, numpy)
- Data handling in dataset loaders (CSV parsing)
- No authentication or API endpoints are exposed in the current version
