# Privacy-Preserving Inference using Paillier Homomorphic Encryption

## Overview
This repository presents a **proof of concept** for computation on encrypted data using **Paillier homomorphic encryption**.  
The project demonstrates **privacy-preserving inference** for a diabetes classification task, where input features remain encrypted during inference and are never revealed to the computing party.

The goal is to explore the **practical feasibility and limitations** of additive homomorphic encryption for machine learning inference.

---

## Key Idea
- The machine learning model (logistic regression) is trained **in plaintext**
- During inference:
  - Feature vectors are **encrypted using Paillier encryption**
  - The server computes the **linear decision function on encrypted data**
  - Only the **final score is decrypted** to obtain the prediction
- No raw feature values are exposed during encrypted computation

This project focuses on **encrypted inference**, not encrypted training.

---

## Project Structure

- ├── diabetes.csv # Diabetes dataset
- ├── LinearRegression.py # Encrypted inference PoC (main script)
- ├── paillier_demo.py # Basic Paillier homomorphic operations demo
- ├── requirements.txt # Python dependencies
- └── README.md


---

## Dependencies
- Python 3.9 – 3.11 (recommended)
- `phe` – Paillier homomorphic encryption
- `numpy`, `pandas`
- `scikit-learn`
- `tqdm` (progress bar for encrypted inference)

Install dependencies:
```bash
pip install -r requirements.txt
#


