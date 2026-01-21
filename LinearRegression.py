"""
Proof of Concept: Privacy-Preserving Inference
Using Paillier Homomorphic Encryption

Task: Diabetes classification (encrypted inference)
"""

import numpy as np
import pandas as pd
from phe import paillier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time


# -----------------------------
# 1. Load and preprocess data
# -----------------------------
df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.astype(np.float32)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Dataset loaded and preprocessed.")
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}\n")


# -----------------------------
# 2. Train plaintext model
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

weights = model.coef_[0]
bias = model.intercept_[0]

print("Plaintext Logistic Regression model trained.\n")


# -----------------------------
# 3. Generate Paillier keys
# -----------------------------
public_key, private_key = paillier.generate_paillier_keypair()
print("Paillier keypair generated.\n")


# -----------------------------
# 4. Helper functions
# -----------------------------

def encrypt_vector(pubkey, x):
    return [pubkey.encrypt(float(v)) for v in x]

def encrypted_dot_product(enc_x, weights):
    """Compute encrypted dot product <w, x>"""
    enc_result = enc_x[0] * weights[0]
    for i in range(1, len(weights)):
        enc_result += enc_x[i] * weights[i]
    return enc_result


# -----------------------------
# 5. Single encrypted inference demo
# -----------------------------
x_sample = X_test[0]
y_true = y_test[0]

x_enc = encrypt_vector(public_key, x_sample)

enc_score = encrypted_dot_product(x_enc, weights)
enc_score += bias

score = private_key.decrypt(enc_score)
prediction = 1 if score >= 0 else 0

print("Single encrypted inference demo:")
print("True label:", y_true)
print("Decrypted linear score:", score)
print("Predicted label:", prediction, "\n")


# -----------------------------
# 6. Encrypted inference on test set
# -----------------------------
def encrypted_predict(X_test, weights, bias, public_key, private_key):
    predictions = []

    start_time = time.time()

    for x in tqdm(X_test, desc="Encrypted inference", unit="sample"):
        x_enc = encrypt_vector(public_key, x)

        enc_score = encrypted_dot_product(x_enc, weights)
        enc_score += bias

        score = private_key.decrypt(enc_score)
        pred = 1 if score >= 0 else 0
        predictions.append(pred)

    total_time = time.time() - start_time
    print(f"\nTotal encrypted inference time: {total_time:.2f} seconds")

    return predictions


# Use small subset for PoC (Paillier is slow by design)
X_test_small = X_test[:10]
y_test_small = y_test[:10]

y_pred_enc = encrypted_predict(
    X_test_small, weights, bias, public_key, private_key
)

enc_acc = accuracy_score(y_test_small, y_pred_enc)

plain_acc = accuracy_score(y_test, model.predict(X_test))

print("Encrypted inference accuracy:", enc_acc)
print("Plaintext inference accuracy:", plain_acc)
