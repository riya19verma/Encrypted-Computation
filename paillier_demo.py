"""
Demonstration of computation on encrypted data using Paillier encryption.
Shows homomorphic addition and encrypted aggregation without revealing
plaintext values.
"""
from phe import paillier

def encrypted_mean(public_key, private_key, values):
    encrypted_vals = [public_key.encrypt(v) for v in values]
    encrypted_sum = sum(encrypted_vals)
    decrypted_sum = private_key.decrypt(encrypted_sum)
    return decrypted_sum / len(values)

def main():
    public_key, private_key = paillier.generate_paillier_keypair()

    m1 = int(input("Enter the first number: "))
    m2 = int(input("Enter the first number: "))

    c1 = public_key.encrypt(m1)
    c2 = public_key.encrypt(m2)

    c_sum = c1 + c2
    c_scaled = c1 * 5

    print("Decrypted (m1 + m2):", private_key.decrypt(c_sum))
    print("Decrypted (m1 * 5):", private_key.decrypt(c_scaled))

    values = [10, 20, 30, 40]
    mean = encrypted_mean(public_key, private_key, values)
    print("Encrypted mean:", mean)

if __name__ == "__main__":
    main()

