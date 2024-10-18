import numpy as np
import matplotlib.pyplot as plt

def entropy(p):
    return -np.sum(p * np.log2(p + 1e-12))

def cross_enctropy(p, q):
    return -np.sum(p * np.log2(q + 1e-12))

def kl_divergence(p, q):
    return np.sum(p * np.log2((p + 1e-12) / (q + 1e-12)))

def validate_distribution(p):
    return np.isclose(np.sum(p), 1.0) and np.all(p >= 0)

def get_distribution(prompt):
    while True:
        try:
            dist = np.array([float(x) for x in input(prompt).split()])
            if validate_distribution(dist):
                return dist
            else:
                print("Invalid distribution. Probabilities must sum to 1 and be non-negative.")
        except ValueError:
            print("Invalid input. Please enter space-separated probabilities.")

def main():
    p = get_distribution("Enter the true distribution (space-separated probabilities): ")
    q = get_distribution("Enter the predicted distribution (space-separated probabilities): ")

    print(f"Entropy of P: {entropy(p):.4f}")
    print(f"Cross-enctropy of P and Q: {cross_enctropy(p, q):.4f}")
    print(f"KL-divergence of P and Q: {kl_divergence(p, q):.4f}")

    # Visualization
    plt.bar(range(len(p)), p, alpha=0.5, label='True (P)')
    plt.bar(range(len(q)), q, alpha=0.5, label='Predicted (Q)')
    plt.legend()
    plt.title('True vs Predicted Distribution')
    plt.savefig('distribution_comparison.png')
    plt.close()

if __name__=="__main__":
    main()