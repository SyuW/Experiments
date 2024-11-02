import matplotlib.pyplot as plt
import numpy as np


def c(n, k):
    k = min(k, n-k)
    num = 1
    denom = 1
    for i in range(1, k+1):
        num *= n - i + 1
        denom *= i
    return num // denom

binDist = lambda N, m, p : c(N, m) * pow((1-p), N-m) * pow(p, m)
normDist = lambda x, s, m: np.exp(-pow(x - m, 2) / (2 * pow(s, 2))) / (np.sqrt(2 * np.pi) * s)
eigenf3 = lambda x, s: normDist(x, s, 0) * (3 * (x / s) - pow(x / s, 3))
eigenf4 = lambda x, s: normDist(x, s, 0) * (3 - 6 * pow(x / s, 2) + pow(x / s, 4))

# Observe that the n=3 eigenfunction did not contribute to the 
# asymptotics. This is because that the n=3 eigenfunction is odd, but both the
# normal and binomial distributions are even, hence their difference is even.
# To respect the parity symmetry, we must have that the n=3 coefficient vanishes.
# If we were to flip a biased coin: p != 0.5, then the binomial distribution becomes
# skewed, breaking the symmetry, so it admits an n=3 mode in this case. 

N2 = 100
prob = 0.5

plt.figure(figsize=(14, 8))

m2 = np.arange(1, N2+1)
plt.subplot(1, 3, 1)
bin2 = np.array([binDist(N2, m2, prob) for m2 in range(1, N2+1)])
norm2 = np.array([normDist(m, np.sqrt(N2 * prob * (1 - prob)), prob * N2) for m in m2])
plt.plot(m2, norm2-bin2, label="Difference")
plt.title(r"Difference between normal and binomial distributions")

plt.subplot(1, 3, 2)
f3 = np.array([eigenf3(m-prob*N2, np.sqrt(N2 * prob * (1-prob))) for m in m2])
plt.plot(m2, f3, label="n=3 eigenfunction")
plt.title(r"$n=3$ eigenfunction of $T$ map linearization")

plt.subplot(1, 3, 3)
f4 = np.array([eigenf4(m-prob*N2, np.sqrt(N2 * prob * (1-prob))) for m in m2])
plt.plot(m2, f4, label="n=4 eigenfunction")
plt.title(r"$n=4$ eigenfunction of $T$ map linearization")
plt.suptitle(fr"Convergence of renormalization group transformation for $N={N2}, p={prob}$ binomial distribution")

# Apply tight layout
plt.tight_layout()

plt.savefig(f"renorm_group_N_{N2}_p_{prob}.png", format="png", dpi=300)

plt.show()
plt.clf()