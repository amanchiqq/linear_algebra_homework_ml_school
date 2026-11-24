# Done by Aman Kumar
# Programming homework

import numpy as np
import matplotlib.pyplot as plt

#P12.Dual bases and interpolation.
print('P12.Dual bases and interpolation.')

# 1. Dual basis
def dual_basis(P):
    return np.linalg.inv(P)

rng = np.random.default_rng(42)

P = rng.normal(size=(5, 5))
D = dual_basis(P)

# Check ri @ bj = delta_ij
print("Dual basis check:", np.allclose(D @ P, np.eye(5)))

x = rng.normal(size=5)
coords = D @ x

# Reconstruct x using the columns of P
x_new = np.zeros(5)

for i in range(5):
    x_new += coords[i] * P[:, i]

print("Reconstruction check:", np.allclose(x, x_new))

# 2.
nodes = np.array([-1., 0., 1.])

# Each row contains [1, t, t**2]
V = np.vander(nodes, increasing=True)
L = np.linalg.inv(V)

print("\nVandermonde matrix:")
print(V)

print("\nLagrange coefficients (each column is one polynomial):")
print(L)

print("Lagrange check:", np.allclose(V @ L, np.eye(3)))


# 3. Interpolation using the Lagrange formula
def interpolate(nodes, values, t):
    t = np.asarray(t, dtype=float)
    p = np.zeros_like(t)

    for k in range(len(nodes)):
        lk = np.ones_like(t)

        for j in range(len(nodes)):
            if j != k:
                lk *= (t - nodes[j]) / (nodes[k] - nodes[j])

        p += values[k] * lk
    return p


# sin(3t): 9 equally spaced nodes on [-1, 1]
nodes = np.linspace(-1, 1, 9)
values = np.sin(3 * nodes)

t = np.linspace(-1, 1, 1000)
f = np.sin(3 * t)
p = interpolate(nodes, values, t)

print("\nsin interpolation check:", np.allclose(interpolate(nodes, values, nodes), values))

plt.figure(figsize=(8, 4))
plt.plot(t, f, label="sin(3t)")
plt.plot(t, p, "--", label="Interpolation")
plt.scatter(nodes, values, color="red", label="Nodes")
plt.title("9 equally spaced nodes")
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

nodes = np.linspace(-5, 5, 21)
values = 1 / (1 + nodes**2)

t = np.linspace(-5, 5, 2000)
f = 1 / (1 + t**2)
p = interpolate(nodes, values, t)

print("Runge interpolation check:", np.allclose(interpolate(nodes, values, nodes), values))

plt.figure(figsize=(8, 4))
plt.plot(t, f, label="1 / (1 + t²)")
plt.plot(t, p, "--", label="Interpolation")
plt.scatter(nodes, values, color="red", label="Nodes")
plt.title("21 equally spaced nodes — Runge phenomenon")
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
