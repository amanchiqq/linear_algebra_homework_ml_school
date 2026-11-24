# Done by Aman Kumar
# Programming homework

import numpy as np
from time import perf_counter


#P16. The LU decomposition and rank-one updates
print('P16. The LU decomposition and rank-one updates')
rng = np.random.default_rng(42)

# 1. LU decomposition without row swaps
def lu(A):
    U = np.array(A, dtype=float)
    n = len(U)
    L = np.eye(n)

    for k in range(n):
        if U[k, k] == 0:
            raise np.linalg.LinAlgError("Zero pivot: row swaps are needed")

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k] # Save the elimination multiplier
            U[i, k:] -= L[i, k] * U[k, k:] # Update the remaining part of the row
            U[i, k] = 0.0
    return L, U

# 2. Solve L U x = b
def solve_lu(L, U, b):
    n = len(b)
    y = np.zeros(n)
    x = np.zeros(n)

    # Forward substitution: L y = b
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]

    # Back substitution: U x = y
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]

    return x

# Example 6.107
A = np.array([
    [2, 1, 1],
    [4, 3, 3],
    [8, 7, 9]
])

b = np.array([5, 13, 31])

L, U = lu(A)

expected_L = np.array([
    [1, 0, 0],
    [2, 1, 0],
    [4, 3, 1]
])

expected_U = np.array([
    [2, 1, 1],
    [0, 1, 1],
    [0, 0, 2]
])

print("L:")
print(L)

print("U:")
print(U)

print("L check:", np.allclose(L, expected_L))
print("U check:", np.allclose(U, expected_U))
print("LU check:", np.allclose(L @ U, A))
print("Solution:", solve_lu(L, U, b))  # [1. 2. 1.]


# Random 6 x 6 matrices and systems
for _ in range(20):
    A = rng.normal(size=(6, 6))
    b = rng.normal(size=6)

    L, U = lu(A)
    x = solve_lu(L, U, b)

    assert np.allclose(L @ U, A)
    assert np.allclose(x, np.linalg.solve(A, b))

print("\nAll 20 random checks passed.")

# Small pivot example
A = np.array([
    [1e-17, 1],
    [1,     1]
])

b = np.array([1., 2.])

L, U = lu(A)

print("\nWithout row swap:", solve_lu(L, U, b))
print("NumPy solution:", np.linalg.solve(A, b))

# Swap both the matrix rows and the corresponding entries of b
A_swap = A[[1, 0]]
b_swap = b[[1, 0]]

L, U = lu(A_swap)

print("With row swap:", solve_lu(L, U, b_swap))

# 3. Sherman-Morrison updates
def update_inverse(G, v):
    left = G @ v
    right = v @ G

    return G - np.outer(left, right) / (1 + v @ left)

X = rng.normal(size=(200, 50))
new_rows = rng.normal(size=(100, 50))

S = X.T @ X
G_start = np.linalg.inv(S)

# Update the inverse after each new row
G = G_start.copy()

start = perf_counter()

for v in new_rows:
    G = update_inverse(G, v)

update_time = perf_counter() - start

S_full = S.copy()

start = perf_counter()

for v in new_rows:
    S_full += np.outer(v, v)
    G_full = np.linalg.inv(S_full)

full_time = perf_counter() - start

# Independent check using the final data matrix
X_final = np.vstack([X, new_rows])
G_final = np.linalg.inv(X_final.T @ X_final)

print("\nInverse update check:", np.allclose(G, G_final))
print("Recomputed inverse check:", np.allclose(G_full, G_final))

print("Update time:", update_time)
print("Recompute time:", full_time)
print("Time ratio (recompute / update):", full_time / update_time)
