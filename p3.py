# Done by Aman Kumar
# Programming homework

import numpy as np

# P3. Row reduction
print('P3. Row reduction')

# 1. Reduced row echelon form
def rref(A, tol=1e-10):
    R = np.array(A, dtype=float)

    if R.ndim != 2:
        raise ValueError("A must be a two-dimensional matrix")

    rows, cols = R.shape
    pivots = []
    r = 0

    for c in range(cols):
        if r == rows:
            break
        # Choosing the largest available pivot in absolute value
        pivot_row = r + np.argmax(np.abs(R[r:, c]))

        if abs(R[pivot_row, c]) < tol:
            continue

        R[[r, pivot_row]] = R[[pivot_row, r]]
        R[r] = R[r] / R[r, c]

        # Eliminate all other entries in the pivot column
        for i in range(rows):
            if i != r:
                factor = R[i, c]
                R[i] = R[i] - factor * R[r]

        pivots.append(c)
        r += 1
    R[np.abs(R) < tol] = 0.0
    return R, pivots

# 2. Matrix rank
def rank(A, tol=1e-10):
    R, pivots = rref(A, tol)
    return len(pivots)

# 3. Null space basis
def null_space_basis(A, tol=1e-10):
    R, pivots = rref(A, tol)
    cols = R.shape[1]
    basis = []

    for free_col in range(cols):
        if free_col in pivots:
            continue

        # Choose one free variable to be 1
        v = np.zeros(cols)
        v[free_col] = 1.0

        for row, pivot_col in enumerate(pivots):
            v[pivot_col] = -R[row, free_col]

        basis.append(v)

    return basis


# 4. Column space basis
def col_space_basis(A, tol=1e-10):
    original = np.array(A, dtype=float)
    R, pivots = rref(original, tol)

    return [original[:, c].copy() for c in pivots]


# Test 1
A = np.array([
    [1,  2, -1],
    [3, -1,  4]
])

expected = np.array([
    [1, 0,  1],
    [0, 1, -1]
])

R, pivots = rref(A)

assert np.allclose(R, expected)
assert pivots == [0, 1]

print("Test 1 passed.")
print("RREF:")
print(R)
print("Pivot columns:", pivots)


# Test 2
F = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

expected_F = np.array([
    [1, 0, -1],
    [0, 1,  2],
    [0, 0,  0]
])

R, pivots = rref(F)

assert np.allclose(R, expected_F)
assert pivots == [0, 1]
assert rank(F) == 2

null_basis = null_space_basis(F)
column_basis = col_space_basis(F)

assert len(null_basis) == 1
assert np.allclose(null_basis[0], [1, -2, 1])
assert np.allclose(column_basis[0], F[:, 0])
assert np.allclose(column_basis[1], F[:, 1])

print("\nTest 2 passed.")
print("RREF:")
print(R)
print("Rank:", rank(F))
print("Null space basis:", null_basis)
print("Column space basis:", column_basis)

# Test 3: twenty random 5 x 7 matrices
rng = np.random.default_rng(42)

for _ in range(20):
    A = rng.normal(size=(5, 7))

    basis = null_space_basis(A)
    matrix_rank = rank(A)

    for v in basis:
        assert np.allclose(A @ v, np.zeros(5))

    # Rank-nullity theorem
    assert matrix_rank + len(basis) == 7

    # Compare with NumPy
    assert matrix_rank == np.linalg.matrix_rank(A)

print("\nTest 3 passed")

# Test 4
B = rng.normal(size=(6, 4))
C = rng.normal(size=(4, 8))
A = B @ C

matrix_rank = rank(A)
basis = null_space_basis(A)

assert matrix_rank == 4
assert np.linalg.matrix_rank(A) == 4
assert len(basis) == 4

for v in basis:
    assert np.allclose(A @ v, np.zeros(6))

assert matrix_rank + len(basis) == 8

print("\nTest 4 passed.")
print("Matrix shape:", A.shape)
print("Rank:", matrix_rank)
print("Null space dimension:", len(basis))
