# Done by Aman Kumar
# Programming homework

import numpy as np

# P15. Tensors with NumPy
print('P15. Tensors with NumPy')

rng = np.random.default_rng(42)

# 1. Tensor indices and unfoldings
T = np.arange(24).reshape(2, 3, 4)
flat = T.reshape(-1)

print("Tensor:")
print(T)

check = True

for i in range(2):
    for j in range(3):
        for k in range(4):
            if T[i, j, k] != flat[12*i + 4*j + k]:
                check = False

print("Index check:", check)

M1 = T.reshape(2, 12)
M2 = T.transpose(1, 0, 2).reshape(3, 8)

print("\nFirst unfolding:")
print(M1)
print("Rank:", np.linalg.matrix_rank(M1))

print("\nSecond unfolding:")
print(M2)
print("Rank:", np.linalg.matrix_rank(M2))

# 2. Operations from Table 1
A = rng.normal(size=(4, 4))
B = rng.normal(size=(4, 4))
x = rng.normal(size=4)
y = rng.normal(size=4)

result = np.einsum("ij,jk->ik", A, B)
print("\nMatrix product:", np.allclose(result, A @ B))

result = np.einsum("ii->", A)
print("Trace:", np.allclose(result, np.trace(A)))

result = np.einsum("i,i->", x, y)
print("Dot product:", np.allclose(result, x @ y))

result = np.einsum("i,j->ij", x, y)
print("Outer product:", np.allclose(result, np.outer(x, y)))

result = np.einsum("ij->ji", A)
print("Transpose:", np.allclose(result, A.T))

result = np.einsum("i,ij,j->", x, A, y)
print("Bilinear form:", np.allclose(result, x @ A @ y))

# Batch of 100 matrices
A = rng.normal(size=(100, 4, 4))
B = rng.normal(size=(100, 4, 4))

traces = np.einsum("bii->b", A)
products = np.einsum("bij,bjk->bik", A, B)

traces_loop = []
products_loop = []

for b in range(100):
    traces_loop.append(np.trace(A[b]))
    products_loop.append(A[b] @ B[b])

print("Batch traces:", np.allclose(traces, traces_loop))
print("Batch products:", np.allclose(products, products_loop))
print("Batch @ check:", np.allclose(products, A @ B))

# 3. Kronecker product
A = rng.normal(size=(3, 4))
B = rng.normal(size=(2, 5))
X = rng.normal(size=(4, 5))

left = (A @ X @ B.T).reshape(-1)
right = np.kron(A, B) @ X.reshape(-1)

print("\nKronecker check:", np.allclose(left, right))

# For two 100 x 100 matrices
count = 100**4

print("Numbers stored:", count)
print("Memory in MB:", count * 8 / 10**6)


# 4. Rank-one tensor
def rank_one(u, v, w):
    return np.einsum("i,j,k->ijk", u, v, w)

u = rng.normal(size=3)
v = rng.normal(size=4)
w = rng.normal(size=5)

T = rank_one(u, v, w)
M = np.outer(u, v)

print("\nRank-one tensor shape:", T.shape)

for k in range(len(w)):
    check = np.allclose(T[:, :, k], w[k] * M)
    print("Slice", k, "check:", check)
