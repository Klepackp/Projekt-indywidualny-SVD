# Reconstruct SVD
import numpy as np
from scipy.linalg import svd

A = np.array([[1, 2], [3, 4], [5, 6]])

def svd_naive(A):
    #dokładność
    eps = 0.00001
    n_og, m_og = A.shape
    k = min(n_og, m_og)
    A_og = A.copy()
    #obliczenie
    if n_og != m_og:
        if n_og > m_og:
            A = A.T @ A
            n, m = A.shape
        else:
            A = A @ A.T
            n, m = A.shape
    else:
        n, m = n_og, m_og
    #normalizacja qr
    Q = np.random.rand(n, k)
    Q, _ = np.linalg.qr(Q)
    Q_prev = Q

    # i można zmienić w celu zmiany dokładności, raczej tylko dla bardzo dużych macierzy
    for i in range(1000):
        Z = A @ Q
        Q, R = np.linalg.qr(Z)
        #sprawdzenie czy wartości zmieniają się na tyle, że znadują się juz poniżej naszej dokładności
        err = ((Q - Q_prev) ** 2).sum()
        Q_prev = Q
        if err < eps:
            break

    singular_values = np.sqrt(np.diag(R))
    # deal with different shape input matrices
    if n_og < m_og:
        left_vecs = Q.T
        # use property Values @ V = U.T@A => V=inv(Values)@U.T@A
        right_vecs = np.linalg.inv(np.diag(singular_values)) @ left_vecs.T @ A_og
    elif n_og == m_og:
        left_vecs = Q.T
        right_vecs = left_vecs
        singular_values = np.square(singular_values)
    else:
        right_vecs = Q.T
        # use property Values @ V = U.T@A => U=A@V@inv(Values)
        left_vecs = A_og @ right_vecs.T @ np.linalg.inv(np.diag(singular_values))

    return left_vecs, singular_values, right_vecs
U_naive, E_naive, V_naive = svd_naive(A)

U_np, E_np, V_np = np.linalg.svd(A, full_matrices=False)
print(U_naive,"\n \n", U_np)
