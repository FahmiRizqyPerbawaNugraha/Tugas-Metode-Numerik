import numpy as np
import scipy.linalg as linalg

def solusi_persamaan_linear_lu_gauss(matriks_A, vektor_b):
    """
    Menyelesaikan sistem persamaan linear Ax = b menggunakan metode LU Gauss.

    Parameters:
        matriks_A (numpy.ndarray): Matriks koefisien (n x n).
        vektor_b (numpy.ndarray): Vektor konstanta (n,).
        
    Returns:
        numpy.ndarray atau None: Solusi dari sistem persamaan linear Ax = b (n,).
        Jika matriks A singular, akan mengembalikan None.
    """
    try:
        # Lakukan dekomposisi LU
        P, L, U = linalg.lu(matriks_A)
        
        # Memecahkan sistem Ly = Pb menggunakan forward substitution
        # Penerapan permutasi matriks P ke vektor b
        vektor_b_permutasi = np.dot(P, vektor_b)
        y = linalg.solve_triangular(L, vektor_b_permutasi, lower=True)
        
        # Memecahkan sistem Ux = y menggunakan backward substitution
        x = linalg.solve_triangular(U, y, lower=False)
        
        return x
    except linalg.LinAlgError:
        # Jika matriks A singular atau terdapat kesalahan dalam dekomposisi LU
        print("Matriks A singular atau terjadi kesalahan dalam dekomposisi LU. Tidak dapat menyelesaikan sistem.")
        return None

# Kode pengujian
if __name__ == "__main__":
    # Kasus uji
    matriks_A = np.array([[1, 1, 1], [0, 2, 5], [2, 5, -1]])
    vektor_b = np.array([6, -4, 27])
    x = solusi_persamaan_linear_lu_gauss(matriks_A, vektor_b)
    
    # Output hasil solusi
    if x is not None:
        print("Solusi menggunakan metode LU Gauss:", x)
    else:
        print("Tidak dapat menyelesaikan sistem persamaan linear.")
