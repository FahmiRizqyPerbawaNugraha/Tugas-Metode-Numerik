import numpy as np

def crout_lu_solve(A, b):
    """
    Menyelesaikan sistem persamaan linear Ax = b menggunakan dekomposisi Crout.

    Parameters:
        A (numpy.ndarray): Matriks koefisien (n x n).
        b (numpy.ndarray): Vektor konstanta (n,).
        
    Returns:
        numpy.ndarray: Solusi dari sistem persamaan linear Ax = b (n,).
        Jika terjadi kesalahan selama dekomposisi atau pemecahan sistem, fungsi akan mengembalikan None.
    """
    n = len(A)
    # Inisialisasi matriks L dan U
    L = np.zeros((n, n))
    U = np.eye(n)  # Matriks identitas untuk U karena metode Crout menggunakan diagonal U bernilai 1

    # Melakukan dekomposisi Crout
    try:
        for i in range(n):
            for j in range(i, n):
                # Menghitung elemen L[j, i]
                L[j, i] = A[j, i] - np.dot(L[j, :i], U[:i, i])
            
            for j in range(i + 1, n):
                # Menghitung elemen U[i, j]
                if L[i, i] == 0:
                    raise ValueError("Singular matrix: L[i, i] = 0")
                U[i, j] = (A[i, j] - np.dot(L[i, :i], U[:i, j])) / L[i, i]
        
        # Memecahkan Ly = b menggunakan forward substitution
        y = np.zeros(n)
        for i in range(n):
            y[i] = b[i] - np.dot(L[i, :i], y[:i])
        
        # Memecahkan Ux = y menggunakan backward substitution
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
        
        return x
    
    except ValueError as e:
        # Jika terjadi kesalahan selama proses dekomposisi atau pemecahan sistem
        print(f"Terjadi kesalahan: {e}")
        return None

def test_crout_lu_solve():
    # Kasus uji 1
    A1 = np.array([[2, 1], [1, -1]])
    b1 = np.array([4, 1])
    expected1 = np.array([3.0, 1.0])
    result1 = crout_lu_solve(A1, b1)
    assert np.allclose(result1, expected1), f"Kasus uji 1 gagal: Hasil yang diharapkan {expected1}, diperoleh {result1}"

    # Kasus uji 2
    A2 = np.array([[3, -2, 5], [2, 6, -8], [1, 5, -6]])
    b2 = np.array([9, 3, 4])
    expected2 = np.array([2.0, 1.0, 1.0])
    result2 = crout_lu_solve(A2, b2)
    assert np.allclose(result2, expected2), f"Kasus uji 2 gagal: Hasil yang diharapkan {expected2}, diperoleh {result2}"

    print("Semua uji kasus berhasil.")

# Menjalankan pengujian
test_crout_lu_solve()
