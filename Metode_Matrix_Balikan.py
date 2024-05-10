import numpy as np

def solve_linear_equations(A, b):
    """
    Menyelesaikan sistem persamaan linear menggunakan numpy.linalg.solve.
    
    Parameters:
        A (numpy.ndarray): Matriks koefisien (m x n).
        b (numpy.ndarray): Vektor konstanta (m x 1).
    
    Returns:
        numpy.ndarray: Vektor solusi (n x 1).
    """
    # Gunakan fungsi numpy.linalg.solve untuk menyelesaikan sistem persamaan
    solution = np.linalg.solve(A, b)
    return solution

def run_tests():
    """
    Melakukan pengujian fungsi solve_linear_equations.
    """
    # Tes kasus pertama
    coeffs_1 = np.array([[2, 1], [1, -1]])
    constants_1 = np.array([4, 1])
    expected_1 = np.array([3, 1])
    output_1 = solve_linear_equations(coeffs_1, constants_1)
    assert np.allclose(output_1, expected_1), f"Tes kasus pertama gagal: Diharapkan {expected_1}, diperoleh {output_1}"

    # Tes kasus kedua
    coeffs_2 = np.array([[3, -2, 5], [2, 6, -8], [1, 5, -6]])
    constants_2 = np.array([9, 3, 4])
    expected_2 = np.array([2, 1, 1])
    output_2 = solve_linear_equations(coeffs_2, constants_2)
    assert np.allclose(output_2, expected_2), f"Tes kasus kedua gagal: Diharapkan {expected_2}, diperoleh {output_2}"

    print("Semua pengujian berhasil.")

if __name__ == "__main__":
    run_tests()
