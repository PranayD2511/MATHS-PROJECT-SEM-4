import numpy as np
import matplotlib.pyplot as plt

def geometric_multiplicity(matrix, eigenvalue):
    """Calculate geometric multiplicity (dimension of null space)."""
    eig_matrix = matrix - eigenvalue * np.eye(matrix.shape[0])
    return matrix.shape[0] - np.linalg.matrix_rank(eig_matrix)

def visualize_matrix(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    unique_eigenvalues, counts = np.unique(eigenvalues, return_counts=True)
    
    # Geometric multiplicities
    geometric_multiplicities = [geometric_multiplicity(matrix, value) for value in unique_eigenvalues]
    
    # Diagonalizability check: algebraic multiplicity == geometric multiplicity for each eigenvalue
    is_diagonalizable = all(gm == count for gm, count in zip(geometric_multiplicities, counts))
    
    # Check if sum of geometric multiplicities equals the matrix size (diagonalizability condition)
    total_geometric_multiplicities = sum(geometric_multiplicities)
    is_diagonalizable = is_diagonalizable and total_geometric_multiplicities == matrix.shape[0]
    
    # Output eigenvalue information
    print("Eigenvalues and their algebraic and geometric multiplicities:")
    for value, count in zip(unique_eigenvalues, counts):
        print(f"Eigenvalue: {value:.2f}, Algebraic multiplicity: {count}, Geometric multiplicity: {geometric_multiplicities[np.where(unique_eigenvalues == value)[0][0]]}")
    
    # Plot eigenvalues
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), color='red', label='Eigenvalues', zorder=5)
    plt.axhline(0, color='black',linewidth=1)
    plt.axvline(0, color='black',linewidth=1)
    plt.grid(True)
    plt.title("Eigenvalues on the Complex Plane")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.legend()
    plt.show()

    # Diagonalizability check output
    if is_diagonalizable:
        print("\nThe matrix is diagonalizable.")
    else:
        print("\nThe matrix is not diagonalizable.")

# Example matrices
A = np.array([[7, 4, -1], [4, 7, -1], [-4, -4, 4]])
print("Matrix A:")
print(A)
visualize_matrix(A)

B = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
print("\nMatrix B:")
print(B)
visualize_matrix(B)
