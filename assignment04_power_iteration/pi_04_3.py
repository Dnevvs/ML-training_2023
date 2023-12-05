import numpy as np


def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    # YOUR CODE HERE

    x = np.random.rand(data.shape[1])
    for _ in range(num_steps):
        y = np.dot(data, x)
        x1 = y / np.linalg.norm(y)
        if np.linalg.norm(x1 - x) < 1e-6:
            break
        x = x1
    return float(np.dot(x1, np.dot(data, x1)) / np.dot(x1, x1)), x1
