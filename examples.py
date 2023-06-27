import numpy as np   

tasks = {
    'right': {
        'dimension': 100,
        'valid_size': 200000,
        'test_size': 100000,
        'target_function': lambda X: 0.5 * (X[:, :2].prod(axis=1) + X[:, :6].prod(axis=1)),
        'mask': np.array([[1,1,0,0,0,0,0, 0], [1,1,1,1,1,1,1, 1]]),
        'print_coefficients': True
    },
    'left': {
        'dimension': 100,
        'valid_size': 200000,
        'test_size': 100000,
        'target_function': lambda X: 0.5 * X[:, :5].prod(axis=1) + X[:, :6].prod(axis=1),
        'mask': np.array([[1,1,1,1,1,0,0, 0], [1,1,1,1,1,1,0, 0]]),
        'print_coefficients': True
    },
    'middle': {
        'dimension': 100,
        'valid_size': 200000,
        'test_size': 100000,
        'target_function': lambda X: X[:, :5].prod(axis=1) + 0.5 * X[:, 5:11].prod(axis=1),
        'mask': np.array([[1,1,1,1,1,0,0, 0,0,0,0], [0,0,0,0,0,1,1,1,1,1,1]]),
        'print_coefficients': True
    }, 
    'parity5': {
        'dimension': 100,
        'valid_size': 200000,
        'test_size': 100000,
        'target_function': lambda X: X[:, :5].prod(axis=1),
        'mask': np.ones((1,5), dtype=int),
        'print_coefficients': True
    }, 
    'parity7': {
        'dimension': 100,
        'valid_size': 200000,
        'test_size': 100000,
        'target_function': lambda X: X[:, :7].prod(axis=1),
        'mask': np.ones((1,7), dtype=int),
        'print_coefficients': True
    }, 
    'parity10': {
        'dimension': 100,
        'valid_size': 200000,
        'test_size': 100000,
        'target_function': lambda X: X[:, :10].prod(axis=1),
        'mask': np.ones((1,10), dtype=int),
        'print_coefficients': True
    }, 
}