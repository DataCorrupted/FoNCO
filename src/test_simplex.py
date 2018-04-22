import numpy as np
# Below defines some test problems.
def Q1():
    # Check this example on pp.40
    # Define A, b:
    A = np.array([
        [ -1, 2, 1, 0, 0],
        [ 2,  3,  0, 1, 0],
        [ 1, -1,  0, 0, 1]
    ], dtype = np.float64)
    b = np.array([[4], [12], [3]], dtype = np.float64)
    # Define the objective function and the initial basis:
    c = np.array([-4, -1, 0, 0, 0])
    basis = np.array([2, 3, 4])
    return c, A, b, basis

def Q2():
    # Check this example on pp.46
    # Define A, b:
    A = np.array([
        [ 1,  1, -2, 1, 0, 0],
        [ 2, -1,  4, 0, 1, 0],
        [-1,  2, -4, 0, 0, 1]
    ], dtype = np.float64)
    b = np.array([[10], [8], [4]], dtype = np.float64)
    # Define the objective function and the initial basis:
    c = np.array([1, -2, 1, 0, 0, 0])
    basis = np.array([3, 4, 5])
    return c, A, b, basis

def Q3():
    # Define A, b:
    A = np.array([
        [ 1, -2, -1,  2, -1, -0,  1,  0,  0,  0,  0,  0],
        [ 1,  4, -1, -4, -0, -1,  0,  1,  0,  0,  0,  0],
        [ 1,  0, -1, -0,  0,  0,  0,  0,  1,  0,  0,  0],
        [ 0,  1, -0, -1,  0,  0,  0,  0,  0,  1,  0,  0],
        [-1, -0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0],
        [-0, -1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  1]
    ], dtype = np.float64)
    b = np.array([[1], [-4], [1], [1], [1], [1]], dtype = np.float64)
    # Define the objective function and the initial basis:
    c = np.array([0, 2, 0, -2, 1, 1, 1, 0, 0, 0, 0, 0])
    basis = np.array([6, 5, 8, 9, 10, 11])
    
    return c, A, b, basis

def Q4():
    # Check this example on pp.46
    # Define A, b:
    A = np.array([
        [ 2,  1,  4, 0, -1,  0],
        [ 2,  2,  0, 4,  0, -1]
    ], dtype = np.float64)
    b = np.array([[2], [3]], dtype = np.float64)
    # Define the objective function and the initial basis:
    c = np.array([12, 8, 16, 12, 0, 0])
    basis = np.array([2, 3])
    return c, A, b, basis

def Q5():
    # This test comes from Hudie Zhou
    # Define A, b:
    A = np.array([
        [-1,  1,  1,  0,  0],
        [ 1,  2,  0,  1,  0],
        [ 3,  1,  0,  0,  1]
    ], dtype = np.float64)
    b = np.array([[2], [10], [5]], dtype = np.float64)
    # Define the objective function and the initial basis:
    c = np.array([-2, -3, 0, 0, 0])
    basis = np.array([2, 3, 4])
    return c, A, b, basis

def Q6():
    # Check this example on pp.130
    # Define A, b:
    A = np.array([
        [ 3, -1,  1, -1,  0],
        [ 1,  2, -3,  0, -1]
    ], dtype = np.float64)
    b = np.array([[1], [2]], dtype = np.float64)
    # Define the objective function and the initial basis:
    c = np.array([2, 3, 1, 0, 0])
    basis = np.array([2, 3, 4])
    return c, A, b, basis