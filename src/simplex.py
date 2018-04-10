import numpy as np

def simplex(of,basis,tableau):
    # Get the number of rows and columns in the tableau:
    n_rows = tableau.shape[0]
    n_cols = tableau.shape[1]
    # Start the simplex algorithm:
    # Compute zj-cj. If zj - cj >= 0 for all columns then current 
    # solution is optimal solution.
    check = np.sum(np.reshape(of[list(basis)],(n_rows,1)) * 
        tableau[:,0:n_cols-1],axis=0) - of
    count = 0
    while ~np.all(check >= 0):
        # Determine the pivot column:
        # The pivot column is the column corresponding to 
        # minimum zj-cj.
        pivot_col = np.argmin(check)
        # Determine the positive elements in the pivot column.
        # If there are no positive elements in the pivot column  
        # then the optimal solution is unbounded.
        positive_rows = np.where(tableau[:,pivot_col] > 0)[0]
        if positive_rows.size == 0:
            print('Unbounded Solution!')
            break
        # Determine the pivot row:
        divide=(tableau[positive_rows,n_cols-1]
            /tableau[positive_rows,pivot_col])
        pivot_row = positive_rows[np.where(divide 
            == divide.min())[0][-1]]
        # Update the basis:
        basis[pivot_row] = pivot_col
        # Perform gaussian elimination to make pivot element one and
        # elements above and below it zero:
        tableau[pivot_row,:]=(tableau[pivot_row,:]
            /tableau[pivot_row,pivot_col])
        for row in range(n_rows):
            if row != pivot_row:
                tableau[row,:] = (tableau[row,:] 
                    - tableau[row,pivot_col]*tableau[pivot_row,:])
        # Determine zj-cj
        check = np.sum(np.reshape(of[list(basis)],(n_rows,1)) * 
            tableau[:,0:n_cols-1],axis=0) - of
        count += 1
        print('Step %d' % count)
        print(tableau)
    return basis,tableau

def get_solution(of,basis,tableau):
    # Get the no of columns in the tableau:
    n_cols = tableau.shape[1]
    # Get the optimal solution:
    solution = np.zeros(of.size)
    solution[list(basis)] = tableau[:,n_cols-1]
    # Determine the optimal value:
    value = np.sum(of[list(basis)] * tableau[:,n_cols-1])
    return solution,value


if __name__ == "__main__":
    # Define the tableau:
    tableau = np.array([
        [1.0,1,1,1,1,0,0,15],
        [7,5,3,2,0,1,0,120],
        [3,5,10,15,0,0,1,100]
    ])
    # Define the objective function and the initial basis:
    of = np.array([4,5,9,11,0,0,0])
    basis = np.array([4,5,6])
    # Run the simplex algorithm:
    basis,tableau = simplex(of,basis,tableau)
    # Get the optimal soultion:
    optimal_solution,optimal_value = get_solution(of,basis,tableau)
    # Print the final tableau:
    print('The final basis is:')
    print(basis)
    # Print the results:
    print('Solution\nx1=%0.2f, x2=%0.2f, x3=%0.2f, x4=%0.2f, z=%0.4f' 
        % (optimal_solution[0],optimal_solution[1],optimal_solution[2],
            optimal_solution[3],optimal_value))
