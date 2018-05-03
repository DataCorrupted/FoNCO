import numpy as np
import numpy.linalg
from debug_utils import pause
from scipy.optimize import linprog

np.set_printoptions(precision = 2, linewidth = 200)

class Simplex:

    # Construct a new Simplex instance. 
    # input: 
    #       c, A, b: the same as the standard simplex form. Using type ndarray.
    #       basis: ndarray with size (m,), the init solution. If not given, it
    #           will be reset to np.arange(m-n, m), as the correct solution 
    #           usally falls there. We DO NOT check if that matrix is invertable.
    def __init__(self, c, A, b, basis = None, iter_max = 100):
        self.iter_max_ = iter_max
        self.inputCheck_(c, A, b, basis)
        m, n = A.shape
        self.n_, self.m_ = (n, m)
        self.c_ = c;
        self.b_ = b;
        self.A_ = A;
        if basis is None:
            self.basis_ = np.arange(m-n, m)
        else:
            self.basis_ = basis
        b_inv = np.linalg.inv(self.A_[:, self.basis_])
        self.tableau_ = b_inv.dot(np.concatenate((A, b), axis=1));

    # Check if the input is legal. Don't worry about it if you
    # are not familiar with numpy's API.
    def inputCheck_(self, c, A, b, basis):
        # Check if primal: cx, constrain Ax = b is of type np.
        if  type(c) != np.ndarray or \
            type(A) != np.ndarray or \
            type(b) != np.ndarray or \
            type(basis) != np.ndarray:
            raise ValueError("Init failed due to non-numpy input type. Abort.")
        m = A.shape[0]
        n = A.shape[1]
        # Check if the size matches.
        if n < m or \
            c.shape != (n,) or \
            b.shape != (m, 1) or \
            (not basis is None and basis.shape != (m,)):
            raise ValueError("Init failed due to mis-matched input size. Abort.")

    def resetC(self, c):
        # If rho is updated, we need new C.
        if (self.c_.size == c.size):
            self.c_ = c;
        else:
            print("Reset object function failed due to type or size mis-match.")

    def getBInv_(self):
        try:
            return np.linalg.inv(self.A_[:, self.basis_])
        except:
#            print self.A_
#            print self.basis_
#            print self.A_[:, self.basis_]
            return np.eye(self.A_.shape[0])
            
    def getNuVar(self, c):
        cb = c[self.basis_]
        return cb.dot(self.getBInv_())

    # Return dual based on primal solution.
    def getDualVar(self):
        cb = np.reshape(self.c_[self.basis_], (1, -1))
        return cb.dot(self.getBInv_())
    # input: none
    # return: 
    #       ndarray with size (1, n)(a vector) indicating z-c
    def zSubC_(self):
        # Compute zj-cj. If zj - cj >= 0 for all columns then current 
        # solution is optimal solution.
        w = self.getDualVar()
        z = np.sum(w.dot(self.A_), axis = 0)
        # Leave this for the sake of debug
        return z - self.c_

    # input: none
    # return:
    #       boolean, determining if the problem is optimal now.
    def isOptimal(self):
        return np.all(self.zSubC_() - 1e-10 <= 0)

    def updateBasis(self):
        tableau = self.tableau_
        # Determine the pivot column:
        # The pivot column is the column corresponding to 
        # minimum zj-cj.
        
        # TODO: rethink where you should put this.
        pivot_col_idx = np.argmax(self.zSubC_())
        # Determine the positive elements in the pivot column.
        # If there are no positive elements in the pivot column  
        # then the optimal solution is unbounded.
        positive_rows = np.where(tableau[:,pivot_col_idx] > 0)[0]
        # We are solving problems that are almost always have bounds.
        # Let't not worry about it first.
        if positive_rows.size == 0: print('Unbounded Solution!')
        # Determine the pivot row:
        divide = \
            (tableau[positive_rows, self.n_] / tableau[positive_rows,pivot_col_idx])
        pivot_row_idx = \
            positive_rows[np.where(divide == divide.min())[0][0]]

        # Update the basis:
        self.basis_[pivot_row_idx] = pivot_col_idx
        # Perform gaussian elimination to make pivot element one and
        # elements above and below it zero:
        pivot, pivot_col, pivot_row = \
            self.getPivotOfTableau_(pivot_col_idx, pivot_row_idx)
        # No easy way to explain why the two lines below performs Gaussian elimination,
        # grasp a matrix, name a pivot and try your self.
        #
        # | 1  3  4  5|   | 3|                        |-5  0 -5  8|
        # | 2 *1* 3 -1| - | 1| / 1 * | 2  1  3  -1| = | 0  0  0  0|
        # |-3 -1  2  1|   |-1|                        |-1  0  5  0| 
        #
        # |-5  0 -5  8|   | 0  0  0  0|   |-5  0 -5  8|
        # | 0  0  0  0| + | 2  1  3 -1| = | 2  1  3 -1|
        # |-1  0  5  0|   | 0  0  0  0|   |-1  0  5  0|
        # 
        # I just made up the example above.
        tableau -= pivot_col.reshape((-1, 1)).dot(pivot_row.reshape((1, -1)))
        tableau[pivot_row_idx, :] += pivot_row
        
    # input: c, r. Int, specifying the position of a pivot.
    # return:
    #       pivot: double.
    #       pivot_row, pivot_col: ndarray with size (n,) (m,)
    def getPivotOfTableau_(self, c, r):
        pivot = self.tableau_[r,c]
        pivot_row = self.tableau_[r, :] / pivot
        pivot_col = self.tableau_[:, c]
        return pivot, pivot_col, pivot_row

    def getPrimalVar(self):
        # Get the optimal solution:
        primal_var = np.zeros(self.c_.size)
        b_inv = np.linalg.inv(self.tableau_[:, self.basis_])
        primal_var[self.basis_] = b_inv.dot(self.tableau_[:, self.n_])
        return primal_var

if __name__ == "__main__":

    from test_simplex import Q1, Q2, Q3, Q4, Q5, Q6

    def callBack(xk, **kwargs):
        if kwargs['phase'] == 2:
            basis = kwargs['basis']
            print(basis)
            print(A)
            print(kwargs['tableau'])
            b_inv = np.linalg.inv(A[:, basis])
            cb = np.reshape(c[basis], (1, -1))
            print(cb.dot(b_inv))
    def printCorrectAns(A, b, c):
        from scipy.optimize import linprog
        ans = linprog(c, A_eq = A, b_eq = b, method = 'simplex', callback = callBack)
        print(ans)
#Q3
    c, A, b, basis = Q3();
    printCorrectAns(A, b, c)
    linsov = Simplex(c, A, b, basis)
    pause(table = linsov.tableau_, zSubC = linsov.zSubC_(), basis = linsov.basis_, primal = linsov.getPrimal(), dual = linsov.getDual())
    while not linsov.isOptimal():
        linsov.updateBasis()
        pause(table = linsov.tableau_, zSubC = linsov.zSubC_(), basis = linsov.basis_, primal = linsov.getPrimal(), dual = linsov.getDual())

