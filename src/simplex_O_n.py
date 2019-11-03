import numpy as np
import numpy.linalg
from debug_utils import pause
from scipy.optimize import linprog

np.set_printoptions(precision = 2, linewidth = 200)

class SimplexON:
    # Construct a new Simplex instance.
    # input:
    #       c, A, b: the same as the standard simplex form. Using type ndarray.
    #       basis: ndarray with size (m,), the init solution. If not given, it
    #           will be reset to np.arange(m-n, m), as the correct solution
    #           usally falls there. We DO NOT check if that matrix is invertable.
    def __init__(self, c, A, b, basis=None, iter_max=100):
        self.name = "SimplexON"
        self.intro = "Simplex with all dual/prime in the table. The table is huge."
        self.iter_max_ = iter_max
        self.inputCheck_(c, A, b, basis)
        m, n = A.shape
        self.n_, self.m_ = (n, m)
        self.c_ = c
        self.b_ = b
        self.A_ = A
        if basis is None:
            self.basis_ = np.arange(m-n, m)
        else:
            self.basis_ = basis
        self.tableau_ =  np.zeros((m+1, n+m+1))

        # line/col idx      --- |      0 ~ n-1             |    n           |    n+1 ~ n+m
        #                 |  f  |  xB  |  xN               |    RHS         |    Dual
        #   0 ~ m-1   xB  |  0  |  I   |  b_inv.N          |    b_inv.b     |    I
        #     m       f   |  1  |  0   |  cB.b_inv.N - cN  |    cB.b_inv.b  |    dual_var
        #
        # Table A, (xB, f) and (f, f) is not in the code.

        b_inv = b_inv = np.linalg.inv(self.A_[:, self.basis_])
        cB = np.reshape(self.c_[self.basis_], (1, -1))
        dual_var = cB.dot(b_inv)

        # (xB, xB) and (xB, xN) in Table A
        self.tableau_[0:m, 0:n] = b_inv.dot(self.A_)
        # (xB, RHS)
        self.tableau_[0:m, n] = b_inv.dot(self.b_).reshape(m)
        # (xB, Dual)
        self.tableau_[0:m, n+1:n+m+1] = np.eye(m, m) 

        # (f, xB) and (f, xN)
        self.tableau_[m, 0:n] = dual_var.dot(self.A_) - self.c_
        # (f, RHS)
        self.tableau_[m, n] = dual_var.dot(self.b_)
        # (f, Dual)
        self.tableau_[m, n+1:n+m+1] = dual_var

    # Check if the input is legal. Don't worry about it if you
    # are not familiar with numpy's API.
    def inputCheck_(self, c, A, b, basis):
        # Check if primal: cx, constrain Ax = b is of type np.
        if type(c) != np.ndarray or \
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
            raise ValueError(
                "Init failed due to mis-matched input size. Abort.")

    def resetC(self, c):
        # If rho is updated, we need new C.
        if (self.c_.size == c.size):
            self.c_ = c
        else:
            print("Reset object function failed due to type or size mis-match.")

    def getNuVar(self, c):
        # In paper we said NuVar is at least DualVar, 
        # so we use DualVal as a substitute.
        return self.getDualVar()

    # return (f, Dual)
    def getDualVar(self):
        return self.tableau_[self.m_, self.n_+1:self.n_+self.m_+1].reshape(1, -1)

    # return (xB, RHS)
    def getPrimalVar(self):
        primal_var = np.zeros(self.n_)
        primal_var[self.basis_] = self.tableau_[0:self.m_, self.n_]
        return primal_var

    # return (f, xB) and (f, xN)
    def zSubC_(self):
        return self.tableau_[self.m_, 0:self.n_]

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
        positive_rows = np.where(tableau[0:self.m_, pivot_col_idx] > 0)[0]
        # We are solving problems that are almost always have bounds.
        # Let't not worry about it first.
        if positive_rows.size == 0:
            return
        # Determine the pivot row:
        divide = \
            (tableau[positive_rows, self.n_] /
             tableau[positive_rows, pivot_col_idx])
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
        pivot = self.tableau_[r, c]
        pivot_row = self.tableau_[r, :] / pivot
        pivot_col = self.tableau_[:, c]
        return pivot, pivot_col, pivot_row
    def getMainTable_(self):
        return self.tableau_[0:self.m_, 0:self.n_+1]

def unimplemented():
    pass

if __name__ == "__main__":

    from test_simplex import Q1, Q2, Q3, Q4, Q5, Q6
    from simplex import Simplex
    import sys

    def prRed(skk): print("\033[91m{}\033[00m".format(skk)) 
    def prCyan(skk): print("\033[96m{}\033[00m".format(skk))

    def array_cmp(a, b, name, verbose):
        ret = True

        if not np.all(np.abs(a - b) < 10**-5):
            ret = False
            if verbose >= 0:
                prRed("Error: disagree on " + name)
                print "SimplexON got: "
                print a
                print "Simplex got: "
                print b
                print
        return ret

    def test_q(q, verbose):
        c, A, b, basis = q()
        linsov = Simplex(c, A, b, basis)
        linsovON = SimplexON(c, A, b, basis)

        ret = True
        if verbose >= 2:
            pause(name=linsov.name, table=linsov.tableau_, zSubC=linsov.zSubC_(
            ), basis=linsov.basis_, getPrimalVar=linsov.getPrimalVar(), getDualVar=linsov.getDualVar())
            pause(name=linsovON.name, table=linsovON.tableau_, zSubC=linsovON.zSubC_(
            ), basis=linsovON.basis_, getPrimalVar=linsovON.getPrimalVar(), getDualVar=linsovON.getDualVar())

        while True:
            if (linsov.isOptimal() != linsovON.isOptimal()):
                prRed("Error: disagree on isOptimal()")
                ret = False 
                break
            else:
                if linsov.isOptimal():
                    break
            linsov.updateBasis()
            linsovON.updateBasis()

            if verbose >= 2:
                pause(name=linsov.name, table=linsov.tableau_, zSubC=linsov.zSubC_(
                ), basis=linsov.basis_, getPrimalVar=linsov.getPrimalVar(), getDualVar=linsov.getDualVar())
                pause(name=linsovON.name, table=linsovON.tableau_, zSubC=linsovON.zSubC_(
                ), basis=linsovON.basis_, getPrimalVar=linsovON.getPrimalVar(), getDualVar=linsovON.getDualVar())

            ret &= array_cmp(linsovON.zSubC_(), linsov.zSubC_(),"zSubC", verbose)
            ret &= array_cmp(linsovON.basis_, linsov.basis_, "basis", verbose)
            ret &= array_cmp(linsovON.getPrimalVar(), linsov.getPrimalVar(), "getPrimalVar", verbose)
            ret &= array_cmp(linsovON.getDualVar(), linsov.getDualVar(), "getDualVar", verbose)
            ret &= array_cmp(linsovON.getMainTable_(), linsov.getMainTable_(), "getMainTable", verbose)

        if not ret and verbose == 1:
            prRed("Taking you through this problem again step by step...\n")
            test_q(q, 2)
        return ret

    def main():
        if "-h" in sys.argv or "--help" in sys.argv:
            print "This is the help message for SimplexON"
            print "run \"python simplex_O_n.py <option>\" to do a testing on this implementation"
            print "Options:"
            print "  -h, --help             Show this help information"
            print "  -v, --verbose          Use default verbose level(0)"
            print "  -v<l>, --verbose=<l>   Set verbose level to <l>"
            print "                         Level 0: Only show what's different"
            print "                         Level 1: Step you through the problem that went wrong."
            print "                         Level 2: Shows everything and you need to manually step through the solving"
            print "                         Leaving out <l> sets it to default(1)"
            return

        verbose = -1
        if "-v" in sys.argv or "--verbose" in sys.argv:
            verbose = 1
        for i in range(3):
            if "-v"+str(i) in sys.argv or "--verbose="+str(i) in sys.argv:
                verbose = i
            
        if verbose >= 1:
            print ""
            pause(info = "From now on you need to press Enter to proceed")

        problems = [Q1, Q2, Q3, Q4, Q5, Q6]
        for i in range(6):
            prCyan("\nTesting on problem {}...".format(i))
            if test_q(problems[i], verbose):
                prCyan("\nPassed!")
            else:
                prRed("\nFailed!")

    main()
