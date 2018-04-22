import numpy as np
import numpy.linalg
from scipy.optimize import linprog
from debug_utils import pause

class SimplexWrapper:

    # Construct a new Simplex instance. 
    # input: 
    #       c, A, b: the same as the standard simplex form. Using type ndarray.
    def __init__(self, c, A, b, g = None, rho = None, equatn = None, iter_max = 100):
        self.iter_max_ = iter_max

        self.inputCheck_(c, A, b)
        self.g_ = g
        self.rho_ = rho
        self.equatn_ = equatn
        self.c_ = c
        self.A_ = A
        self.b_ = b

    # Check if the input is legal. Don't worry about it if you
    # are not familiar with numpy's API.
    def inputCheck_(self, c, A, b):
        # Check if obj: cx, constrain Ax = b is of type np.
        if  type(c) != np.ndarray or \
            type(A) != np.ndarray or \
            type(b) != np.ndarray:
            raise ValueError("Init failed due to non-numpy input type. Abort.")
        m = A.shape[0]
        n = A.shape[1]
        if n < m or \
            c.size != n or b.size != m:
            raise ValueError("Init failed due to mis-matched input size. Abort.")

    def resetC(self, rho):
        if self.equatn_ is None or self.g_ is None:
            raise ValueError("You can't reset c without valid equatn and g")
        self.rho = rho
        self.c_ = makeC_(self.g_*self.rho, self.equatn)

    def simplexCallback_(self, d, **kwargs):
        self.basis_ = kwargs['basis']
        if kwargs['phase'] == 1:
            # We don't care about phase 1.
            return
        self.iter_cnt_ += 1;
        
        self.dual_var_ = self.calcDual()

        self.primal_var_ = np.zeros(self.c_.size)
        # The last row of the table is z-c part.
        self.primal_var_[self.basis_] = kwargs['tableau'][:-1, -1]
        
        self.primal_obj_ = self.c_.dot(self.primal_var_)
        self.calcDual()


        # dual_var_, d_k, _, _, rho, ratio_complementary, 
        # ratio_opt, ratio_fea, sub_iter, H_rho \

    def solve(self):
        self.iter_cnt_ = 0;
        ans = linprog(
            self.c_, A_eq = self.A_, b_eq = self.b_, 
            method = 'simplex', callback = self.simplexCallback_)
        self.primal_var_ = ans.x
        self.primal_obj_ = ans.fun
        self.calcDual()
        return ans

    def calcDual(self): 
        b_inv = np.linalg.inv(self.A_[:, self.basis_])
        self.dual_var_ = -self.c_[self.basis_].dot(b_inv)
        self.dual_obj_ = -self.dual_var_.dot(self.b_)[0]

    def getIterCnt(self): return self.iter_cnt_
    def getDualObj(self): return self.dual_obj_
    def getDualVar(self): return self.dual_var_
    def getPrimalObj(self): return self.primal_obj_
    def getPrimalVar(self): return self.primal_var_

def standardize(g, rho, A, b, delta, equatn):
    #                                                   | d+(n,) |
    #                                                   | d-(n,) |
    # | A(mxn) -A(mxn) -I(mxm) I(mxm)               |   | r(m,)  |   | -b(m,)     |
    # | I(nxn) -I(nxn)                I(nxn)        | * | s(m,)  | = |  delta(n,) |
    # |-I(nxn)  I(nxn)                       I(nxn) |   | u(n,)  |   |  delta(n,) |
    #                                                   | u(n,)  |
    #
    #                 A                             x     =       b
    #  
    # c = | rho*g, -rho*g, e(1, m), e(1, #eq) |
    m, n = A.shape
    return makeC_(g, equatn), makeA_(A), makeB_(b, delta, n), makeBasis_(b, n)


def makeA_(A):
    m, n = A.shape
    A_ = np.zeros((m+2*n, 2*m+4*n))
    A_[0: m, 0: n] = A.copy();                      # Row1:  A(mxn)
    A_[0: m, n: 2*n] = -A.copy();                   # Row1: -A(mxn)
    A_[0: m, 2*n: m+2*n] = -np.eye(m)               # Row1: -I(mxm)
    A_[0: m, m+2*n: 2*m+2*n] = np.eye(m)            # Row1:  I(mxm)

    A_[m: m+n, 0: n] = np.eye(n)                    # Row2:  I(nxn)
    A_[m: m+n, n: 2*n] = -np.eye(n)                 # Row2: -I(nxn)
    A_[m: m+n, 2*m+2*n: 2*m+3*n] = np.eye(n)        # Row2:  I(nxn)

    A_[m+n: m+2*n, 0: n] = -np.eye(n)               # Row3:  I(nxn)
    A_[m+n: m+2*n, n: 2*n] = np.eye(n)              # Row3:  I(nxn)
    A_[m+n: m+2*n, 2*m+3*n: 2*m+4*n] = np.eye(n)    # Row3:  I(nxn)

    return A_

def makeB_(b, delta, n):
    m = b.shape[0]
    b_ = np.zeros((m+2*n, 1))           
    b_[0:m, :] = -b.copy()              # -b
    b_[m:m+2*n, 0] = delta              # delta
    return b_

def makeC_(g, equatn):
    n = g.shape[0]
    m = equatn.shape[0]
    c_ = np.zeros((2*m+4*n, 1))

    c_[0:n, :] = g.copy()
    c_[n:2*n, :] = -g.copy()
    c_[2*n:m+2*n, :] = 1                    
    c_[m+2*n: 2*m+2*n, 0] = equatn.copy()   # Not sure why
    # Other entries of c is 0.

    # Transpose as simplex solves prefers c with shape (1, -1)
    return c_.reshape((2*m+4*n,))

def makeBasis_(b, n):
    m = b.shape[0]
    basis = np.concatenate(                                         \
        (np.arange(2*n, m+2*n),np.arange(2*m+2*n, 2*m+4*n)),                \
        axis = 0                                                    \
    )
    for i in range(m):
        basis[i] += (b[i] <=0) *m
    return basis

if __name__ == "__main__":
    def testMakeA():
        A = np.array([
            [1, -2],
            [1,  4]         
            ])
        A_real = np.array([
            [ 1, -2, -1,  2,   -1,  0,    1, 0,    0, 0,    0, 0],
            [ 1,  4, -1, -4,    0, -1,    0, 1,    0, 0,    0, 0],
            ##########################################################
            [ 1,  0, -1,  0,    0,  0,    0, 0,    1, 0,    0, 0],
            [ 0,  1,  0, -1,    0,  0,    0, 0,    0, 1,    0, 0],
            ##########################################################
            [-1,  0,  1,  0,    0,  0,    0, 0,    0, 0,    1, 0],
            [ 0, -1,  0,  1,    0,  0,    0, 0,    0, 0,    0, 1],
            ])
        A_calc = makeA_(A)
        assert np.equal(A_calc, A_real).all(), "Got wrong A!"
        print("Got correct A!")
    def testMakeB():
        b = np.array([
            [-1],
            [4],
            ])
        b_real = np.array([
            [1],
            [-4],
            #######
            [1],
            [1],
            [1],
            [1]         
            ])
        b_calc = makeB_(b, 1, 2)
        assert np.equal(b_calc, b_real).all(), "Got wrong b!"
        print("Got correct b!")
    def testMakeC():
        g = np.array([
            [ 0],
            [ 2]
            ])
        equatn = np.array([True, False])
        c_real = np.array([0, 2, 0, -2, 1, 1, 1, 0, 0, 0, 0, 0])
        c_calc = makeC_(g, equatn)
        assert np.equal(c_calc, c_real).all(), "Got wrong c!"
        print("Got correct c!")

    def testMakeBasis():
        b = np.array([
            [-1],
            [ 4],
            ])
        basis_real = np.array([6, 5,    8, 9, 10, 11])
        basis_calc = makeBasis_(b, 2)
        assert np.equal(basis_calc, basis_real).all(), "Got wrong basis!"
        print("Got correct basis!")
    print("Running unit tests for matrix maker...")
    testMakeA()
    testMakeB()
    testMakeC()
    testMakeBasis()

    print("Running unit tests for solver...")
    from test_simplex import Q5 as Q
    c, A, b, _ = Q()
    sw = SimplexWrapper(c, A, b)
    print(sw.solve())
    
    print(sw.getPrimalVar())
    print(sw.getPrimalObj())
    print(sw.getDualObj())
    print(sw.getDualVar())
    
