import numpy as np
import numpy.linalg
from debug_utils import pause

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
    return makeC(g*rho, equatn), makeA(A), makeB(b, delta, n), makeBasis(b, n)


def makeA(A):
    #
    #     | A(mxn) -A(mxn) -I(mxm) I(mxm)               |   
    # A = | I(nxn) -I(nxn)                I(nxn)        |
    #     |-I(nxn)  I(nxn)                       I(nxn) |   
    #                                                
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

def makeB(b, delta, n):
    #     | -b(m,)     |
    # b = |  delta(n,) |
    #     |  delta(n,) |
    m = b.shape[0]
    b_ = np.zeros((m+2*n, 1))           
    b_[0:m, :] = -b.copy()              # -b
    b_[m:m+2*n, 0] = delta              # delta
    return b_

def makeC(g, equatn):
    # c = | rho*g, -rho*g, e(1, m), e(1, #eq) |
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

def makeBasis(b, n):
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
        A_calc = makeA(A)
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
        b_calc = makeB(b, 1, 2)
        assert np.equal(b_calc, b_real).all(), "Got wrong b!"
        print("Got correct b!")
    def testMakeC():
        g = np.array([
            [ 0],
            [ 2]
            ])
        equatn = np.array([True, False])
        c_real = np.array([0, 2, 0, -2, 1, 1, 1, 0, 0, 0, 0, 0])
        c_calc = makeC(g, equatn)
        assert np.equal(c_calc, c_real).all(), "Got wrong c!"
        print("Got correct c!")

    def testMakeBasis():
        b = np.array([
            [-1],
            [ 4],
            ])
        basis_real = np.array([6, 5,    8, 9, 10, 11])
        basis_calc = makeBasis(b, 2)
        assert np.equal(basis_calc, basis_real).all(), "Got wrong basis!"
        print("Got correct basis!")
    print("Running unit tests for matrix maker...")
    testMakeA()
    testMakeB()
    testMakeC()
    testMakeBasis()
