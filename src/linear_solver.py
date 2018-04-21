import numpy as np
import numpy.linalg
from scipy.optimize import linprog


class SimplexWrapper:

	# Construct a new Simplex instance. 
	# input: 
	#       c, A, b: the same as the standard simplex form. Using type ndarray.
	def __init__(self, c, A, b, iter_max = 100):
		self.iter_max_ = iter_max
		self.resetProblem(c, A, b)

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
		# Check if the size matches.
		if n < m or \
			c.shape != (n,) or b.shape != (m, 1):
			raise ValueError("Init failed due to mis-matched input size. Abort.")

	# Used to update c and A, as they will change with rho.
	# See the input param in __init__
	def resetProblem(self, c, A, b):
		self.inputCheck_(c, A, b)
		self.c_ = c;
		self.A_ = A;
		self.b_ = b;

	# Changes the object function without changing anything else.
	# You can always do this as the tableau won't change except
	# the last row(z-c).
	# input: 
	#       c: ndarray with size (n,)
	def resetC(self, c):
		if type(c) == np.ndarray and c.shape == (self.n_,):
			self.c_ = c;
		else:
			print("Reset object function failed due to type or size mis-match.")

	def simplexCallback_(self, d, **kwargs):
		if kwargs['phase'] == 2:
			self.iter_cnt_ += 1;
			self.dual_ = cb.dot(b_inv)
			self.primal_ = np.zeros(self.c_.size)
			self.primal_[basis] = kwargs['tableau'][:, -1]
			self.obj_ = self.c_.dot(self.primal_)


	def solve(self):
		self.iter_cnt_ = 0;
		print linprog(
			self.c_, A_eq = self.A_, b_eq = self.b_, 
			method = 'simplex', callback = self.simplexCallback_)

	def getObj(self): return self.obj_
	def getDual(self): return self.dual_
	def getPrimal(self): return self.primal_
	def getIterCnt(self): return self.iter_cnt_

def standardize(A, b, g, delta, equatn):
	#										            | d+(n,) |
	# 											        | d-(n,) |
	# | A(mxn) -A(mxn) -I(mxm) I(mxm)               |   | r(m,)  |   | -b(m,)     |
	# | I(nxn) -I(nxn)                I(nxn)        | * | s(m,)  | = |  delta(n,) |
	# |-I(nxn)  I(nxn)                       I(nxn) |   | u(n,)  |   |  delta(n,) |
	# 										            | u(n,)  |
	#
	#                 A                             x     =       b
	#  
	# c = | rho*g, -rho*g, e(1, m), e(1, #eq) |
	m, n = A.shape
	return makeA_(A), makeBasis_(b, n), makeB_(b, delta, n), makeC_(g, equatn) 


def makeA_(A):
	m, n = A.shape
	A_ = np.zeros((m+2*n, 2*m+4*n))
	A_[0: m, 0: n] = A.copy(); 						# Row1:  A(mxn)
	A_[0: m, n: 2*n] = -A.copy(); 					# Row1: -A(mxn)
	A_[0: m, 2*n: m+2*n] = -np.eye(m) 				# Row1: -I(mxm)
	A_[0: m, m+2*n: 2*m+2*n] = np.eye(m)			# Row1:  I(mxm)

	A_[m: m+n, 0: n] = np.eye(n)					# Row2:  I(nxn)
	A_[m: m+n, n: 2*n] = -np.eye(n) 				# Row2: -I(nxn)
	A_[m: m+n, 2*m+2*n: 2*m+3*n] = np.eye(n) 		# Row2:  I(nxn)

	A_[m+n: m+2*n, 0: n] = -np.eye(n)				# Row3:  I(nxn)
	A_[m+n: m+2*n, n: 2*n] = np.eye(n) 				# Row3:  I(nxn)
	A_[m+n: m+2*n, 2*m+3*n: 2*m+4*n] = np.eye(n)	# Row3:  I(nxn)

	return A_

def makeB_(b, delta, n):
	m = b.shape[0]
	b_ = np.zeros((m+2*n, 1)) 			
	b_[0:m, :] = -b.copy() 				# -b
	b_[m:m+2*n, 0] = delta 				# delta
	return b_

def makeC_(g, equatn):
	n = g.shape[0]
	m = equatn.shape[0]
	c_ = np.zeros((2*m+4*n, 1))

	c_[0:n, :] = g.copy()
	c_[n:2*n, :] = -g.copy()
	c_[2*n:m+2*n, :] = 1 					
	c_[m+2*n: 2*m+2*n, 0] = equatn.copy() 	# Not sure why
	# Other entries of c is 0.

	# Transpose as simplex solves prefers c with shape (1, -1)
	return c_.reshape((2*m+4*n,))

def makeBasis_(b, n):
	m = b.shape[0]
	basis = np.concatenate(											\
		(np.arange(2*n, m+2*n),np.arange(2*m+2*n, 2*m+4*n)), 				\
		axis = 0													\
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
	