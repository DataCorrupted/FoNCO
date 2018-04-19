import numpy as np
def standardize(A, b, g, delta, equatn):
	#										    | d(n,) |
	# | A(mxn) -I(mxm) I(mxm)               |   | r(m,) |   | -b(m,)     |
	# | I(nxn)                I(nxn)        | * | s(m,) | = |  delta(n,) |
	# |-I(nxn)                       I(nxn) |   | u(n,) |   |  delta(n,) |
	# 										    | u(n,) |
	#
	#                 A                             x     =       b
	#  
	# <c, x> = | rho*g, e(1*, |
 	m, n = A.shape
	return makeA_(A), makeBasis_(equatn), makeA_(b, delta, n), makeC_(g, equatn) 


def makeA_(A):
	m, n = A.shape
	A_ = np.zeros((m+2*n, 2*m+3*n))
	A_[0: m, 0: n] = A.copy(); 						# Row1:  A(mxn)
	A_[0: m, n: m+n] = -np.eye(m) 					# Row1: -I(mxm)
	A_[0: m, n+m: 2*m+n] = np.eye(m)				# Row1:  I(mxm)

	A_[m: m+n, 0: n] = np.eye(n)					# Row2:  I(nxn)
	A_[m: m+n, 2*m+n: 2*m+2*n] = np.eye(n) 			# Row2:  I(nxn)

	A_[m+n: m+2*n, 0: n] = -np.eye(n)				# Row3:  I(nxn)
	A_[m+n: m+2*n, 2*m+2*n: 2*m+3*n] = np.eye(n)	# Row3:  I(nxn)

	return A_

def makeB_(b, delta, n):
	m = b.shape[0]
	b_ = np.zeros((m+2*n, 1)) 			# -b
	b_[0:m, :] = -b.copy() 				# delta
	b_[m:m+2*n, 0] = delta 				# delta
	return b_

def makeC_(g, equatn):
	n = g.shape[0]
	m = equatn.shape[0]
	c_ = np.zeros((2*m+3*n, 1))

	c_[0:n, :] = g.copy()
	c_[n:m+n, :] = 1 					
	c_[m+n: 2*m+n, 0] = equatn.copy() 	# Not sure why
	# Other entries of c is 0.

	# Transpose as simplex solves prefers c with shape (1, -1)
	return c_.T

def makeBasis_(equatn):
	pass

if __name__ == "__main__":
	def testMakeA():
		A = np.array([
			[-1, -1, -1],
			[.03, -6, -4],
			[-1, 0, 0],
			[0, -1, 0],
			[0, 0, -1]
			])
		A_real = np.array([
			[-1, -1, -1,   -1, 0, 0, 0, 0,    1, 0, 0, 0, 0,    0, 0, 0,    0, 0, 0],
			[.03, -6, -4,   0,-1, 0, 0, 0,    0, 1, 0, 0, 0,    0, 0, 0,    0, 0, 0],
			[-1, 0, 0,      0, 0,-1, 0, 0,    0, 0, 1, 0, 0,    0, 0, 0,    0, 0, 0],
			[0, -1, 0,      0, 0, 0,-1, 0,    0, 0, 0, 1, 0,    0, 0, 0,    0, 0, 0],
			[0, 0, -1,      0, 0, 0, 0,-1,    0, 0, 0, 0, 1,    0, 0, 0,    0, 0, 0],
			#########################################################################
			[1, 0, 0,       0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    1, 0, 0,    0, 0, 0],
			[0, 1, 0,       0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 1, 0,    0, 0, 0],
			[0, 0, 1,       0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 1,    0, 0, 0],
			#########################################################################
			[-1, 0, 0,      0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0,    1, 0, 0],
			[0, -1, 0,      0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0,    0, 1, 0],
			[0, 0, -1,      0, 0, 0, 0, 0,    0, 0, 0, 0, 0,    0, 0, 0,    0, 0, 1],
			])
		A_calc = makeA_(A)
		assert np.equal(A_calc, A_real).all(), "Got wrong A!"
		print("Got correct A!")
	def testMakeB():
		b = np.array([
			[0],
			[0],
			[-.1],
			[-.7],
			[-.2],
			])
		b_real = np.array([
			[0],
			[0],
			[.1],
			[.7],
			[.2],
			[1],			
			[1],
			[1],
			[1],
			[1],
			[1],
			])
		b_calc = makeB_(b, 1, 3)
		assert np.equal(b_calc, b_real).all(), "Got wrong b!"
		print("Got correct b!")
	def testMakeC():
		g = np.array([
			[  0. ],
			[ 19.2],
			[  4.8]
			])
		equatn = np.array([True, False, False, False, False])
		c_real = np.array([0, 19.2, 4.8, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
		c_calc = makeC_(g, equatn)
		assert np.equal(c_calc, c_real).all(), "Got wrong c!"
		print("Got correct c!")
	testMakeA()
	testMakeB()
	testMakeC()