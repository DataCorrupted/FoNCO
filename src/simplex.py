import numpy as np
import numpy.linalg

# Simple function used to debug. 
# input:
#		anything. It will print out anything you give.
def pause(*args, **kwargs):
	for item in args:
		print(item)
	for (key, item) in kwargs.items():
		print(key + ":", item)
	# Press Enter to continue.
	a = input()

class Simplex:

	# Construct a new Simplex instance. 
	# input: 
	# 		c, A, b: the same as the standard simplex form. Using type ndarray.
	# 		basis: ndarray with size (m,), the init solution. If not given, it
	#			will be reset to np.arange(m-n, m), as the correct solution 
	# 			usally falls there. We DO NOT check if that matrix is invertable.
	def __init__(self, c, A, b, basis = None, iter_max = 100):
		self.iter_max_ = iter_max
		self.resetProblem(c, A, b, basis)

	# Check if the input is legal. Don't worry about it if you
	# are not familiar with numpy's API.
	def inputCheck_(self, c, A, b, basis):
		# Check if obj: cx, constrain Ax = b is of type np.
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

	# Used to update c and A, as they will change with rho.
	# See the input param in __init__
	def resetProblem(self, c, A, b, basis = None):
		self.inputCheck_(c, A, b, basis)
		m, n = A.shape
		self.n_, self.m_ = (n, m)
		self.c_ = c;
		self.tableau_ = np.concatenate((A, b), axis=1);
		if basis is None:
			self.basis_ = np.arange(m-n, m)
		else:
			self.basis_ = basis

	# Changes the object function without changing anything else.
	# You can always do this as the tableau won't change except
	# the last row(z-c).
	# input: 
	# 		c: ndarray with size (n,)
	def resetC(c):
		if type(c) == np.ndarray and c.shape == (self.n_,):
			self.c_ = c;
		else:
			print("Reset object function failed due to type or size mis-match.")

	# Return dual based on primal solution.
	def getDual(self):
		cb = np.reshape(self.c_[self.basis_], (1, -1))
		b_inv = np.linalg.inv(self.tableau_[:, self.basis_])
		return cb.dot(b_inv)

	# input: none
	# return: 
	# 		ndarray with size (1, n)(a vector) indicating z-c
	def zSubC_(self):
		# Compute zj-cj. If zj - cj >= 0 for all columns then current 
		# solution is optimal solution.
		A = self.tableau_[:, 0: self.n_]
		w = self.getDual()
		z = np.sum(w.dot(A), axis = 0)
		# Leave this for the sake of debug
		return z - self.c_

	# input: none
	# return:
	# 		boolean, determining if the problem is optimal now.
	def isOptimal(self):
		return np.all(self.zSubC_() <= 0)

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
		if positive_rows.size == 0:	print('Unbounded Solution!')
		# Determine the pivot row:
		divide = \
			(tableau[positive_rows, self.n_] / tableau[positive_rows,pivot_col_idx])
		pivot_row_idx = \
			positive_rows[np.where(divide == divide.min())[0][-1]]
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

	# input: 
	# 		verbose: boolean, determing whether to print debug information.
	# return: none
	def solve(self, verbose = False):
		print('Original problem: \n', self.tableau_, '\n', self.c_)
		pause()
		# Start the simplex algorithm:
		iter_cnt = 0
		while not self.isOptimal() and iter_cnt < self.iter_max_:
			self.updateBasis()
			if verbose:
				print('Step %d' % iter_cnt)
				print(self.tableau_)
				print(self.getStatus())
				print(self.isOptimal())
				pause()
			iter_cnt += 1

	# input: c, r. Int, specifying the position of a pivot.
	# return:
	# 		pivot: double.
	# 		pivot_row, pivot_col: ndarray with size (n,) (m,)
	def getPivotOfTableau_(self, c, r):
		pivot = self.tableau_[r,c]
		pivot_row = self.tableau_[r, :] / pivot
		pivot_col = self.tableau_[:, c]
		return pivot, pivot_col, pivot_row

	def getSolution(self):
		# Re-solve it in case user changed something or forget to solve it.
		# In case the user has solved, the overhead will be 0 as solve() 
		# will return almost instantly.
		self.solve()
		return self.getStatus()

	def getStatus(self):
		# Get the optimal solution:
		x = np.zeros(self.c_.size)
		x[self.basis_] = self.tableau_[:,self.n_]
		# Determine the optimal value:
		obj = np.sum(self.c_[self.basis_] * self.tableau_[:,self.n_])
		return x, obj

if __name__ == "__main__":
	# Below defines some test problems.
	def Q1():
		# Check this example on pp.40
		# Define A, b:
		A = np.array([
			[ 1, -2, -1, 0, 0],
			[ 2,  3,  0, 1, 0],
			[ 1, -1,  0, 0, 1]
		], dtype = np.float64)
		b = np.array([[-4], [12], [3]], dtype = np.float64)
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
		# Check this example on pp.136
		# Define A, b:
		A = np.array([
			[2, 1, 4, 0, -1,  0],
			[2, 2, 0, 4,  0, -1]
		], dtype = np.float64)
		b = np.array([[2], [3]], dtype = np.float64)
		# Define the objective function and the initial basis:
		c = np.array([12, 8, 16, 12, 0, 0])
		basis = np.array([4, 5])
		return c, A, b, basis

	c, A, b, basis = Q2();
	# Run the simplex algorithm:
	s = Simplex(c, A, b, basis);
	s.solve(verbose = True)
	# Get the optimal soultion:
	while not s.isOptimal():
		optimal_solution,optimal_value = s.getStatus()
		# Print the results:
		# change print later.
		print('Solution:\nx1=%0.2f, x2=%0.2f, x3=%0.2f, x4=%0.2f\nz=%0.4f' 
			% (optimal_solution[0],optimal_solution[1],optimal_solution[2],
				optimal_solution[3],optimal_value))
		pause()
		s.updateBasis()

