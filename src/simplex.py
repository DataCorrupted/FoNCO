import numpy as np

# Simple function used to debug. 
# Used to break
def pause():
	a = input()

class Simplex:
	# Check if the input is legal. Don't worry about it if you
	# are not familiar with numpy's API.
	def inputCheck(self, c, A, b, basis):
		# Check if obj: cx, constrain Ax = b is of type np.
		if  type(c) != np.ndarray or \
			type(A) != np.ndarray or \
			type(b) != np.ndarray or \
			type(basis) != np.ndarray:
			raise ValueError("Init failed due to non-numpy input type. Abort.")
		n = A.shape[1]
		m = A.shape[0]
		# Check if the size matches.
		if n < m or \
			c.shape != (n,) or \
			b.shape != (m, 1) or \
			(not basis is None and basis.shape != (m,)):
			raise ValueError("Init failed due to mis-matched input size. Abort.")

	# Construct a new Simplex instance. 
	# input: 
	# 		c, A, b: the same as the standard simplex form. Using type ndarray.
	# 		basis: ndarray with size (m,), the init solution. If not given, it
	#			will be reset to np.arange(m-n, m), as the easiest solution 
	# 			usally falls there.
	def __init__(self, c, A, b, basis = None):
		self.inputCheck(c, A, b, basis)
		m, n = A.shape
		self.n, self.m = (n, m)
		self.c = c;
		self.tableau = np.concatenate((A, b), axis=1);
		if basis is None:
			self.basis = np.arange(m-n, m)
		else:
			self.basis = basis

	# Used to update c and A, as they will change with rho.
	def updateC(c):	self.c = c
	def updateA(A):	self.A = A

	# input: none
	# return: 
	# 		ndarray with size (1, n)(a vector) indicating z-c
	def zSubC(self):
		# Compute zj-cj. If zj - cj >= 0 for all columns then current 
		# solution is optimal solution.
		A = self.tableau[:, 0: self.n]
		cb = np.reshape(self.c[list(basis)], (self.m, 1))
		z = np.sum(cb * A, axis = 0)
		# Leave this for the sake of debug
		print(cb)
		print(A)
		print(cb * A)
		print(z)
		print(z - self.c)
		pause()
		return z - self.c

	# input: 
	# 		verbose: boolean, determing whether to print debug information.
	# return: none
	def solve(self, verbose = False):
		tableau = self.tableau
		# Start the simplex algorithm:
		check = self.zSubC()
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
			divide=(tableau[positive_rows, self.n]
				/tableau[positive_rows,pivot_col])
			pivot_row = positive_rows[np.where(divide 
				== divide.min())[0][-1]]
			# Update the basis:
			self.basis[pivot_row] = pivot_col
			# Perform gaussian elimination to make pivot element one and
			# elements above and below it zero:
			tableau[pivot_row,:]=(tableau[pivot_row,:]
				/tableau[pivot_row,pivot_col])
			for row in range(self.m):
				if row != pivot_row:
					tableau[row,:] = (tableau[row,:] 
						- tableau[row,pivot_col]*tableau[pivot_row,:])
			# Determine zj-cj
			check = self.zSubC();
			count += 1
			if verbose:
				print('Step %d' % count)
				print(tableau)


	def getSolution(self):
		# Re-solve it in case user changed something or forget to solve it.
		# In case the user has solved, the overhead will be 0 as solve() 
		# will return almost instantly.
		self.solve()
		# Get the no of columns in the tableau:
		n_cols = self.tableau.shape[1]
		# Get the optimal solution:
		solution = np.zeros(self.c.size)
		solution[list(self.basis)] = self.tableau[:,n_cols-1]
		# Determine the optimal value:
		value = np.sum(self.c[list(self.basis)] * self.tableau[:,n_cols-1])
		return solution,value

	# Return dual based on primal solution.
	def getDual():
		pass

if __name__ == "__main__":
	# Below defines some test problems.
	def Q1():
		# Check this example on pp.40
		# Define A, b:
		A = np.array([
			[-1,  2, 1, 0, 0],
			[ 2,  3, 0, 1, 0],
			[ 1, -1, 0, 0, 1]
		], dtype = np.float64)
		b = np.array([[4], [12], [3]], dtype = np.float64)
		# Define the objective function and the initial basis:
		c = np.array([-4, -1, 0, 0, 0])
		basis = np.array([2, 3, 4])
		return c, A, b, basis

	def Q2():
		# Define A, b:
		A = np.array([
			[1, 1, 1,  1,  1, 0, 0],
			[7, 5, 3,  2,  0, 1, 0],
			[3, 5, 10, 15, 0, 0, 1]
		], dtype = np.float64)
		b = np.array([[15], [120], [100]], dtype = np.float64)
		# Define the objective function and the initial basis:
		c = np.array([4, 5, 9, 11, 0, 0, 0])
		basis = np.array([4, 5, 6])
		return c, A, b, basis

	def Q3():
		# Define A, b:
		A = np.array([
			[ 1,  1, -2, 1, 0, 0],
			[ 2, -1,  4, 0, 1, 0],
			[-1,  2, -4, 0, 0, 1]
		], dtype = np.float64)
		b = np.array([[10], [8], [4]], dtype = np.float64)
		# Define the objective function and the initial basis:
		c = np.array([-1, 2, -1, 0, 0, 0])
		basis = np.array([3, 4, 5])
		return c, A, b, basis

	c, A, b, basis = Q1();
	# Run the simplex algorithm:
	s = Simplex(c, A, b, basis);
	s.solve()
	# Get the optimal soultion:
	optimal_solution,optimal_value = s.getSolution()
	# Print the results:
	# change print later.
	print('Solution:\nx1=%0.2f, x2=%0.2f, x3=%0.2f, x4=%0.2f\nz=%0.4f' 
		% (optimal_solution[0],optimal_solution[1],optimal_solution[2],
			optimal_solution[3],optimal_value))

