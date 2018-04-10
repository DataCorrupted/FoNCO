import numpy as np

class Simplex:
	def inputCheck(self, c, A, b, basis):
		if  type(c) != np.ndarray or \
			type(A) != np.ndarray or \
			type(b) != np.ndarray or \
			type(basis) != np.ndarray:
			raise ValueError("Init failed due to non-numpy input type. Abort.")
		n = A.shape[1]
		m = A.shape[0]
		if n < m or \
			c.shape != (n,) or \
			b.shape != (m, 1) or \
			(not basis is None and basis.shape != (m,)):
			raise ValueError("Init failed due to mis-matched input size. Abort.")

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

	def updateC(c):	self.c = c
	def updateA(A):	self.A = A

	def zSubC(self):
		A = self.tableau[:, 0: self.n]
		cb = np.reshape(self.c[list(basis)], (self.m, 1))
		z = np.sum(cb * A, axis = 0)
		# Leave this for the sake of debug
		#print(cb)
		#print(A)
		#print(cb * A)
		#print(z)
		#print(z - self.c)
		#pause()
		return z - self.c

	def solve(self):
		tableau = self.tableau
		basis = self.basis
		# Get the number of rows and columns in the tableau:
		n_rows = tableau.shape[0]
		n_cols = tableau.shape[1]
		# Start the simplex algorithm:
		# Compute zj-cj. If zj - cj >= 0 for all columns then current 
		# solution is optimal solution.
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
			check = self.zSubC();
			count += 1
			print('Step %d' % count)
			print(tableau)

	def getSolution(self):
		self.solve()
		# Get the no of columns in the tableau:
		n_cols = self.tableau.shape[1]
		# Get the optimal solution:
		solution = np.zeros(self.c.size)
		solution[list(self.basis)] = self.tableau[:,n_cols-1]
		# Determine the optimal value:
		value = np.sum(self.c[list(self.basis)] * self.tableau[:,n_cols-1])
		return solution,value

def pause():
	a = input()

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
		c = np.array([4, 1, 0, 0, 0])
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

	c, A, b, basis = Q3();
	# Run the simplex algorithm:
	s = Simplex(c, A, b, basis);
	s.solve()
	# Get the optimal soultion:
	optimal_solution,optimal_value = s.getSolution()
	# Print the final tableau:
	print('The final basis is:')
	# Print the results:
	print('Solution\nx1=%0.2f, x2=%0.2f, x3=%0.2f, x4=%0.2f, z=%0.4f' 
		% (optimal_solution[0],optimal_solution[1],optimal_solution[2],
			optimal_solution[3],optimal_value))

