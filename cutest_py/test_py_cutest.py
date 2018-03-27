#!/usr/bin/python
import os, sys
from ctypes import *
#from v4_cord import *
import numpy as np
import scipy.sparse as sp

PtD = POINTER(c_double)
PtI = POINTER(c_int)
PtB = POINTER(c_bool)


def func(x, *args):
	handle, nvar, ncon, status  = args[:4]
	grad = PtB(c_bool(0))
	f = PtD(c_double(0))
	handle.CUTEST_cofg_1(status, nvar, x.ctypes.data_as(PtD), \
						 f, PtD(c_double(0)), grad)

	return f[0]


def fprime(x, *args):
	handle, nvar, ncon, status  = args[:4]
	grad = POINTER(c_bool)(c_bool(1))
	f = PtD(c_double(0))
	g = np.zeros((nvar[0], 1))
	handle.CUTEST_cofg_1(status, nvar, x.ctypes.data_as(PtD), \
			             f, g.ctypes.data_as(PtD), grad)

	return g 

def fconstr(x, *args):
	handle, nvar, ncon, status, lj, cjac, indvar, indfun  = args
	grad = POINTER(c_bool)(c_bool(0))
	c = np.zeros((ncon[0], 1))
	nnx = PtI(c_int(0))

	handle.CUTEST_ccfsg_1( status,  nvar, ncon, x.ctypes.data_as(PtD),\
			  c.ctypes.data_as(PtD), nnx, PtI(c_int(nnzj[0])), \
		      PtD(c_double(0)), PtI(c_int(0)), PtI(c_int(0)), grad)

	return c


def Jaconstr(x, *args):
	handle, nvar, ncon, status, lj, cjac, indvar, indfun  = args
	grad = POINTER(c_bool)(c_bool(1))
	c = np.zeros((ncon[0], 1))

	nnx = PtI(c_int(0))

	handle.CUTEST_ccfsg_1( status,  nvar, ncon, x.ctypes.data_as(PtD),\
			c.ctypes.data_as(PtD), nnx, lj, \
		      cjac.ctypes.data_as(PtD), \
			  indvar.ctypes.data_as(PtI), \
			  indfun.ctypes.data_as(PtI), grad)
	
	const = nnx[0]
	row = indfun[:const] - 1
	row = row.flatten()
	col = indvar[:const] - 1
	col = col.flatten()

	A = sp.coo_matrix((cjac[:const].flatten(), (row, col)), shape = (ncon[0], nvar[0]))

	return A.tocsr()


def solve(sifDir, beta_fea = 0.5, beta_opt = 0.7, subMaxIter = 50, \
		maxIter = 100, rho = 1.0, sigma = 10, gamma = 0.1):
	'''
	Solve the optimization problem in sifDir
	The input file is the sifdecoded OUTSDIF.d 
	and the compiled py_cute.so

	Parameters for nonlinear solvers
	beta_fea = 0.5
	beta_opt = 0.7
	subMaxIter = 50
	maxIter = 100 
	rho = 1.0
	sigma = 10 
	gamma = 0.1
	'''
	z1 = CDLL(sifDir+'py_cute.so')

	
	status = PtI(c_int(0))
	funit = PtI(c_int(42))
	fname = c_char_p(sifDir+'OUTSDIF.d')
	iout = PtI(c_int(6))
	io_buffer = PtI(c_int(11))
	e_order = PtI(c_int(1))
	l_order = PtI(c_int(0))
	v_order = PtI(c_int(0))
	n = PtI(c_int(0))
	m = PtI(c_int(0))
	
	#Open the file and get the dimension of the problem
	z1._open(fname, funit)
	z1.CUTEST_cdimen_1(status, funit, n, m)

	#if m[0] == 0:
	#	return 'Unconstrained problem'

	x = np.zeros((n[0], 1))
	bl = np.zeros((n[0], 1))
	bu = np.zeros((n[0], 1))
	v = np.zeros((m[0], 1))
	cl = np.zeros((m[0], 1))
	cu = np.zeros((m[0], 1))
	equatn = np.array([0]*m[0], dtype=np.bool)
	linear = np.array([0]*m[0], dtype=np.bool)
	x1 = np.random.randn(n[0], 1)
	f = PtD(c_double(0))

	#Number of equations
	nequation = np.sum(equatn)
	
	#Setup the problem
	z1.CUTEST_csetup_1(status, funit, iout, io_buffer, n, m, \
		               x.ctypes.data_as(PtD), \
					   bl.ctypes.data_as(PtD), \
					   bu.ctypes.data_as(PtD), \
					   v.ctypes.data_as(PtD), \
					   cl.ctypes.data_as(PtD), \
					   cu.ctypes.data_as(PtD), \
					   equatn.ctypes.data_as(POINTER(c_bool)), \
					   linear.ctypes.data_as(POINTER(c_bool)), \
					   e_order, l_order, v_order)
	
	#x1 = np.random.randn(n[0], 1)
	#z1.CUTEST_cfn_1(status, n, m, \
	#		        x1.ctypes.data_as(PtD), f, \
	#				v.ctypes.data_as(PtD));
	
	
	g = np.zeros((n[0], 1))
	
	
	#z1.CUTEST_cofg_1(status, n, x1.ctypes.data_as(PtD), \
	#		f, g.ctypes.data_as(PtD), POINTER(c_bool)(c_bool(1)))
	
	#Get the sparsity of Jacobian
	global nnzj
	nnzj = PtI(c_int(0))
	
	z1.CUTEST_cdimsj_1(status, nnzj)
	
	grad = POINTER(c_bool)(c_bool(0))
	cjac = np.zeros((nnzj[0], 1))
	indvar = np.zeros((nnzj[0], 1), dtype = np.int32) #column
	indfun = np.zeros((nnzj[0], 1), dtype = np.int32) #row
	lj = PtI(c_int(nnzj[0]))
	


	f1 = func(x1, z1, n, m, status, lj, cjac, indvar, indfun)
	g1 = fprime(x1, z1, n, m, status, lj, cjac, indvar, indfun)
	c = fconstr(x1, z1, n, m, status, lj, cjac, indvar, indfun)	
	A = Jaconstr(x1, z1, n, m, status, lj, cjac, indvar, indfun)

	
	
	z1.CUTEST_cterminate_1(status)
	z1._close(funit)


if __name__ == "__main__":
	if len(sys.argv) == 1:
		print("Error, no folder specified.")
	print("Solving...")
	solve(sys.argv[1])
	print("Done. The whole thing works fine.")